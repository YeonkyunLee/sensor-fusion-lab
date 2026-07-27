"""Monte Carlo Localization (파티클 필터): 비모수 베이즈 필터로 자기위치추정.

(E)KF는 믿음(belief)을 하나의 가우시안 pose로 표현한다. 반면 파티클 필터(MCL)는
믿음을 **가중치가 달린 수많은 파티클(가설 pose)의 집합**으로 표현하는 비모수 필터다.
가우시안이 아니어도, 봉우리가 여러 개인 다봉(multimodal) 분포여도 그대로 담을 수 있다.

거리(range)만 관측하는 문제는 파티클 필터의 강점을 잘 드러낸다. 랜드마크 하나까지의
거리 하나는 그 랜드마크를 중심으로 한 '원'을 만들어 위치가 다봉으로 애매해진다.
EKF는 이 애매함을 하나의 가우시안으로 뭉개 엉뚱한 봉우리에 갇히기 쉽지만, MCL은
여러 랜드마크 관측과 이동을 거치며 파티클 구름이 참 위치로 수렴한다.

MCL 한 스텝:
  (1) 이동 갱신  : 각 파티클을 잡음 섞인 이동모델로 전파(예측)
  (2) 관측 갱신  : 각 파티클을 측정 우도로 가중(거리 오차가 작을수록 높은 가중치)
  (3) 리샘플링   : 저분산(systematic) 리샘플링으로 가중치 비례 복제 + 소량 roughening
  (4) 추정       : 가중 평균 pose(+ 파티클 구름으로 불확실도 시각화)

전역 위치추정(global/kidnapped-robot): 초기 pose를 모른 채 파티클을 지도 전체에
퍼뜨려 시작해도 관측이 쌓이며 참 위치로 수렴한다 — EKF가 못 하는 MCL의 대표 강점.

    python scripts/36_particle_filter.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
SENSOR_RANGE = 22.0        # 거리센서 유효반경[m] — 초기엔 1~2개만 관측(거리전용 → 다봉)
R_STD = 0.6                # 실제 거리 관측 잡음[m]
R_FILTER = 2.5             # 필터 우도용 거리 잡음(실제보다 크게) — 첫 갱신에서 파티클
                           #   고갈(전멸) 방지: 우도를 넓혀 더 많은 파티클을 살림
V_STD, W_STD = 0.12, 0.04  # 오도메트리(제어) 잡음: 선속도[m/s], 각속도[rad/s]

WORLD = (100.0, 80.0)      # 지도 크기(x,y)
N_PARTICLES = 3000         # 파티클 수 (많을수록 안정, 대신 계산비용↑)
# roughening 하한[m,m,rad]: 리샘플 후 최소 지터를 보장해 파티클 다양성 유지(고갈 방지)
ROUGH_FLOOR = np.array([0.5, 0.5, 0.05])
CONV_THRESH = 3.0          # 전역 위치추정 '수렴' 판정 오차[m]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def motion(pose, v, w):
    """단륜(unicycle) 이동모델. pose=(x,y,theta)."""
    x, y, th = pose[..., 0], pose[..., 1], pose[..., 2]
    out = np.empty_like(pose)
    out[..., 0] = x + v * DT * np.cos(th)
    out[..., 1] = y + v * DT * np.sin(th)
    out[..., 2] = wrap(th + w * DT)
    return out


def make_world():
    """알려진 지도(랜드마크)와 참 궤적/제어 시퀀스를 만든다(결정적)."""
    landmarks = np.array([
        [10, 12], [30, 8], [55, 14], [80, 10], [92, 30],
        [78, 52], [88, 70], [60, 68], [40, 74], [18, 66],
        [8, 44], [26, 40], [50, 44], [70, 34], [44, 22],
    ], float)

    # 지도 안을 도는 폐곡선: 직진 4구간 + 90° 좌선회 4번(모서리 둥근 사각 루프)
    V = 6.0
    W = 0.5               # 각속도[rad/s] → 스텝당 0.05rad, 31스텝 ≈ 90°
    straight, turn = 60, 31
    controls = []
    for _ in range(4):
        controls += [(V, 0.0)] * straight
        controls += [(V, W)] * turn
    return landmarks, controls


def simulate(landmarks, controls, seed):
    """참 궤적을 굴리며 스텝별 (참pose, 잡음제어, 관측[idx,range])을 생성."""
    rng = np.random.default_rng(seed)
    start = np.array([18.0, 15.0, 0.0])   # 개활지 시작(초기 관측 1~2개 → 다봉 애매성)
    true_pose = start.copy()
    truth, noisy_ctrl, obs = [], [], []
    for (v, w) in controls:
        true_pose = motion(true_pose, v, w)
        v_n = v + rng.normal(0, V_STD)
        w_n = w + rng.normal(0, W_STD)
        d = landmarks - true_pose[:2]
        r = np.hypot(d[:, 0], d[:, 1])
        vis = np.where(r <= SENSOR_RANGE)[0]
        z = r[vis] + rng.normal(0, R_STD, size=vis.size)
        truth.append(true_pose.copy())
        noisy_ctrl.append((v_n, w_n))
        obs.append((vis, z))
    return start, np.array(truth), noisy_ctrl, obs


def systematic_resample(weights, rng):
    """저분산(systematic) 리샘플링: 균등 간격 포인터 하나로 O(N) 복제."""
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def run_mcl(landmarks, start, noisy_ctrl, obs, mode, seed):
    """MCL 실행. mode='track'(초기 pose 대략 앎) 또는 'global'(전역, 초기 pose 모름).

    반환: (추정궤적[N,3], 스텝별 파티클 스냅샷 dict, 스텝별 추정오차[N])
    """
    rng = np.random.default_rng(seed)
    N = N_PARTICLES

    if mode == "track":
        # 추적: 참 시작점 근처에 좁게 뿌림
        P = np.empty((N, 3))
        P[:, 0] = start[0] + rng.normal(0, 2.0, N)
        P[:, 1] = start[1] + rng.normal(0, 2.0, N)
        P[:, 2] = wrap(start[2] + rng.normal(0, 0.3, N))
    else:
        # 전역(kidnapped): 지도 전체 + 모든 방향에 균등 분포 — 초기 pose 정보 없음
        P = np.empty((N, 3))
        P[:, 0] = rng.uniform(0, WORLD[0], N)
        P[:, 1] = rng.uniform(0, WORLD[1], N)
        P[:, 2] = rng.uniform(-np.pi, np.pi, N)

    w = np.full(N, 1.0 / N)
    snap_steps = {-1: P.copy()}         # -1 = 갱신 전 초기 구름(전역이면 지도 전체)
    want_snaps = [4, 8, len(obs) - 1]   # 수렴 과정(붕괴 중 → 갓 수렴 → 최종)
    est_traj = []

    for t, ((v_n, w_n), (vis, z)) in enumerate(zip(noisy_ctrl, obs)):
        # (1) 이동 갱신: 파티클마다 잡음 섞은 제어로 전파
        v_p = v_n + rng.normal(0, V_STD, N)
        w_p = w_n + rng.normal(0, W_STD, N)
        P = motion(P, v_p, w_p)

        # (2) 관측 갱신: 거리 우도로 로그가중 누적(언더플로 회피)
        if vis.size > 0:
            logw = np.log(w + 1e-300)
            lp = landmarks[vis]                                   # (K,2)
            dx = P[:, 0:1] - lp[:, 0][None, :]
            dy = P[:, 1:2] - lp[:, 1][None, :]
            r_pred = np.hypot(dx, dy)                             # (N,K)
            logw += np.sum(-0.5 * ((z[None, :] - r_pred) / R_FILTER) ** 2, axis=1)
            logw -= logw.max()
            w = np.exp(logw)
            w /= w.sum()

        # 추정: 가중 평균(각도는 원형 평균)
        mean_xy = w @ P[:, :2]
        mean_th = np.arctan2(w @ np.sin(P[:, 2]), w @ np.cos(P[:, 2]))
        est = np.array([mean_xy[0], mean_xy[1], mean_th])
        est_traj.append(est)

        # (3) 리샘플링: Neff가 낮을 때만(다양성 유지)
        neff = 1.0 / np.sum(w ** 2)
        if neff < N / 2:
            idx = systematic_resample(w, rng)
            P = P[idx]
            w = np.full(N, 1.0 / N)
            # roughening: 파티클 고갈 방지용 지터. 상태 폭 비례 + 절대 하한(붕괴 방지)
            spread = P.max(axis=0) - P.min(axis=0)
            sigma = np.maximum(0.05 * spread * N ** (-1.0 / 3.0), ROUGH_FLOOR)
            P = P + rng.normal(0, 1.0, P.shape) * sigma
            P[:, 2] = wrap(P[:, 2])

        if t in want_snaps:
            snap_steps[t] = P.copy()

    est_traj = np.array(est_traj)
    return est_traj, snap_steps


def run_ekf_rangeonly(landmarks, init_pose, noisy_ctrl, obs):
    """같은 거리전용 문제를 EKF로(초기 pose는 대략 앎). 국소 추적엔 EKF도 잘 동작 —
    EKF의 한계는 정확도가 아니라, 초기 사전믿음(prior)이 반드시 있어야 한다는 점이다.
    전역/다봉 믿음을 단일 가우시안으로는 표현할 수 없다(아래 ring 데모)."""
    x = init_pose.astype(float).copy()
    Pcov = np.diag([4.0, 4.0, 0.2])
    Q = np.diag([(2 * V_STD) ** 2, (2 * V_STD) ** 2, (2 * W_STD) ** 2])
    est = []
    for (v_n, w_n), (vis, z) in zip(noisy_ctrl, obs):
        th = x[2]
        x = motion(x, v_n, w_n)
        F = np.array([[1, 0, -v_n * DT * np.sin(th)],
                      [0, 1,  v_n * DT * np.cos(th)],
                      [0, 0, 1]])
        Pcov = F @ Pcov @ F.T + Q
        for j, zr in zip(vis, z):
            dx, dy = x[0] - landmarks[j, 0], x[1] - landmarks[j, 1]
            r = np.hypot(dx, dy)
            if r < 1e-3:
                continue
            H = np.array([[dx / r, dy / r, 0.0]])
            S = H @ Pcov @ H.T + R_STD ** 2
            K = (Pcov @ H.T) / S
            x = x + (K[:, 0] * (zr - r))
            x[2] = wrap(x[2])
            Pcov = (np.eye(3) - K @ H) @ Pcov
        est.append(x.copy())
    return np.array(est)


def rmse(est_xy, true_xy):
    return float(np.sqrt(np.mean(np.sum((est_xy - true_xy) ** 2, axis=1))))


def ring_demo(seed=0):
    """단일 거리 관측 하나 → '고리(ring)' 모양 사후분포. 가우시안(EKF)로는 표현 불가.

    파티클을 균등하게 뿌리고 랜드마크까지의 거리 하나로만 가중/리샘플하면, 파티클은
    그 거리를 반지름으로 하는 원 위에 남는다(다봉·비가우시안). 파티클 필터는 이 믿음을
    그대로 담지만, 평균+공분산 하나로 근사하는 EKF는 원의 '빈 중심'에 확률을 몰아준다.
    """
    rng = np.random.default_rng(seed)
    lm = np.array([0.0, 0.0])
    r0 = 20.0                                   # 참 거리(관측)
    N = 4000
    P = rng.uniform(-28, 28, size=(N, 2))
    d = np.hypot(P[:, 0] - lm[0], P[:, 1] - lm[1])
    w = np.exp(-0.5 * ((r0 - d) / 1.0) ** 2)
    w /= w.sum()
    idx = systematic_resample(w, rng)
    P = P[idx] + rng.normal(0, 0.3, size=(N, 2))   # 소량 roughening
    gmean = P.mean(axis=0)
    gcov = np.cov(P.T)
    return lm, r0, P, gmean, gcov


def main(plot: bool = True):
    landmarks, controls = make_world()
    start, truth, noisy_ctrl, obs = simulate(landmarks, controls, seed=1)
    true_xy = truth[:, :2]

    # 순수 오도메트리(dead reckoning): 시작 pose는 알되 관측 미사용 → 드리프트
    odo = start.copy()
    odo_traj = []
    for (v_n, w_n) in noisy_ctrl:
        odo = motion(odo, v_n, w_n)
        odo_traj.append(odo.copy())
    odo_traj = np.array(odo_traj)
    odo_rmse = rmse(odo_traj[:, :2], true_xy)

    # MCL 추적(초기 pose 대략 앎)
    mcl_track, snaps_track = run_mcl(landmarks, start, noisy_ctrl, obs, "track", seed=7)
    mcl_rmse = rmse(mcl_track[:, :2], true_xy)

    # MCL 전역 위치추정(kidnapped): 지도 전체 스프레드 → 수렴
    mcl_glob, snaps_glob = run_mcl(landmarks, start, noisy_ctrl, obs, "global", seed=7)
    glob_err = np.linalg.norm(mcl_glob[:, :2] - true_xy, axis=1)
    below = glob_err < CONV_THRESH
    conv_step = int(np.argmax(below)) if below.any() else len(glob_err)
    # 수렴 후 구간 RMSE(정착 성능)
    glob_rmse_after = rmse(mcl_glob[conv_step:, :2], true_xy[conv_step:]) if conv_step < len(truth) else float("nan")

    # EKF(거리전용, 올바른 초기 pose) — 국소 추적은 EKF도 정상 동작
    ekf = run_ekf_rangeonly(landmarks, start.copy(), noisy_ctrl, obs)
    ekf_rmse = rmse(ekf[:, :2], true_xy)

    print("=== Monte Carlo Localization (파티클 필터) 결과 ===")
    print(f"파티클 수                       : {N_PARTICLES}")
    print(f"순수 오도메트리 RMSE            : {odo_rmse:6.3f} m  (dead reckoning, 관측 미사용)")
    print(f"MCL 추적 RMSE                   : {mcl_rmse:6.3f} m  → 오도메트리 대비 {odo_rmse/mcl_rmse:.1f}배 개선")
    print(f"EKF 추적 RMSE(올바른 초기 pose) : {ekf_rmse:6.3f} m  (국소 추적은 EKF도 OK)")
    print(f"MCL 전역(kidnapped) 수렴 스텝   : {conv_step} 스텝에서 <{CONV_THRESH}m 도달")
    print(f"MCL 전역 수렴후 RMSE            : {glob_rmse_after:6.3f} m")
    print("→ EKF는 정확도가 아니라 '초기 사전믿음 필요 + 단일 가우시안'이 한계:")
    print("  거리전용 단일 관측의 고리형(다봉) 믿음을 EKF는 표현 못 함(ring 데모 참조).")

    if plot:
        _plot(landmarks, truth, odo_traj, mcl_track, mcl_glob, snaps_glob,
              glob_err, conv_step, odo_rmse, mcl_rmse, ekf_rmse, ekf)

    return odo_rmse, mcl_rmse, conv_step


def _cov_ellipse(ax, mean, cov, n_std, **kw):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(t), np.sin(t)])
    pts = (vecs @ (n_std * np.sqrt(vals)[:, None] * circle)) + mean[:, None]
    ax.plot(pts[0], pts[1], **kw)


def _plot(landmarks, truth, odo_traj, mcl_track, mcl_glob, snaps_glob,
          glob_err, conv_step, odo_rmse, mcl_rmse, ekf_rmse, ekf):
    true_xy = truth[:, :2]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (1) 지도 + 국소 추적: true vs MCL vs EKF vs 오도메트리
    ax = axes[0, 0]
    ax.plot(landmarks[:, 0], landmarks[:, 1], "k*", ms=14, label="landmarks (known map)")
    ax.plot(true_xy[:, 0], true_xy[:, 1], "g-", lw=2.4, label="true trajectory")
    ax.plot(mcl_track[:, 0], mcl_track[:, 1], "b-", lw=1.5, label=f"MCL ({mcl_rmse:.2f}m)")
    ax.plot(ekf[:, 0], ekf[:, 1], color="orange", lw=1.3, alpha=0.9, label=f"EKF, good init ({ekf_rmse:.2f}m)")
    ax.plot(odo_traj[:, 0], odo_traj[:, 1], "r-", lw=1.2, alpha=0.6, label=f"odometry only ({odo_rmse:.1f}m)")
    ax.set_title("Range-only tracking: MCL & EKF both beat dead-reckoning\n(with a prior, EKF tracks fine — its limit is elsewhere)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # (2) 전역 위치추정: 초기 스프레드 → 수렴 (파티클 구름 스냅샷)
    ax = axes[0, 1]
    ax.plot(landmarks[:, 0], landmarks[:, 1], "k*", ms=14)
    ax.plot(true_xy[:, 0], true_xy[:, 1], "g-", lw=2.0, label="true trajectory", zorder=5)
    keys = sorted(snaps_glob.keys())
    colors = ["#cccccc", "#ff9999", "#9999ff", "#0000cc"]
    for c, k in zip(colors, keys):
        P = snaps_glob[k]
        lbl = "initial spread (whole map)" if k < 0 else f"particles @ step {k}"
        ax.scatter(P[:, 0], P[:, 1], s=4, c=c, alpha=0.35, label=lbl)
    ax.set_title(f"Global (kidnapped-robot) localization: no initial pose\n"
                 f"belief collapses from whole map to true pose by step {conv_step}")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # (3) 수렴 곡선: MCL 전역 오차 vs 순수 오도메트리 오차
    ax = axes[1, 0]
    odo_err = np.linalg.norm(odo_traj[:, :2] - true_xy, axis=1)
    ax.plot(glob_err, "b-", lw=1.8, label="MCL global: estimate error")
    ax.plot(odo_err, "r-", lw=1.2, alpha=0.7, label="odometry-only error (grows)")
    ax.axhline(CONV_THRESH, color="k", ls="--", lw=1, label=f"converged (<{CONV_THRESH}m)")
    ax.axvline(conv_step, color="g", ls=":", lw=1.5, label=f"convergence @ step {conv_step}")
    ax.set_yscale("log")
    ax.set_title("MCL converges from map-wide ambiguity, then stays locked on")
    ax.set_xlabel("step"); ax.set_ylabel("position error [m] (log)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    # (4) 다봉 믿음 데모: 단일 거리 관측 → 고리형 사후분포 (가우시안 불가)
    ax = axes[1, 1]
    lm, r0, Pr, gmean, gcov = ring_demo(seed=0)
    ax.scatter(Pr[:, 0], Pr[:, 1], s=5, c="#1f77b4", alpha=0.25, label="particles (PF belief)")
    ax.plot(lm[0], lm[1], "k*", ms=16, label="landmark")
    _cov_ellipse(ax, gmean, gcov, 2.0, color="orange", lw=2.2, label="single Gaussian (EKF belief), 2σ")
    ax.plot(gmean[0], gmean[1], "x", color="orange", ms=10)
    ax.set_title(f"One range measurement (r={r0:.0f}m) → ring-shaped, non-Gaussian belief\n"
                 "PF represents it exactly; a Kalman filter cannot")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    fig.suptitle("Monte Carlo Localization (Particle Filter): nonparametric, multimodal belief",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for p in ("outputs/36_particle_filter.png", "assets/36_particle_filter.png"):
        Path(p).parent.mkdir(exist_ok=True)
        fig.savefig(p, dpi=125, bbox_inches="tight")
    print("\n[plot] outputs/36_particle_filter.png, assets/36_particle_filter.png")


if __name__ == "__main__":
    main()
