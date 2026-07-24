"""장애물 회피 MPC: 추종 최적화 안에 충돌회피 제약을 녹인다.

exp24의 MPC는 기준궤적을 액추에이터 한계 하에서 조밀하게 추종하지만, 경로 위에
장애물이 놓이면 그대로 뚫고 지나가 충돌한다(장애물을 모른다). exp20의 DWA는
반응적으로 장애물을 피하지만 매 스텝 속도공간을 격자 샘플링하는 지역계획기라
기준궤적 추종과 제약을 정면으로 다루지 못한다. exp26은 둘을 하나의 원리적
최적화로 통합한다: MPC의 지평 비용에 '각 예측 스텝에서 장애물 안전반경 밖에
머물라'는 항을 더해, 추종오차 + 제어노력 + 충돌회피를 동시에 최소화한다.

정식화: 제어열 u_{0..N-1}을 결정변수로 두고 유니사이클을 '완전 비선형'으로
롤아웃한다(exp24의 기준궤적 선형화와 달리, 크게 우회할 때도 예측 위치가 정확).
비용 = 추종 이차식 + 제어 이차식 + 장애물 소프트배리어. 배리어는 여유거리가
안전여유 아래로 떨어질 때만 켜지는 부드러운 힌지(0.5*beta*max(0, margin-clear)^2)
라 미분가능하고 항상 실현가능(하드제약과 달리 solver 비실현 위험이 없음)하다.
대신 안전거리를 '보장'하진 않고 강하게 유인하므로 beta로 유인강도를 조절한다.
비용의 해석적 그래디언트를 애드조인트(backprop)로 구해 L-BFGS-B로 빠르게 풀고,
|v|,|w| 절대한계와 첫 스텝 가속한계는 박스제약으로 건다(receding horizon).

움직이는 장애물은 지평 동안 등속으로 예측해(exp20처럼) 미래 위치를 회피에 반영한다.
비교: 동일 시나리오(장애물이 기준경로 위에 놓임)에서 장애물 무지 MPC(=exp24식)는
충돌하고, 장애물 인지 MPC는 부드럽게 우회 후 경로로 복귀함을 보인다.

    python scripts/26_mpc_obstacle.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DT = 0.1
N = 18                       # 예측 지평(스텝) — 회피는 여유롭게 앞을 봐야 하므로 exp24보다 길게
V_MAX, W_MAX = 2.6, 1.5      # 액추에이터 한계: 병진속도[m/s], 각속도[rad/s]
A_V, A_W = 4.0, 6.0          # 가속 한계: [m/s^2], [rad/s^2]

ROBOT_R = 0.3                # 로봇 반경 [m]
BETA = 200.0                 # 소프트배리어 유인강도
MARGIN = 0.6                 # 안전반경 위로 이만큼 더 떨어지려 함 [m]


# --------------------------------------------------------------------------
# 기준 궤적: exp24와 동일한 베르누이 8자(lemniscate). 등호길이 재샘플 후 등속.
# --------------------------------------------------------------------------
def make_reference(a=8.0, v_nom=2.0):
    phi = np.linspace(0, 2 * np.pi, 20000)
    den = 1 + np.sin(phi) ** 2
    x = a * np.cos(phi) / den
    y = a * np.sin(phi) * np.cos(phi) / den
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    nref = int((L / v_nom) / DT)
    snew = np.linspace(0, L, nref)
    xr = np.interp(snew, s, x)
    yr = np.interp(snew, s, y)
    th = np.unwrap(np.arctan2(np.gradient(yr), np.gradient(xr)))
    v = np.hypot(np.diff(xr), np.diff(yr)) / DT
    w = np.diff(th) / DT
    v = np.append(v, v[-1])
    w = np.append(w, w[-1])
    Xref = np.stack([xr, yr, th], axis=1)
    Uref = np.stack([v, w], axis=1)
    return Xref, Uref


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_unicycle(X, u):
    x, y, th = X
    v, w = u
    return np.array([x + v * np.cos(th) * DT,
                     y + v * np.sin(th) * DT,
                     th + w * DT])


# --------------------------------------------------------------------------
# 장애물: 등속 이동하는 원. pos(t) = p0 + vel*t, 안전반경 rs(=장애물반경+로봇반경).
# 여유거리 clearance = ||robot - obstacle|| - rs (음수면 충돌).
# --------------------------------------------------------------------------
class Obstacle:
    def __init__(self, p0, vel, r_obs):
        self.p0 = np.asarray(p0, float)
        self.vel = np.asarray(vel, float)
        self.r_obs = float(r_obs)
        self.rs = float(r_obs) + ROBOT_R      # 중심간 충돌 임계거리

    def pos(self, t):
        return self.p0 + self.vel * t


# --------------------------------------------------------------------------
# 비선형 MPC 지평 비용 + 해석적 그래디언트(애드조인트/backprop).
#   J = Σ 추종이차식 + Σ 제어이차식 + Σ 장애물 소프트배리어
# 결정변수 u_flat = [v0,w0, v1,w1, ...] (N*2).
# --------------------------------------------------------------------------
def _cost_grad(u_flat, x0, Xr_h, Ur_h, Qd, Qfd, Rd, obs, t0, beta, margin):
    u = u_flat.reshape(N, 2)

    # ---- forward rollout (완전 비선형) ----
    X = np.empty((N + 1, 3))
    X[0] = x0
    for k in range(N):
        X[k + 1] = step_unicycle(X[k], u[k])

    J = 0.0
    dcx = np.zeros((N + 1, 3))     # dJ/dX[k]
    dcu = np.zeros((N, 2))         # dJ/du[k]

    # ---- 추종 + 장애물 항 (k=1..N) ----
    for k in range(1, N + 1):
        Qk = Qfd if k == N else Qd
        err = X[k] - Xr_h[k]
        err[2] = wrap(err[2])
        J += 0.5 * float(err @ (Qk * err))
        dcx[k] += Qk * err

        # 장애물 소프트배리어
        t = t0 + k * DT
        for ob in obs:
            o = ob.pos(t)
            diff = X[k, :2] - o
            dist = np.hypot(diff[0], diff[1]) + 1e-9
            clear = dist - ob.rs
            if clear < margin:
                slack = margin - clear
                J += 0.5 * beta * slack * slack
                # dpen/dpos = -beta*slack * (diff/dist)
                g = (-beta * slack / dist) * diff
                dcx[k, 0] += g[0]
                dcx[k, 1] += g[1]

    # ---- 제어 항 (k=0..N-1) ----
    for k in range(N):
        eu = u[k] - Ur_h[k]
        J += 0.5 * float(eu @ (Rd * eu))
        dcu[k] += Rd * eu

    # ---- 애드조인트 backprop ----
    grad = np.zeros((N, 2))
    lam = dcx[N].copy()
    for k in range(N - 1, -1, -1):
        th = X[k, 2]
        v = u[k, 0]
        A = np.array([[1.0, 0.0, -v * np.sin(th) * DT],
                      [0.0, 1.0,  v * np.cos(th) * DT],
                      [0.0, 0.0, 1.0]])
        B = np.array([[np.cos(th) * DT, 0.0],
                      [np.sin(th) * DT, 0.0],
                      [0.0, DT]])
        grad[k] = dcu[k] + B.T @ lam
        lam = dcx[k] + A.T @ lam

    return J, grad.ravel()


def mpc_control(X, t, Xref, Uref, Qd, Qfd, Rd, obs, warm, u_prev, beta, margin):
    nref = len(Xref)
    idx = [min(t + k, nref - 1) for k in range(N + 1)]
    Xr_h = np.array([Xref[i] for i in idx])          # (N+1,3), Xr_h[0]=현재 기준
    Ur_h = np.array([Uref[idx[k]] for k in range(N)])  # (N,2)

    # 박스제약: |v|,|w| 절대한계 + 첫 스텝 가속한계
    bounds = []
    for k in range(N):
        lo_v, hi_v = -V_MAX, V_MAX
        lo_w, hi_w = -W_MAX, W_MAX
        if k == 0:
            lo_v = max(lo_v, u_prev[0] - A_V * DT)
            hi_v = min(hi_v, u_prev[0] + A_V * DT)
            lo_w = max(lo_w, u_prev[1] - A_W * DT)
            hi_w = min(hi_w, u_prev[1] + A_W * DT)
        bounds += [(lo_v, hi_v), (lo_w, hi_w)]

    res = minimize(_cost_grad, warm, args=(X, Xr_h, Ur_h, Qd, Qfd, Rd, obs, t * DT, beta, margin),
                   jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 40})
    u = res.x.reshape(N, 2)
    u0 = np.array([np.clip(u[0, 0], -V_MAX, V_MAX),
                   np.clip(u[0, 1], -W_MAX, W_MAX)])
    warm_next = np.vstack([u[1:], u[-1]]).ravel()     # 시프트 warm start
    return u0, warm_next


def run_mpc(Xref, Uref, obs, beta, margin):
    Qd = np.array([9.0, 9.0, 0.25])
    Qfd = np.array([14.0, 14.0, 0.35])
    Rd = np.array([0.25, 0.04])
    X = Xref[0].copy()
    traj = [X[:2].copy()]
    ctrl = []
    warm = np.tile(Uref[0], N)
    u_prev = Uref[0].copy()
    for t in range(len(Xref)):
        u, warm = mpc_control(X, t, Xref, Uref, Qd, Qfd, Rd, obs, warm, u_prev, beta, margin)
        X = step_unicycle(X, u)
        u_prev = u
        traj.append(X[:2].copy())
        ctrl.append(u)
    return np.array(traj), np.array(ctrl)


# --------------------------------------------------------------------------
# 평가: 실제(현재 시각) 장애물 여유거리 시계열, 최소 여유, 추종 RMSE.
# --------------------------------------------------------------------------
def clearance_series(traj, obs):
    """traj[i]는 스텝 i(시각 i*DT) 종료 위치. 각 스텝의 최소 여유거리."""
    out = np.empty(len(traj))
    for i, p in enumerate(traj):
        t = i * DT
        gaps = [np.hypot(*(p - ob.pos(t))) - ob.rs for ob in obs]
        out[i] = min(gaps)
    return out


def cross_track(traj, Xref):
    tree = cKDTree(Xref[:, :2])
    d, _ = tree.query(traj)
    return d


def main():
    Xref, Uref = make_reference()
    nref = len(Xref)

    # --- 장애물 배치: 기준경로 '위'에 놓아 무지 MPC가 정면충돌하게 함 ---
    # 로봇은 v_nom 등속 재샘플 경로를 대략 스텝당 1점씩 따라가므로, 기준인덱스 i에
    # 도달하는 시각 ~ i*DT. 움직이는 장애물은 그 도달시각에 경로점에 오도록 타이밍.
    i1 = int(0.17 * nref)
    i2 = int(0.50 * nref)
    i3 = int(0.80 * nref)
    pA = Xref[i1, :2]
    pB = Xref[i2, :2]
    # 움직이는 장애물 C: 경로를 가로질러 드리프트, t3=i3*DT에 Xref[i3] 통과
    t3 = i3 * DT
    velC = np.array([-0.35, 0.20])
    pC0 = Xref[i3, :2] - velC * t3
    obs = [
        Obstacle(pA, [0.0, 0.0], 0.55),          # 정적
        Obstacle(pB, [0.0, 0.0], 0.55),          # 정적
        Obstacle(pC0, velC, 0.55),               # 이동(지평 예측)
    ]

    # --- 장애물 인지 vs 무지(=exp24식, beta=0) MPC, 동일 시나리오 ---
    aware_traj, aware_u = run_mpc(Xref, Uref, obs, beta=BETA, margin=MARGIN)
    plain_traj, plain_u = run_mpc(Xref, Uref, obs, beta=0.0, margin=MARGIN)

    aware_clear = clearance_series(aware_traj, obs)
    plain_clear = clearance_series(plain_traj, obs)
    aware_min = float(aware_clear.min())
    plain_min = float(plain_clear.min())

    # 추종 품질: 장애물에서 충분히 먼(여유>1.2m) 구간의 교차추종 RMSE
    aware_ct = cross_track(aware_traj, Xref)
    far = aware_clear > 1.2
    rmse_far = float(np.sqrt(np.mean(aware_ct[far] ** 2)))
    rmse_all = float(np.sqrt(np.mean(aware_ct ** 2)))

    av_vmax, av_wmax = np.abs(aware_u[:, 0]).max(), np.abs(aware_u[:, 1]).max()
    dv = np.abs(np.diff(aware_u[:, 0])).max() / DT
    dw = np.abs(np.diff(aware_u[:, 1])).max() / DT
    limits_ok = (av_vmax <= V_MAX + 1e-6 and av_wmax <= W_MAX + 1e-6
                 and dv <= A_V + 1e-6 and dw <= A_W + 1e-6)
    collision_avoided = aware_min > 0.0

    print("=== 장애물 회피 MPC vs 장애물 무지 MPC(exp24식): 8자 궤적 + 장애물 3개 ===")
    print(f"기준궤적 {nref}점, 지평 N={N}, dt={DT}s, 한계 |v|<={V_MAX} |w|<={W_MAX}")
    print(f"장애물: 정적 2 + 이동 1(등속 예측), 안전반경 rs={obs[0].rs:.2f} m, 유인여유 {MARGIN} m")
    print(f"최소 여유거리   인지 MPC {aware_min:+.3f} m   |   무지 MPC {plain_min:+.3f} m")
    print(f"  -> 인지: {'충돌 없음' if aware_min > 0 else '충돌!'}   "
          f"무지: {'충돌 없음' if plain_min > 0 else '충돌(경로 관통)!'}")
    print(f"추종 RMSE(장애물 밖 구간) {rmse_far:.3f} m   (전체 {rmse_all:.3f} m)")
    print(f"인지 MPC 제어 한계  |v|={av_vmax:.3f}<= {V_MAX}, |w|={av_wmax:.3f}<= {W_MAX}, "
          f"|dv|={dv:.2f}<= {A_V}, |dw|={dw:.2f}<= {A_W}  "
          f"({'준수' if limits_ok else '위반!'})")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(13.5, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(Xref[:, 0], Xref[:, 1], "k--", lw=1.2, alpha=0.55, label="reference (figure-8)")
    ax.plot(plain_traj[:, 0], plain_traj[:, 1], color="#d9534f", lw=2.0, alpha=0.9,
            label=f"plain MPC / obstacle-unaware (min clear {plain_min:+.2f} m, collides)")
    ax.plot(aware_traj[:, 0], aware_traj[:, 1], color="#1f77b4", lw=2.2,
            label=f"obstacle-aware MPC (min clear {aware_min:+.2f} m, avoids)")

    labelled_c = labelled_s = False
    for ob in obs:
        moving = np.hypot(*ob.vel) > 1e-6
        if moving:
            # 이동 장애물: 경로선 + 시간 스냅샷
            ts = np.linspace(0, (nref - 1) * DT, 60)
            pts = np.array([ob.pos(tt) for tt in ts])
            ax.plot(pts[:, 0], pts[:, 1], ":", color="#8a6d3b", lw=1.0, alpha=0.6)
            for si, tt in enumerate(np.linspace(0, (nref - 1) * DT, 5)):
                a = 0.12 + 0.5 * si / 4
                lab = "moving obstacle (predicted)" if (si == 4 and not labelled_c) else None
                ax.add_patch(plt.Circle(ob.pos(tt), ob.rs, color="#8a6d3b", alpha=a, label=lab))
                labelled_c = labelled_c or lab is not None
        else:
            lab = "static obstacle (safety radius)" if not labelled_s else None
            ax.add_patch(plt.Circle(ob.p0, ob.rs, color="#555555", alpha=0.35, label=lab))
            ax.add_patch(plt.Circle(ob.p0, ob.rs + MARGIN, color="#555555", alpha=0.12, ls="--", fill=False))
            labelled_s = labelled_s or lab is not None

    ax.plot(*Xref[0, :2], "ko", ms=8, label="start")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("Obstacle-aware MPC bends around on-path obstacles;\nplain MPC drives straight through", fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)

    tvec = np.arange(len(aware_clear)) * DT
    axc = fig.add_subplot(gs[0, 1])
    axc.plot(tvec, plain_clear, color="#d9534f", lw=1.6, label="plain MPC")
    axc.plot(tvec, aware_clear, color="#1f77b4", lw=1.8, label="obstacle-aware")
    axc.axhline(0.0, color="k", ls="-", lw=1.0)
    axc.axhline(MARGIN, color="gray", ls=":", lw=1.0)
    axc.fill_between(tvec, -2, 0, color="#d9534f", alpha=0.08)
    axc.set_ylim(min(-1.0, plain_min - 0.2), max(2.5, aware_clear.max() * 0.6))
    axc.set_ylabel("clearance [m]")
    axc.set_title("Clearance over time (0 = collision boundary)", fontsize=9)
    axc.legend(fontsize=7, loc="upper right")
    axc.grid(alpha=0.2)

    axu = fig.add_subplot(gs[1, 1])
    tu = np.arange(len(aware_u)) * DT
    axu.plot(tu, aware_u[:, 0], color="#1f77b4", label="v [m/s]")
    axu.plot(tu, aware_u[:, 1], color="#2ca02c", label="w [rad/s]")
    axu.axhline(V_MAX, color="gray", ls=":", lw=1)
    axu.axhline(-W_MAX, color="gray", ls=":", lw=1)
    axu.axhline(W_MAX, color="gray", ls=":", lw=1)
    axu.set_ylabel("control")
    axu.set_xlabel("time [s]")
    axu.set_title("Obstacle-aware controls within |v|,|w| limits", fontsize=9)
    axu.legend(fontsize=7, loc="lower right")
    axu.grid(alpha=0.2)

    fig.tight_layout()
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(f"{d}/26_mpc_obstacle.png", dpi=130)
    print("\n[plot] outputs/26_mpc_obstacle.png, assets/26_mpc_obstacle.png")

    return {
        "collision_avoided": collision_avoided,
        "aware_min_clearance": aware_min,
        "plain_min_clearance": plain_min,
        "tracking_rmse_far": rmse_far,
        "limits_ok": limits_ok,
    }


if __name__ == "__main__":
    main()
