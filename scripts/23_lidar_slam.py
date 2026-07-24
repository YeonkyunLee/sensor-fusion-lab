"""전체 2D LiDAR SLAM: ICP 스캔매칭 프론트엔드 + pose-graph 백엔드(루프클로저).

exp 21(ICP 스캔매칭 오도메트리)과 exp 07(SE(2) pose-graph 최적화)을 하나로 잇는
통합편이다. 프론트엔드가 연속 스캔을 정렬해 상대 이동을 뽑아 오도메트리 궤적을 만들지만,
스캔당 미세오차가 누적되며 여러 바퀴를 돌수록 궤적이 벌어진다(drift). 백엔드는
장소재인식으로 루프클로저를 찾아 이 드리프트를 한 번에 정렬한다.

파이프라인:
  1. 프론트엔드 — 연속 스캔 k, k+1에 ICP → 상대 SE(2) → 오도메트리 엣지(누적 시 드리프트).
  2. 장소재인식 — 로봇이 이전에 지난 곳 근처로 돌아오면(오도메트리 추정 위치 반경 검색)
     후보를 뽑고, 그 시점 스캔과 과거 스캔을 ICP로 정렬해 잔차가 작을 때만 채택 → 루프클로저 엣지.
  3. 백엔드 — 오도메트리 + 루프클로저 엣지를 SE(2) pose-graph(Gauss-Newton)에 넣어 전체 궤적 최적화.

세 궤적을 비교한다:
  - true              : 지상 진실(2바퀴 루프)
  - ICP odometry      : 스캔매칭 상대이동 누적(드리프트)
  - graph-optimized   : 루프클로저로 전체 정렬(드리프트 제거)

백엔드가 프론트엔드 드리프트를 뚜렷이 줄이고 루프를 닫는다.

    python scripts/23_lidar_slam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import Edge, optimize  # noqa: E402

SENSOR_RANGE = 18.0      # LiDAR 최대 사거리(m)
SCAN_NOISE = 0.05        # 스캔 점당 가우시안 잡음(m) — 스캔당 정렬 미세오차의 원천
KEEP_FRAC = 0.85         # pose마다 보이는 점 중 스캔에 담기는 비율(부분 관측)


# --------------------------------------------------------------------------- #
# SE(2) · ICP 프론트엔드 (exp 21에서 채택·자기완결)
# --------------------------------------------------------------------------- #
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def se2(x, y, th):
    """SE(2) 동차변환 행렬."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def se2_params(T):
    """SE(2) 행렬 → (x, y, theta)."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def transform(pts, T):
    """(N,2) 점군에 SE(2) T 적용."""
    return pts @ T[:2, :2].T + T[:2, 2]


def rel(a, b):
    """pose a 기준 pose b의 상대 pose (a→b)."""
    return se2_params(np.linalg.inv(se2(*a)) @ se2(*b))


def best_fit_transform(src, dst):
    """대응된 두 점군의 최적 강체변환(Umeyama/Kabsch)을 SVD로."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(S)
    D = np.diag([1.0, np.linalg.det(U @ Vt)])   # reflection 방지
    R = U @ D @ Vt
    t = mu_d - R @ mu_s
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def icp(src, dst, init=None, max_iter=60, tol=1e-6, max_corr_dist=5.0):
    """2D 점-대-점 ICP. src를 dst에 정렬하는 SE(2)를 추정. 반환 (T, mean_err, n_iter)."""
    T = np.eye(3) if init is None else init.copy()
    tree = cKDTree(dst)
    cur = transform(src, T)
    prev_err = np.inf
    n = 0
    for n in range(1, max_iter + 1):
        dist, idx = tree.query(cur)
        m = dist < max_corr_dist                # 대응 거리 게이트(outlier 제거)
        if m.sum() < 3:
            break
        dT = best_fit_transform(cur[m], dst[idx[m]])
        T = dT @ T
        cur = transform(src, T)
        err = float(dist[m].mean())
        if abs(prev_err - err) < tol:
            prev_err = err
            break
        prev_err = err
    return T, prev_err, n


# --------------------------------------------------------------------------- #
# 환경 · 스캔 · 궤적 시뮬레이션
# --------------------------------------------------------------------------- #
def build_environment():
    """방 윤곽(사각형 벽) + 흩뿌린 원형 기둥. 점-대-점 ICP에 이상적인 특징."""
    pts = []
    W, H, ds = 30.0, 22.0, 0.20
    for (x0, y0, x1, y1) in [(0, 0, W, 0), (W, 0, W, H), (W, H, 0, H), (0, H, 0, 0)]:
        n = int(np.hypot(x1 - x0, y1 - y0) / ds)
        pts.append(np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1))
    pillars = [(5, 5, 0.6), (5, 12, 0.5), (6, 18, 0.7), (11, 8, 0.5),
               (12, 16, 0.6), (15, 4, 0.7), (16, 11, 0.5), (18, 19, 0.6),
               (21, 7, 0.6), (22, 15, 0.5), (25, 5, 0.7), (26, 18, 0.6),
               (9, 3, 0.4), (24, 11, 0.5)]
    for (cx, cy, r) in pillars:
        n = max(14, int(2 * np.pi * r / ds))
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1))
    return np.vstack(pts)


def simulate_scan(env, pose, rng):
    """pose=(x,y,th)에서 사거리 내 환경 점의 잡음 섞인 스캔을 로봇 프레임으로 반환."""
    d = env - pose[:2]
    r = np.hypot(d[:, 0], d[:, 1])
    idx = np.where(r < SENSOR_RANGE)[0]
    if len(idx) == 0:
        return np.empty((0, 2))
    keep = rng.random(len(idx)) < KEEP_FRAC          # 부분 관측(occlusion 근사)
    idx = idx[keep]
    world = env[idx] + rng.normal(0, SCAN_NOISE, (len(idx), 2))
    R = se2(*pose)[:2, :2]
    return (world - pose[:2]) @ R                     # 월드 → 로봇 프레임


def make_lap():
    """방 안을 도는 둥근 사각형 한 바퀴(제어입력 v, w의 열). 시작 자세로 되돌아옴."""
    turn = [(0.9, (np.pi / 2) / 10)] * 10            # 90° 좌회전(둥근 모서리)
    c = []
    c += [(1.0, 0.0)] * 13                            # 바닥 직진
    c += turn
    c += [(1.0, 0.0)] * 8                             # 우측 직진
    c += turn
    c += [(1.0, 0.0)] * 13                            # 상단 직진
    c += turn
    c += [(1.0, 0.0)] * 8                             # 좌측 직진
    c += turn
    return c


def integrate(pose, v, w, dt=1.0):
    x, y, th = pose
    return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), wrap(th + w * dt)])


# --------------------------------------------------------------------------- #
# 프론트엔드 · 장소재인식 · 백엔드
# --------------------------------------------------------------------------- #
def run_frontend(seed=1, laps=2):
    """스캔 시뮬 + ICP 오도메트리. 반환: env, scans, true_poses, x_odo, odom_edges."""
    rng = np.random.default_rng(seed)
    env = build_environment()
    controls = make_lap() * laps                     # 여러 바퀴 → 드리프트 누적

    start = np.array([6.0, 5.0, 0.0])
    true_poses = [start.copy()]
    p = start.copy()
    for (v, w) in controls:
        p = integrate(p, v, w)
        true_poses.append(p.copy())
    true_poses = np.array(true_poses)
    scans = [simulate_scan(env, tp, rng) for tp in true_poses]

    # ICP 오도메트리: 연속 스캔 정렬 → 상대이동 → 누적 + 오도메트리 엣지
    odom_sigma = np.array([0.05, 0.05, 0.02])
    info_odo = np.diag(1.0 / odom_sigma ** 2)
    x_odo = [start.copy()]
    Tglob = se2(*start)
    edges = []
    for k in range(len(scans) - 1):
        src, dst = scans[k + 1], scans[k]            # 새 스캔을 이전 스캔에 정렬
        v, w = controls[k]
        rel0 = se2(v, 0.0, wrap(w))                  # dead-reckoning 초기값
        T_rel, _, _ = icp(src, dst, init=rel0, max_corr_dist=0.8)
        z = se2_params(T_rel)                        # k→k+1 상대 pose
        edges.append(Edge(k, k + 1, z, info_odo))
        Tglob = Tglob @ T_rel
        x_odo.append(se2_params(Tglob))
    return env, scans, true_poses, np.array(x_odo), edges


def detect_loop_closures(scans, x_odo, min_gap=25, radius=4.0,
                         err_thresh=0.15, max_corr_dist=1.0, step=2):
    """장소재인식 + ICP 검증. 반환: 루프클로저 엣지 리스트와 (j,k) 링크 리스트.

    장소재인식 후보는 오도메트리 추정 위치의 반경 검색으로 뽑고, 그 시점 스캔과 과거
    스캔을 ICP로 정렬해 (1) 잔차가 임계 미만이고 (2) 정렬된 상대 이동이 검색 반경 내로
    합당할 때만 채택한다. 두 조건이 거짓 클로저(잘못 미끄러진 정렬)를 걸러낸다.
    """
    positions = x_odo[:, :2]
    lc_sigma = np.array([0.04, 0.04, 0.02])          # ICP 검증 클로저(정밀 측정)
    info_lc = np.diag(1.0 / lc_sigma ** 2)
    edges, links = [], []
    N = len(scans)
    for k in range(min_gap, N, step):
        # 오도메트리 추정 위치 기준으로 min_gap 이상 과거의 근접 후보
        cand = [j for j in range(0, k - min_gap)
                if np.hypot(*(positions[k] - positions[j])) < radius]
        if not cand:
            continue
        best = None
        for j in cand:
            init = se2(*rel(x_odo[j], x_odo[k]))      # 오도메트리 초기값
            T, err, _ = icp(scans[k], scans[j], init=init, max_corr_dist=max_corr_dist)
            z = se2_params(T)
            if err < err_thresh and np.hypot(z[0], z[1]) < radius \
                    and (best is None or err < best[1]):
                best = (j, err, z)
        if best is not None:
            j, _, z = best
            edges.append(Edge(j, k, z, info_lc))       # j→k 상대 pose
            links.append((j, k))
    return edges, links


def rmse(est, truth):
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - truth[:, :2]) ** 2, axis=1))))


def main(seed=1, plot=True):
    env, scans, true_poses, x_odo, odom_edges = run_frontend(seed=seed, laps=2)
    lc_edges, links = detect_loop_closures(scans, x_odo)
    edges = odom_edges + lc_edges

    x_opt, hist = optimize(x_odo, edges, iters=40)

    odom_rmse = rmse(x_odo, true_poses)
    opt_rmse = rmse(x_opt, true_poses)
    chi2_before, chi2_after = hist[0], hist[-1]
    end_gap_odo = float(np.hypot(*(x_odo[-1, :2] - true_poses[-1, :2])))
    end_gap_opt = float(np.hypot(*(x_opt[-1, :2] - true_poses[-1, :2])))

    print("=== 23. 전체 2D LiDAR SLAM (ICP 프론트엔드 + pose-graph 백엔드) ===")
    print(f"pose {len(true_poses)}개, 스캔 {len(scans)}개(점/스캔 평균 "
          f"{np.mean([len(s) for s in scans]):.0f}), 2바퀴 루프")
    print(f"엣지: 오도메트리 {len(odom_edges)} + 루프클로저 {len(lc_edges)}")
    print(f"ICP 오도메트리    : 궤적 RMSE={odom_rmse:.3f} m,  종점오차={end_gap_odo:.3f} m")
    print(f"그래프 최적화 후  : 궤적 RMSE={opt_rmse:.3f} m,  종점오차={end_gap_opt:.3f} m")
    print(f"→ 드리프트를 {odom_rmse / max(opt_rmse, 1e-6):.1f}배 줄임")
    print(f"chi2: {chi2_before:.1f} → {chi2_after:.3g} ({len(hist)} iters)")

    if plot:
        fig = plt.figure(figsize=(14, 6.5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(env[:, 0], env[:, 1], ".", color="0.78", ms=2, label="map (LiDAR points)")
        # 루프클로저 링크(최적화 궤적 위에)
        for (j, k) in links:
            ax1.plot([x_opt[j, 0], x_opt[k, 0]], [x_opt[j, 1], x_opt[k, 1]],
                     "-", color="orange", lw=0.7, alpha=0.7, zorder=1)
        if links:
            ax1.plot([], [], "-", color="orange", lw=1.2,
                     label=f"loop closures ({len(links)})")
        ax1.plot(true_poses[:, 0], true_poses[:, 1], "g-", lw=2.4, label="true (2 laps)")
        ax1.plot(x_odo[:, 0], x_odo[:, 1], "r--", lw=1.5,
                 label=f"ICP odometry (RMSE {odom_rmse:.2f}m)")
        ax1.plot(x_opt[:, 0], x_opt[:, 1], "b-", lw=1.8,
                 label=f"graph-optimized (RMSE {opt_rmse:.2f}m)")
        ax1.plot(*true_poses[0, :2], "ko", ms=8, label="start")
        ax1.set_aspect("equal")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.set_title("Full 2D LiDAR SLAM: ICP front-end + pose-graph back-end")
        ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.semilogy(range(len(hist)), hist, "b.-", lw=1.5)
        ax2.set_title("Back-end convergence (Gauss-Newton)")
        ax2.set_xlabel("iteration"); ax2.set_ylabel(r"$\chi^2$ (log)")
        ax2.grid(alpha=0.3, which="both")
        ax2.text(0.5, 0.9, f"loop closures snap drift shut\n"
                           f"RMSE {odom_rmse:.2f}m → {opt_rmse:.2f}m",
                 transform=ax2.transAxes, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

        fig.suptitle("23. Full 2D LiDAR SLAM (scan-matching front-end + graph back-end)")
        fig.tight_layout()
        for d in ("outputs", "assets"):
            Path(d).mkdir(exist_ok=True)
            fig.savefig(Path(d) / "23_lidar_slam.png", dpi=130)
        plt.close(fig)
        print("\n[plot] outputs/23_lidar_slam.png, assets/23_lidar_slam.png")

    return odom_rmse, opt_rmse, len(lc_edges), chi2_before, chi2_after


if __name__ == "__main__":
    main()
