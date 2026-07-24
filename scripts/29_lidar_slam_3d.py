"""전체 3D LiDAR SLAM: 점-대-평면 ICP 프론트엔드 + SE(3) pose-graph 백엔드.

exp 23(2D LiDAR SLAM)을 3차원으로 끌어올린 통합편이다. 실전 드론·다리로봇은 6-DOF로
움직이고, 3D LiDAR는 방 벽·바닥·천장·기둥의 표면 점군을 훑는다. 프론트엔드는 연속
스캔을 정렬해 상대 SE(3) 이동을 뽑아 오도메트리 궤적을 만들지만, 스캔당 미세오차가
누적되며 궤적이 벌어진다(drift). 백엔드는 재방문(루프클로저)을 찾아 이 드리프트를
SE(3) pose-graph 최적화로 한 번에 정렬한다.

파이프라인:
  1. 프론트엔드 — 연속 스캔 k, k+1에 점-대-평면 ICP → 상대 SE(3) → 오도메트리 엣지.
     대상 스캔의 국소 표면 법선을 k-NN PCA(공분산 최소고유벡터)로 추정하고, 점-대-평면
     잔차 n·(Tp - q)를 se(3) 접공간에서 선형최소제곱으로 최소화(Low 2004 선형화)한다.
  2. 장소재인식 — 오도메트리 추정 위치의 반경 검색으로 과거 근접 후보를 뽑고, 현 스캔과
     과거 스캔을 ICP로 정렬해 평면잔차가 임계 미만이고 상대이동이 합당할 때만 채택.
  3. 백엔드 — 오도메트리 + 루프클로저 엣지를 SE(3) pose-graph(Gauss-Newton, 매니폴드)에
     넣어 전체 6-DOF 궤적을 최적화(posegraph3d.optimize_se3 재사용).

세 궤적을 비교한다:
  - true            : 지상 진실(기울어진 원 3바퀴)
  - ICP odometry    : 스캔매칭 상대이동 누적(3D 드리프트)
  - graph-optimized : 루프클로저로 전체 정렬(드리프트 제거)

점-대-평면은 벽·바닥이 서로 다른 방향의 법선을 주어 6-DOF가 잘 관측될 때 점-대-점보다
빠르고 정확히 수렴한다. 잔차·법선·3D 관측성의 한계는 모듈 하단 주석에 정리했다.

    python scripts/29_lidar_slam_3d.py
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

from sensor_fusion.posegraph3d import Edge3D, optimize_se3  # noqa: E402
from sensor_fusion.se3 import se3_inv, se3_log, so3_exp  # noqa: E402

SENSOR_RANGE = 16.0      # 3D LiDAR 최대 사거리(m)
SCAN_NOISE = 0.05        # 스캔 점당 가우시안 잡음(m) — 스캔당 정렬 미세오차의 원천
KEEP_FRAC = 0.50         # 스캔에 담기는 가시점 비율(부분 관측/occlusion 근사)


# --------------------------------------------------------------------------- #
# SE(3) 유틸 · 점군 변환
# --------------------------------------------------------------------------- #
def apply_T(T, pts):
    """(N,3) 점군에 4x4 SE(3) T 적용 (robot→world 또는 정렬용)."""
    return pts @ T[:3, :3].T + T[:3, 3]


def pose_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --------------------------------------------------------------------------- #
# 점-대-평면 ICP 프론트엔드 (3D, se(3) 접공간 선형화)
# --------------------------------------------------------------------------- #
def estimate_normals(pts, k=12):
    """국소 k-NN PCA로 표면 법선 추정 — 공분산 최소고유벡터가 법선. (N,3) 반환."""
    n = len(pts)
    if n < k + 1:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)          # (N,k) 이웃 인덱스(자기 포함)
    nb = pts[idx]                          # (N,k,3)
    c = nb - nb.mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", c, c) / k   # (N,3,3) 국소 공분산
    _, V = np.linalg.eigh(C)              # 오름차순 고유값
    return V[:, :, 0]                     # 최소고유값 고유벡터 = 법선


def point_to_plane_icp(src, dst, dst_normals=None, init=None,
                       max_iter=40, tol=1e-5, max_corr_dist=1.5):
    """3D 점-대-평면 ICP. src를 dst에 정렬하는 SE(3)를 추정. 반환 (T, rms_plane, n_iter).

    각 반복: 현재 T로 옮긴 src의 최근접 대응을 찾고, 점-대-평면 잔차
    r_i = n_i·(t_i - q_i)를 se(3) 증분 δ=[ω, ν]로 선형화한다. 옮긴 점 q에 대해
    T'q ≈ q + ω×q + ν 이므로 자코비안 행은 [q×n, n], 정규방정식 (JᵀJ)δ = Jᵀr 를 풀어
    ΔT=[exp(ω̂), ν]를 좌곱 retraction(T ← ΔT·T)한다.
    """
    T = np.eye(4) if init is None else init.copy()
    tree = cKDTree(dst)
    if dst_normals is None:
        dst_normals = estimate_normals(dst)
    prev = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        q = apply_T(T, src)
        dist, idx = tree.query(q)
        m = dist < max_corr_dist                  # 대응 거리 게이트(outlier 제거)
        if m.sum() < 12:
            break
        qm = q[m]
        tm = dst[idx[m]]
        nm = dst_normals[idx[m]]
        cxn = np.cross(qm, nm)                     # (M,3)
        J = np.hstack([cxn, nm])                  # (M,6), 순서 [ω, ν]
        r = np.einsum("ij,ij->i", nm, tm - qm)    # n·(t - q)
        A = J.T @ J
        b = J.T @ r
        delta = np.linalg.solve(A + 1e-9 * np.eye(6), b)
        omega, nu = delta[:3], delta[3:]
        dT = pose_T(so3_exp(omega), nu)
        T = dT @ T
        err = float(np.sqrt(np.mean(r ** 2)))     # RMS 평면잔차
        if abs(prev - err) < tol:
            prev = err
            break
        prev = err
    return T, prev, it


# --------------------------------------------------------------------------- #
# 3D 환경 · 스캔 · 궤적 시뮬레이션
# --------------------------------------------------------------------------- #
def build_environment(ds=0.7):
    """방(벽 4면 + 바닥 + 천장) + 몇 개의 기둥/박스. 서로 다른 방향의 법선을 주어
    점-대-평면 ICP의 6-DOF 관측성을 확보한다. (N,3) 표면 점군 반환."""
    W, H, D = 30.0, 22.0, 8.0
    pts = []

    def grid(a_rng, b_rng, fixed_axis, fixed_val):
        a = np.arange(0, a_rng + 1e-9, ds)
        b = np.arange(0, b_rng + 1e-9, ds)
        ga, gb = np.meshgrid(a, b)
        ga, gb = ga.ravel(), gb.ravel()
        f = np.full_like(ga, fixed_val)
        cols = {0: (f, ga, gb), 1: (ga, f, gb), 2: (ga, gb, f)}[fixed_axis]
        return np.stack(cols, axis=1)

    # 바닥/천장 (법선 ±z)
    pts.append(grid(W, H, 2, 0.0))
    pts.append(grid(W, H, 2, D))
    # 벽 x=0, x=W (법선 ±x)
    pts.append(grid(H, D, 0, 0.0))
    pts.append(grid(H, D, 0, W))
    # 벽 y=0, y=H (법선 ±y)
    pts.append(grid(W, D, 1, 0.0))
    pts.append(grid(W, D, 1, H))

    # 실내 기둥(수직 원기둥) — 국소 특징으로 정렬 보강
    for (cx, cy, r) in [(9, 7, 0.7), (10, 16, 0.6), (20, 8, 0.7),
                        (21, 15, 0.6), (15, 11, 0.8)]:
        na = max(10, int(2 * np.pi * r / ds))
        nz = int(D / ds)
        a = np.linspace(0, 2 * np.pi, na, endpoint=False)
        zc = np.linspace(0.3, D - 0.3, nz)
        aa, zz = np.meshgrid(a, zc)
        aa, zz = aa.ravel(), zz.ravel()
        pts.append(np.stack([cx + r * np.cos(aa), cy + r * np.sin(aa), zz], axis=1))

    # 낮은 박스 2개(윗면 + 옆면) — 천장·바닥 외 z 특징
    for (x0, y0, x1, y1, h) in [(24, 3, 27, 6, 2.5), (3, 17, 6, 20, 3.0)]:
        xs = np.arange(x0, x1 + 1e-9, ds)
        ys = np.arange(y0, y1 + 1e-9, ds)
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.stack([gx.ravel(), gy.ravel(),
                             np.full(gx.size, h)], axis=1))   # 윗면
        for (a_rng, ax, av) in [(ys, 0, x0), (ys, 0, x1)]:
            zz = np.arange(0, h + 1e-9, ds)
            ga, gz = np.meshgrid(a_rng, zz)
            pts.append(np.stack([np.full(ga.size, av), ga.ravel(), gz.ravel()], axis=1))

    return np.vstack(pts)


def make_trajectory(per_lap=60, laps=2, tilt=0.32, radius=7.0):
    """기울어진 평면 위의 원을 여러 바퀴 — 같은 3D 경로를 재방문하므로 루프클로저 성립.
    각 pose는 접선 방향을 향하는 4x4 SE(3). (list[4x4], per_lap) 반환."""
    center = np.array([15.0, 11.0, 4.0])
    ct, st = np.cos(tilt), np.sin(tilt)
    Rtilt = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    n = per_lap * laps
    poses = []
    for kk in range(n + 1):
        t = 2 * np.pi * (kk % per_lap) / per_lap
        p = Rtilt @ np.array([radius * np.cos(t), radius * np.sin(t), 0.0]) + center
        yaw = t + np.pi / 2                       # 진행 접선 방향
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        poses.append(pose_T(Rtilt @ Rz, p))
    return poses, per_lap


def simulate_scan(env, T_pose, rng):
    """T_pose(robot→world)에서 사거리 내 환경 점의 잡음 섞인 스캔을 로봇 프레임으로 반환."""
    t = T_pose[:3, 3]
    R = T_pose[:3, :3]
    d = env - t
    r = np.linalg.norm(d, axis=1)
    idx = np.where(r < SENSOR_RANGE)[0]
    if len(idx) == 0:
        return np.empty((0, 3))
    keep = rng.random(len(idx)) < KEEP_FRAC
    idx = idx[keep]
    world = env[idx] + rng.normal(0, SCAN_NOISE, (len(idx), 3))
    return (world - t) @ R                         # world → robot 프레임


# --------------------------------------------------------------------------- #
# 프론트엔드 · 장소재인식 · 백엔드
# --------------------------------------------------------------------------- #
def run_frontend(seed=1, laps=3):
    """스캔 시뮬 + 점-대-평면 ICP 오도메트리.
    반환: env, scans, normals, poses_true, X_odo, odom_edges, per_lap."""
    rng = np.random.default_rng(seed)
    env = build_environment()
    poses_true, per_lap = make_trajectory(laps=laps)
    scans = [simulate_scan(env, T, rng) for T in poses_true]
    normals = [estimate_normals(s) for s in scans]

    odo_sigma = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])  # [rho, phi]
    Om = np.diag(1.0 / odo_sigma ** 2)
    X_odo = [poses_true[0].copy()]
    edges = []
    rel_prev = np.eye(4)                            # 등속 초기값
    for k in range(len(scans) - 1):
        # scan[k+1](src)을 scan[k](dst)에 정렬 → Z = T_k^{-1} T_{k+1}
        T_rel, _, _ = point_to_plane_icp(
            scans[k + 1], scans[k], dst_normals=normals[k],
            init=rel_prev, max_corr_dist=1.2)
        edges.append(Edge3D(k, k + 1, T_rel, Om))
        X_odo.append(X_odo[-1] @ T_rel)
        rel_prev = T_rel
    return env, scans, normals, poses_true, X_odo, edges, per_lap


def detect_loop_closures(scans, normals, X_odo, per_lap,
                         min_gap=30, radius=3.5, err_thresh=0.14,
                         max_corr_dist=1.0, step=2):
    """장소재인식 + ICP 검증. 반환: 루프클로저 엣지 리스트, (j,k) 링크 리스트."""
    pos = np.array([T[:3, 3] for T in X_odo])
    lc_sigma = np.array([0.04, 0.04, 0.04, 0.02, 0.02, 0.02])
    Om_lc = np.diag(1.0 / lc_sigma ** 2)
    edges, links = [], []
    N = len(scans)
    for k in range(min_gap, N, step):
        cand = [j for j in range(0, k - min_gap)
                if np.linalg.norm(pos[k] - pos[j]) < radius]
        if not cand:
            continue
        best = None
        for j in cand:
            init = se3_inv(X_odo[j]) @ X_odo[k]     # 오도메트리 초기값
            T, err, _ = point_to_plane_icp(
                scans[k], scans[j], dst_normals=normals[j],
                init=init, max_corr_dist=max_corr_dist)
            trans = float(np.linalg.norm(T[:3, 3]))
            if err < err_thresh and trans < radius and (best is None or err < best[1]):
                best = (j, err, T)
        if best is not None:
            j, _, T = best
            edges.append(Edge3D(j, k, T, Om_lc))    # Z = T_j^{-1} T_k
            links.append((j, k))
    return edges, links


def positions(Xs):
    return np.array([T[:3, 3] for T in Xs])


def rmse_3d(Xs, poses_true):
    P = positions(Xs)
    G = positions(poses_true)
    return float(np.sqrt(np.mean(np.sum((P - G) ** 2, axis=1))))


def main(seed=1, plot=True):
    env, scans, normals, poses_true, X_odo, odom_edges, per_lap = run_frontend(seed=seed)
    lc_edges, links = detect_loop_closures(scans, normals, X_odo, per_lap)
    edges = odom_edges + lc_edges

    X_opt, hist = optimize_se3(X_odo, edges, iters=30)

    odom_rmse = rmse_3d(X_odo, poses_true)
    opt_rmse = rmse_3d(X_opt, poses_true)
    chi2_before, chi2_after = hist[0], hist[-1]
    gt = positions(poses_true)
    end_gap_odo = float(np.linalg.norm(positions(X_odo)[-1] - gt[-1]))
    end_gap_opt = float(np.linalg.norm(positions(X_opt)[-1] - gt[-1]))

    print("=== 29. 전체 3D LiDAR SLAM (점-대-평면 ICP + SE(3) pose-graph) ===")
    print(f"pose {len(poses_true)}개, 스캔 {len(scans)}개(점/스캔 평균 "
          f"{np.mean([len(s) for s in scans]):.0f}), 환경점 {len(env)}, 3바퀴 루프")
    print(f"엣지: 오도메트리 {len(odom_edges)} + 루프클로저 {len(lc_edges)}")
    print(f"ICP 오도메트리   : 3D RMSE={odom_rmse:.3f} m,  종점오차={end_gap_odo:.3f} m")
    print(f"SE(3) 최적화 후  : 3D RMSE={opt_rmse:.3f} m,  종점오차={end_gap_opt:.3f} m")
    print(f"→ 드리프트를 {odom_rmse / max(opt_rmse, 1e-6):.1f}배 줄임")
    print(f"chi2: {chi2_before:.1f} → {chi2_after:.3g} ({len(hist)} iters)")

    if plot:
        tp, op, xp = gt, positions(X_odo), positions(X_opt)
        env_s = env[::7]                            # 배경 점군 서브샘플
        fig = plt.figure(figsize=(14, 6.5))
        for idx, (P, title, col) in enumerate(
                [(op, f"ICP odometry (RMSE {odom_rmse:.2f} m)", "r"),
                 (xp, f"SE(3) optimized (RMSE {opt_rmse:.2f} m)", "b")]):
            ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
            ax.scatter(env_s[:, 0], env_s[:, 1], env_s[:, 2],
                       c="0.8", s=1, alpha=0.25)
            ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], "g-", lw=2.4, label="true (3 laps)")
            ax.plot(P[:, 0], P[:, 1], P[:, 2], col + "-", lw=1.4, label="estimate")
            if idx == 1:
                for (j, k) in links:
                    ax.plot([xp[j, 0], xp[k, 0]], [xp[j, 1], xp[k, 1]],
                            [xp[j, 2], xp[k, 2]], "-", color="orange",
                            lw=0.6, alpha=0.6)
                if links:
                    ax.plot([], [], "-", color="orange",
                            label=f"loop closures ({len(links)})")
            ax.scatter(*tp[0], c="k", s=45, label="start")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
            ax.view_init(elev=22, azim=-60)
        fig.suptitle("29. Full 3D LiDAR SLAM: point-to-plane ICP front-end "
                     "+ SE(3) pose-graph back-end", fontsize=12)
        fig.tight_layout()
        for d in ("outputs", "assets"):
            Path(d).mkdir(exist_ok=True)
            fig.savefig(Path(d) / "29_lidar_slam_3d.png", dpi=130)
        plt.close(fig)
        print("\n[plot] outputs/29_lidar_slam_3d.png, assets/29_lidar_slam_3d.png")

    return odom_rmse, opt_rmse, len(lc_edges), chi2_before, chi2_after


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 점-대-평면 vs 점-대-점: 벽·바닥·천장의 법선이 세 축을 모두 덮어 6-DOF가 잘
#     관측될 때 점-대-평면이 더 빠르고 정확히 수렴한다. 특징이 한 평면(예: 바닥만)에
#     몰리면 접선 방향 이동이 관측 불가(sliding)라 점-대-점 SVD 폴백이 나을 수 있다.
#   - 법선 추정: k-NN PCA는 국소 곡률·잡음·밀도에 민감하다. k가 작으면 잡음에, 크면
#     모서리에서 법선이 뭉개진다. 여기선 표면이 대부분 평면이라 안정적.
#   - 3D 관측성: 이 궤적은 x·y·z와 요를 모두 자극하지만 롤·피치 여기가 약하면 해당
#     자유도의 오도메트리 신뢰도가 떨어진다. 루프클로저가 이를 전역에서 보정한다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
