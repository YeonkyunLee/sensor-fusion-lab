"""ICP scan-matching으로 2D LiDAR 오도메트리를 만든다.

SLAM 프론트엔드의 고전: 연속한 두 LiDAR 스캔(점군)을 정렬해 그 사이 로봇의
상대 이동 SE(2)를 추정하고, 이를 누적해 궤적을 복원한다.

ICP(Iterative Closest Point) 점-대-점 반복:
  {최근접 대응(NN) → SVD로 최적 강체변환(Umeyama/Kabsch) → 적용 → 수렴까지 반복}

데모: 방 윤곽 + 흩뿌린 기둥 환경에서 로봇이 궤적을 따라 이동하며, 각 pose에서
사거리 내 점 + 가우시안 잡음으로 노이즈 스캔을 만든다(이동에 따라 점이 사거리를
드나들어 부분 겹침이 자연스레 생김). 연속 스캔에
ICP를 걸어 상대 이동을 추정 → 오도메트리로 적분. 세 궤적을 비교한다:
  - true            : 지상 진실
  - ICP odometry    : 스캔매칭 상대이동 누적
  - raw/dead-reckon  : 잡음 섞인 제어입력만으로 적분(무보정 기준선)

ICP 스캔매칭이 무보정 dead-reckoning보다 궤적을 훨씬 잘 복원함을 보인다.

    python scripts/21_icp_scan_matching.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

SENSOR_RANGE = 18.0      # LiDAR 최대 사거리(m)
SCAN_NOISE = 0.03        # 스캔 점당 가우시안 잡음(m)
KEEP_FRAC = 1.0          # pose마다 보이는 점 중 스캔에 담기는 비율(1.0=사거리 내 전부)


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


def best_fit_transform(src, dst):
    """대응된 두 점군의 최적 강체변환(Umeyama/Kabsch, 회전+평행이동)을 SVD로."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (dst - mu_d).T @ (src - mu_s)          # 2x2 상관행렬
    U, _, Vt = np.linalg.svd(S)
    D = np.diag([1.0, np.linalg.det(U @ Vt)])  # reflection 방지
    R = U @ D @ Vt
    t = mu_d - R @ mu_s
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def icp(src, dst, init=None, max_iter=60, tol=1e-6, max_corr_dist=5.0):
    """2D 점-대-점 ICP. src를 dst에 정렬하는 SE(2) 변환을 추정한다.

    반환: (T, mean_error, n_iter)  — T는 src를 dst 좌표계로 보내는 변환.
    """
    T = np.eye(3) if init is None else init.copy()
    tree = cKDTree(dst)
    cur = transform(src, T)
    prev_err = np.inf
    n = 0
    for n in range(1, max_iter + 1):
        dist, idx = tree.query(cur)
        # 대응 거리 게이트: outlier(부분 겹침·occlusion) 제거
        m = dist < max_corr_dist
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
# 데모: 환경 · 스캔 시뮬레이션 · 오도메트리
# --------------------------------------------------------------------------- #
def build_environment():
    """방 윤곽(사각형 벽) + 방 전체에 흩뿌린 원형 기둥들을 점군으로.

    흩어진 기둥은 점-대-점 ICP에 이상적인 특징이다: 어느 방향으로도 대응을 고정해
    벽만 보일 때 생기는 aperture(벽-미끄러짐) 문제를 깨준다.
    """
    pts = []
    W, H, ds = 30.0, 22.0, 0.20
    # 바깥 벽 사각형
    for (x0, y0, x1, y1) in [(0, 0, W, 0), (W, 0, W, H), (W, H, 0, H), (0, H, 0, 0)]:
        n = int(np.hypot(x1 - x0, y1 - y0) / ds)
        pts.append(np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1))
    # 방 전체에 흩뿌린 원형 기둥(반경 다양) — 특징이 풍부해 정렬이 유일해짐
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
    vis = r < SENSOR_RANGE
    idx = np.where(vis)[0]
    if len(idx) == 0:
        return np.empty((0, 2))
    keep = rng.random(len(idx)) < KEEP_FRAC       # 부분 관측(occlusion 근사)
    idx = idx[keep]
    world = env[idx] + rng.normal(0, SCAN_NOISE, (len(idx), 2))
    # 월드 → 로봇 프레임: p_robot = R(th)^T (p_world - t)
    # 행벡터 규약에서 (p_world - t) @ R 가 곧 R^T (p_world - t) 이다.
    R = se2(*pose)[:2, :2]
    return (world - pose[:2]) @ R


def make_trajectory():
    """방 안을 도는 둥근 사각형 루프(제어입력 v, w의 열)."""
    dt = 1.0
    turn = [(0.6, (np.pi / 2) / 10)] * 10          # 약 90° 좌회전(둥근 모서리)
    controls = []
    controls += [(1.0, 0.0)] * 14                   # 바닥 직진
    controls += turn
    controls += [(1.0, 0.0)] * 7                    # 우측 직진
    controls += turn
    controls += [(1.0, 0.0)] * 14                   # 상단 직진
    controls += turn
    controls += [(1.0, 0.0)] * 7                    # 좌측 직진
    controls += turn[:5]
    return dt, controls


def integrate(pose, v, w, dt):
    x, y, th = pose
    return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), wrap(th + w * dt)])


def run(seed=0):
    rng = np.random.default_rng(seed)
    env = build_environment()
    dt, controls = make_trajectory()

    # 지상진실 궤적 + 각 pose 스캔
    start = np.array([6.0, 5.0, 0.0])
    true_poses = [start.copy()]
    p = start.copy()
    for (v, w) in controls:
        p = integrate(p, v, w, dt)
        true_poses.append(p.copy())
    true_poses = np.array(true_poses)

    scans = [simulate_scan(env, tp, rng) for tp in true_poses]

    # 무보정(dead-reckoning): 잡음 섞인 제어입력만 적분(실제 오도메트리 드리프트)
    V_STD, W_STD = 0.08, 0.05
    raw = [start.copy()]
    p = start.copy()
    for (v, w) in controls:
        p = integrate(p, v + rng.normal(0, V_STD), w + rng.normal(0, W_STD), dt)
        raw.append(p.copy())
    raw = np.array(raw)

    # ICP 오도메트리: 연속 스캔 정렬로 상대 이동 추정 후 누적
    icp_poses = [start.copy()]
    Tglob = se2(*start)
    errs = []
    for k in range(len(scans) - 1):
        src, dst = scans[k + 1], scans[k]     # 새 스캔을 이전 스캔에 정렬
        # dead-reckoning 상대이동을 ICP 초기값으로(수렴 가속)
        v, w = controls[k]
        # ICP가 찾는 T: scan_{k+1} -> scan_k = 프레임k에서 본 pose k+1 = T_k^{-1} T_{k+1}
        rel0 = se2(v * dt, 0.0, wrap(w * dt))  # 로봇k 프레임에서 본 로봇k+1
        T_rel, err, _ = icp(src, dst, init=rel0, max_corr_dist=0.8)
        errs.append(err)
        Tglob = Tglob @ T_rel                  # 상대이동 누적
        icp_poses.append(se2_params(Tglob))
    icp_poses = np.array(icp_poses)

    return env, scans, true_poses, raw, icp_poses, np.array(errs)


def rmse(est, truth):
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - truth[:, :2]) ** 2, axis=1))))


def main():
    env, scans, true_poses, raw, icp_poses, errs = run(seed=1)

    icp_rmse = rmse(icp_poses, true_poses)
    raw_rmse = rmse(raw, true_poses)
    per_scan = float(errs.mean())

    print("=== ICP scan-matching LiDAR 오도메트리 ===")
    print(f"스캔 수            : {len(scans)}  (점/스캔 평균 {np.mean([len(s) for s in scans]):.0f})")
    print(f"스캔당 평균 정렬오차 : {per_scan:.3f} m")
    print(f"ICP 오도메트리 RMSE : {icp_rmse:.3f} m")
    print(f"무보정(raw) RMSE    : {raw_rmse:.3f} m")
    print(f"→ ICP가 무보정 대비 {raw_rmse / max(icp_rmse, 1e-6):.1f}배 정확")

    # 그림 ---------------------------------------------------------------- #
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(env[:, 0], env[:, 1], ".", color="0.75", ms=2, label="environment")
    ax1.plot(true_poses[:, 0], true_poses[:, 1], "g-", lw=2.2, label="true")
    ax1.plot(icp_poses[:, 0], icp_poses[:, 1], "b-", lw=1.6,
             label=f"ICP odometry (RMSE {icp_rmse:.2f}m)")
    ax1.plot(raw[:, 0], raw[:, 1], "r--", lw=1.4,
             label=f"raw dead-reckoning (RMSE {raw_rmse:.2f}m)")
    ax1.plot(*true_poses[0, :2], "ko", ms=8, label="start")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("LiDAR odometry: ICP scan-matching vs dead-reckoning")
    ax1.grid(alpha=0.3)

    # 우측: 한 쌍의 스캔 정렬 before/after -------------------------------- #
    k = len(scans) // 2
    src, dst = scans[k + 1], scans[k]
    v, w = make_trajectory()[1][k]
    rel0 = se2(v * 1.0, 0.0, wrap(w * 1.0))
    T_rel, _, _ = icp(src, dst, init=rel0, max_corr_dist=0.8)
    src_aligned = transform(src, T_rel)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(dst[:, 0], dst[:, 1], "k.", ms=4, label="target scan (k)")
    ax2.plot(src[:, 0], src[:, 1], "r.", ms=4, alpha=0.5, label="source scan (k+1), before")
    ax2.plot(src_aligned[:, 0], src_aligned[:, 1], "b.", ms=4, label="source, after ICP")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8)
    ax2.set_title(f"Scan alignment (pair k={k}): before vs after ICP")
    ax2.grid(alpha=0.3)

    fig.suptitle("21. ICP scan-matching for 2D LiDAR odometry")
    fig.tight_layout()
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "21_icp_scan_matching.png", dpi=130)
    plt.close(fig)
    print("\n[plot] outputs/21_icp_scan_matching.png, assets/21_icp_scan_matching.png")

    return icp_rmse, raw_rmse


if __name__ == "__main__":
    main()
