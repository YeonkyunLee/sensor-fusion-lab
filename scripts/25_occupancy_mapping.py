"""점유 격자 지도(occupancy-grid mapping, scan-to-map): SLAM의 '지도 만들기' 절반.

exp 23은 ICP 프론트엔드 + pose-graph 백엔드로 로봇 '궤적'을 복원하지만, 명시적인
지도 격자는 만들지 않는다. 이 실험은 그 나머지 절반 — 주어진 로봇 pose와 LiDAR 스캔으로
환경의 확률적 **점유 격자**를 세운다.

핵심은 **로그오즈(log-odds) 갱신 + 광선투사(ray casting)**다.
  - 각 스캔의 점마다 로봇에서 명중점까지 광선을 쏜다(DDA 격자 순회).
  - 광선이 지나는 셀은 '비어있음'(음의 로그오즈)으로, 명중 셀은 '점유'(양의 로그오즈)로 갱신.
  - 로그오즈를 확률 p = 1/(1+e^-l) 로 바꿔 회색조로 표시한다.
  - 로그오즈는 누적형이라 잡음 섞인 스캔을 여러 번 관측할수록 벽·기둥이 또렷해진다.

pose가 잡음(오도메트리 드리프트)일 때는 새 스캔을 지금까지의 지도에 먼저 정렬하는
**scan-to-map 보정**(현재 지도의 점유 셀 점군에 스캔을 ICP)으로 각 스캔을 끼워 넣어
지도 번짐을 줄인다. 네 지도를 비교한다:
  - ground-truth 환경(벽 + 기둥)
  - 참 궤적으로 세운 지도(상한 기준)
  - 잡음 궤적 그대로 세운 naive 지도(번짐)
  - 잡음 궤적 + scan-to-map 보정 지도(또렷해짐)

지도 품질은 점유 셀의 IoU(관측된 영역 한정, 지상진실을 1셀 팽창해 관대하게)로 잰다.

    python scripts/25_occupancy_mapping.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

SENSOR_RANGE = 18.0      # LiDAR 최대 사거리(m)
SCAN_NOISE = 0.04        # 스캔 점당 가우시안 잡음(m)
KEEP_FRAC = 0.85         # pose마다 보이는 점 중 스캔에 담기는 비율(부분 관측)

# 로그오즈 파라미터
L_OCC = 0.85            # 명중 셀 점유 증거(p≈0.70)
L_FREE = 0.40           # 광선 통과 셀 비점유 증거(p≈0.40)
L_CLAMP = 6.0           # 로그오즈 포화(과확신 방지)

# 격자 설정
RES = 0.15              # 셀 크기(m)
PAD = 1.5               # 방 바깥 여유(m)


# --------------------------------------------------------------------------- #
# SE(2) · ICP (exp 21/23에서 채택·자기완결)
# --------------------------------------------------------------------------- #
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def se2(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def se2_params(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def transform(pts, T):
    return pts @ T[:2, :2].T + T[:2, 2]


def best_fit_transform(src, dst):
    """대응된 두 점군의 최적 강체변환(Kabsch/SVD)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(S)
    D = np.diag([1.0, np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    t = mu_d - R @ mu_s
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def icp(src, dst, init=None, max_iter=40, tol=1e-6, max_corr_dist=1.2):
    """2D 점-대-점 ICP. src를 dst에 정렬하는 SE(2)를 추정. 반환 (T, mean_err)."""
    T = np.eye(3) if init is None else init.copy()
    tree = cKDTree(dst)
    cur = transform(src, T)
    prev_err = np.inf
    for _ in range(max_iter):
        dist, idx = tree.query(cur)
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
    return T, prev_err


# --------------------------------------------------------------------------- #
# 환경 · 스캔 · 궤적 (exp 23 규약 재사용)
# --------------------------------------------------------------------------- #
ROOM_W, ROOM_H = 30.0, 22.0
PILLARS = [(5, 5, 0.6), (5, 12, 0.5), (6, 18, 0.7), (11, 8, 0.5),
           (12, 16, 0.6), (15, 4, 0.7), (16, 11, 0.5), (18, 19, 0.6),
           (21, 7, 0.6), (22, 15, 0.5), (25, 5, 0.7), (26, 18, 0.6),
           (9, 3, 0.4), (24, 11, 0.5)]


def build_environment():
    """방 윤곽(사각형 벽) + 흩뿌린 원형 기둥의 표면 점군."""
    pts = []
    ds = 0.20
    for (x0, y0, x1, y1) in [(0, 0, ROOM_W, 0), (ROOM_W, 0, ROOM_W, ROOM_H),
                             (ROOM_W, ROOM_H, 0, ROOM_H), (0, ROOM_H, 0, 0)]:
        n = int(np.hypot(x1 - x0, y1 - y0) / ds)
        pts.append(np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1))
    for (cx, cy, r) in PILLARS:
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
    keep = rng.random(len(idx)) < KEEP_FRAC
    idx = idx[keep]
    world = env[idx] + rng.normal(0, SCAN_NOISE, (len(idx), 2))
    R = se2(*pose)[:2, :2]
    return (world - pose[:2]) @ R                     # 월드 → 로봇 프레임


def make_lap():
    """방 안을 도는 둥근 사각형 한 바퀴. 시작 자세로 되돌아옴."""
    turn = [(0.9, (np.pi / 2) / 10)] * 10
    c = []
    c += [(1.0, 0.0)] * 13
    c += turn
    c += [(1.0, 0.0)] * 8
    c += turn
    c += [(1.0, 0.0)] * 13
    c += turn
    c += [(1.0, 0.0)] * 8
    c += turn
    return c


def integrate(pose, v, w, dt=1.0):
    x, y, th = pose
    return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), wrap(th + w * dt)])


def simulate_trajectory(seed=1, laps=2):
    """참 궤적 + 스캔(로봇 프레임)을 생성."""
    rng = np.random.default_rng(seed)
    env = build_environment()
    controls = make_lap() * laps
    start = np.array([6.0, 5.0, 0.0])
    true_poses = [start.copy()]
    p = start.copy()
    for (v, w) in controls:
        p = integrate(p, v, w)
        true_poses.append(p.copy())
    true_poses = np.array(true_poses)
    scans = [simulate_scan(env, tp, rng) for tp in true_poses]
    return env, scans, true_poses


def make_noisy_poses(true_poses, seed=1):
    """오도메트리 드리프트를 흉내낸 잡음 궤적(누적 랜덤워크)."""
    rng = np.random.default_rng(seed + 7)
    step = rng.normal(0.0, [0.028, 0.028, 0.010], (len(true_poses), 3))
    drift = np.cumsum(step, axis=0)
    noisy = true_poses.copy()
    noisy[:, :2] += drift[:, :2]
    noisy[:, 2] = wrap(noisy[:, 2] + drift[:, 2])
    return noisy


# --------------------------------------------------------------------------- #
# 점유 격자 · 로그오즈 · 광선투사
# --------------------------------------------------------------------------- #
class GridSpec:
    def __init__(self):
        self.x0 = -PAD
        self.y0 = -PAD
        self.res = RES
        self.nx = int(np.ceil((ROOM_W + 2 * PAD) / RES))
        self.ny = int(np.ceil((ROOM_H + 2 * PAD) / RES))

    def world_to_cell(self, x, y):
        ix = ((np.asarray(x) - self.x0) / self.res).astype(int)
        iy = ((np.asarray(y) - self.y0) / self.res).astype(int)
        return ix, iy

    def cell_centers(self, ix, iy):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res


def integrate_scan(logodds, gs, world_hits, robot_xy):
    """스캔 한 장을 로그오즈 격자에 통합. 광선 통과=free, 명중=occupied.

    광선을 격자 해상도 간격으로 표본화(DDA)해 통과 셀은 -L_FREE, 명중 셀은 +L_OCC.
    벡터화로 스캔 전체를 한 번에 처리한다.
    """
    if len(world_hits) == 0:
        return
    flat = logodds.reshape(-1)
    nx, ny = gs.nx, gs.ny
    p0 = robot_xy
    d = world_hits - p0
    dist = np.hypot(d[:, 0], d[:, 1])
    dist = np.maximum(dist, 1e-3)
    dirs = d / dist[:, None]

    # --- free 셀: 광선을 따라 명중 직전까지 표본화 ---
    max_s = int(np.ceil(dist.max() / gs.res)) + 1
    steps = (np.arange(max_s) * gs.res)[None, :]            # (1,S) 거리
    valid = steps < (dist[:, None] - gs.res)               # 명중점 직전에서 멈춤
    sx = p0[0] + dirs[:, 0:1] * steps
    sy = p0[1] + dirs[:, 1:2] * steps
    ix, iy = gs.world_to_cell(sx, sy)
    inb = valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(flat, (iy[inb] * nx + ix[inb]), -L_FREE)

    # --- occupied 셀: 명중점 ---
    hix, hiy = gs.world_to_cell(world_hits[:, 0], world_hits[:, 1])
    hinb = (hix >= 0) & (hix < nx) & (hiy >= 0) & (hiy < ny)
    np.add.at(flat, hiy[hinb] * nx + hix[hinb], L_OCC)

    np.clip(logodds, -L_CLAMP, L_CLAMP, out=logodds)


def map_point_cloud(logodds, gs, thr=1.5):
    """현재 지도의 점유 셀 중심을 점군으로(scan-to-map 정렬 대상)."""
    iy, ix = np.where(logodds > thr)
    x, y = gs.cell_centers(ix, iy)
    return np.stack([x, y], axis=1)


def build_map(scans, poses, gs):
    """주어진 pose로 스캔을 순차 통합한 로그오즈 격자."""
    logodds = np.zeros((gs.ny, gs.nx))
    for scan, pose in zip(scans, poses):
        if len(scan) == 0:
            continue
        world = transform(scan, se2(*pose))
        integrate_scan(logodds, gs, world, pose[:2])
    return logodds


def build_map_scan_to_map(scans, noisy_poses, gs, warmup=8):
    """잡음 pose를 scan-to-map ICP로 보정하며 통합. 반환 (logodds, refined_poses)."""
    logodds = np.zeros((gs.ny, gs.nx))
    refined = []
    for k, (scan, npose) in enumerate(zip(scans, noisy_poses)):
        if len(scan) == 0:
            refined.append(npose.copy())
            continue
        T_guess = se2(*npose)
        world_guess = transform(scan, T_guess)
        T_corr = np.eye(3)
        if k >= warmup:
            mpts = map_point_cloud(logodds, gs)
            if len(mpts) >= 20:
                T_corr, _ = icp(world_guess, mpts, max_corr_dist=1.2)
        T_ref = T_corr @ T_guess
        world_corr = transform(world_guess, T_corr)
        integrate_scan(logodds, gs, world_corr, T_ref[:2, 2])
        refined.append(se2_params(T_ref))
    return logodds, np.array(refined)


# --------------------------------------------------------------------------- #
# 지상진실 격자 · 품질 지표
# --------------------------------------------------------------------------- #
def truth_surface(gs):
    """지상진실 벽·기둥 표면 셀(래스터화, 팽창 없음)."""
    env = build_environment()
    occ = np.zeros((gs.ny, gs.nx), dtype=bool)
    ix, iy = gs.world_to_cell(env[:, 0], env[:, 1])
    inb = (ix >= 0) & (ix < gs.nx) & (iy >= 0) & (iy < gs.ny)
    occ[iy[inb], ix[inb]] = True
    return occ


def occupancy_iou(logodds, truth, occ_thr=0.7, tol=1):
    """관측된 영역 한정 점유 셀 IoU + 정확분류율.

    벽은 두께가 얇고 스캔·격자 양자화 오차가 있어, 예측/진실을 tol셀(=tol×해상도)만큼
    허용해 대응시킨다. 관측 못 한 진실 셀(가림)은 제외해 정직하게 센다.
    """
    known = np.abs(logodds) > 1e-6                     # 광선이 닿은 셀만
    pred = (logodds > occ_thr) & known
    tk = truth & known                                 # 관측된 진실 표면 셀
    dt = binary_dilation(tk, iterations=tol)
    dp = binary_dilation(pred, iterations=tol)
    tp_pred = np.count_nonzero(pred & dt)              # 벽 근처의 예측(정답)
    tp_truth = np.count_nonzero(tk & dp)               # 예측이 덮은 진실 표면
    inter = 0.5 * (tp_pred + tp_truth)
    union = pred.sum() + tk.sum() - inter
    iou = inter / max(union, 1.0)
    # 정확분류: 벽 근처 점유 예측 + 벽에서 먼 free 예측
    occ_correct = pred & dt
    free_correct = (~pred) & (~dt) & known
    frac = np.count_nonzero(occ_correct | free_correct) / max(np.count_nonzero(known), 1)
    return iou, frac


def logodds_to_prob(logodds):
    return 1.0 / (1.0 + np.exp(-logodds))


# --------------------------------------------------------------------------- #
def main(seed=1, plot=True):
    gs = GridSpec()
    env, scans, true_poses = simulate_trajectory(seed=seed, laps=2)
    noisy_poses = make_noisy_poses(true_poses, seed=seed)
    truth_occ = truth_surface(gs)

    # 세 가지 지도
    lo_true = build_map(scans, true_poses, gs)                 # 상한 기준(참 pose)
    lo_naive = build_map(scans, noisy_poses, gs)               # 잡음 pose 그대로
    lo_ref, refined_poses = build_map_scan_to_map(scans, noisy_poses, gs)

    iou_true, frac_true = occupancy_iou(lo_true, truth_occ)
    iou_naive, frac_naive = occupancy_iou(lo_naive, truth_occ)
    iou_ref, frac_ref = occupancy_iou(lo_ref, truth_occ)

    def pose_rmse(est):
        return float(np.sqrt(np.mean(np.sum((est[:, :2] - true_poses[:, :2]) ** 2, axis=1))))

    rmse_noisy = pose_rmse(noisy_poses)
    rmse_ref = pose_rmse(refined_poses)

    print("=== 25. 점유 격자 지도(occupancy-grid mapping, scan-to-map) ===")
    print(f"격자 {gs.nx}x{gs.ny} 셀(해상도 {gs.res} m), pose {len(true_poses)}개, 2바퀴 루프")
    print(f"스캔 {len(scans)}장(점/스캔 평균 {np.mean([len(s) for s in scans]):.0f})")
    print(f"로그오즈: L_occ={L_OCC}, L_free={L_FREE}, clamp=±{L_CLAMP}")
    print("-" * 58)
    print(f"참 pose 지도            : 점유 IoU={iou_true:.3f}, 정확분류={frac_true*100:.1f}%")
    print(f"잡음 pose naive 지도    : 점유 IoU={iou_naive:.3f}, 정확분류={frac_naive*100:.1f}%"
          f"  (pose RMSE={rmse_noisy:.3f} m)")
    print(f"잡음 pose scan-to-map   : 점유 IoU={iou_ref:.3f}, 정확분류={frac_ref*100:.1f}%"
          f"  (pose RMSE={rmse_ref:.3f} m)")
    print(f"→ scan-to-map 보정이 IoU {iou_naive:.3f} → {iou_ref:.3f} 로 개선"
          f" (pose RMSE {rmse_noisy:.3f} → {rmse_ref:.3f} m)")

    if plot:
        prob_true = logodds_to_prob(lo_true)
        prob_naive = logodds_to_prob(lo_naive)
        prob_ref = logodds_to_prob(lo_ref)
        extent = [gs.x0, gs.x0 + gs.nx * gs.res, gs.y0, gs.y0 + gs.ny * gs.res]

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # (0,0) 지상진실 환경
        ax = axes[0, 0]
        ax.plot(env[:, 0], env[:, 1], ".", color="0.25", ms=2)
        for (cx, cy, r) in PILLARS:
            ax.add_patch(plt.Circle((cx, cy), r, color="0.55", zorder=2))
        ax.plot(true_poses[:, 0], true_poses[:, 1], "g-", lw=2.0, label="true trajectory")
        ax.plot(*true_poses[0, :2], "ko", ms=7, label="start")
        ax.set_title("Ground-truth environment (walls + pillars)")
        ax.legend(fontsize=8, loc="upper right")

        # (0,1) 참 pose 지도(상한 기준)
        ax = axes[0, 1]
        ax.imshow(prob_true, origin="lower", extent=extent, cmap="gray_r",
                  vmin=0, vmax=1)
        ax.plot(true_poses[:, 0], true_poses[:, 1], "-", color="lime", lw=1.6)
        ax.set_title(f"Occupancy grid — true poses (IoU {iou_true:.2f})")

        # (1,0) 잡음 pose naive
        ax = axes[1, 0]
        ax.imshow(prob_naive, origin="lower", extent=extent, cmap="gray_r",
                  vmin=0, vmax=1)
        ax.plot(noisy_poses[:, 0], noisy_poses[:, 1], "-", color="red", lw=1.4)
        ax.set_title(f"Noisy poses — naive (smeared, IoU {iou_naive:.2f})")

        # (1,1) 잡음 pose + scan-to-map
        ax = axes[1, 1]
        ax.imshow(prob_ref, origin="lower", extent=extent, cmap="gray_r",
                  vmin=0, vmax=1)
        ax.plot(refined_poses[:, 0], refined_poses[:, 1], "-", color="dodgerblue", lw=1.4)
        ax.set_title(f"Noisy poses — scan-to-map refined (IoU {iou_ref:.2f})")

        for ax in axes.ravel():
            ax.set_aspect("equal")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.grid(alpha=0.2)

        fig.suptitle("25. Occupancy-grid mapping (log-odds ray casting + scan-to-map)",
                     fontsize=13)
        fig.tight_layout()
        for d in ("outputs", "assets"):
            Path(d).mkdir(exist_ok=True)
            fig.savefig(Path(d) / "25_occupancy_mapping.png", dpi=125)
        plt.close(fig)
        print("\n[plot] outputs/25_occupancy_mapping.png, assets/25_occupancy_mapping.png")

    return iou_true, iou_naive, iou_ref


if __name__ == "__main__":
    main()
