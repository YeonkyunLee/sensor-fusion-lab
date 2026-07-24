"""내비게이션 종합(capstone): A* 전역계획 + 장애물 인지 MPC 지역제어를 하나로 묶는다.

계획/제어 트랙(exp19 A*, exp24 MPC, exp26 장애물 회피 MPC)을 하나의 '출발→목표'
자율주행 스택으로 통합한다. 정적 지도(벽 + no-go 존)는 알지만 나중에 길을 가로지르는
'움직이는 장애물'은 전역계획 시점에 알 수 없다 — 실제 자율주행이 늘 마주치는 상황이다.

구성:
  (1) 전역계획 A*(exp19): 로봇 반경만큼 인플레이션한 점유격자 위에서 출발→목표
      최단 경로를 찾는다. 정적 벽과 no-go 존을 피하지만 이동 장애물은 모른다.
  (2) 전역경로 → 기준궤적: A* 꺾은선을 평활화하고 등호길이로 재샘플해 등속 기준궤적
      (Xref, Uref)으로 만든다. MPC가 추종할 목표.
  (3) 지역제어 장애물 인지 MPC(exp26): 기준궤적을 액추에이터 한계 하에서 추종하되,
      지평 동안 등속 예측한 '이동 장애물'을 소프트배리어로 반응 회피한다. 전역계획이
      몰랐던 동적 장애물을 지역에서 실시간으로 비껴간다.

핵심 대비(ablation): 동일한 A* 경로를 '장애물 무지' 추종기(beta=0, exp24식)로 따라가면
길을 가로지르는 이동 장애물과 정면충돌한다. 반면 전체 시스템(A* + 장애물 인지 MPC)은
전역경로를 유지한 채 이동 장애물을 비껴 목표에 무충돌 도달한다. 두 경우의 최소 여유거리를
보고해 '전역계획만으론 부족, 지역 반응성이 안전을 만든다'를 정량적으로 보인다.

한계(정직하게): 전역경로는 고정이고 재계획(replanning)은 하지 않는다. 회피는 소프트배리어라
안전거리를 '보장'하진 않고 강하게 유인한다. 이동 장애물이 좁은 통로를 막으면 지역
최소점(local minima)에 걸릴 수 있다 — 그때는 전역 재계획이 답이며 여기선 다루지 않는다.

    python scripts/28_full_navigation.py
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- 제어(로봇) 파라미터: exp26과 동일 계열 ----
DT = 0.1
N = 18                        # 예측 지평(스텝)
V_MAX, W_MAX = 2.6, 1.5       # 액추에이터 한계: 병진[m/s], 각속도[rad/s]
A_V, A_W = 4.0, 6.0           # 가속 한계: [m/s^2], [rad/s^2]
ROBOT_R = 0.3                 # 제어용 로봇 반경 [m] (여유거리 계산)
BETA = 220.0                  # 소프트배리어 유인강도
MARGIN = 0.6                  # 안전반경 위로 더 벌리려는 여유 [m]

# ---- 전역계획(A*) 파라미터: exp19 계열 ----
GRID = 0.5                    # 점유격자 해상도 [m]
PLAN_ROBOT_R = 0.5            # 계획용 인플레이션 반경 [m]
V_NOM = 2.0                   # 기준궤적 등속 [m/s]
PAD_STEPS = 70                # 목표에서 정착하도록 지평 이후 여유 스텝


# ==========================================================================
# 1) 정적 지도 + 전역계획 A* (exp19 방식 채용)
# ==========================================================================
def make_map():
    W, H = 40.0, 30.0
    # (x, y, w, h) 벽 — 출발(좌하)→목표(우상)로 가는 경로를 굽이치게 함
    walls = [
        (6, 0, 2, 18),
        (16, 12, 2, 18),
        (26, 0, 2, 20),
        (32, 14, 6, 2),
    ]
    nogo = (21.0, 7.0, 3.2)   # 민감 장비 = no-go 원 (cx, cy, r)
    return W, H, walls, nogo


def occupancy(W, H, walls, nogo):
    nx, ny = int(W / GRID), int(H / GRID)
    occ = np.zeros((ny, nx), bool)
    for (ox, oy, ow, oh) in walls:
        i0, i1 = int(oy / GRID), int((oy + oh) / GRID)
        j0, j1 = int(ox / GRID), int((ox + ow) / GRID)
        occ[i0:i1, j0:j1] = True
    cx, cy, r = nogo
    for i in range(ny):
        for j in range(nx):
            if np.hypot((j + 0.5) * GRID - cx, (i + 0.5) * GRID - cy) < r:
                occ[i, j] = True
    # 로봇 반경만큼 인플레이션
    infl = occ.copy()
    rad = int(np.ceil(PLAN_ROBOT_R / GRID))
    ys, xs = np.where(occ)
    for y, x in zip(ys, xs):
        infl[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = True
    return occ, infl


def astar(infl, start, goal):
    ny, nx = infl.shape

    def c2(p):
        return (int(p[1] / GRID), int(p[0] / GRID))   # (row, col)

    s, g = c2(start), c2(goal)
    if infl[s] or infl[g]:
        return None
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    openh = [(0, s)]
    came = {}
    gsc = {s: 0}
    while openh:
        _, cur = heapq.heappop(openh)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return np.array([[(c + 0.5) * GRID, (r + 0.5) * GRID] for (r, c) in path])
        for dr, dc in nbrs:
            nr, ncc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < ny and 0 <= ncc < nx) or infl[nr, ncc]:
                continue
            step = np.hypot(dr, dc)
            ng = gsc[cur] + step
            if ng < gsc.get((nr, ncc), 1e18):
                gsc[(nr, ncc)] = ng
                came[(nr, ncc)] = cur
                h = np.hypot(nr - g[0], ncc - g[1])
                heapq.heappush(openh, (ng + h, (nr, ncc)))
    return None


# ==========================================================================
# 2) 전역경로(A* 꺾은선) → 기준궤적: 평활화 + 등호길이 재샘플 후 등속
# ==========================================================================
def _push_clear(p, walls, nogo, d_safe):
    """점 p를 벽/no-go에서 최소 d_safe 만큼 떨어지게 바깥으로 밀어낸다(평활화 코너컷 보정)."""
    p = p.copy()
    for _ in range(3):
        for (ox, oy, ow, oh) in walls:
            qx = min(max(p[0], ox), ox + ow)
            qy = min(max(p[1], oy), oy + oh)
            dx, dy = p[0] - qx, p[1] - qy
            dist = np.hypot(dx, dy)
            if dist < 1e-9:                         # 사각형 내부 → 가장 가까운 변으로
                cand = [(ox - d_safe, p[1]), (ox + ow + d_safe, p[1]),
                        (p[0], oy - d_safe), (p[0], oy + oh + d_safe)]
                p = np.array(min(cand, key=lambda c: (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2))
            elif dist < d_safe:
                p = np.array([qx + dx / dist * d_safe, qy + dy / dist * d_safe])
        cx, cy, r = nogo
        dx, dy = p[0] - cx, p[1] - cy
        dd = np.hypot(dx, dy)
        if dd < r + d_safe:
            p = np.array([cx + dx / dd * (r + d_safe), cy + dy / dd * (r + d_safe)])
    return p


def smooth_path(path, walls, nogo, iters=10, alpha=0.2, d_safe=0.7):
    """끝점을 고정하고 내부점을 반복 평균해 격자 꺾임을 부드럽게 하되,
    각 반복 후 벽/no-go에서 d_safe 이상 떨어지도록 밀어내 코너컷 침범을 막는다."""
    p = path.astype(float).copy()
    for _ in range(iters):
        q = p.copy()
        q[1:-1] = p[1:-1] + alpha * (p[:-2] + p[2:] - 2 * p[1:-1])
        for i in range(1, len(q) - 1):
            q[i] = _push_clear(q[i], walls, nogo, d_safe)
        p = q
    return p


def make_reference(path, walls, nogo, v_nom=V_NOM):
    p = smooth_path(path, walls, nogo)
    d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    nref = max(int((L / v_nom) / DT), 2)
    snew = np.linspace(0, L, nref)
    xr = np.interp(snew, s, p[:, 0])
    yr = np.interp(snew, s, p[:, 1])
    th = np.unwrap(np.arctan2(np.gradient(yr), np.gradient(xr)))
    v = np.hypot(np.diff(xr), np.diff(yr)) / DT
    w = np.diff(th) / DT
    v = np.append(v, v[-1])
    w = np.append(w, w[-1])
    Xref = np.stack([xr, yr, th], axis=1)
    Uref = np.stack([v, w], axis=1)
    return Xref, Uref


# ==========================================================================
# 3) 장애물 인지 MPC (exp26 코어 채용: 완전 비선형 롤아웃 + 애드조인트 그래디언트)
# ==========================================================================
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_unicycle(X, u):
    x, y, th = X
    v, w = u
    return np.array([x + v * np.cos(th) * DT,
                     y + v * np.sin(th) * DT,
                     th + w * DT])


class Obstacle:
    """등속 이동 원 장애물. 안전반경 rs = 장애물반경 + 로봇반경."""

    def __init__(self, p0, vel, r_obs):
        self.p0 = np.asarray(p0, float)
        self.vel = np.asarray(vel, float)
        self.r_obs = float(r_obs)
        self.rs = float(r_obs) + ROBOT_R

    def pos(self, t):
        return self.p0 + self.vel * t


def _cost_grad(u_flat, x0, Xr_h, Ur_h, Qd, Qfd, Rd, obs, t0, beta, margin):
    u = u_flat.reshape(N, 2)

    X = np.empty((N + 1, 3))
    X[0] = x0
    for k in range(N):
        X[k + 1] = step_unicycle(X[k], u[k])

    J = 0.0
    dcx = np.zeros((N + 1, 3))
    dcu = np.zeros((N, 2))

    for k in range(1, N + 1):
        Qk = Qfd if k == N else Qd
        err = X[k] - Xr_h[k]
        err[2] = wrap(err[2])
        J += 0.5 * float(err @ (Qk * err))
        dcx[k] += Qk * err

        t = t0 + k * DT
        for ob in obs:
            o = ob.pos(t)
            diff = X[k, :2] - o
            dist = np.hypot(diff[0], diff[1]) + 1e-9
            clear = dist - ob.rs
            if clear < margin:
                slack = margin - clear
                J += 0.5 * beta * slack * slack
                g = (-beta * slack / dist) * diff
                dcx[k, 0] += g[0]
                dcx[k, 1] += g[1]

    for k in range(N):
        eu = u[k] - Ur_h[k]
        J += 0.5 * float(eu @ (Rd * eu))
        dcu[k] += Rd * eu

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
    Xr_h = np.array([Xref[i] for i in idx])
    Ur_h = np.array([Uref[idx[k]] for k in range(N)])

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

    res = minimize(_cost_grad, warm,
                   args=(X, Xr_h, Ur_h, Qd, Qfd, Rd, obs, t * DT, beta, margin),
                   jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 40})
    u = res.x.reshape(N, 2)
    u0 = np.array([np.clip(u[0, 0], -V_MAX, V_MAX),
                   np.clip(u[0, 1], -W_MAX, W_MAX)])
    warm_next = np.vstack([u[1:], u[-1]]).ravel()
    return u0, warm_next


def run_mpc(Xref, Uref, obs, beta, margin, pad=PAD_STEPS):
    Qd = np.array([9.0, 9.0, 0.25])
    Qfd = np.array([14.0, 14.0, 0.35])
    Rd = np.array([0.25, 0.04])
    X = Xref[0].copy()
    traj = [X[:2].copy()]
    ctrl = []
    warm = np.tile(Uref[0], N)
    u_prev = Uref[0].copy()
    nsteps = len(Xref) + pad          # 목표에서 정착하도록 여유 스텝
    for t in range(nsteps):
        u, warm = mpc_control(X, t, Xref, Uref, Qd, Qfd, Rd, obs, warm, u_prev, beta, margin)
        X = step_unicycle(X, u)
        u_prev = u
        traj.append(X[:2].copy())
        ctrl.append(u)
    return np.array(traj), np.array(ctrl)


# ==========================================================================
# 평가: 이동 장애물 여유거리 시계열, 정적 지도 여유거리
# ==========================================================================
def clearance_series(traj, obs):
    out = np.empty(len(traj))
    for i, p in enumerate(traj):
        t = i * DT
        out[i] = min(np.hypot(*(p - ob.pos(t))) - ob.rs for ob in obs)
    return out


def static_clearance(traj, walls, nogo):
    """정적 벽/ no-go 존에 대한 최소 여유(음수면 침범). 정직성 확인용."""
    gaps = []
    for p in traj:
        dmin = 1e9
        for (ox, oy, ow, oh) in walls:
            dx = max(ox - p[0], 0, p[0] - (ox + ow))
            dy = max(oy - p[1], 0, p[1] - (oy + oh))
            dmin = min(dmin, np.hypot(dx, dy))
        cx, cy, r = nogo
        dmin = min(dmin, np.hypot(p[0] - cx, p[1] - cy) - r)
        gaps.append(dmin - ROBOT_R)
    return float(min(gaps))


def make_moving_obstacles(Xref):
    """A* 경로를 가로지르는 이동 장애물 3개. 각기 다른 기준인덱스에서 경로점을
    통과하도록 타이밍을 맞춘다(로봇은 스텝당 ~1 기준점 진행 → 도달시각 ~ i*DT)."""
    nref = len(Xref)
    specs = [
        (0.42, 0.75, +1),   # (기준인덱스 비율, 속력 m/s, 횡단 방향 부호)
        (0.65, 0.70, -1),   # 벽에서 충분히 떨어진 개활 구간에서 횡단 → 우회 여유 확보
        (0.85, 0.80, +1),
    ]
    obs = []
    for frac, speed, sgn in specs:
        i = int(frac * nref)
        th = Xref[i, 2]
        perp = np.array([-np.sin(th), np.cos(th)]) * sgn   # 경로 접선의 수직
        vel = speed * perp
        t_i = i * DT
        p0 = Xref[i, :2] - vel * t_i                       # t_i에 경로점 통과
        obs.append(Obstacle(p0, vel, 0.55))
    return obs


def main():
    W, H, walls, nogo = make_map()
    occ, infl = occupancy(W, H, walls, nogo)
    start, goal = (2.0, 2.0), (37.0, 27.0)

    path = astar(infl, start, goal)
    if path is None:
        print("A* 경로 없음"); return {"reached_goal": False}

    Xref, Uref = make_reference(path, walls, nogo)
    nref = len(Xref)
    obs = make_moving_obstacles(Xref)

    # --- 전체 시스템(장애물 인지) vs 무지 추종기(beta=0) ---
    full_traj, full_u = run_mpc(Xref, Uref, obs, beta=BETA, margin=MARGIN)
    plain_traj, plain_u = run_mpc(Xref, Uref, obs, beta=0.0, margin=MARGIN)

    full_clear = clearance_series(full_traj, obs)
    plain_clear = clearance_series(plain_traj, obs)
    full_min = float(full_clear.min())
    plain_min = float(plain_clear.min())

    goal_arr = np.array(goal)
    full_end_err = float(np.hypot(*(full_traj[-1] - goal_arr)))
    plain_end_err = float(np.hypot(*(plain_traj[-1] - goal_arr)))
    reached_goal = full_end_err < 1.0

    stat_full = static_clearance(full_traj, walls, nogo)
    nogo_full = float(min(np.hypot(full_traj[:, 0] - nogo[0],
                                   full_traj[:, 1] - nogo[1])) - nogo[2])

    fv_max, fw_max = np.abs(full_u[:, 0]).max(), np.abs(full_u[:, 1]).max()
    dv = np.abs(np.diff(full_u[:, 0])).max() / DT
    dw = np.abs(np.diff(full_u[:, 1])).max() / DT
    limits_ok = (fv_max <= V_MAX + 1e-6 and fw_max <= W_MAX + 1e-6
                 and dv <= A_V + 1e-6 and dw <= A_W + 1e-6)

    print("=== 내비게이션 종합: A* 전역계획 + 장애물 인지 MPC 지역제어 ===")
    print(f"지도 {W:.0f}x{H:.0f} m, 격자 {GRID} m, 인플레이션 {PLAN_ROBOT_R} m")
    print(f"A* 경로점 {len(path)} → 기준궤적 {nref}점(등속 {V_NOM} m/s), 지평 N={N}")
    print(f"이동 장애물 {len(obs)}개(전역계획 미지), 안전반경 rs={obs[0].rs:.2f} m, 유인여유 {MARGIN} m")
    print(f"목표 도달   전체 시스템 {'예' if reached_goal else '아니오'}"
          f"(끝오차 {full_end_err:.2f} m)   |   무지 추종기 끝오차 {plain_end_err:.2f} m")
    print(f"이동장애물 최소 여유   전체 {full_min:+.3f} m   |   무지 {plain_min:+.3f} m")
    print(f"  -> 전체: {'충돌 없음' if full_min > 0 else '충돌!'}   "
          f"무지: {'충돌 없음' if plain_min > 0 else '충돌(경로 위 이동장애물 관통)!'}")
    print(f"전체 시스템 정적지도 최소 여유 {stat_full:+.2f} m (no-go {nogo_full:+.2f} m) "
          f"({'안전' if stat_full > 0 else '침범!'})")
    print(f"전체 제어 한계  |v|={fv_max:.2f}<= {V_MAX}, |w|={fw_max:.2f}<= {W_MAX}, "
          f"|dv|={dv:.2f}<= {A_V}, |dw|={dw:.2f}<= {A_W} ({'준수' if limits_ok else '위반!'})")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(13.5, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    ax = fig.add_subplot(gs[:, 0])

    for (ox, oy, ow, oh) in walls:
        ax.add_patch(plt.Rectangle((ox, oy), ow, oh, color="#555555"))
    cx, cy, r = nogo
    ax.add_patch(plt.Circle((cx, cy), r, color="#d9534f", alpha=0.35))
    ax.text(cx, cy, "NO-GO", ha="center", va="center", color="#7a1a1a",
            fontsize=7.5, weight="bold")

    ax.plot(path[:, 0], path[:, 1], color="#7a7a7a", ls=":", lw=1.0, alpha=0.7)
    ax.plot(Xref[:, 0], Xref[:, 1], "b--", lw=1.3, alpha=0.65, label="A* global path (reference)")
    ax.plot(plain_traj[:, 0], plain_traj[:, 1], color="#d9534f", lw=1.9, alpha=0.9,
            label=f"plain tracker (min clear {plain_min:+.2f} m, collides)")
    ax.plot(full_traj[:, 0], full_traj[:, 1], color="#1f77b4", lw=2.2,
            label=f"full system: A*+obstacle-aware MPC (min clear {full_min:+.2f} m, safe)")

    # 이동 장애물: 경로선 + 시간 스냅샷(도달시각 부근)
    labelled = False
    for ob in obs:
        i_star = None
        # 대략 통과 시각을 스냅샷 중심으로
        ts_line = np.linspace(0, (nref - 1) * DT, 60)
        pts = np.array([ob.pos(tt) for tt in ts_line])
        ax.plot(pts[:, 0], pts[:, 1], ":", color="#8a6d3b", lw=1.0, alpha=0.55)
        for si, tt in enumerate(np.linspace(0.15 * nref * DT, 0.95 * nref * DT, 4)):
            a = 0.14 + 0.5 * si / 3
            lab = "moving obstacle (predicted, unknown to A*)" if (not labelled and si == 3) else None
            ax.add_patch(plt.Circle(ob.pos(tt), ob.rs, color="#8a6d3b", alpha=a, label=lab))
            labelled = labelled or lab is not None

    ax.plot(*start, "ko", ms=9, label="start")
    ax.plot(*goal, "g*", ms=17, label="goal")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.3)
    ax.set_title("Full navigation: A* global path + obstacle-aware MPC bends around\n"
                 "moving obstacles the plan didn't know; plain tracker drives into one", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    tvec = np.arange(len(full_clear)) * DT
    axc = fig.add_subplot(gs[0, 1])
    axc.plot(tvec, plain_clear, color="#d9534f", lw=1.6, label="plain tracker")
    axc.plot(tvec, full_clear, color="#1f77b4", lw=1.8, label="full system")
    axc.axhline(0.0, color="k", lw=1.0)
    axc.axhline(MARGIN, color="gray", ls=":", lw=1.0)
    axc.fill_between(tvec, -2, 0, color="#d9534f", alpha=0.08)
    axc.set_ylim(min(-1.0, plain_min - 0.2), max(2.5, full_clear.max() * 0.6))
    axc.set_ylabel("clearance [m]")
    axc.set_title("Moving-obstacle clearance (0 = collision)", fontsize=9)
    axc.legend(fontsize=7, loc="upper right")
    axc.grid(alpha=0.2)

    axu = fig.add_subplot(gs[1, 1])
    tu = np.arange(len(full_u)) * DT
    axu.plot(tu, full_u[:, 0], color="#1f77b4", label="v [m/s]")
    axu.plot(tu, full_u[:, 1], color="#2ca02c", label="w [rad/s]")
    axu.axhline(V_MAX, color="gray", ls=":", lw=1)
    axu.axhline(W_MAX, color="gray", ls=":", lw=1)
    axu.axhline(-W_MAX, color="gray", ls=":", lw=1)
    axu.set_ylabel("control"); axu.set_xlabel("time [s]")
    axu.set_title("Full-system controls within limits", fontsize=9)
    axu.legend(fontsize=7, loc="lower right")
    axu.grid(alpha=0.2)

    fig.tight_layout()
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(f"{d}/28_full_navigation.png", dpi=130)
    print("\n[plot] outputs/28_full_navigation.png, assets/28_full_navigation.png")

    return {
        "reached_goal": bool(reached_goal),
        "full_min_clearance": full_min,
        "plain_min_clearance": plain_min,
        "static_clearance": stat_full,
        "limits_ok": bool(limits_ok),
    }


if __name__ == "__main__":
    main()
