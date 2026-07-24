"""동적 장애물 회피: DWA(Dynamic Window Approach) 지역 경로계획.

19번의 A*는 '정적' 지도에서 미리 경로를 깐다. 하지만 실제 랩/물류 환경엔
사람·다른 로봇처럼 '움직이는' 장애물이 있다. 미리 깐 경로는 곧 무효가 된다.
DWA는 매 스텝 로봇의 속도공간(v, w)에서 가속한계로 도달 가능한 창(dynamic
window)만 샘플링하고, 각 후보로 짧은 궤적을 예측(roll-out)한 뒤
  점수 = heading(목표 지향) + clearance(장애물 여유) + velocity(전진 선호)
로 최적 (v, w)를 골라 '반응적으로' 한 스텝 나아간다. 움직이는 장애물의
미래 위치까지 예측에 반영해 실시간 충돌 회피를 수행한다.

시나리오: 유니사이클 로봇이 등속으로 움직이는 장애물 3개를 피해 목표에 도달.

    python scripts/20_dwa_dynamic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1                    # 제어 주기 [s]

# --- 로봇 운동/제어 한계 ---
V_MAX = 1.6                 # 최대 전진속도 [m/s]
V_MIN = 0.0
W_MAX = 2.8                 # 최대 각속도 [rad/s]
A_V = 2.5                   # 선가속 한계 [m/s^2]
A_W = 4.0                   # 각가속 한계 [rad/s^2]
V_RES = 0.08               # 속도 샘플 해상도
W_RES = 0.10               # 각속도 샘플 해상도
PREDICT_T = 2.2            # 궤적 예측 시간 [s]

ROBOT_R = 0.35             # 로봇 반경 [m]
SAFE_DIST = 1.2            # 이 거리 안으로 들어오면 clearance 페널티 부과 [m]

# --- 점수 가중치 ---
W_HEADING = 0.6            # 종단에서 목표를 바라보는 정도
W_GOAL = 0.6              # 종단이 목표에 가까운 정도(지역최소 탈출)
W_CLEAR = 2.4             # 장애물 여유
W_VEL = 0.2              # 전진 선호


def motion(state, v, w, dt):
    """유니사이클 1스텝 적분: state = [x, y, theta, v, w]."""
    x, y, th, _, _ = state
    x += v * np.cos(th) * dt
    y += v * np.sin(th) * dt
    th += w * dt
    return np.array([x, y, th, v, w])


def dynamic_window(state):
    """가속한계로 다음 DT 내 도달 가능한 (v, w) 창."""
    v, w = state[3], state[4]
    vs = [V_MIN, V_MAX, -W_MAX, W_MAX]                       # 절대 한계
    vd = [v - A_V * DT, v + A_V * DT, w - A_W * DT, w + A_W * DT]  # 동역학 창
    return [max(vs[0], vd[0]), min(vs[1], vd[1]),
            max(vs[2], vd[2]), min(vs[3], vd[3])]


def obstacle_positions(obstacles, t):
    """시각 t에서 등속 장애물들의 위치 (N,2)."""
    return np.array([[ox + vx * t, oy + vy * t] for (ox, oy, vx, vy, r) in obstacles])


def rollout(state, v, w):
    """(v, w) 유지 시 PREDICT_T 동안의 예측 궤적 (K,5)."""
    traj = [state]
    s = state
    n = int(PREDICT_T / DT)
    for _ in range(n):
        s = motion(s, v, w, DT)
        traj.append(s)
    return np.array(traj)


def clearance_cost(traj, obstacles, t0):
    """예측 궤적과 '움직이는' 장애물의 최소 거리 -> 비용.

    장애물도 예측 구간 동안 등속 이동하므로 각 예측 스텝의 시각에 맞춰
    장애물 위치를 갱신해 최소 여유거리를 구한다. 충돌 시 무한대.
    """
    radii = np.array([r for (*_, r) in obstacles])
    min_gap = np.inf
    for k, s in enumerate(traj):
        t = t0 + k * DT
        opos = obstacle_positions(obstacles, t)
        d = np.hypot(opos[:, 0] - s[0], opos[:, 1] - s[1]) - radii - ROBOT_R
        gap = float(np.min(d))
        if gap < 0.0:
            return np.inf, gap                    # 충돌 예측
        min_gap = min(min_gap, gap)
    # SAFE_DIST 밖이면 페널티 0(개활지에선 목표로 직진), 안이면 선형 증가
    pen = max(0.0, SAFE_DIST - min_gap) / SAFE_DIST
    return pen, min_gap


def heading_cost(traj, goal):
    """예측 종단에서 목표를 바라보는 정도 (0~1, 작을수록 좋음)."""
    end = traj[-1]
    ang = np.arctan2(goal[1] - end[1], goal[0] - end[0])
    err = abs((ang - end[2] + np.pi) % (2 * np.pi) - np.pi)
    return err / np.pi


def goal_dist_cost(traj, goal, d0):
    """예측 종단이 목표에 가까운 정도 (0~1, 시작 거리 d0로 정규화)."""
    end = traj[-1]
    return np.hypot(goal[0] - end[0], goal[1] - end[1]) / (d0 + 1e-9)


def plan(state, goal, obstacles, t0):
    """DWA 한 스텝: 최적 (v, w)와 그 예측 궤적을 반환."""
    dw = dynamic_window(state)
    best = (np.inf, None, None)
    vs = np.arange(dw[0], dw[1] + 1e-9, V_RES)
    ws = np.arange(dw[2], dw[3] + 1e-9, W_RES)
    d0 = np.hypot(goal[0] - state[0], goal[1] - state[1])
    for v in vs:
        for w in ws:
            traj = rollout(state, v, w)
            c_clear, _ = clearance_cost(traj, obstacles, t0)
            if not np.isfinite(c_clear):
                continue                          # 충돌 후보 제외
            c_head = heading_cost(traj, goal)
            c_goal = goal_dist_cost(traj, goal, d0)
            c_vel = (V_MAX - v) / V_MAX            # 빠를수록 좋음
            cost = (W_HEADING * c_head + W_GOAL * c_goal
                    + W_CLEAR * c_clear + W_VEL * c_vel)
            if cost < best[0]:
                best = (cost, (v, w), traj)
    return best[1], best[2]


def main():
    goal = np.array([10.0, 10.0])
    start = np.array([0.0, 0.0, np.pi / 4, 0.0, 0.0])

    # 장애물: (x0, y0, vx, vy, r) — 등속 이동. 로봇 대각선 통로를
    # '서로 다른 시각·지점'에서 가로질러 회피 여지를 남긴다.
    obstacles = [
        (5.0, -1.0, 0.0, 0.85, 0.6),     # 하단에서 위로 통로 횡단(초반)
        (0.5, 6.0, 0.85, 0.0, 0.6),      # 좌측에서 오른쪽 횡단(중반)
        (10.5, 6.5, -0.55, 0.25, 0.6),   # 목표 근처로 접근(후반)
    ]

    state = start.copy()
    traj = [state.copy()]
    min_clear = np.inf
    reached = False
    T_MAX = 40.0
    steps = int(T_MAX / DT)

    for i in range(steps):
        t = i * DT
        if np.hypot(state[0] - goal[0], state[1] - goal[1]) < 0.5:
            reached = True
            break
        u, _ = plan(state, goal, obstacles, t)
        if u is None:                    # 모든 후보가 충돌 예측 -> 정지(감속)
            u = (max(0.0, state[3] - A_V * DT), 0.0)
        state = motion(state, u[0], u[1], DT)
        traj.append(state.copy())
        # 실제(현재 시각) 클리어런스 기록
        opos = obstacle_positions(obstacles, t + DT)
        radii = np.array([r for (*_, r) in obstacles])
        gap = float(np.min(np.hypot(opos[:, 0] - state[0], opos[:, 1] - state[1]) - radii - ROBOT_R))
        min_clear = min(min_clear, gap)

    traj = np.array(traj)
    t_final = (len(traj) - 1) * DT

    print("=== DWA 동적 장애물 회피 ===")
    print(f"목표 도달: {'예' if reached else '아니오'}  (소요 {t_final:.1f} s)")
    print(f"최소 장애물 클리어런스: {min_clear:.2f} m ({'충돌 없음' if min_clear > 0 else '충돌!'})")
    plen = float(np.sum(np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]))))
    print(f"주행 거리: {plen:.1f} m")

    # --- 그림 ---
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(traj[:, 0], traj[:, 1], "g-", lw=2.2, label="robot trajectory (DWA)", zorder=5)

    snaps = np.linspace(0, t_final, 5)
    colors = ["#1f77b4", "#ff7f0e", "#9467bd"]
    for oi, (ox, oy, vx, vy, r) in enumerate(obstacles):
        # 장애물 경로선
        xs = ox + vx * snaps
        ys = oy + vy * snaps
        ax.plot(ox + vx * np.linspace(0, t_final, 50),
                oy + vy * np.linspace(0, t_final, 50),
                "--", color=colors[oi], lw=1.0, alpha=0.6)
        # 시간 스냅샷(원)
        for si, ts in enumerate(snaps):
            alpha = 0.15 + 0.6 * si / (len(snaps) - 1)
            ax.add_patch(plt.Circle((ox + vx * ts, oy + vy * ts), r,
                                    color=colors[oi], alpha=alpha,
                                    label=f"obstacle {oi+1}" if si == len(snaps) - 1 else None))

    ax.plot(start[0], start[1], "ko", ms=11, label="start", zorder=6)
    ax.plot(goal[0], goal[1], "g*", ms=20, label="goal", zorder=6)
    ax.add_patch(plt.Circle((goal[0], goal[1]), 0.5, color="green", alpha=0.15))

    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    status = "reached" if reached else "FAILED"
    ax.set_title(f"DWA local planner vs moving obstacles: {status}, "
                 f"min clearance {min_clear:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()

    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(f"{d}/20_dwa_dynamic.png", dpi=130)
    print("\n[plot] outputs/20_dwa_dynamic.png, assets/20_dwa_dynamic.png")

    return reached, min_clear


if __name__ == "__main__":
    main()
