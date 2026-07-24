"""계획(Planning) + 제어(Control): 측위 위에 '목표로 가는' 능력을 얹는다.

측위/SLAM은 '나는 어디에?'를 푼다. 그 위에 경로계획(A*)과 추종제어(pure-pursuit)를
얹으면 로봇이 장애물과 no-go 존(의료/랩의 민감 장비 = 안전구역)을 피해 목표까지
자율 주행한다. 로봇 스택의 estimation → action 고리를 완성.

시나리오: 랩 자동화 로봇이 no-go 존을 피해 샘플 스테이션(목표)으로 이동.

    python scripts/19_plan_control.py
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
GRID = 0.5          # 격자 해상도 [m]
ROBOT_R = 0.5       # 로봇 반경(인플레이션)


def make_map():
    W, H = 40.0, 30.0
    obstacles = [(6, 0, 2, 18), (16, 12, 2, 18), (26, 0, 2, 20), (32, 14, 6, 2)]  # x,y,w,h 벽
    nogo = (21, 7, 3.2)  # 민감 장비 = no-go 원 (cx,cy,r)
    return W, H, obstacles, nogo


def occupancy(W, H, obstacles, nogo):
    nx, ny = int(W/GRID), int(H/GRID)
    occ = np.zeros((ny, nx), bool)
    for (ox, oy, ow, oh) in obstacles:
        i0, i1 = int(oy/GRID), int((oy+oh)/GRID); j0, j1 = int(ox/GRID), int((ox+ow)/GRID)
        occ[i0:i1, j0:j1] = True
    cx, cy, r = nogo
    for i in range(ny):
        for j in range(nx):
            if np.hypot((j+0.5)*GRID-cx, (i+0.5)*GRID-cy) < r:
                occ[i, j] = True
    # 로봇 반경만큼 인플레이션
    infl = occ.copy(); rad = int(np.ceil(ROBOT_R/GRID))
    ys, xs = np.where(occ)
    for y, x in zip(ys, xs):
        infl[max(0, y-rad):y+rad+1, max(0, x-rad):x+rad+1] = True
    return occ, infl


def astar(infl, start, goal):
    ny, nx = infl.shape
    def c2(p): return (int(p[1]/GRID), int(p[0]/GRID))  # (row,col)
    s, g = c2(start), c2(goal)
    if infl[s] or infl[g]:
        return None
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    openh = [(0, s)]; came = {}; gsc = {s: 0}
    while openh:
        _, cur = heapq.heappop(openh)
        if cur == g:
            path = [cur]
            while cur in came: cur = came[cur]; path.append(cur)
            path.reverse()
            return np.array([[(c+0.5)*GRID, (r+0.5)*GRID] for (r, c) in path])
        for dr, dc in nbrs:
            nr, ncc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < ny and 0 <= ncc < nx) or infl[nr, ncc]:
                continue
            step = np.hypot(dr, dc)
            ng = gsc[cur] + step
            if ng < gsc.get((nr, ncc), 1e18):
                gsc[(nr, ncc)] = ng; came[(nr, ncc)] = cur
                h = np.hypot(nr-g[0], ncc-g[1])
                heapq.heappush(openh, (ng+h, (nr, ncc)))
    return None


def pure_pursuit(path, start_pose, v=3.0, Ld=2.5, goal_tol=0.6, max_steps=4000):
    x = np.array(start_pose, float); traj = [x[:2].copy()]
    gi = 0
    for _ in range(max_steps):
        if np.hypot(*(path[-1]-x[:2])) < goal_tol:
            break
        # 전방주시점: 현재 위치에서 Ld 이상 떨어진 첫 경로점
        while gi < len(path)-1 and np.hypot(*(path[gi]-x[:2])) < Ld:
            gi += 1
        tgt = path[gi]
        ang = np.arctan2(tgt[1]-x[1], tgt[0]-x[0])
        derr = (ang - x[2] + np.pi) % (2*np.pi) - np.pi
        w = 2.0*derr                    # 비례 조향
        x = np.array([x[0]+v*DT*np.cos(x[2]), x[1]+v*DT*np.sin(x[2]), x[2]+w*DT])
        traj.append(x[:2].copy())
    return np.array(traj)


def main():
    W, H, obstacles, nogo = make_map()
    occ, infl = occupancy(W, H, obstacles, nogo)
    start, goal = (2, 2), (37, 27)
    path = astar(infl, start, goal)
    if path is None:
        print("경로 없음"); return
    traj = pure_pursuit(path, [start[0], start[1], 0.0])

    reached = np.hypot(*(traj[-1]-np.array(goal))) < 1.0
    # no-go 존 최소 클리어런스
    cx, cy, r = nogo
    clr = min(np.hypot(traj[:, 0]-cx, traj[:, 1]-cy)) - r
    plen = np.sum(np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1])))
    print("=== 계획 + 제어: 랩 로봇 자율 주행 ===")
    print(f"A* 경로점 {len(path)}, 목표 도달: {'예' if reached else '아니오'}")
    print(f"no-go 존 최소 클리어런스: {clr:.2f} m ({'안전' if clr > 0 else '침범!'})")
    print(f"주행 거리 {plen:.1f} m")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for (ox, oy, ow, oh) in obstacles:
        ax.add_patch(plt.Rectangle((ox, oy), ow, oh, color="#555"))
    ax.add_patch(plt.Circle((cx, cy), r, color="#d9534f", alpha=0.4))
    ax.text(cx, cy, "NO-GO", ha="center", va="center", color="#7a1a1a", fontsize=8, weight="bold")
    ax.plot(path[:, 0], path[:, 1], "b--", lw=1.2, alpha=0.7, label="A* planned path")
    ax.plot(traj[:, 0], traj[:, 1], "g-", lw=2, label="executed (pure-pursuit)")
    ax.plot(*start, "ko", ms=10, label="start"); ax.plot(*goal, "g*", ms=16, label="goal (sample station)")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Plan (A*) + Control (pure-pursuit): reach goal, avoid no-go zone (clearance {clr:.1f}m)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig("outputs/19_plan_control.png", dpi=130)
    print("\n[plot] outputs/19_plan_control.png")
    return reached, clr


if __name__ == "__main__":
    main()
