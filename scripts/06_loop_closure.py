"""루프 클로저 SLAM: 출발점 재방문이 누적 드리프트를 교정한다.

로봇이 원형 루프를 돌며 랜드마크로 SLAM을 한다. 루프를 도는 동안 지도·pose에 미세
드리프트가 쌓인다. 마지막에 출발점으로 돌아와 **초기 랜드마크를 재관측**하면, 공분산
상관을 타고 보정이 과거로 전파돼 전체 궤적·지도가 정렬된다(loop closure).

두 모드를 비교:
  - no-closure : 복귀 시 초기 랜드마크 재관측을 무시(루프 미검출) → 드리프트 잔존
  - closure    : 재관측 반영 → 드리프트 일괄 교정

    python scripts/06_loop_closure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
SENSOR_RANGE = 13.0        # 좁게 → 랜드마크 사이 공백에선 dead-reckoning
R_STD, B_STD = 0.5, 0.02
V_STD, W_STD = 0.12, 0.08  # 헤딩 잡음(W)으로 루프 도는 동안 드리프트 누적
START_LMS = 2              # 출발점 근처 '앵커' 랜드마크 수(재관측=루프클로저)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def motion(p, v, w):
    x, y, th = p
    return np.array([x + v * DT * np.cos(th), y + v * DT * np.sin(th), wrap(th + w * DT)])


def run(landmarks, controls, loop_closure: bool, seed=0):
    rng = np.random.default_rng(seed)
    M = len(landmarks)
    dim = 3 + 2 * M
    x = np.zeros(dim); P = np.zeros((dim, dim))
    P[:3, :3] = np.diag([0.01, 0.01, 0.01]); P[3:, 3:] = np.eye(2 * M) * 1e4
    seen = [False] * M
    Rmot = np.diag([(3 * V_STD)**2, (3 * W_STD)**2])
    Robs = np.diag([R_STD**2, B_STD**2])
    tp = np.zeros(3); traj_t, traj_e = [], []
    N = len(controls)

    for step, (v, w) in enumerate(controls):
        tp = motion(tp, v, w)
        vn, wn = v + rng.normal(0, V_STD), w + rng.normal(0, W_STD)

        th = x[2]
        x[:3] = motion(x[:3], vn, wn)
        Gr = np.array([[1, 0, -vn*DT*np.sin(th)], [0, 1, vn*DT*np.cos(th)], [0, 0, 1]])
        Vr = np.array([[DT*np.cos(th), 0], [DT*np.sin(th), 0], [0, DT]])
        G = np.eye(dim); G[:3, :3] = Gr
        P = G @ P @ G.T; P[:3, :3] += Vr @ Rmot @ Vr.T

        in_return = step > 0.75 * N   # 복귀 구간
        for j in range(M):
            d = landmarks[j] - tp[:2]; r = np.hypot(*d)
            if r > SENSOR_RANGE or r < 1.5:
                continue
            # no-closure: 복귀 구간에서 초기 앵커 랜드마크 재관측을 무시
            if (not loop_closure) and in_return and j < START_LMS:
                continue
            z = np.array([r + rng.normal(0, R_STD),
                          wrap(np.arctan2(d[1], d[0]) - tp[2] + rng.normal(0, B_STD))])
            li = 3 + 2 * j
            if not seen[j]:
                rx, ry, rth = x[:3]; rr, bb = z
                cb, sb = np.cos(bb+rth), np.sin(bb+rth)
                x[li] = rx + rr*cb; x[li+1] = ry + rr*sb
                Grp = np.array([[1, 0, -rr*sb], [0, 1, rr*cb]]); Gz = np.array([[cb, -rr*sb], [sb, rr*cb]])
                P[li:li+2, li:li+2] = Grp @ P[:3, :3] @ Grp.T + Gz @ Robs @ Gz.T
                P[li:li+2, :li] = Grp @ P[:3, :li]; P[:li, li:li+2] = P[li:li+2, :li].T
                seen[j] = True; continue
            dx, dy = x[li]-x[0], x[li+1]-x[1]; q = dx*dx+dy*dy; sq = np.sqrt(q)
            if sq < 2.0:
                continue
            zhat = np.array([sq, wrap(np.arctan2(dy, dx) - x[2])])
            Hl = np.array([[-sq*dx, -sq*dy, 0, sq*dx, sq*dy], [dy, -dx, -q, -dy, dx]]) / q
            H = np.zeros((2, dim)); H[:, :3] = Hl[:, :3]; H[:, li:li+2] = Hl[:, 3:5]
            y = z - zhat; y[1] = wrap(y[1])
            S = H @ P @ H.T + Robs; Sinv = np.linalg.inv(S)
            # 루프 클로저(복귀 시 앵커 재관측)는 큰 이노베이션이 정상 → 게이트 우회
            is_closure = loop_closure and in_return and j < START_LMS
            if (not is_closure) and y @ Sinv @ y > 16:
                continue
            K = P @ H.T @ Sinv; x = x + K @ y; x[2] = wrap(x[2])
            IKH = np.eye(dim) - K @ H; P = IKH @ P @ IKH.T + K @ Robs @ K.T; P = 0.5*(P+P.T)

        traj_t.append(tp[:2].copy()); traj_e.append(x[:2].copy())

    return np.array(traj_t), np.array(traj_e), x[3:].reshape(-1, 2), np.array(seen)


def main() -> None:
    # 원형 루프: 출발(0,0) 헤딩0 → 좌회전 CCW → 중심 (0,R) 원 → 한 바퀴+α 후 출발점 재방문
    v = 6.0; R = 22.0; w = v / R
    N = int(2 * np.pi * R / v / DT) + 20
    controls = [(v, w)] * N
    center = np.array([0.0, R])

    # 링 위 랜드마크 8개(로봇이 5m 안쪽 통과). 첫 2개=출발점(바닥, -90°) 근처 앵커
    ang = np.linspace(-np.pi/2, 3*np.pi/2, 9)[:-1]
    landmarks = center + (R + 5) * np.stack([np.cos(ang), np.sin(ang)], axis=1)

    tt, ee_no, lm_no, seen = run(landmarks, controls, loop_closure=False, seed=1)
    _, ee_yes, lm_yes, _ = run(landmarks, controls, loop_closure=True, seed=1)

    def rmse(e, idx=slice(None)):
        return float(np.sqrt(np.mean(np.sum((e[idx] - tt[idx])**2, axis=1))))
    ret = slice(int(0.75 * len(tt)), None)  # 복귀 구간(드리프트 최대·클로저 발생)
    print("=== 루프 클로저 효과 ===")
    print(f"no-closure  : 전체 RMSE={rmse(ee_no):.2f} m,  복귀구간 RMSE={rmse(ee_no, ret):.2f} m")
    print(f"closure     : 전체 RMSE={rmse(ee_yes):.2f} m,  복귀구간 RMSE={rmse(ee_yes, ret):.2f} m")
    print(f"→ 복귀 구간 드리프트를 {rmse(ee_no, ret)/max(rmse(ee_yes, ret),1e-6):.1f}배 줄임")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for ax, ee, lm, title in [(ax1, ee_no, lm_no, f"NO loop closure (RMSE {rmse(ee_no):.2f}m)"),
                              (ax2, ee_yes, lm_yes, f"WITH loop closure (RMSE {rmse(ee_yes):.2f}m)")]:
        ax.plot(tt[:, 0], tt[:, 1], "g-", lw=2, label="true")
        ax.plot(ee[:, 0], ee[:, 1], "b-", lw=1.3, label="EKF-SLAM")
        ax.plot(landmarks[seen, 0], landmarks[seen, 1], "g*", ms=12)
        ax.plot(lm[seen, 0], lm[seen, 1], "bx", ms=8)
        ax.plot(ee[0, 0], ee[0, 1], "ko", ms=8, label="start")
        ax.plot(ee[-1, 0], ee[-1, 1], "rs", ms=8, label="end (est)")
        ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title(title); ax.grid(alpha=0.3)
    fig.suptitle("Loop closure: revisiting the start corrects accumulated drift")
    fig.tight_layout()
    fig.savefig(Path("outputs") / "06_loop_closure.png", dpi=130)
    print("\n[plot] outputs/06_loop_closure.png")
    return rmse(ee_no, ret), rmse(ee_yes, ret)


if __name__ == "__main__":
    main()
