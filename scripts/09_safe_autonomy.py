"""불확실도-인지 안전 정지 (수술 로봇 'No-Fly Zone'의 상태추정 버전).

자율 로봇이 금지구역(critical structure) 앞까지 접근한다. 센서가 근처에서 끊겨
위치 추정 불확실도가 커지는데, 두 정지 규칙을 비교한다:
  - naive     : 추정 위치만 보고 정지 판단 (불확실도 무시)
  - uncertainty-aware : 추정 위치 + k·σ(불확실도 마진)로 정지 판단

추정이 과대낙관일 때 naive는 금지구역을 침범한다. 불확실도-인지 게이트는 '모를 때
멈춘다' — Task Autonomy 시대 의료/수술 로봇의 안전 배치 원리.

    python scripts/09_safe_autonomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
V = 1.0                 # 접근 속도 [m/s]
ZONE = 20.0            # 금지구역 경계 [m]
BUFFER = 1.0          # 안전 버퍼 → 19m 전에 멈춰야 함
POS_STD = 0.3
DROPOUT_X = 15.0     # 이 지점부터 위치센서 끊김(근처에서 관측 악화)
K_SIGMA = 3.0        # 불확실도 마진 계수


def trial(seed, aware):
    rng = np.random.default_rng(seed)
    p_true = 0.0
    # 1D EKF: 상태 [pos, vel]
    x = np.array([0.0, V]); P = np.diag([0.04, 0.01])
    F = np.array([[1, DT], [0, 1]]); Q = np.diag([1e-4, 1e-3])
    H = np.array([[1.0, 0.0]]); R = np.array([[POS_STD**2]])
    stopped = False; violated = False
    for _ in range(400):
        # 참 이동(+ 약간의 실제 요동)
        p_true += V * DT + rng.normal(0, 0.01)
        # 예측
        x = F @ x; P = F @ P @ F.T + Q
        # 위치 측정(단, DROPOUT 이후 끊김)
        if p_true < DROPOUT_X:
            z = p_true + rng.normal(0, POS_STD)
            y = z - (H @ x)[0]
            S = (H @ P @ H.T)[0, 0] + R[0, 0]
            Kk = (P @ H.T / S).ravel()
            x = x + Kk * y; P = (np.eye(2) - np.outer(Kk, H)) @ P
        # 정지 판단
        sigma = np.sqrt(P[0, 0])
        margin = K_SIGMA * sigma if aware else 0.0
        if x[0] + margin >= ZONE - BUFFER:
            stopped = True
            break
        # 실제 침범(참 위치가 안전선을 넘음)
        if p_true >= ZONE - BUFFER:
            violated = True
            break
    # 멈췄더라도 그 시점 참 위치가 이미 안전선을 넘었는지 확인
    if p_true >= ZONE - BUFFER:
        violated = True
    return violated, p_true, x[0], np.sqrt(P[0, 0])


def main():
    N = 300
    for aware in [False, True]:
        res = [trial(s, aware) for s in range(N)]
        viol = np.mean([r[0] for r in res])
        final_true = np.mean([r[1] for r in res])
        label = "uncertainty-aware" if aware else "naive"
        print(f"[{label:18s}] 침범률={viol*100:5.1f}%  평균 정지 참위치={final_true:.2f} m (안전선 {ZONE-BUFFER})")

    # 대표 1회 궤적 시각화(불확실도 성장 + 정지점)
    def run_trace(aware, seed=7):
        rng = np.random.default_rng(seed)
        p_true = 0.0; x = np.array([0.0, V]); P = np.diag([0.04, 0.01])
        F = np.array([[1, DT], [0, 1]]); Q = np.diag([1e-4, 1e-3]); H = np.array([[1.0, 0.0]]); R = np.array([[POS_STD**2]])
        ts, tp, ep, es = [], [], [], []
        for k in range(400):
            p_true += V*DT + rng.normal(0, 0.01)
            x = F @ x; P = F @ P @ F.T + Q
            if p_true < DROPOUT_X:
                z = p_true + rng.normal(0, POS_STD); y = z-(H@x)[0]; S=(H@P@H.T)[0,0]+R[0,0]
                Kk=(P@H.T/S).ravel(); x=x+Kk*y; P=(np.eye(2)-np.outer(Kk,H))@P
            ts.append(k*DT); tp.append(p_true); ep.append(x[0]); es.append(np.sqrt(P[0,0]))
            margin = K_SIGMA*np.sqrt(P[0,0]) if aware else 0.0
            if x[0]+margin >= ZONE-BUFFER or p_true >= ZONE-BUFFER:
                break
        return map(np.array, (ts, tp, ep, es))

    fig, ax = plt.subplots(figsize=(9, 5))
    for aware, col in [(False, "r"), (True, "b")]:
        ts, tp, ep, es = run_trace(aware)
        lab = "uncertainty-aware" if aware else "naive"
        ax.plot(ts, tp, col+"-", lw=1.6, label=f"{lab}: true pos (stop {tp[-1]:.1f}m)")
        ax.fill_between(ts, ep-K_SIGMA*es, ep+K_SIGMA*es, color=col, alpha=0.15)
    ax.axhline(ZONE-BUFFER, color="k", ls="--", label="safety line (19m)")
    ax.axhspan(ZONE-BUFFER, ZONE+1, color="red", alpha=0.08)
    ax.axvline(DROPOUT_X/V, color="gray", ls=":", lw=0.8, label="sensor dropout")
    ax.set_xlabel("time [s]"); ax.set_ylabel("position [m]")
    ax.set_title("Uncertainty-aware safe-stop vs naive (surgical No-Fly-Zone analog)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path("outputs") / "09_safe_autonomy.png", dpi=130)
    print("\n[plot] outputs/09_safe_autonomy.png")
    # 반환: (naive 침범률, aware 침범률)
    return (np.mean([trial(s, False)[0] for s in range(N)]),
            np.mean([trial(s, True)[0] for s in range(N)]))


if __name__ == "__main__":
    main()
