"""학습 기반 IMU 프론트엔드: 1D-CNN 디노이저로 dead-reckoning 드리프트 감소.

2026 로봇 트렌드 = 학습 + 추정. IMU를 적분하면 잡음이 드리프트로 쌓인다. 작은 1D-CNN이
raw IMU를 정제해 넣으면 드리프트가 줄어드는지, 고전 저역통과와 비교한다.

현실적 IMU 잡음 = 백색 + 랜덤워크 바이어스 + 스파이크(비가우시안). 스펙트럼으로 안
갈리는 이 구조를 학습이 고전 필터보다 잘 제거하는지 본다(신호처리 통찰의 재확인).

필요: torch (선택 의존성). python scripts/16_learned_imu_frontend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("torch 필요: pip install torch (CPU)"); sys.exit(0)

DT = 0.02   # 50Hz IMU
WIN = 256


def true_accel(n, rng):
    """부드러운 1D 가속도 프로파일(사인 합)."""
    t = np.arange(n) * DT
    a = np.zeros(n)
    for _ in range(4):
        f = rng.uniform(0.2, 1.5); ph = rng.uniform(0, 2*np.pi); amp = rng.uniform(0.3, 1.0)
        a += amp * np.sin(2*np.pi*f*t + ph)
    return a.astype(np.float32)


def corrupt(a, rng):
    """백색 + 랜덤워크 바이어스 + 스파이크."""
    n = a.size
    white = rng.normal(0, 0.4, n)
    bias = np.cumsum(rng.normal(0, 0.006, n))      # 랜덤워크 드리프트
    spikes = np.zeros(n)
    idx = rng.choice(n, size=max(1, n//40), replace=False)
    spikes[idx] = rng.normal(0, 3.0, idx.size)     # 비가우시안 스파이크
    return (a + white + bias + spikes).astype(np.float32)


class Denoiser(nn.Module):
    def __init__(self, ch=32, k=9):
        super().__init__()
        p = k//2
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, k, padding=p), nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=p), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=p), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, 1, k, padding=p))

    def forward(self, x):
        return x - self.net(x)   # 잔차: 입력 - 추정잡음


def lowpass(x, alpha=0.15):
    y = np.empty_like(x); acc = x[0]
    for i in range(x.size):
        acc = alpha*x[i] + (1-alpha)*acc; y[i] = acc
    return y


def main():
    rng = np.random.default_rng(0)
    # 학습 데이터: (noisy, clean) 윈도우
    N = 3000
    C = np.empty((N, WIN), np.float32); Z = np.empty((N, WIN), np.float32)
    for i in range(N):
        a = true_accel(WIN, rng); C[i] = a; Z[i] = corrupt(a, rng)

    model = Denoiser()
    opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.MSELoss()
    xb = torch.from_numpy(Z).unsqueeze(1); yb = torch.from_numpy(C).unsqueeze(1)
    ds = torch.utils.data.TensorDataset(xb, yb)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    print("[train] 1D-CNN IMU 디노이저...")
    for ep in range(15):
        for a, b in dl:
            opt.zero_grad(); lossf(model(a), b).backward(); opt.step()
    model.eval()

    # 테스트 궤적 → dead-reckoning(이중적분) 위치 드리프트
    n = 1500
    a_true = true_accel(n, np.random.default_rng(99))
    a_noisy = corrupt(a_true, np.random.default_rng(100))
    a_lp = lowpass(a_noisy)
    with torch.no_grad():
        a_ml = model(torch.from_numpy(a_noisy).view(1, 1, -1)).view(-1).numpy()

    def deadreckon(a):
        v = np.cumsum(a) * DT
        return np.cumsum(v) * DT
    p_true = deadreckon(a_true)
    def rmse(a): return float(np.sqrt(np.mean((deadreckon(a) - p_true)**2)))

    print("=== IMU dead-reckoning 위치 드리프트 RMSE ===")
    print(f"raw IMU        : {rmse(a_noisy):.2f} m")
    print(f"classical LP   : {rmse(a_lp):.2f} m")
    print(f"learned (1DCNN): {rmse(a_ml):.2f} m")
    print(f"→ 학습이 raw 대비 {rmse(a_noisy)/max(rmse(a_ml),1e-6):.1f}x, 고전 대비 {rmse(a_lp)/max(rmse(a_ml),1e-6):.1f}x")

    t = np.arange(n)*DT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(t[:400], a_true[:400], "g-", lw=1.5, label="true accel")
    ax1.plot(t[:400], a_noisy[:400], color="r", lw=0.5, alpha=0.5, label="noisy IMU")
    ax1.plot(t[:400], a_ml[:400], "b-", lw=1.0, label="learned denoised")
    ax1.set_title("IMU signal (first 8s)"); ax1.legend(fontsize=8); ax1.set_xlabel("t [s]")
    ax2.plot(t, p_true, "g-", lw=2, label="true position")
    ax2.plot(t, deadreckon(a_noisy), "r-", lw=1, alpha=0.7, label=f"raw ({rmse(a_noisy):.0f}m)")
    ax2.plot(t, deadreckon(a_lp), color="orange", lw=1, label=f"low-pass ({rmse(a_lp):.0f}m)")
    ax2.plot(t, deadreckon(a_ml), "b-", lw=1.2, label=f"learned ({rmse(a_ml):.0f}m)")
    ax2.set_title("Dead-reckoned position (double integration)"); ax2.legend(fontsize=8); ax2.set_xlabel("t [s]")
    fig.suptitle("Learned IMU front-end reduces dead-reckoning drift (ML + estimation)")
    fig.tight_layout(); fig.savefig("outputs/16_learned_imu_frontend.png", dpi=130)
    print("\n[plot] outputs/16_learned_imu_frontend.png")
    return rmse(a_noisy), rmse(a_lp), rmse(a_ml)


if __name__ == "__main__":
    main()
