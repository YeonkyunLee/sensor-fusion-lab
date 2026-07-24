"""생리적 수전증(hand tremor) 제거: 미세수술 로봇의 손떨림 상쇄.

미세수술/현미수술 로봇(steady-hand, tremor-canceling tool)은 집도의의 손에서
나오는 '의도한 움직임'(저주파, 크게)은 그대로 따르면서, 생리적 수전증
(physiological tremor, 약 8~12 Hz, 진폭 수백 마이크론)만 걷어내야 한다.
떨림을 그대로 툴에 전달하면 조직 손상·정밀도 저하로 이어지기 때문이다.

이 실험은 집도의의 손 움직임을 합성한다:
    측정 = 의도한 부드러운 리칭 궤적(저주파) + ~10 Hz 수전증 + 센서 노이즈
그리고 네 가지 상쇄 기법을 비교한다:
  (1) 저역통과(Butterworth, 영위상 filtfilt)      — 비인과(오프라인)
  (2) 노치/대역저지(~10 Hz band-stop, filtfilt)   — 비인과(오프라인)
  (3) 등속(constant-velocity) 칼만 스무더           — 인과(실시간)
  (4) 적응형 푸리에 선형결합기(FLC, 적응 노치)      — 인과(실시간)

각 기법의 잔여 떨림(8~12 Hz 대역 RMS)과 의도 궤적 대비 추종오차를 비교하고,
'떨림 억제 vs 지연(lag)'의 트레이드오프를 정직하게 보고한다.

    python scripts/22_surgical_tremor.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

FS = 500.0              # 샘플링 [Hz] (수술 로봇 제어 루프의 현실적 대역)
T = 6.0                # 관측 시간 [s]
F_TREMOR = 10.0        # 생리적 수전증 중심 주파수 [Hz]
A_TREMOR = 0.20        # 수전증 진폭 [mm] (~200 마이크론)
MEAS_STD = 0.02        # 센서 노이즈 [mm] (~20 마이크론)
TREMOR_BAND = (8.0, 12.0)


# ---------------------------------------------------------------- 신호 합성
def intended_motion(t):
    """의도한 2D 리칭 궤적 [mm]: 최소저크 리치 + 마지막 빠른 미세보정."""
    tau = np.clip(t / T, 0, 1)
    mj = 10 * tau**3 - 15 * tau**4 + 6 * tau**5      # 최소저크 프로파일 0->1
    ix = 30.0 * mj                                    # x: 0 -> 30 mm 리치
    iy = 15.0 * np.sin(np.pi * mj)                    # y: 곡선 아크(0->15->0)
    # 4.2~4.8 s 사이 빠른 미세보정(집도의가 위치를 다듬는 동작) -> 인과필터 지연 노출
    w = np.clip((t - 4.2) / 0.6, 0, 1)
    bump = 10 * w**3 - 15 * w**4 + 6 * w**5
    ix = ix + 4.0 * bump
    return ix, iy


def synth(seed=0):
    rng = np.random.default_rng(seed)
    n = int(T * FS)
    t = np.arange(n) / FS
    ix, iy = intended_motion(t)

    # 수전증: 진폭 변조 + 미세한 주파수 흔들림(현실성) — 두 축 위상 다르게
    fjit = F_TREMOR + 0.3 * np.sin(2 * np.pi * 0.4 * t)      # 주파수 미세 요동
    phase = 2 * np.pi * np.cumsum(fjit) / FS
    amp = A_TREMOR * (1.0 + 0.3 * np.sin(2 * np.pi * 1.3 * t))
    tremor_x = amp * np.sin(phase)
    tremor_y = 0.85 * amp * np.sin(phase + 1.1)

    nx = rng.normal(0, MEAS_STD, n)
    ny = rng.normal(0, MEAS_STD, n)
    rx = ix + tremor_x + nx
    ry = iy + tremor_y + ny
    return t, (ix, iy), (rx, ry)


# ---------------------------------------------------------------- 지표
def _bandpass(x):
    b, a = signal.butter(4, np.array(TREMOR_BAND) / (FS / 2), btype="band")
    return signal.filtfilt(b, a, x)


def tremor_rms(x, y):
    """8~12 Hz 대역에 남은 떨림 에너지의 RMS(두 축 합성) [mm]."""
    bx, by = _bandpass(x), _bandpass(y)
    return float(np.sqrt(np.mean(bx**2 + by**2)))


def tracking_rms(ox, oy, ix, iy):
    """의도 궤적 대비 추종오차 RMS [mm]."""
    return float(np.sqrt(np.mean((ox - ix) ** 2 + (oy - iy) ** 2)))


# ---------------------------------------------------------------- 기법들
def lowpass(x, y, fc=5.0):
    b, a = signal.butter(4, fc / (FS / 2), btype="low")
    return signal.filtfilt(b, a, x), signal.filtfilt(b, a, y)


def bandstop(x, y):
    b, a = signal.butter(4, np.array([7.0, 13.0]) / (FS / 2), btype="bandstop")
    return signal.filtfilt(b, a, x), signal.filtfilt(b, a, y)


def kalman_cv(z, meas_std=0.10, sa=10.0):
    """등속 모델 칼만 필터(인과). meas_std를 크게 잡아 모델을 신뢰 -> 떨림 스무딩.

    z: (n,) 위치 측정. 반환: 위치 추정 (n,). 지연(lag)이 대가로 따라온다.
    """
    dt = 1.0 / FS
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1.0, 0.0]])
    q = sa**2
    Q = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    R = np.array([[meas_std**2]])
    x = np.array([z[0], 0.0])
    P = np.diag([meas_std**2, 1.0])
    out = np.empty(len(z))
    I = np.eye(2)
    for k, zk in enumerate(z):
        x = F @ x
        P = F @ P @ F.T + Q
        S = (H @ P @ H.T)[0, 0] + R[0, 0]
        K = (P @ H.T / S).ravel()
        x = x + K * (zk - (H @ x)[0])
        P = (I - np.outer(K, H)) @ P
        out[k] = x[0]
    return out


def flc(z, f0, mu=0.05, n_harm=1, fc=4.0):
    """적응형 푸리에 선형결합기(FLC, 인과·실시간) — 적응 노치.

    수전증을 f0의 사인/코사인(+고조파) 선형결합으로 모델링하고 LMS로 진폭을
    적응 추정한 뒤 측정에서 빼준다:  출력 = 측정 - 떨림추정.

    핵심: 의도 움직임(30 mm)은 떨림(0.2 mm)보다 ~150배 커서, 순수 LMS는 이
    거대한 저주파 성분이 그래디언트를 지배해 가중치를 흔들어 놓는다. 그래서
    가중치 갱신에는 오차를 고역통과(fc Hz)한 성분만 사용해(=떨림 대역만 보고
    적응) 의도 움직임의 오염을 제거한다. 출력은 위상 왜곡 없이 전대역 그대로.

    z: (n,) 위치 측정. 반환: 떨림 제거된 위치 (n,).
    """
    n = len(z)
    dt = 1.0 / FS
    bh, ah = signal.butter(2, fc / (FS / 2), btype="high")   # 그래디언트용 고역통과
    v = np.zeros(2)                                            # 고역통과 상태(DF2T)
    w = np.zeros(2 * n_harm)                                   # [a1,b1,a2,b2,...]
    out = np.empty(n)
    norm = float(n_harm)                                       # ||r||^2 = n_harm
    for k in range(n):
        t = k * dt
        r = np.empty(2 * n_harm)
        for h in range(n_harm):
            r[2 * h] = np.sin(2 * np.pi * f0 * (h + 1) * t)
            r[2 * h + 1] = np.cos(2 * np.pi * f0 * (h + 1) * t)
        y = float(w @ r)                                      # 떨림 추정
        err = z[k] - y                                        # 출력 = 의도 + 노이즈(떨림 제거)
        out[k] = err
        # 오차를 고역통과 -> 떨림 대역만 남긴 갱신 신호(DF2T 1스텝)
        ehp = bh[0] * err + v[0]
        v[0] = bh[1] * err - ah[1] * ehp + v[1]
        v[1] = bh[2] * err - ah[2] * ehp
        w = w + 2 * mu * ehp * r / norm                       # LMS 진폭 적응
    return out


def estimate_tremor_freq(x):
    """대역통과 후 FFT 피크로 수전증 중심 주파수 추정 (7~14 Hz 내)."""
    xb = _bandpass(x)
    freqs = np.fft.rfftfreq(len(xb), 1.0 / FS)
    mag = np.abs(np.fft.rfft(xb * np.hanning(len(xb))))
    band = (freqs >= 7) & (freqs <= 14)
    return float(freqs[band][np.argmax(mag[band])])


# ---------------------------------------------------------------- main
def main():
    t, (ix, iy), (rx, ry) = synth(seed=0)

    f0 = estimate_tremor_freq(rx)     # 실측에서 떨림 주파수 자동 추정

    methods = {}
    methods["low-pass (filtfilt, 5Hz)"] = lowpass(rx, ry)
    methods["band-stop (filtfilt, 7-13Hz)"] = bandstop(rx, ry)
    methods["Kalman CV (causal)"] = (kalman_cv(rx), kalman_cv(ry))
    methods["adaptive FLC (causal)"] = (flc(rx, f0), flc(ry, f0))

    raw_tremor = tremor_rms(rx, ry)
    raw_track = tracking_rms(rx, ry, ix, iy)

    print("=== 수술 로봇 생리적 수전증 제거 ===")
    print(f"샘플링 {FS:.0f} Hz, 관측 {T:.0f} s, 추정 떨림 주파수 f0 = {f0:.2f} Hz")
    print(f"의도 리칭 진폭 ~30 mm,  주입 수전증 진폭 ~{A_TREMOR*1000:.0f} um")
    print(f"\n[raw]  잔여떨림 RMS = {raw_tremor*1000:7.1f} um   추종오차 RMS = {raw_track*1000:7.1f} um")
    print("-" * 74)
    print(f"{'method':32s}{'잔여떨림[um]':>14s}{'억제배율':>10s}{'추종오차[um]':>14s}")
    print("-" * 74)

    results = {}
    for name, (ox, oy) in methods.items():
        tr = tremor_rms(ox, oy)
        te = tracking_rms(ox, oy, ix, iy)
        results[name] = (tr, te)
        print(f"{name:32s}{tr*1000:14.1f}{raw_tremor/tr:9.1f}x{te*1000:14.1f}")
    print("-" * 74)

    # 헤드라인: 실시간(인과) 최적 기법 = 적응형 FLC
    best_name = "adaptive FLC (causal)"
    filt_tremor, track_err = results[best_name]
    print(f"\n권장(실시간): {best_name}")
    print(f"  떨림 {raw_tremor*1000:.0f} um -> {filt_tremor*1000:.1f} um "
          f"({raw_tremor/filt_tremor:.1f}배 억제),  추종오차 {track_err*1000:.1f} um")
    print("트레이드오프:")
    print("  - filtfilt(저역/대역저지)는 영위상이라 억제·추종 모두 최고지만 비인과(오프라인 후처리).")
    print("  - Kalman CV는 실시간이나 전대역 저역 스무딩이라 빠른 의도동작에서 지연(lag)이 크다")
    print("    -> 억제는 되지만 추종오차가 상대적으로 큼.")
    print("  - 적응형 FLC는 떨림 대역만 노치처럼 제거 -> 실시간이면서 의도동작을 잘 보존(최적 실시간).")

    _plot(t, (ix, iy), (rx, ry), methods, results, raw_tremor, raw_track, f0)

    # (raw 떨림, 필터후 떨림, 추종오차) [mm]
    return raw_tremor, filt_tremor, track_err


# ---------------------------------------------------------------- 시각화
def _plot(t, intended, raw, methods, results, raw_tremor, raw_track, f0):
    ix, iy = intended
    rx, ry = raw
    flc_x, flc_y = methods["adaptive FLC (causal)"]
    lp_x, lp_y = methods["low-pass (filtfilt, 5Hz)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (A) x축 시계열 (전체 + 확대 삽입)
    ax = axes[0, 0]
    ax.plot(t, rx, color="#c0c0c0", lw=0.8, label="raw (tremulous hand)")
    ax.plot(t, ix, "k--", lw=1.4, label="intended motion")
    ax.plot(t, flc_x, color="#2ca02c", lw=1.2, label="adaptive FLC (causal)")
    ax.set_title("(A) x-position: hand tremor cancelled, intended motion preserved")
    ax.set_xlabel("time [s]"); ax.set_ylabel("x [mm]")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    axi = ax.inset_axes([0.55, 0.08, 0.42, 0.42])
    m = (t >= 2.0) & (t <= 2.4)
    axi.plot(t[m], rx[m], color="#c0c0c0", lw=1.0)
    axi.plot(t[m], ix[m], "k--", lw=1.2)
    axi.plot(t[m], flc_x[m], color="#2ca02c", lw=1.2)
    axi.set_title("zoom 2.0-2.4s", fontsize=7); axi.tick_params(labelsize=6)

    # (B) 2D 경로
    ax = axes[0, 1]
    ax.plot(rx, ry, color="#d0d0d0", lw=0.7, label="raw")
    ax.plot(ix, iy, "k--", lw=1.6, label="intended")
    ax.plot(flc_x, flc_y, color="#2ca02c", lw=1.1, label="FLC output")
    ax.plot(ix[0], iy[0], "ko", ms=7); ax.plot(ix[-1], iy[-1], "g*", ms=14)
    ax.set_title("(B) 2D tool path (surgical reach)")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal")

    # (C) 스펙트럼: 10 Hz 피크 제거 확인
    ax = axes[1, 0]
    for sig_, col, lab in [(rx, "#999999", "raw"),
                           (flc_x, "#2ca02c", "FLC"),
                           (lp_x, "#1f77b4", "low-pass")]:
        f, pxx = signal.welch(sig_ - np.mean(sig_), FS, nperseg=1024)
        ax.semilogy(f, pxx, color=col, lw=1.3, label=lab)
    ax.axvspan(TREMOR_BAND[0], TREMOR_BAND[1], color="red", alpha=0.10)
    ax.axvline(f0, color="red", ls=":", lw=1.0, label=f"f0={f0:.1f}Hz")
    ax.set_xlim(0, 30); ax.set_title("(C) spectrum (x): 10 Hz tremor peak removed")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("PSD [mm^2/Hz]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (D) 기법별 잔여떨림 / 추종오차 막대
    ax = axes[1, 1]
    names = list(results.keys())
    short = [n.split(" (")[0] for n in names]
    tr = [results[n][0] * 1000 for n in names]
    te = [results[n][1] * 1000 for n in names]
    xpos = np.arange(len(names)); wbar = 0.38
    ax.bar(xpos - wbar / 2, tr, wbar, color="#d62728", label="residual tremor [um]")
    ax.bar(xpos + wbar / 2, te, wbar, color="#1f77b4", label="tracking error [um]")
    ax.axhline(raw_tremor * 1000, color="#d62728", ls="--", lw=1.0, label=f"raw tremor {raw_tremor*1000:.0f}um")
    ax.set_xticks(xpos); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("[um]"); ax.set_title("(D) residual tremor vs tracking error by method")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Physiological hand-tremor cancellation for surgical robots", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "22_surgical_tremor.png", dpi=130)
    plt.close(fig)
    print("\n[plot] outputs/22_surgical_tremor.png, assets/22_surgical_tremor.png")


if __name__ == "__main__":
    main()
