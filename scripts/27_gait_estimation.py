"""보행 위상(gait-phase) 추정: 재활 외골격(exoskeleton)의 보조 토크 타이밍.

하지 재활 로봇/보행 보조 외골격은 착용자의 보행 주기(gait cycle) 안에서
'지금이 입각기(stance)인가 유각기(swing)인가', 그리고 발뒤꿈치 착지
(heel-strike)·발끝 이지(toe-off) 같은 사건이 언제 일어나는지를 실시간으로
알아야 한다. 그래야 유각기 초기에 발을 들어 올리고 입각 말기에 밀어주는
보조 토크를 옳은 위상에 얹을 수 있다. 위상이 어긋난 보조력은 오히려
보행을 방해하거나 낙상 위험을 키운다.

이 실험은 발/정강이에 부착한 관성센서(IMU) 신호만으로 보행 위상을 추정한다.
    측정 = 발 각속도(gyro) + 가속도(accel), 200 Hz, 바이어스·노이즈 포함
핵심 관찰: 입각기에는 발이 지면에 정지 -> 각속도·가속도 편차가 0에 가깝고,
유각기에는 발이 크게 흔들려 각속도가 커진다. 이 대비를 이용한다.

  (1) 영속도(zero-velocity)/저각속도 검출기 — 짧은 창(window)에서 gyro 크기와
      중력 대비 가속도 편차를 함께 보고 입각기를 판정, 그 경계에서
      heel-strike(유각->입각) / toe-off(입각->유각) 사건을 뽑는다.
  (2) ZUPT-보정 stride 추정 — 전진 가속도를 적분해 속도를, 다시 적분해 이동거리를
      얻되, 검출된 입각기마다 속도를 0으로 되돌리는 영속도갱신(ZUPT)으로
      누적 드리프트를 잘라낸다. 순진한 이중적분(가속도 바이어스가 2차로
      쌓여 폭주)과 stride 길이 오차를 정직하게 비교한다.

지표: 입각/유각 분류 정확도, heel-strike/toe-off 타이밍 오차[ms],
그리고 ZUPT 보정 stride 오차 vs 순진 이중적분 stride 오차.
한계: 합성 보행이라 실제 보행의 개인차·불규칙성·센서 정렬 오차는 단순화했고,
전진축은 중력 보상(레벨링)이 된 항법 프레임으로 가정한다.

    python scripts/27_gait_estimation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 200.0              # 샘플링 [Hz] (발부착 IMU의 현실적 대역)
N_STRIDES = 8           # 보행 stride 수
T_STRIDE = 1.10         # 한 stride 주기 [s]
STANCE_FRAC = 0.60      # 입각기 비율 (보행에서 통상 ~60%)
STRIDE_LEN = 0.70       # 한 stride 전진 거리 [m] (ground truth)
G = 9.81                # 중력가속도 [m/s^2]

# 유각기 발 운동 파라미터
W_PEAK = 4.2            # 발 pitch 각속도 피크 [rad/s]
Z_CLEAR = 0.12          # 유각기 발 지면 이격(수직 clearance) [m]

# 센서 결함
GYRO_BIAS = 0.03        # gyro 바이어스 [rad/s]
GYRO_STD = 0.04         # gyro 노이즈 [rad/s]
ACC_BIAS = 0.18         # 전진 가속도계 바이어스 [m/s^2] (드리프트의 주범)
ACC_STD = 0.12          # 가속도 노이즈 [m/s^2]


# ---------------------------------------------------------------- 신호 합성
def synth(seed=0):
    """발부착 IMU 신호 + ground-truth 위상/사건을 합성한다.

    반환: t, gyro_meas, acc_fwd_meas, acc_mag_meas,
          stance_true(bool), hs_true(idx), to_true(idx), v_true
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / FS
    ts = STANCE_FRAC * T_STRIDE              # 입각기 지속시간
    tw = (1.0 - STANCE_FRAC) * T_STRIDE      # 유각기 지속시간
    vpk = 2.0 * STRIDE_LEN / tw              # sin^2 속도프로파일 피크(적분=STRIDE_LEN)

    # 세그먼트 순서: 시작 정지(입각) 이후 [유각, 입각] 반복
    segs = [("stance", 0.40)]
    for _ in range(N_STRIDES):
        segs.append(("swing", tw))
        segs.append(("stance", ts))

    gyro, a_fwd, a_vert, v_true, stance = [], [], [], [], []
    hs_true, to_true = [], []                # 사건 샘플 인덱스
    idx = 0
    for kind, dur in segs:
        n = int(round(dur * FS))
        if kind == "stance":
            # 입각기: 발 정지 -> 모든 운동량 0
            gyro.append(np.zeros(n))
            a_fwd.append(np.zeros(n))
            a_vert.append(np.zeros(n))
            v_true.append(np.zeros(n))
            stance.append(np.ones(n, bool))
            hs_true.append(idx)              # 입각 시작 = heel-strike
        else:
            tau = (np.arange(n) + 0.5) / n   # 유각 진행도 0..1
            # 전진: 속도 sin^2 프로파일(양끝 0), 가속도는 해석적 미분
            v = vpk * np.sin(np.pi * tau) ** 2
            af = vpk * (np.pi / tw) * np.sin(2 * np.pi * tau)
            # 발 각속도: 유각기 큰 스윙(1차+2차 로브)
            w = W_PEAK * (np.sin(np.pi * tau) + 0.35 * np.sin(2 * np.pi * tau))
            # 수직: 발 들어올림 z=H sin^2 -> 가속도 2차미분
            av = Z_CLEAR * (2 * np.pi**2 / tw**2) * np.cos(2 * np.pi * tau)
            gyro.append(w)
            a_fwd.append(af)
            a_vert.append(av)
            v_true.append(v)
            stance.append(np.zeros(n, bool))
            to_true.append(idx)              # 유각 시작 = toe-off
        idx += n

    gyro = np.concatenate(gyro)
    a_fwd = np.concatenate(a_fwd)
    a_vert = np.concatenate(a_vert)
    v_true = np.concatenate(v_true)
    stance_true = np.concatenate(stance)
    n = len(gyro)
    t = np.arange(n) / FS

    # 첫 입각(idx=0)은 heel-strike로 세지 않음(보행 시작 정지)
    hs_true = np.array(hs_true[1:], int)
    to_true = np.array(to_true, int)

    # 측정: 바이어스 + 백색노이즈
    gyro_meas = gyro + GYRO_BIAS + rng.normal(0, GYRO_STD, n)
    acc_fwd_meas = a_fwd + ACC_BIAS + rng.normal(0, ACC_STD, n)
    # 가속도 크기(비력): 정지 시 ~g, 유각 시 편차 발생
    acc_mag_true = np.sqrt(a_fwd**2 + (a_vert + G) ** 2)
    acc_mag_meas = acc_mag_true + rng.normal(0, ACC_STD, n)

    return t, gyro_meas, acc_fwd_meas, acc_mag_meas, stance_true, hs_true, to_true, v_true


# ---------------------------------------------------------------- (1) 입각 검출
def _moving_avg(x, w):
    """중심 이동평균(위상지연 없음)."""
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def detect_stance(gyro_meas, acc_mag_meas, win_s=0.05,
                  gyro_thr=0.6, acc_thr=1.2, min_stance_s=0.15, min_swing_s=0.12):
    """영속도/저각속도 입각 검출기 (gyro 크기 + 중력 대비 가속도 편차).

    입각기 판정: 짧은 창에서 |gyro| 평균과 |accel|-g 편차 평균이 모두 임계 이하.
    (Skog 등의 SHOE 영속도 검출기를 gyro·accel 두 임계로 단순화한 형태)
    이후 너무 짧은 입각/유각 조각을 정리해 채터링을 없앤다.
    """
    win = max(1, int(round(win_s * FS)))
    gyro_sm = _moving_avg(np.abs(gyro_meas), win)
    acc_dev_sm = _moving_avg(np.abs(acc_mag_meas - G), win)
    stance = (gyro_sm < gyro_thr) & (acc_dev_sm < acc_thr)
    stance = _clean_runs(stance, int(round(min_stance_s * FS)),
                         int(round(min_swing_s * FS)))
    return stance


def _clean_runs(stance, min_stance, min_swing):
    """최소 지속 길이보다 짧은 입각/유각 구간을 뒤집어 안정화."""
    s = stance.copy()
    # 짧은 입각(True) 조각 제거 -> False
    for target, min_len in ((True, min_stance), (False, min_swing)):
        i = 0
        n = len(s)
        while i < n:
            if s[i] == target:
                j = i
                while j < n and s[j] == target:
                    j += 1
                if (j - i) < min_len:
                    s[i:j] = not target
                i = j
            else:
                i += 1
    return s


def transitions(stance):
    """입각 bool 배열에서 heel-strike(False->True), toe-off(True->False) 인덱스."""
    d = np.diff(stance.astype(int))
    hs = np.where(d == 1)[0] + 1           # 유각->입각
    to = np.where(d == -1)[0] + 1          # 입각->유각
    return hs, to


# ---------------------------------------------------------------- (2) ZUPT stride
def integrate_zupt(acc_fwd_meas, stance):
    """전진가속도 -> 속도 -> 위치, 입각기마다 영속도갱신(ZUPT)으로 드리프트 절단."""
    dt = 1.0 / FS
    n = len(acc_fwd_meas)
    v = np.zeros(n)
    p = np.zeros(n)
    vk = 0.0
    pk = 0.0
    for k in range(n):
        vk += acc_fwd_meas[k] * dt
        if stance[k]:
            vk = 0.0                        # ZUPT: 입각기 속도는 0
        pk += vk * dt
        v[k] = vk
        p[k] = pk
    return v, p


def integrate_naive(acc_fwd_meas):
    """순진한 이중적분(ZUPT 없음): 가속도 바이어스가 2차로 누적되어 폭주."""
    dt = 1.0 / FS
    v = np.cumsum(acc_fwd_meas) * dt
    p = np.cumsum(v) * dt
    return v, p


def stride_lengths(pos, hs_idx):
    """heel-strike 경계 사이의 위치 증가분 = stride 길이."""
    return np.diff(pos[hs_idx])


# ---------------------------------------------------------------- 채점
def stance_accuracy(stance_det, stance_true):
    return float(np.mean(stance_det == stance_true))


def event_timing_error(det_idx, true_idx):
    """각 참사건을 가장 가까운 검출사건에 매칭한 평균 |Δt| [ms]."""
    if len(det_idx) == 0:
        return float("inf")
    errs = []
    for ti in true_idx:
        j = np.argmin(np.abs(det_idx - ti))
        errs.append(abs(det_idx[j] - ti) / FS * 1000.0)
    return float(np.mean(errs))


# ---------------------------------------------------------------- main
def main():
    (t, gyro_meas, acc_fwd_meas, acc_mag_meas,
     stance_true, hs_true, to_true, v_true) = synth(seed=0)

    # (1) 입각 검출 + 사건
    stance_det = detect_stance(gyro_meas, acc_mag_meas)
    hs_det, to_det = transitions(stance_det)

    acc = stance_accuracy(stance_det, stance_true)
    hs_err = event_timing_error(hs_det, hs_true)
    to_err = event_timing_error(to_det, to_true)
    ev_err = 0.5 * (hs_err + to_err)

    # (2) ZUPT vs 순진 적분
    v_zupt, p_zupt = integrate_zupt(acc_fwd_meas, stance_det)
    v_naive, p_naive = integrate_naive(acc_fwd_meas)

    # 참 heel-strike 경계로 stride 분할(적분 품질만 공정 비교)
    zupt_sl = stride_lengths(p_zupt, hs_true)
    naive_sl = stride_lengths(p_naive, hs_true)
    zupt_err = float(np.mean(np.abs(zupt_sl - STRIDE_LEN)))
    naive_err = float(np.mean(np.abs(naive_sl - STRIDE_LEN)))

    true_total = N_STRIDES * STRIDE_LEN
    zupt_total = p_zupt[hs_true[-1]] - p_zupt[hs_true[0]] + STRIDE_LEN
    naive_total = p_naive[-1]

    print("=== 재활 외골격 보행 위상(gait-phase) 추정 ===")
    print(f"샘플링 {FS:.0f} Hz, stride {N_STRIDES}개, 주기 {T_STRIDE:.2f} s, "
          f"입각비율 {STANCE_FRAC*100:.0f}%")
    print(f"참 stride 길이 {STRIDE_LEN*100:.0f} cm, gyro 피크 ~{W_PEAK:.1f} rad/s, "
          f"accel 바이어스 {ACC_BIAS:.2f} m/s^2")
    print("-" * 60)
    print("[1] 위상 검출")
    print(f"  입각/유각 분류 정확도      : {acc*100:6.2f} %")
    print(f"  heel-strike 타이밍 오차     : {hs_err:6.1f} ms  "
          f"(검출 {len(hs_det)} / 참 {len(hs_true)})")
    print(f"  toe-off 타이밍 오차         : {to_err:6.1f} ms  "
          f"(검출 {len(to_det)} / 참 {len(to_true)})")
    print(f"  평균 사건 타이밍 오차       : {ev_err:6.1f} ms")
    print("-" * 60)
    print("[2] stride 길이 추정 (ZUPT vs 순진 이중적분)")
    print(f"  {'':14s}{'평균 stride오차':>16s}{'추정 총거리':>14s}")
    print(f"  {'ZUPT 보정':14s}{zupt_err*100:13.1f} cm{zupt_total:12.2f} m")
    print(f"  {'순진 적분':14s}{naive_err*100:13.1f} cm{naive_total:12.2f} m")
    print(f"  {'참값':14s}{'-':>13s}  {true_total:12.2f} m")
    print(f"  -> ZUPT가 순진 적분보다 stride 오차 {naive_err/zupt_err:.0f}배 작음")
    print("-" * 60)
    print("한계: 합성 보행이라 개인차·불규칙 보행·센서 정렬오차는 단순화.")
    print("전진축은 레벨링(중력보상)된 항법 프레임 가정.")

    _plot(t, gyro_meas, acc_mag_meas, stance_true, stance_det,
          hs_true, to_true, hs_det, to_det,
          v_zupt, v_naive, p_zupt, p_naive, hs_true)

    # (분류정확도, 평균사건타이밍오차[ms], ZUPT stride오차[m], 순진 stride오차[m])
    return acc, ev_err, zupt_err, naive_err


# ---------------------------------------------------------------- 시각화
def _plot(t, gyro_meas, acc_mag_meas, stance_true, stance_det,
          hs_true, to_true, hs_det, to_det,
          v_zupt, v_naive, p_zupt, p_naive, hs_idx):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    def shade(ax, stance, color, alpha, label=None):
        d = np.diff(np.concatenate(([0], stance.astype(int), [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(t[max(s, 0)], t[min(e, len(t) - 1)], color=color,
                       alpha=alpha, label=label if i == 0 else None)

    # (A) gyro + 검출 입각 음영 + 참/검출 사건
    ax = axes[0, 0]
    ax.plot(t, gyro_meas, color="#4a4a4a", lw=0.8, label="foot gyro |meas|")
    shade(ax, stance_det, "#2ca02c", 0.15, "detected stance")
    for i, k in enumerate(hs_true):
        ax.axvline(t[k], color="#1f77b4", ls="--", lw=1.0,
                   label="true heel-strike" if i == 0 else None)
    for i, k in enumerate(to_true):
        ax.axvline(t[k], color="#d62728", ls=":", lw=1.0,
                   label="true toe-off" if i == 0 else None)
    ax.plot(t[hs_det], gyro_meas[hs_det], "v", color="#1f77b4", ms=7,
            label="detected HS")
    ax.plot(t[to_det], gyro_meas[to_det], "^", color="#d62728", ms=7,
            label="detected TO")
    ax.set_title("(A) foot angular rate: stance shaded, gait events marked")
    ax.set_xlabel("time [s]"); ax.set_ylabel("gyro [rad/s]")
    ax.legend(fontsize=7, loc="upper right", ncol=2); ax.grid(alpha=0.3)

    # (B) accel 크기 + 참/검출 입각 비교
    ax = axes[0, 1]
    ax.plot(t, acc_mag_meas, color="#7a4a9a", lw=0.8, label="|accel| [m/s^2]")
    ax.axhline(G, color="k", ls="--", lw=0.8, label="g (foot at rest)")
    shade(ax, stance_true, "#888888", 0.18, "true stance")
    shade(ax, stance_det, "#2ca02c", 0.15, "detected stance")
    ax.set_title("(B) accel magnitude: |a|~g in stance, deviates in swing")
    ax.set_xlabel("time [s]"); ax.set_ylabel("[m/s^2]")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

    # (C) 속도: ZUPT(입각마다 0 리셋) vs 순진(바이어스로 램프)
    ax = axes[1, 0]
    ax.plot(t, v_naive, color="#d62728", lw=1.3,
            label="naive integration (drifts)")
    ax.plot(t, v_zupt, color="#2ca02c", lw=1.3, label="ZUPT-aided (reset each stance)")
    shade(ax, stance_det, "#2ca02c", 0.10)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(C) forward velocity: ZUPT resets vs drifting naive")
    ax.set_xlabel("time [s]"); ax.set_ylabel("velocity [m/s]")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    # (D) 누적 이동거리: ZUPT vs 순진 vs 참 계단
    ax = axes[1, 1]
    true_stairs_x, true_stairs_y = [t[0]], [0.0]
    for i, k in enumerate(hs_idx):
        true_stairs_x += [t[k], t[k]]
        true_stairs_y += [true_stairs_y[-1], (i + 1) * STRIDE_LEN]
    true_stairs_x.append(t[-1]); true_stairs_y.append(true_stairs_y[-1])
    ax.plot(true_stairs_x, true_stairs_y, color="#1f77b4", lw=1.4,
            label="true distance (stairs)")
    ax.plot(t, p_zupt, color="#2ca02c", lw=1.4, label="ZUPT-aided distance")
    ax.plot(t, p_naive, color="#d62728", lw=1.2, label="naive distance (drift)")
    ax.set_title("(D) cumulative travelled distance")
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance [m]")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    fig.suptitle("Gait-phase estimation for a rehabilitation exoskeleton "
                 "(foot-IMU: stance detection + ZUPT stride)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "27_gait_estimation.png", dpi=130)
    plt.close(fig)
    print("\n[plot] outputs/27_gait_estimation.png, assets/27_gait_estimation.png")


if __name__ == "__main__":
    main()
