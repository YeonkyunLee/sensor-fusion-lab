"""오차상태 칼만필터(ESKF, indirect KF)로 3D 자세 추정 — VIO/INS 표준 정식화.

자이로는 각속도를 직접 주지만 바이어스+백색잡음 때문에 적분하면 자세가 드리프트한다
(특히 헤딩/요). 가속도계(중력 방향)와 지자기계(자기장 방향)는 절대 기준을 주지만 자세를
비선형으로 관측한다. 두 정보를 어떻게 합치나?

**오차상태(간접) 칼만필터**의 아이디어:
  - 상태를 둘로 쪼갠다. (1) 큰 값을 담는 *공칭(nominal)* 상태 — 자세 R(회전행렬)과
    자이로 바이어스 b. R은 자이로로 그냥 적분한다(비선형·특이점 없이 매니폴드 위에서).
    (2) 작은 값을 담는 *오차(error)* 상태 — 접공간의 3D 회전오차 δθ 와 바이어스오차 δb(6-DOF).
  - 칼만필터는 '작은 오차'만 다룬다. 오차는 항상 0 근처라 선형화(야코비안)가 정확하고,
    회전을 3-벡터로 최소표현해 특이점·과잉파라미터(쿼터니언 4개, 정규화 제약) 문제가 없다.
  - **예측**: 공칭 R을 (자이로−바이어스)로 SO(3) 위에서 전파하고, 오차상태 공분산만
    선형 오차동역학으로 전파한다(오차 평균은 0 유지).
  - **보정**: 가속도/지자기 측정의 잔차(innovation)로 오차상태 δθ,δb 를 추정.
  - **주입(inject)&리셋**: 추정된 오차를 exp 사상으로 공칭에 되먹이고(R←R·Exp(δθ)),
    오차상태는 다시 0으로 리셋. 매 스텝 반복.

바로 이 정식화가 실제 VIO/INS(예: MSCKF, ROVIO, VINS, 항공 INS)의 사실상 표준이다:
회전을 매니폴드에서 정확히 적분하면서, 불확실성은 최소차원 접공간에서 선형 KF로 다룬다.

정직한 한계(관측성): 가속도계는 중력 *방향*만 주므로 롤·피치 2-DOF만 관측하고 요(헤딩)는
중력축 회전에 불변 → 관측 불가. 요를 잡으려면 지자기계(또는 다른 헤딩 기준)가 반드시 필요하다.
또한 여기서는 준정적(비가속) 가정으로 가속도계가 순수 중력만 본다고 둔다(실제엔 선형가속이 섞임).

    python scripts/37_error_state_kf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.se3 import hat, so3_exp, so3_log  # noqa: E402

DT = 0.02                     # 50 Hz
T_END = 60.0                  # 60 s
GYRO_WHITE = 0.004            # 자이로 백색잡음 std [rad/s]
BIAS_RW = 8e-4                # 자이로 바이어스 random-walk std [rad/s/√s]
ACC_STD = 0.03               # 가속도계 방향 측정 잡음 std (단위벡터)
MAG_STD = 0.03               # 지자기계 방향 측정 잡음 std (단위벡터)

# 세계좌표 기준 방향(단위벡터): 중력은 아래(-z), 자기장은 북+하향(요 관측용, 중력과 비평행)
G_DIR = np.array([0.0, 0.0, -1.0])
_incl = np.radians(60.0)
MAG_DIR = np.array([np.cos(_incl), 0.0, -np.sin(_incl)])
MAG_DIR = MAG_DIR / np.linalg.norm(MAG_DIR)


def rot_to_euler(R):
    """회전행렬 → ZYX 오일러각 (roll, pitch, yaw) [rad]."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def att_error_deg(R_est, R_true):
    """두 자세 사이 측지 각오차 [deg]."""
    return np.degrees(np.linalg.norm(so3_log(R_est.T @ R_true)))


def simulate(rng):
    """회전하는 IMU 참 궤적 + 센서 측정 생성."""
    N = int(T_END / DT)
    t = np.arange(N) * DT

    # 시변 각속도(모든 축, 지속 요회전 포함) → 참 자세 R_true 를 SO(3) 적분
    omega_true = np.stack([
        0.6 * np.sin(0.5 * t),
        0.4 * np.cos(0.3 * t + 0.5),
        0.3 + 0.25 * np.sin(0.2 * t),      # 지속 요회전(요 드리프트를 부각)
    ], axis=1)

    # 느리게 변하는 참 자이로 바이어스(초기 오프셋 + random walk)
    bias_true = np.zeros((N, 3))
    b = np.array([0.02, -0.015, 0.025])    # ~1 deg/s 수준 초기 바이어스
    for k in range(N):
        bias_true[k] = b
        b = b + rng.normal(0, BIAS_RW * np.sqrt(DT), 3)

    R = np.eye(3)
    R_true = np.zeros((N, 3, 3))
    gyro = np.zeros((N, 3))
    accel = np.zeros((N, 3))
    mag = np.zeros((N, 3))
    for k in range(N):
        R_true[k] = R
        # 자이로 측정 = 참각속도 + 바이어스 + 백색잡음
        gyro[k] = omega_true[k] + bias_true[k] + rng.normal(0, GYRO_WHITE, 3)
        # 방향 센서: 세계기준을 body로 회전(R^T)해 측정(+잡음), 단위정규화
        a = R.T @ G_DIR + rng.normal(0, ACC_STD, 3)
        m = R.T @ MAG_DIR + rng.normal(0, MAG_STD, 3)
        accel[k] = a / np.linalg.norm(a)
        mag[k] = m / np.linalg.norm(m)
        # 참 자세 전파(다음 스텝)
        R = R @ so3_exp(omega_true[k] * DT)

    return t, R_true, bias_true, gyro, accel, mag


def run_gyro_only(gyro):
    """순수 자이로 적분(바이어스 미보정) — 드리프트 기준선."""
    N = len(gyro)
    R = np.eye(3)
    out = np.zeros((N, 3, 3))
    for k in range(N):
        out[k] = R
        R = R @ so3_exp(gyro[k] * DT)
    return out


def run_eskf(gyro, accel, mag):
    """오차상태 칼만필터. body-frame 우측오차 규약: R_true = R_nom · Exp(δθ)."""
    N = len(gyro)

    R_nom = np.eye(3)                 # 공칭 자세
    b_nom = np.zeros(3)               # 공칭 자이로 바이어스
    # 오차상태 공분산 P (6x6): [δθ(3), δb(3)]
    P = np.diag([np.radians(10)**2] * 3 + [0.05**2] * 3).astype(float)

    # 프로세스 잡음(이산): δθ엔 자이로 백색잡음, δb엔 바이어스 random-walk
    Q = np.diag([GYRO_WHITE**2 * DT] * 3 + [BIAS_RW**2 * DT] * 3)
    # 측정 잡음: accel 3 + mag 3
    Rm = np.diag([ACC_STD**2] * 3 + [MAG_STD**2] * 3)

    out_R = np.zeros((N, 3, 3))
    out_b = np.zeros((N, 3))
    I3 = np.eye(3)

    for k in range(N):
        # --- 예측: 공칭 전파 + 오차 공분산 전파 ---
        u = gyro[k] - b_nom
        Phi = so3_exp(u * DT)
        F = np.eye(6)
        F[0:3, 0:3] = Phi.T          # 우측오차의 회전 전이 = Exp(u·dt)^T
        F[0:3, 3:6] = -I3 * DT       # 바이어스오차 → 회전오차 결합
        P = F @ P @ F.T + Q
        R_nom = R_nom @ Phi          # 매니폴드 위 자세 전파

        # --- 보정: accel(중력) + mag(자기장) 방향 측정 ---
        gh = R_nom.T @ G_DIR
        mh = R_nom.T @ MAG_DIR
        h = np.concatenate([gh, mh])
        z = np.concatenate([accel[k], mag[k]])
        H = np.zeros((6, 6))
        H[0:3, 0:3] = hat(gh)        # ∂(R^T v)/∂δθ = hat(R^T v) (우측오차)
        H[3:6, 0:3] = hat(mh)
        y = z - h
        S = H @ P @ H.T + Rm
        K = P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        P = (np.eye(6) - K @ H) @ P

        # --- 주입 & 리셋 ---
        dtheta, dbias = dx[0:3], dx[3:6]
        R_nom = R_nom @ so3_exp(dtheta)   # exp 사상으로 매니폴드에 되먹임
        b_nom = b_nom + dbias
        # 공분산 리셋 야코비안(우측오차): θ블록 G = I - 0.5·hat(δθ)
        G = np.eye(6)
        G[0:3, 0:3] = I3 - 0.5 * hat(dtheta)
        P = G @ P @ G.T
        P = 0.5 * (P + P.T)               # 대칭 유지

        out_R[k] = R_nom
        out_b[k] = b_nom

    return out_R, out_b


def main(plot: bool = True):
    rng = np.random.default_rng(7)
    t, R_true, bias_true, gyro, accel, mag = simulate(rng)
    N = len(t)

    R_gyro = run_gyro_only(gyro)
    R_eskf, b_eskf = run_eskf(gyro, accel, mag)

    err_gyro = np.array([att_error_deg(R_gyro[k], R_true[k]) for k in range(N)])
    err_eskf = np.array([att_error_deg(R_eskf[k], R_true[k]) for k in range(N)])

    # 정상상태(뒤쪽 절반) RMSE로 헤드라인 산정
    half = N // 2
    gyro_rmse = float(np.sqrt(np.mean(err_gyro[half:] ** 2)))
    eskf_rmse = float(np.sqrt(np.mean(err_eskf[half:] ** 2)))
    final_bias_err = float(np.linalg.norm(b_eskf[-1] - bias_true[-1]))

    print("=== ESKF 3D 자세 추정 (자이로 + 가속도 + 지자기) ===")
    print(f"자세 RMSE(정상상태): gyro-only = {gyro_rmse:8.3f} deg")
    print(f"                     ESKF      = {eskf_rmse:8.3f} deg")
    print(f"개선율               = {gyro_rmse / eskf_rmse:6.1f}x")
    print(f"\n자이로 바이어스 [rad/s]:")
    print(f"  참값(최종)   = [{bias_true[-1,0]:+.4f}, {bias_true[-1,1]:+.4f}, {bias_true[-1,2]:+.4f}]")
    print(f"  ESKF 추정    = [{b_eskf[-1,0]:+.4f}, {b_eskf[-1,1]:+.4f}, {b_eskf[-1,2]:+.4f}]")
    print(f"  최종 바이어스 오차 = {final_bias_err:.4f} rad/s "
          f"({np.degrees(final_bias_err):.3f} deg/s)")

    if plot:
        eul_true = np.array([rot_to_euler(R_true[k]) for k in range(N)])
        eul_eskf = np.array([rot_to_euler(R_eskf[k]) for k in range(N)])
        eul_gyro = np.array([rot_to_euler(R_gyro[k]) for k in range(N)])

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        (ax1, ax2), (ax3, ax4) = axes

        # (1) 자세 오차 vs 시간
        ax1.plot(t, err_gyro, "C3", lw=1.2, label="gyro-only (drifts)")
        ax1.plot(t, err_eskf, "C0", lw=1.4, label="ESKF (bounded)")
        ax1.set_xlabel("time [s]"); ax1.set_ylabel("attitude error [deg]")
        ax1.set_title("Attitude error: gyro integration drifts vs ESKF stays bounded")
        ax1.legend(); ax1.grid(alpha=0.3)

        # (2) 바이어스 추정 수렴
        for i, c in enumerate(["C0", "C1", "C2"]):
            ax2.axhline(bias_true[-1, i], color=c, ls="--", lw=0.8)
            ax2.plot(t, b_eskf[:, i], c, lw=1.2, label=f"est b{'xyz'[i]}")
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("gyro bias [rad/s]")
        ax2.set_title("Online gyro-bias estimate converging to truth (dashed)")
        ax2.legend(fontsize=8, ncol=3); ax2.grid(alpha=0.3)

        # (3) 요(헤딩) 추적 — 자이로는 바이어스로 발산, ESKF는 지자기로 붙잡음
        ax3.plot(t, np.degrees(np.unwrap(eul_true[:, 2])), "g-", lw=2, label="true yaw")
        ax3.plot(t, np.degrees(np.unwrap(eul_gyro[:, 2])), "C3", lw=1, alpha=0.8, label="gyro-only")
        ax3.plot(t, np.degrees(np.unwrap(eul_eskf[:, 2])), "C0", lw=1.3, label="ESKF")
        ax3.set_xlabel("time [s]"); ax3.set_ylabel("yaw [deg]")
        ax3.set_title("Heading (yaw): needs magnetometer to be observable")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

        # (4) 롤/피치 추적(ESKF)
        ax4.plot(t, np.degrees(eul_true[:, 0]), "g-", lw=2, label="true roll")
        ax4.plot(t, np.degrees(eul_eskf[:, 0]), "C0", lw=1, label="ESKF roll")
        ax4.plot(t, np.degrees(eul_true[:, 1]), "m-", lw=2, label="true pitch")
        ax4.plot(t, np.degrees(eul_eskf[:, 1]), "C1", lw=1, label="ESKF pitch")
        ax4.set_xlabel("time [s]"); ax4.set_ylabel("angle [deg]")
        ax4.set_title("Roll/pitch tracking (observable from accel/gravity)")
        ax4.legend(fontsize=8, ncol=2); ax4.grid(alpha=0.3)

        fig.suptitle("Error-State Kalman Filter (ESKF) for 3D attitude — the VIO/INS standard",
                     fontsize=13)
        fig.tight_layout()
        for out in ("outputs", "assets"):
            Path(out).mkdir(exist_ok=True)
            fig.savefig(Path(out) / "37_error_state_kf.png", dpi=130)
        plt.close(fig)
        print("\n[plot] outputs/37_error_state_kf.png, assets/37_error_state_kf.png")

    return gyro_rmse, eskf_rmse, final_bias_err


if __name__ == "__main__":
    main()
