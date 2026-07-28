"""수술 팔 sim-to-real 폐루프: 시뮬에서 준비 → 실기 배치 → 문제 감지 → 식별 → 재배치.

exp 42의 오차 예산에서 가장 불편한 결과는 이것이었다: **도구를 갈아 끼워 생긴 3%
페이로드 오차만으로 서보 몫이 정합오차의 10배가 되어 예산 순위가 뒤집힌다.** 그
실험은 거기서 멈췄다("캘리브레이션이 1순위") — 이 실험은 그 캘리브레이션을 **닫힌
루프로 자동화**한다. exp 31(SimOpt, 카트폴)의 절차를 매니퓰레이터로 옮긴 것이다.

    [시뮬] 공칭 모델로 계산토크 제어기 준비
        ↓  배치
    [실기] 모델에 없는 것들(도구 페이로드 + 관절 마찰)로 추종 붕괴
        ↓  감지: 추종 잔차가 임계 초과 → 사람이 아니라 시스템이 스스로 플래그
    [식별] 여기(excitation) 궤적을 돌려 로그 수집 → 선형 회귀 최소자승
        ↓
    [갱신] 식별된 파라미터로 제어기 모델 교체 → 재배치 → 다시 측정 (반복)

--- 왜 최소자승으로 식별이 되는가 ---
강체 매니퓰레이터 동역학은 **관성 파라미터에 대해 선형**이다. 마찰 항을 포함해도
그대로다.

    τ = Y(q, q̇, q̈) · π,   π = [a, b, d, G1, G2, fv1, fv2, fc1, fc2]

  a = I1 + I2 + m1 lc1² + m2(l1² + lc2²),  b = m2 l1 lc2,  d = I2 + m2 lc2²
  G1 = (m1 lc1 + m2 l1) g,  G2 = m2 lc2 g,  fv = 점성마찰,  fc = 쿨롱마찰

즉 미지의 도구 질량·무게중심·마찰이 π 안에 흡수되므로, 실기에서 (q, q̇, q̈, τ)만
로깅하면 **회귀 한 번으로** 전부 복원된다. 물리 파라미터를 하나씩 재는 것이 아니라
'모델이 실기를 재현하도록' 맞추는 것 — 이것이 시스템 식별이다.

--- 실데이터 앵커: 합성 팔이 아니라 공개된 실제 로봇 ---
링크 파라미터는 지어내지 않고 **Universal Robots UR5의 공개 사양**을 쓴다. UR5의
관절 2(shoulder-lift)·3(elbow)은 실제로 **수직 평면 안의 2링크 팔**을 이루므로,
평면 축약이 임의 가정이 아니라 이 로봇의 실제 부분구조다.

--- 정직하게 확인하는 것: 관측성 ---
식별은 '충분히 여기된' 데이터를 요구한다. 임상 삽입 궤적(느리고 매끄러움)만으로
식별하면 회귀행렬이 병조건이 되어 파라미터가 복원되지 않는다. 전용 여기 궤적
(다중 사인)과 비교해 이를 정량화한다 — 이 저장소의 관측성 주제(IMU 바이어스,
EKF-SLAM 헤딩)가 동역학 식별에서 반복되는 지점이다.

    python scripts/43_sim_to_real_arm.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, filtfilt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
kin = import_module("39_manipulator_kinematics")     # FK · DLS IK
tgt = import_module("42_image_guided_targeting")     # 표적 시나리오·경로 생성

# --------------------------------------------------------------------------- #
# 플랜트: UR5 관절2·3의 공개 사양으로 만든 수직평면 2링크 팔
#   출처: Universal Robots 공개 DH·질량·무게중심 표 (UR5).
#   링크 길이 a2 = 0.425 m, a3 = 0.39225 m / 질량 8.393 kg, 2.275 kg
#   무게중심(링크 좌표계 x) 0.2125 m, 0.15 m
#   회전관성은 공개표의 값이 출처마다 달라, 균일 봉 근사 I = m l²/12 를 명시적으로 쓴다.
# --------------------------------------------------------------------------- #
G = 9.81
L1, L2 = 0.425, 0.39225
M1, M2 = 8.393, 2.275
LC1, LC2 = 0.2125, 0.15
I1, I2 = M1 * L1 ** 2 / 12, M2 * L2 ** 2 / 12
L_ARM = np.array([L1, L2])

# 실기에만 있는 것(공칭 모델은 모른다):
#  - 수술 도구 + 손목 어셈블리 페이로드: UR5 링크4~6 공개 질량 합(1.219+1.219+0.1879)
#    에 준하는 2.6 kg 을 링크2 끝단에 장착
#  - 관절 마찰(점성·쿨롱) — 강체 모델에는 아예 없는 항
TOOL_MASS = 2.626
FV_TRUE = np.array([2.5, 1.2])          # 점성마찰 [N·m·s/rad]
FC_TRUE = np.array([1.8, 0.9])          # 쿨롱마찰 [N·m]
# 구조적 간극(선택): 스트라이벡 스틱션 — 저속에서 마찰이 쿨롱값보다 커진다.
# 회귀자에 대응 열이 없으므로 **어떤 π 로도 표현할 수 없다**. 파라미터 간극과
# 구조 간극이 루프에서 어떻게 다르게 끝나는지를 비교하기 위한 항.
FS_RATIO = 1.8                          # 정지마찰/쿨롱 비
V_STRIBECK = 0.05                       # 스트라이벡 속도 [rad/s]

DT = 0.002
KP, KD = 900.0, 60.0                    # 무거운 팔이라 exp 42보다 높은 이득
TORQUE_NOISE = 0.05                     # 토크 센서 잡음 σ [N·m]
ENC_NOISE = 2e-5                        # 엔코더 잡음 σ [rad]
RESIDUAL_TRIGGER = 5e-4                 # 추종 잔차 감지 임계 [m] (0.5 mm)
N_ITERS = 3                             # sim-to-real 루프 반복 횟수


def inertial_params(tool_mass=0.0):
    """(a, b, d, G1, G2) — 링크2 끝단에 점질량 tool_mass 를 달았을 때의 값."""
    m2_eff = M2 + tool_mass
    # 점질량은 링크2 끝(거리 L2)에 붙으므로 등가 무게중심·관성이 밀린다
    lc2_eff = (M2 * LC2 + tool_mass * L2) / m2_eff
    i2_eff = I2 + M2 * (lc2_eff - LC2) ** 2 + tool_mass * (L2 - lc2_eff) ** 2
    a = I1 + i2_eff + M1 * LC1 ** 2 + m2_eff * (L1 ** 2 + lc2_eff ** 2)
    b = m2_eff * L1 * lc2_eff
    d = i2_eff + m2_eff * lc2_eff ** 2
    g1 = (M1 * LC1 + m2_eff * L1) * G
    g2 = m2_eff * lc2_eff * G
    return np.array([a, b, d, g1, g2])


def pack(inertial, fv=(0.0, 0.0), fc=(0.0, 0.0)):
    """π = [a, b, d, G1, G2, fv1, fv2, fc1, fc2]."""
    return np.concatenate([inertial, np.asarray(fv, float), np.asarray(fc, float)])


PI_TRUE = pack(inertial_params(TOOL_MASS), FV_TRUE, FC_TRUE)   # 실기(숨겨진 진실)
PI_NOMINAL = pack(inertial_params(0.0))                        # 시뮬이 믿는 모델


# --------------------------------------------------------------------------- #
# π 로 표현한 동역학 — 식별과 제어가 같은 파라미터화를 공유한다
# --------------------------------------------------------------------------- #
def M_of(pi, q):
    a, b, d = pi[0], pi[1], pi[2]
    c2 = np.cos(q[1])
    return np.array([[a + 2 * b * c2, d + b * c2],
                     [d + b * c2, d]])


def Cqd_of(pi, q, qd):
    b = pi[1]
    s2 = np.sin(q[1])
    return np.array([-b * s2 * (2 * qd[0] * qd[1] + qd[1] ** 2),
                     b * s2 * qd[0] ** 2])


def g_of(pi, q):
    g1, g2 = pi[3], pi[4]
    return np.array([g1 * np.cos(q[0]) + g2 * np.cos(q[0] + q[1]),
                     g2 * np.cos(q[0] + q[1])])


def friction_of(pi, qd):
    fv, fc = pi[5:7], pi[7:9]
    return fv * qd + fc * np.tanh(qd / 0.01)     # tanh = 수치적으로 매끈한 sign


def friction_real(pi, qd, stribeck=False):
    """실기의 마찰. stribeck=True면 저속에서 커지는 정지마찰이 추가된다(모델 밖 항)."""
    if not stribeck:
        return friction_of(pi, qd)
    fv, fc = pi[5:7], pi[7:9]
    fc_eff = fc * (1.0 + (FS_RATIO - 1.0) * np.exp(-(qd / V_STRIBECK) ** 2))
    return fv * qd + fc_eff * np.tanh(qd / 0.01)


def inverse_dynamics(pi, q, qd, qdd):
    return M_of(pi, q) @ qdd + Cqd_of(pi, q, qd) + g_of(pi, q) + friction_of(pi, qd)


def forward_dynamics(pi, q, qd, tau, stribeck=False):
    rhs = tau - Cqd_of(pi, q, qd) - g_of(pi, q) - friction_real(pi, qd, stribeck)
    return np.linalg.solve(M_of(pi, q), rhs)


def regressor(q, qd, qdd):
    """Y(q,q̇,q̈) (2×9). 항등식 Y·π ≡ inverse_dynamics(π,·) 를 만족한다."""
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    Y = np.zeros((2, 9))
    # a, b, d
    Y[0, 0] = qdd[0]
    Y[0, 1] = 2 * c2 * qdd[0] + c2 * qdd[1] - s2 * (2 * qd[0] * qd[1] + qd[1] ** 2)
    Y[0, 2] = qdd[1]
    Y[1, 1] = c2 * qdd[0] + s2 * qd[0] ** 2
    Y[1, 2] = qdd[0] + qdd[1]
    # 중력
    Y[0, 3] = np.cos(q[0])
    Y[0, 4] = np.cos(q[0] + q[1])
    Y[1, 4] = np.cos(q[0] + q[1])
    # 마찰(관절별 대각)
    Y[0, 5] = qd[0]
    Y[1, 6] = qd[1]
    Y[0, 7] = np.tanh(qd[0] / 0.01)
    Y[1, 8] = np.tanh(qd[1] / 0.01)
    return Y


# --------------------------------------------------------------------------- #
# 제어 · 시뮬레이션
# --------------------------------------------------------------------------- #
def computed_torque(pi_hat, q, qd, q_d, qd_d, qdd_d):
    """제어기는 자기가 '아는' 모델 pi_hat 만 쓴다(마찰 보상 포함)."""
    e, edot = q_d - q, qd_d - qd
    return (M_of(pi_hat, q) @ (qdd_d + KP * e + KD * edot)
            + Cqd_of(pi_hat, q, qd) + g_of(pi_hat, q) + friction_of(pi_hat, qd))


def rollout(pi_true, pi_hat, Q_des, Qd_des, Qdd_des, rng=None, log=False,
            stribeck=False):
    """실기(pi_true)에서 궤적 추종. log=True면 식별용 (q,q̇,q̈,τ) 기록.

    로그의 q̈ 는 '측정된' 관절각을 수치미분해 얻는다(가속도계가 따로 없는 실제 팔과
    동일). 여기에 센서 잡음이 실려, 식별이 잡음에 얼마나 견디는지도 함께 시험된다."""
    n = len(Q_des)
    q, qd = Q_des[0].copy(), Qd_des[0].copy()
    Q = np.zeros((n, 2))
    TIP = np.zeros((n, 2))
    TAU = np.zeros((n, 2))
    for k in range(n):
        tau = computed_torque(pi_hat, q, qd, Q_des[k], Qd_des[k], Qdd_des[k])
        Q[k], TAU[k] = q, tau
        TIP[k] = kin.fk(q, L=L_ARM)[:2]
        if k == n - 1:
            break

        def deriv(s):
            return np.concatenate(
                [s[2:], forward_dynamics(pi_true, s[:2], s[2:], tau, stribeck)])

        s = np.concatenate([q, qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * DT * k1)
        k3 = deriv(s + 0.5 * DT * k2)
        k4 = deriv(s + DT * k3)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = s[:2], s[2:]

    if not log:
        return Q, TIP, TAU
    rng = np.random.default_rng(0) if rng is None else rng
    Q_meas = Q + rng.normal(0, ENC_NOISE, Q.shape)           # 엔코더 잡음
    TAU_meas = TAU + rng.normal(0, TORQUE_NOISE, TAU.shape)  # 토크 센서 잡음
    return Q, TIP, TAU, smooth_derivatives(Q_meas, TAU_meas)


def smooth_derivatives(Q_meas, TAU_meas, cutoff_hz=12.0, trim=0.05):
    """관절각을 두 번 미분해 q̈ 를 얻되, **영위상 저역통과** 후에 미분한다.

    생 엔코더 신호를 그냥 두 번 차분하면 잡음이 1/dt² 로 증폭돼(σ=20 µrad, dt=2 ms →
    q̈ 잡음 ~10 rad/s²) 식별이 완전히 무너진다. 실제 로봇 식별에서 쓰는 표준 처리:
    q·τ 를 같은 필터로 통과시킨 뒤 미분하고, filtfilt 의 경계 왜곡 구간은 잘라낸다.
    비인과(filtfilt)이지만 **식별은 오프라인 후처리**라 정당하다 — 실시간 제어 루프에
    쓰는 필터와 구분해야 한다(exp 22의 인과성 논의 참조)."""
    b, a = butter(4, cutoff_hz / (0.5 / DT), btype="low")
    q = filtfilt(b, a, Q_meas, axis=0)
    qd = filtfilt(b, a, np.gradient(q, DT, axis=0), axis=0)
    qdd = filtfilt(b, a, np.gradient(qd, DT, axis=0), axis=0)
    tau = filtfilt(b, a, TAU_meas, axis=0)
    k = max(int(trim * len(q)), 1)
    return q[k:-k], qd[k:-k], qdd[k:-k], tau[k:-k]


# --------------------------------------------------------------------------- #
# 식별: 누적 로그에 대한 최소자승 (+ 조건수로 여기 품질 진단)
# --------------------------------------------------------------------------- #
def identify(logs, decimate=5):
    """logs = [(q, qd, qdd, tau), ...] 를 쌓아 π 를 최소자승 추정.

    반환 (π̂, cond(YᵀY), 토크 잔차 RMS). decimate: 인접 샘플은 정보가 겹치므로 솎아
    회귀 크기를 줄인다(결과에 영향 미미, 속도만 개선)."""
    rows, rhs = [], []
    for q, qd, qdd, tau in logs:
        for k in range(0, len(q), decimate):
            rows.append(regressor(q[k], qd[k], qdd[k]))
            rhs.append(tau[k])
    A = np.vstack(rows)
    y = np.concatenate(rhs)
    pi_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
    cond = float(np.linalg.cond(A.T @ A))
    resid = float(np.sqrt(np.mean((A @ pi_hat - y) ** 2)))
    return pi_hat, cond, resid


def excitation_trajectory(t_end=6.0, seed=0):
    """식별 전용 여기 궤적: 관절별 다중 사인(주파수·위상 다름).

    관성·코리올리·중력·마찰이 **서로 다른 방식으로** 여기되도록 속도 부호가 바뀌고
    가속도가 큰 구간을 포함한다(쿨롱 마찰은 부호 반전 없이는 관측되지 않는다)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, t_end, DT)
    Q = np.zeros((len(t), 2))
    Qd = np.zeros_like(Q)
    Qdd = np.zeros_like(Q)
    for j in range(2):
        base = [0.6, -1.1][j]
        Q[:, j] = base
        for w, amp, ph in [(1.3 + 0.4 * j, 0.55, rng.uniform(0, 6.28)),
                           (2.7 + 0.5 * j, 0.30, rng.uniform(0, 6.28)),
                           (4.9 + 0.3 * j, 0.12, rng.uniform(0, 6.28))]:
            Q[:, j] += amp * np.sin(w * t + ph)
            Qd[:, j] += amp * w * np.cos(w * t + ph)
            Qdd[:, j] += -amp * w ** 2 * np.sin(w * t + ph)
    return Q, Qd, Qdd


# --------------------------------------------------------------------------- #
# 임상 과제: exp 42 의 삽입 과제를 UR5 팔로 (표적오차가 최종 지표)
# --------------------------------------------------------------------------- #
def clinical_task():
    """정합은 완벽하다고 두고(정합 몫 0), 서보 몫만 남긴 표적 도달 과제."""
    T_map = tgt.se2(np.deg2rad(6.0), np.array([0.62, 0.10]))   # UR5 리치에 맞춘 배치
    entry_img, _ = tgt.plan_entry()
    entry_rob = tgt.apply_se2(T_map, entry_img)
    target_rob = tgt.apply_se2(T_map, tgt.TUMOR_IMG)
    _, path, phases = tgt.cartesian_path(entry_rob, target_rob)

    q = np.array([0.6, -1.2])
    Q = np.zeros((len(path), 2))
    for k, p in enumerate(path):
        q, _, _, _ = kin.ik_dls(p, q, L=L_ARM, lam=1e-3, max_iters=60,
                                tol=1e-9, step_cap=0.15)
        Q[k] = q
    Qd = np.gradient(Q, DT, axis=0)
    Qdd = np.gradient(Qd, DT, axis=0)
    return Q, Qd, Qdd, path, target_rob, phases


def deploy(pi_hat, task, stribeck=False):
    """실기에 배치해 표적오차·추종잔차를 측정. 반환 (표적오차, 잔차 RMS, 팁궤적)."""
    Q_des, Qd_des, Qdd_des, path, target_rob, _ = task
    _, TIP, _ = rollout(PI_TRUE, pi_hat, Q_des, Qd_des, Qdd_des, stribeck=stribeck)
    tip_err = np.linalg.norm(TIP - path, axis=1)
    if not np.all(np.isfinite(TIP)):        # 모델이 너무 틀리면 제어가 발산한다
        return np.inf, np.inf, TIP, tip_err
    return (float(np.linalg.norm(TIP[-1] - target_rob)),
            float(np.sqrt(np.mean(tip_err ** 2))), TIP, tip_err)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(7)
    task = clinical_task()

    print("=== 43. 수술 팔 sim-to-real 폐루프 (공개 UR5 파라미터 + 도구/마찰 식별) ===")
    print(f"[플랜트] UR5 관절2·3 공개 사양: l={L1:.3f}/{L2:.5f} m, m={M1}/{M2} kg, "
          f"lc={LC1}/{LC2} m (관성은 균일봉 근사)")
    print(f"[실기에만 있는 것] 도구+손목 페이로드 {TOOL_MASS} kg, "
          f"점성마찰 {FV_TRUE.tolist()}, 쿨롱마찰 {FC_TRUE.tolist()} — 공칭 모델엔 없음")

    # 회귀자 항등식 자기검증: Y·π 가 역동역학과 일치해야 식별이 성립한다
    qc, qdc, qddc = np.array([0.3, -0.7]), np.array([0.9, -0.4]), np.array([1.1, 0.6])
    ident_err = float(np.max(np.abs(regressor(qc, qdc, qddc) @ PI_TRUE
                                    - inverse_dynamics(PI_TRUE, qc, qdc, qddc))))
    print(f"[검증] 회귀자 항등식 max|Y·π − τ| = {ident_err:.2e}")

    def run_loop(stribeck, rng, verbose=True):
        """배치 → 감지 → 식별 → 재배치 루프. 반환 (hist, π̂, conds, 배치 기록)."""
        pi_hat = PI_NOMINAL.copy()
        hist, logs, conds = [], [], []
        err0, resid0, TIP0, tiperr0 = deploy(pi_hat, task, stribeck)
        hist.append(dict(it=0, err=err0, resid=resid0,
                         perr=float(np.linalg.norm(pi_hat - PI_TRUE))))
        if verbose:
            trig = "감지됨 → 루프 시작" if resid0 > RESIDUAL_TRIGGER else "정상"
            print(f"반복 0 (공칭 모델 배치): 표적오차 {err0*1e3:8.3f} mm | "
                  f"추종잔차 {resid0*1e3:7.3f} mm | {trig}")
        for it in range(1, N_ITERS + 1):
            Qe, Qde, Qdde = excitation_trajectory(seed=it)
            *_, meas = rollout(PI_TRUE, pi_hat, Qe, Qde, Qdde, rng=rng, log=True,
                               stribeck=stribeck)
            logs.append(meas)
            pi_hat, cond, tres = identify(logs)
            conds.append(cond)
            err, resid, _, _ = deploy(pi_hat, task, stribeck)
            hist.append(dict(it=it, err=err, resid=resid,
                             perr=float(np.linalg.norm(pi_hat - PI_TRUE))))
            if verbose:
                print(f"반복 {it} (여기 {it}회 누적 → 식별): 표적오차 {err*1e3:8.3f} mm | "
                      f"추종잔차 {resid*1e3:7.3f} mm | π 오차 {hist[-1]['perr']:7.3f} | "
                      f"cond(YᵀY) {cond:.1e} | 토크잔차 {tres:.3f} N·m")
        return hist, pi_hat, conds, (TIP0, tiperr0)

    # ---- 루프 A: 파라미터 간극만 (도구 페이로드 + 점성·쿨롱 마찰) ----
    print("-" * 78)
    print("[루프 A] 파라미터 간극 — 실기의 모든 항이 π 안에 표현 가능")
    hist, pi_hat, conds, (TIP0, tiperr0) = run_loop(False, rng)
    err0, resid0 = hist[0]["err"], hist[0]["resid"]
    err_final, resid_final, TIP_final, tiperr_final = deploy(pi_hat, task)
    print("-" * 78)
    print(f"루프 효과: 표적오차 {err0*1e3:.3f} → {err_final*1e3:.3f} mm "
          f"({err0/max(err_final,1e-12):.0f}배), 추종잔차 {resid0*1e3:.3f} → "
          f"{resid_final*1e3:.3f} mm")
    print(f"파라미터 복원: 페이로드 유효질량 오차 {abs(pi_hat[4]-PI_TRUE[4])/PI_TRUE[4]*100:.2f}% "
          f"(G2 기준), 쿨롱마찰 추정 {np.round(pi_hat[7:9],3).tolist()} "
          f"vs 실제 {FC_TRUE.tolist()}")

    # ---- 루프 B: 구조 간극 (모델에 없는 스틱션) ----
    print("-" * 78)
    print("[루프 B] 구조 간극 — 실기에 스트라이벡 스틱션 추가(회귀자에 대응 열 없음)")
    hist_s, pi_hat_s, _, _ = run_loop(True, np.random.default_rng(21))
    err_s = hist_s[-1]["err"]
    print(f"  → 같은 루프인데 {hist_s[0]['err']*1e3:.3f} → {err_s*1e3:.3f} mm 에서 "
          f"멈춘다(A의 {err_s/max(err_final,1e-12):.0f}배). 파라미터를 아무리 잘 맞춰도 "
          "**모델에 없는 물리**는 식별이 흡수하지 못한다 — 루프가 알려주는 건 "
          "'더 볼 것이 남았다'는 사실이고, 다음 수는 재식별이 아니라 모델 구조 확장이다.")

    # ---- 관측성: 임상 삽입 궤적만으로 식별하면? ----
    Q_des, Qd_des, Qdd_des = task[0], task[1], task[2]
    *_, meas_task = rollout(PI_TRUE, PI_NOMINAL, Q_des, Qd_des, Qdd_des,
                            rng=np.random.default_rng(1), log=True)
    pi_task, cond_task, _ = identify([meas_task])
    err_task, resid_task, _, _ = deploy(pi_task, task)
    print("-" * 78)
    print("[관측성] 무엇으로 식별했는가가 결과를 가른다")
    print(f"  임상 삽입 궤적만 : cond(YᵀY) {cond_task:.1e} | π 오차 "
          f"{np.linalg.norm(pi_task-PI_TRUE):8.3f} | 표적오차 {err_task*1e3:8.3f} mm")
    print(f"  전용 여기 궤적   : cond(YᵀY) {conds[-1]:.1e} | π 오차 "
          f"{np.linalg.norm(pi_hat-PI_TRUE):8.3f} | 표적오차 {err_final*1e3:8.3f} mm")
    print("  → 느리고 매끄러운 임상 궤적은 관성·마찰을 여기하지 못해 회귀가 병조건. "
          "식별에는 '일부러 흔드는' 궤적이 필요하다(=관측성).")

    # ---- exp 42 예산으로의 환원 ----
    print("-" * 78)
    print(f"[exp 42 예산 재구성] 정합 몫 94 µm 기준")
    print(f"  루프 전        : 서보 {err0*1e6:6.0f} µm = 정합의 {err0/94e-6:.0f}배 → 서보 지배")
    print(f"  루프 후(A)     : 서보 {err_final*1e6:6.0f} µm = 정합의 "
          f"{err_final/94e-6:.2f}배 → 정합이 다시 지배항, 예산 순위 복귀")
    print(f"  루프 후(B, 구조간극): 서보 {err_s*1e6:6.0f} µm = 정합의 {err_s/94e-6:.1f}배 "
          "→ 여전히 서보 지배. 자동 루프만으로는 예산이 돌아오지 않는다")

    # ---- 그림 ----
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    its = [h["it"] for h in hist]

    ax = axes[0, 0]
    ax.semilogy(its, [h["err"] * 1e3 for h in hist], "-o", color="tab:green",
                label="A: parametric gap (payload+friction)")
    ax.semilogy([h["it"] for h in hist_s], [h["err"] * 1e3 for h in hist_s], "-s",
                color="tab:red", label="B: + structural gap (stiction)")
    ax.axhline(0.094, color="0.5", ls="--", lw=1, label="registration share (exp 42)")
    ax.set_xlabel("sim-to-real iteration"); ax.set_ylabel("target error [mm]")
    ax.set_title("Closing the loop: target error", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8); ax.set_xticks(its)

    ax = axes[0, 1]
    ax.semilogy(its, [max(h["perr"], 1e-6) for h in hist], "-o", color="tab:purple")
    ax.set_xlabel("iteration"); ax.set_ylabel("‖π̂ − π_true‖")
    ax.set_title("Parameter error (payload + friction identified)", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.set_xticks(its)

    ax = axes[0, 2]
    names = ["a", "b", "d", "G1", "G2", "fv1", "fv2", "fc1", "fc2"]
    xs = np.arange(len(names))
    ax.bar(xs - 0.26, PI_NOMINAL, 0.25, label="nominal (sim)", color="0.75")
    ax.bar(xs, pi_hat, 0.25, label="identified", color="tab:blue")
    ax.bar(xs + 0.26, PI_TRUE, 0.25, label="true (real)", color="tab:orange")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
    ax.set_title("Inertial + friction parameters", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    t_task = np.arange(len(tiperr0)) * DT
    ax.semilogy(t_task, np.maximum(tiperr0, 1e-9) * 1e3, color="crimson",
                label="before loop (nominal model)")
    ax.semilogy(t_task, np.maximum(tiperr_final, 1e-9) * 1e3, color="tab:green",
                label="after loop (identified)")
    n_app, n_ins, _ = task[5]
    ax.axvspan(n_app * DT, (n_app + n_ins) * DT, color="0.9", label="insertion")
    ax.set_xlabel("t [s]"); ax.set_ylabel("tip path error [mm]")
    ax.set_title("Tracking along the clinical insertion", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(*task[3].T, color="0.6", lw=2, label="planned path")
    ax.plot(*TIP0.T, color="crimson", lw=1.2, label="executed: before")
    ax.plot(*TIP_final.T, color="tab:green", lw=1.2, ls="--", label="executed: after")
    ax.scatter(*task[4], marker="*", s=140, color="k", zorder=5, label="target")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Executed vs planned (UR5 planar 2R)", fontsize=10); ax.legend(fontsize=7)

    ax = axes[1, 2]
    vals = [err_task * 1e3, err_final * 1e3]
    ax.bar(["clinical trajectory\n(low excitation)", "dedicated excitation"], vals,
           color=["crimson", "seagreen"])
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:.3f} mm", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log"); ax.set_ylabel("target error after identification [mm]")
    ax.set_title(f"Observability: cond {cond_task:.0e} vs {conds[-1]:.0e}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("43. Sim-to-real loop on a surgical arm — deploy, detect, identify, redeploy "
                 "(UR5 published parameters)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "43_sim_to_real_arm.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/43_sim_to_real_arm.png, assets/43_sim_to_real_arm.png")

    return dict(hist=hist, hist_stribeck=hist_s, pi_hat=pi_hat, err0=err0,
                err_final=err_final, err_stribeck=err_s, cond_task=cond_task,
                cond_exc=conds[-1], err_task=err_task,
                detected=bool(resid0 > RESIDUAL_TRIGGER))


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - '실기'도 결국 시뮬레이션이다. 다만 모델 구조가 아니라 **파라미터와 누락 항**
#     (페이로드·마찰)이 어긋나게 두어, 실제 배치에서 겪는 실패 모드를 재현했다.
#     진짜 로봇에는 백래시·유연성·감속기 비선형·관절 탄성이 더 있다.
#   - 링크 파라미터는 UR5 공개 사양이지만 회전관성은 균일봉 근사다(공개표의 관성값이
#     출처마다 달라 재현 가능한 근사를 명시 채택). 결론(식별로 페이로드·마찰 복원)은
#     이 근사에 의존하지 않는다.
#   - 최소자승 식별은 여기(excitation)에 의존한다. 관측성 실험이 그 한계를 정량화한다.
#     실무에서는 조건수를 최소화하는 '최적 여기 궤적 설계'가 별도 주제다.
#   - 마찰은 점성+쿨롱(tanh 근사)만 모델링했다. 스틱션·스트라이벡 효과는 저속에서
#     남는 오차가 되며, 삽입처럼 느린 구간에서 특히 문제가 된다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
