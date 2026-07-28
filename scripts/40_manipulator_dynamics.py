"""매니퓰레이터 동역학 + 계산토크 제어: 모델을 알면 왜 더 정확한가.

exp39가 매니퓰레이터의 '기구학'(관절각↔말단위치)을 다뤘다면, 여기서는 나머지 절반인
'동역학'과 제어를 다룬다. 팔은 질량과 관성을 가진 강체 사슬이므로, 관절 토크와 운동의
관계는 다음 강체 매니퓰레이터 방정식(라그랑주 유도)으로 지배된다.

    M(q) q̈ + C(q, q̇) q̇ + g(q) = τ

    M(q)      : 관성행렬(대칭 양정치). 자세 q에 따라 유효관성이 변한다.
    C(q, q̇)  : 코리올리/원심 항. 관절이 서로 밀고 당기는 '커플링'을 담는다.
    g(q)      : 중력 벡터. 팔을 뻗을수록 아래로 처지게 하는 토크.
    τ         : 관절 구동 토크(입력).

2링크 평면 팔(canonical example)에 대해 위 세 항을 라그랑지안에서 직접 유도해 구현한다
(아래 M/C/g 함수 주석에 수식 명시). 순동역학은 q̈ = M⁻¹(τ − C q̇ − g)로 적분한다.

--- 왜 동역학을 아는 제어가 필요한가 ---
같은 관절궤적을 세 제어기로 추종시켜 비교한다.

  (1) PD 제어           : τ = Kp e + Kd ė.  중력·커플링·관성을 무시하므로 처짐/지연 오차.
  (2) PD + 중력보상     : τ = Kp e + Kd ė + g(q).  정적 처짐은 없애지만 커플링/관성은 남음.
  (3) 계산토크(역동역학) : τ = M(q)(q̈_d + Kp e + Kd ė) + C(q, q̇) q̇ + g(q).

계산토크는 모델을 그대로 상쇄해 넣어, 폐루프 오차동역학을 ë + Kd ė + Kp e = 0 이라는
'선형·비커플링' 형태로 만든다(피드백 선형화). 각 관절이 독립 2차 시스템처럼 거동하므로
빠른 동작·큰 중력에서도 오차가 사실상 0으로 수렴한다.

--- 의료/수술 매니퓰레이터에서의 의미 ---
수술 팔은 조직과 부드럽게(compliant) 상호작용하면서도 서브밀리미터로 정확해야 한다.
중력·관성·커플링을 모델로 상쇄해두면 낮은 피드백 이득으로도 정밀 추종이 가능해져,
'단단한 위치제어'가 아니라 '부드럽지만 정확한' 힘/위치 협조가 가능하다. 반대로 모델을
모르면 이득을 높여 뻣뻣하게 눌러야 하고, 이는 조직 손상 위험을 키운다.

정직한 한계: 계산토크의 우수함은 '정확한 모델'을 전제로 한다. 실제 팔에는 여기서
모델링하지 않은 마찰·관절 유연성·페이로드 변화가 있어, 모델이 틀리면 성능이 떨어진다.
이를 확인하려고 질량을 20% 틀리게 준 계산토크(mismatch)도 함께 비교한다.

    python scripts/40_manipulator_dynamics.py
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# --------------------------------------------------------------------------
# 2링크 평면 팔의 물리 파라미터 (공개/합성 값, SI 단위)
#   m  : 링크 질량[kg]   l : 링크 길이[m]   lc : 링크 무게중심까지 거리[m]
#   I  : 무게중심 기준 링크 관성[kg·m²]   G : 중력가속도[m/s²]
# --------------------------------------------------------------------------
class ArmParams:
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0,
                 lc1=0.5, lc2=0.5, I1=1.0 / 12, I2=1.0 / 12, g=9.81):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.g = g


DT = 0.002          # 적분 스텝[s] (RK4, 고정)
T_END = 5.0         # 시뮬레이션 시간[s]


# --------------------------------------------------------------------------
# 라그랑주 유도 결과 — 2링크 평면 팔의 M(q), C(q,q̇), g(q)
#
# 축약상수:
#   a = I1 + I2 + m1 lc1² + m2 (l1² + lc2²)
#   b = m2 l1 lc2
#   d = I2 + m2 lc2²
#
# 관성행렬 (대칭 양정치):
#   M = [[ a + 2 b cos q2 ,  d + b cos q2 ],
#        [ d + b cos q2   ,  d            ]]
#
# 코리올리/원심 행렬 (C q̇ 가 실제 항):
#   C = [[ -b sin q2 · q̇2 , -b sin q2 · (q̇1 + q̇2) ],
#        [  b sin q2 · q̇1 ,  0                     ]]
#
# 중력 벡터:
#   g1 = (m1 lc1 + m2 l1) g cos q1 + m2 lc2 g cos(q1 + q2)
#   g2 = m2 lc2 g cos(q1 + q2)
# --------------------------------------------------------------------------
def mass_matrix(q, p: ArmParams):
    q2 = q[1]
    a = p.I1 + p.I2 + p.m1 * p.lc1 ** 2 + p.m2 * (p.l1 ** 2 + p.lc2 ** 2)
    b = p.m2 * p.l1 * p.lc2
    d = p.I2 + p.m2 * p.lc2 ** 2
    c2 = np.cos(q2)
    return np.array([[a + 2 * b * c2, d + b * c2],
                     [d + b * c2,     d]])


def coriolis_matrix(q, qd, p: ArmParams):
    q2 = q[1]
    b = p.m2 * p.l1 * p.lc2
    s2 = np.sin(q2)
    return np.array([[-b * s2 * qd[1], -b * s2 * (qd[0] + qd[1])],
                     [ b * s2 * qd[0],  0.0]])


def gravity_vector(q, p: ArmParams):
    q1, q2 = q
    g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.cos(q1) \
        + p.m2 * p.lc2 * p.g * np.cos(q1 + q2)
    g2 = p.m2 * p.lc2 * p.g * np.cos(q1 + q2)
    return np.array([g1, g2])


def forward_dynamics(q, qd, tau, p: ArmParams):
    """순동역학: q̈ = M⁻¹ (τ − C q̇ − g)."""
    M = mass_matrix(q, p)
    C = coriolis_matrix(q, qd, p)
    g = gravity_vector(q, p)
    return np.linalg.solve(M, tau - C @ qd - g)


def forward_kinematics(q, p: ArmParams):
    """말단위치(x, y) — 추종오차를 작업공간에서도 보기 위해."""
    x = p.l1 * np.cos(q[0]) + p.l2 * np.cos(q[0] + q[1])
    y = p.l1 * np.sin(q[0]) + p.l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


# --------------------------------------------------------------------------
# 기준 관절궤적: 두 관절 모두 매끄러운 정현파. 속도/가속도 해석적으로 얻는다.
# 중력이 크게 걸리는 자세를 지나가도록 오프셋을 준다(팔을 수평으로 뻗음).
# --------------------------------------------------------------------------
def desired_trajectory(t):
    w = 2.0                     # 각진동수[rad/s] (주기 ~3.1s) — 관성/커플링을 자극
    A1, A2 = 0.7, 0.9           # 진폭[rad]
    off1, off2 = 0.3, 0.5       # 중심 자세[rad] (중력토크가 실리는 영역)
    q1 = off1 + A1 * np.sin(w * t)
    q2 = off2 + A2 * np.sin(w * t + 0.5)
    qd1 = A1 * w * np.cos(w * t)
    qd2 = A2 * w * np.cos(w * t + 0.5)
    qdd1 = -A1 * w ** 2 * np.sin(w * t)
    qdd2 = -A2 * w ** 2 * np.sin(w * t + 0.5)
    return (np.array([q1, q2]), np.array([qd1, qd2]), np.array([qdd1, qdd2]))


# --------------------------------------------------------------------------
# 세 가지 제어기. 모두 e = q_d − q, ė = q̇_d − q̇.
# Kp, Kd 는 세 제어기가 동일 → 공정한 비교. 계산토크는 M/C/g 를 '모델'로 사용하며,
# 모델오차 확인을 위해 제어용 파라미터 p_ctrl 을 실제 팔 p_true 와 분리해 받는다.
# --------------------------------------------------------------------------
def ctrl_pd(q, qd, qd_d, qdd_d, qd_ref, e, edot, Kp, Kd, p_ctrl):
    return Kp * e + Kd * edot


def ctrl_pd_gravcomp(q, qd, qd_d, qdd_d, qd_ref, e, edot, Kp, Kd, p_ctrl):
    return Kp * e + Kd * edot + gravity_vector(q, p_ctrl)


def ctrl_computed_torque(q, qd, qd_d, qdd_d, qd_ref, e, edot, Kp, Kd, p_ctrl):
    M = mass_matrix(q, p_ctrl)
    C = coriolis_matrix(q, qd, p_ctrl)
    g = gravity_vector(q, p_ctrl)
    return M @ (qdd_d + Kp * e + Kd * edot) + C @ qd + g


def simulate(controller, Kp, Kd, p_true, p_ctrl=None):
    """고정스텝 RK4로 팔을 적분. 제어토크는 스텝 시작에서 계산해 스텝 내내 유지(ZOH).

    반환: t, 실제 q(N,2), 기준 q_d(N,2), 관절오차 e(N,2), 토크 τ(N,2)
    """
    if p_ctrl is None:
        p_ctrl = p_true
    n = int(round(T_END / DT))
    t = np.arange(n + 1) * DT

    q_d0, qd_d0, _ = desired_trajectory(0.0)
    q = q_d0.copy()             # 기준에서 출발(초기오차 0) → 정상상태 추종오차만 비교
    qd = qd_d0.copy()

    Q = np.zeros((n + 1, 2))
    QD_des = np.zeros((n + 1, 2))
    E = np.zeros((n + 1, 2))
    TAU = np.zeros((n + 1, 2))

    for k in range(n + 1):
        tk = t[k]
        q_d, qd_d, qdd_d = desired_trajectory(tk)
        e = q_d - q
        edot = qd_d - qd
        tau = controller(q, qd, qd_d, qdd_d, qd, e, edot, Kp, Kd, p_ctrl)

        Q[k] = q
        QD_des[k] = q_d
        E[k] = e
        TAU[k] = tau

        if k == n:
            break

        # RK4 적분 (토크 ZOH, 실제 팔 p_true 로 전진)
        def deriv(state):
            qq = state[:2]
            qqd = state[2:]
            qdd = forward_dynamics(qq, qqd, tau, p_true)
            return np.concatenate([qqd, qdd])

        s = np.concatenate([q, qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * DT * k1)
        k3 = deriv(s + 0.5 * DT * k2)
        k4 = deriv(s + DT * k3)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = s[:2], s[2:]

    return t, Q, QD_des, E, TAU


def joint_rmse(E):
    """관절공간 RMSE[rad] — 두 관절·전 시간에 대한 RMS."""
    return float(np.sqrt(np.mean(E ** 2)))


def ee_rmse(Q, QD_des, p: ArmParams):
    """말단 위치 RMSE[m]."""
    ee = np.array([forward_kinematics(q, p) for q in Q])
    ee_d = np.array([forward_kinematics(q, p) for q in QD_des])
    return float(np.sqrt(np.mean(np.sum((ee - ee_d) ** 2, axis=1))))


def main():
    p = ArmParams()

    # 동일 이득(임계감쇠 목표: wn=20, ζ=1 → Kp=wn², Kd=2ζwn)
    Kp = np.array([400.0, 400.0])
    Kd = np.array([40.0, 40.0])

    t, Q_pd, Qd, E_pd, Tau_pd = simulate(ctrl_pd, Kp, Kd, p)
    _, Q_gc, _, E_gc, Tau_gc = simulate(ctrl_pd_gravcomp, Kp, Kd, p)
    _, Q_ct, _, E_ct, Tau_ct = simulate(ctrl_computed_torque, Kp, Kd, p)

    # 모델오차: 제어기가 질량을 20% 과대추정한 경우의 계산토크
    p_wrong = ArmParams(m1=1.2 * p.m1, m2=1.2 * p.m2)
    _, Q_mm, _, E_mm, Tau_mm = simulate(ctrl_computed_torque, Kp, Kd, p, p_ctrl=p_wrong)

    pd_rmse = joint_rmse(E_pd)
    gc_rmse = joint_rmse(E_gc)
    ct_rmse = joint_rmse(E_ct)
    mm_rmse = joint_rmse(E_mm)

    pd_ee = ee_rmse(Q_pd, Qd, p)
    gc_ee = ee_rmse(Q_gc, Qd, p)
    ct_ee = ee_rmse(Q_ct, Qd, p)

    # M(q) 대칭 양정치 점검(수치)
    Mtest = mass_matrix(np.array([0.3, 0.7]), p)
    sym_ok = np.allclose(Mtest, Mtest.T)
    pd_ok = np.all(np.linalg.eigvalsh(Mtest) > 0)

    print("=== 2링크 팔 동역학: PD vs 중력보상 vs 계산토크 (관절궤적 추종) ===")
    print(f"적분 RK4 dt={DT}s, T={T_END}s, 동일이득 Kp={Kp[0]:.0f} Kd={Kd[0]:.0f}")
    print(f"M(q) 대칭={sym_ok}, 양정치={pd_ok}, 고윳값={np.linalg.eigvalsh(Mtest)}")
    print("-" * 60)
    print(f"{'controller':<24}{'joint RMSE[rad]':>16}{'EE RMSE[m]':>14}")
    print(f"{'PD only':<24}{pd_rmse:>16.5f}{pd_ee:>14.5f}")
    print(f"{'PD + gravity comp':<24}{gc_rmse:>16.5f}{gc_ee:>14.5f}")
    print(f"{'computed-torque':<24}{ct_rmse:>16.5f}{ct_ee:>14.5f}")
    print("-" * 60)
    print(f"계산토크 개선율(vs PD)         : {100 * (pd_rmse - ct_rmse) / pd_rmse:.1f}%")
    print(f"모델오차(+20% 질량) 계산토크   : joint RMSE {mm_rmse:.5f} rad "
          f"({mm_rmse / ct_rmse:.0f}배 악화 — 정확한 모델이 관건)")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.0], hspace=0.42, wspace=0.22)
    cP, cG, cC = "#d9534f", "#f0ad4e", "#1f77b4"

    # (1,2) 관절1·관절2 추종
    for j, name in enumerate(["joint 1", "joint 2"]):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(t, np.degrees(Qd[:, j]), "k--", lw=1.4, alpha=0.7, label="desired")
        ax.plot(t, np.degrees(Q_pd[:, j]), color=cP, lw=1.3, alpha=0.9, label="PD")
        ax.plot(t, np.degrees(Q_gc[:, j]), color=cG, lw=1.3, alpha=0.9, label="PD+grav")
        ax.plot(t, np.degrees(Q_ct[:, j]), color=cC, lw=1.6, label="computed-torque")
        ax.set_title(f"{name}: desired vs actual", fontsize=10)
        ax.set_ylabel("angle [deg]")
        ax.grid(alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    # (3) 관절오차 크기(‖e‖) 시간이력
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, np.degrees(np.linalg.norm(E_pd, axis=1)), color=cP, lw=1.3, label=f"PD ({pd_rmse:.4f})")
    ax.plot(t, np.degrees(np.linalg.norm(E_gc, axis=1)), color=cG, lw=1.3, label=f"PD+grav ({gc_rmse:.4f})")
    ax.plot(t, np.degrees(np.linalg.norm(E_ct, axis=1)), color=cC, lw=1.6, label=f"comp-torque ({ct_rmse:.4f})")
    ax.set_title("joint tracking error ‖e‖ over time (comp-torque ≈ 0)", fontsize=10)
    ax.set_ylabel("‖e‖ [deg]")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="upper right", title="RMSE [rad]")

    # (4) 말단 궤적(작업공간)
    ax = fig.add_subplot(gs[1, 1])
    ee_d = np.array([forward_kinematics(q, p) for q in Qd])
    ee_pd = np.array([forward_kinematics(q, p) for q in Q_pd])
    ee_ct = np.array([forward_kinematics(q, p) for q in Q_ct])
    ax.plot(ee_d[:, 0], ee_d[:, 1], "k--", lw=1.4, alpha=0.7, label="desired")
    ax.plot(ee_pd[:, 0], ee_pd[:, 1], color=cP, lw=1.3, alpha=0.9, label="PD")
    ax.plot(ee_ct[:, 0], ee_ct[:, 1], color=cC, lw=1.6, label="computed-torque")
    ax.plot(0, 0, "ks", ms=6)
    ax.set_aspect("equal")
    ax.set_title("end-effector path (workspace)", fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")

    # (5) 토크 명령(관절1)
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, Tau_pd[:, 0], color=cP, lw=1.1, alpha=0.9, label="PD")
    ax.plot(t, Tau_gc[:, 0], color=cG, lw=1.1, alpha=0.9, label="PD+grav")
    ax.plot(t, Tau_ct[:, 0], color=cC, lw=1.4, label="computed-torque")
    ax.set_title("joint-1 torque command", fontsize=10)
    ax.set_ylabel("τ₁ [N·m]")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="upper right")

    # (6) 모델오차 영향
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, np.degrees(np.linalg.norm(E_ct, axis=1)), color=cC, lw=1.5,
            label=f"exact model ({ct_rmse:.4f})")
    ax.plot(t, np.degrees(np.linalg.norm(E_mm, axis=1)), color="#6f42c1", lw=1.3,
            label=f"+20% mass err ({mm_rmse:.4f})")
    ax.set_title("computed-torque needs an accurate model", fontsize=10)
    ax.set_ylabel("‖e‖ [deg]")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="upper right", title="RMSE [rad]")

    fig.suptitle("Manipulator dynamics & computed-torque control (2-link planar arm)",
                 fontsize=13, y=0.98)
    for pth in ("outputs/40_manipulator_dynamics.png", "assets/40_manipulator_dynamics.png"):
        fig.savefig(pth, dpi=125, bbox_inches="tight")
    print("\n[plot] outputs/40_manipulator_dynamics.png, assets/40_manipulator_dynamics.png")

    return pd_rmse, gc_rmse, ct_rmse


if __name__ == "__main__":
    main()
