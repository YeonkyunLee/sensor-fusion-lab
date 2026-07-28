"""UR5 6축 공간 매니퓰레이터: 기구학 · 야코비안 · 동역학 (공개 사양 기반).

exp 39·40·43 은 평면(2R) 팔이었다. 이 모듈은 같은 이론을 **공간 6-DOF** 로 올린다.
링크 파라미터는 지어내지 않고 **Universal Robots UR5 의 공개 DH·질량·무게중심 표**를
쓴다. 회전관성 텐서는 출처마다 값이 달라, 각 링크를 **균일 원기둥**으로 근사하고
그 사실을 명시한다(결론은 이 근사에 의존하지 않는다).

구성:
  - 표준 DH 순기구학: T_i = Rz(θ) Tz(d) Tx(a) Rx(α)
  - 기하 야코비안(6×6): 회전관절 i 에 대해 J_v = z_i × (p_e − p_i), J_ω = z_i
  - 감쇠최소자승 6-DOF IK: 위치 3 + 자세 3(so(3) 로그) 오차를 동시에 줄인다
  - 동역학: 링크별 야코비안으로 M(q) 를 조립하고(운동에너지 정의 그대로),
    g(q) 는 위치에너지의 해석적 기울기, C(q,q̇)q̇ 는 M 의 수치 미분에서 나오는
    크리스토펠 기호로 계산한다. RNEA 보다 느리지만 **정의에서 바로 유도**되어
    검증이 쉽다(에너지 보존 테스트로 확인).
"""

from __future__ import annotations

import numpy as np

from .se3 import so3_log

# --------------------------------------------------------------------------- #
# UR5 공개 사양
#   DH: Universal Robots 가 공개한 UR5 표준 DH 표
#   질량·무게중심: UR 공개 동역학 파라미터 표(링크 좌표계 기준)
# --------------------------------------------------------------------------- #
DH_D = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
DH_A = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

MASS = np.array([3.7, 8.393, 2.275, 1.219, 1.219, 0.1879])
COM = np.array([
    [0.0, -0.02561, 0.00193],
    [0.2125, 0.0, 0.11336],
    [0.15, 0.0, 0.0265],
    [0.0, -0.0018, 0.01634],
    [0.0, 0.0018, 0.01634],
    [0.0, 0.0, -0.001159],
])
GRAVITY = np.array([0.0, 0.0, -9.81])

# 균일 원기둥 근사용 치수(반경, 길이) — 공개 관성텐서가 출처마다 달라 명시적 근사
_CYL = np.array([[0.06, 0.15], [0.055, 0.425], [0.045, 0.392],
                 [0.045, 0.10], [0.045, 0.10], [0.035, 0.06]])


def link_inertias():
    """링크 좌표계 기준 관성텐서 6개 (3,3). 균일 원기둥(축 = z) 근사."""
    out = []
    for m, (r, L) in zip(MASS, _CYL):
        ixx = iyy = m * (3 * r ** 2 + L ** 2) / 12.0
        izz = m * r ** 2 / 2.0
        out.append(np.diag([ixx, iyy, izz]))
    return out


INERTIA = link_inertias()


# --------------------------------------------------------------------------- #
# 기구학
# --------------------------------------------------------------------------- #
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_all(q):
    """base→링크 i 변환 6개를 누적해 반환 (6,4,4)."""
    T = np.eye(4)
    out = []
    for i in range(6):
        T = T @ dh_transform(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
        out.append(T.copy())
    return np.array(out)


def fk(q, tool=None):
    """말단(또는 도구 팁) 포즈 4×4. tool 은 플랜지→도구 팁 고정 변환."""
    T = fk_all(q)[-1]
    return T if tool is None else T @ tool


def jacobian(q, tool=None):
    """기하 야코비안 (6,6): 위 3행 선속도, 아래 3행 각속도 (base 좌표계)."""
    Ts = fk_all(q)
    p_e = (Ts[-1] if tool is None else Ts[-1] @ tool)[:3, 3]
    J = np.zeros((6, 6))
    z_prev, p_prev = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    for i in range(6):
        J[:3, i] = np.cross(z_prev, p_e - p_prev)
        J[3:, i] = z_prev
        z_prev, p_prev = Ts[i][:3, 2], Ts[i][:3, 3]
    return J


def pose_error(T_cur, T_des):
    """6벡터 오차 [위치, 자세]. 자세는 so(3) 로그(회전벡터)."""
    e_p = T_des[:3, 3] - T_cur[:3, 3]
    e_r = so3_log(T_des[:3, :3] @ T_cur[:3, :3].T)
    return np.concatenate([e_p, e_r])


def manipulability(q, tool=None):
    J = jacobian(q, tool)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def ik_dls(T_des, q0, tool=None, lam=0.05, max_iters=300, tol=1e-6,
           step_cap=0.3, w_rot=1.0):
    """6-DOF 감쇠최소자승 IK. 반환 (q, iters, 위치오차[m], 자세오차[rad]).

    특이점(UR5 손목 정렬 등)에서 감쇠가 관절속도를 유계로 유지한다 — exp 39 의
    평면 결론이 공간에서도 그대로다."""
    q = np.array(q0, float).copy()
    W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
    it = 0
    for it in range(1, max_iters + 1):
        e = W @ pose_error(fk(q, tool), T_des)
        if np.linalg.norm(e) < tol:
            break
        J = W @ jacobian(q, tool)
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > step_cap:
            dq *= step_cap / n
        q = q + dq
    err = pose_error(fk(q, tool), T_des)
    return q, it, float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))


# --------------------------------------------------------------------------- #
# 동역학 — 정의에서 바로 조립(검증 용이)
# --------------------------------------------------------------------------- #
def _link_jacobians(q):
    """링크별 무게중심 야코비안 (J_v, J_ω) 와 회전행렬. 운동에너지 조립용."""
    Ts = fk_all(q)
    origins = [np.zeros(3)] + [T[:3, 3] for T in Ts]
    axes = [np.array([0.0, 0.0, 1.0])] + [T[:3, 2] for T in Ts]
    Jvs, Jws, Rs = [], [], []
    for i in range(6):
        R_i = Ts[i][:3, :3]
        p_ci = Ts[i][:3, 3] + R_i @ COM[i]
        Jv = np.zeros((3, 6))
        Jw = np.zeros((3, 6))
        for j in range(i + 1):
            Jv[:, j] = np.cross(axes[j], p_ci - origins[j])
            Jw[:, j] = axes[j]
        Jvs.append(Jv)
        Jws.append(Jw)
        Rs.append(R_i)
    return Jvs, Jws, Rs


def mass_matrix(q):
    """M(q) = Σ_i [ m_i J_vi^T J_vi + J_ωi^T (R_i I_i R_iᵀ) J_ωi ] — 운동에너지 정의."""
    Jvs, Jws, Rs = _link_jacobians(q)
    M = np.zeros((6, 6))
    for i in range(6):
        M += MASS[i] * (Jvs[i].T @ Jvs[i])
        M += Jws[i].T @ (Rs[i] @ INERTIA[i] @ Rs[i].T) @ Jws[i]
    return 0.5 * (M + M.T)


def gravity_torque(q):
    """g(q) = −Σ_i m_i J_viᵀ g  (위치에너지 기울기)."""
    Jvs, _, _ = _link_jacobians(q)
    g = np.zeros(6)
    for i in range(6):
        g -= MASS[i] * (Jvs[i].T @ GRAVITY)
    return g


def coriolis_torque(q, qd, eps=1e-5):
    """C(q,q̇)q̇ — M 의 수치 편미분에서 나오는 크리스토펠 기호로 계산."""
    dM = np.empty((6, 6, 6))
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = eps
        dM[k] = (mass_matrix(q + dq) - mass_matrix(q - dq)) / (2 * eps)
    # c_k = Σ_ij 1/2 (∂M_kj/∂q_i + ∂M_ki/∂q_j − ∂M_ij/∂q_k) q̇_i q̇_j
    c = np.einsum("ikj,i,j->k", dM, qd, qd) \
        - 0.5 * np.einsum("kij,i,j->k", dM, qd, qd)
    return c


def inverse_dynamics(q, qd, qdd):
    return mass_matrix(q) @ qdd + coriolis_torque(q, qd) + gravity_torque(q)


def forward_dynamics(q, qd, tau):
    return np.linalg.solve(mass_matrix(q),
                           tau - coriolis_torque(q, qd) - gravity_torque(q))


# --------------------------------------------------------------------------- #
# 재귀 뉴턴-오일러(RNEA) — 같은 동역학의 O(n) 구현
#
# 위의 라그랑주 조립(M, C, g)은 정의에 가까워 읽기 쉽지만, C 를 수치미분으로 얻어
# 호출당 M 을 12번 평가한다(≈5 ms). 시뮬레이션에는 너무 느리므로 표준 RNEA 를 함께
# 둔다. 둘은 **독립 구현**이라 서로의 검증기가 된다(tests 에서 일치 확인).
#
#   전진: 링크를 따라 ω, ω̇, 무게중심 가속도를 전파 (중력은 베이스 가속도 −g 로 주입)
#   후진: 링크에 걸리는 힘·모멘트를 되짚어 관절 토크 τ_i = n_i · z 를 얻는다
# --------------------------------------------------------------------------- #
_Z = np.array([0.0, 0.0, 1.0])


def _cross(a, b):
    """3벡터 외적 — np.cross 는 일반화 오버헤드가 커서 내부 루프에선 직접 계산한다."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def rnea(q, qd, qdd, gravity=True):
    """역동역학 τ = M(q)q̈ + C(q,q̇)q̇ + g(q) 를 O(n) 으로. gravity=False면 중력 제외.

    **표준 DH 주의**: T_i^{i-1} = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i) 이므로 관절 i 의 축은
    프레임 i 가 아니라 **프레임 i−1 의 z** 다. 따라서 관절 속도·가속도는 프레임 i−1
    에서 더한 뒤 회전시켜야 하고, 토크도 그 축에 사영해야 한다(τ_i = (R_i n_i)·z).
    이 한 칸을 틀리면 토크 벡터가 통째로 한 관절씩 밀린다."""
    Ts = [dh_transform(q[i], DH_D[i], DH_A[i], DH_ALPHA[i]) for i in range(6)]
    omega = np.zeros(3)      # 프레임 i-1 기준 각속도
    alpha = np.zeros(3)
    acc = -GRAVITY if gravity else np.zeros(3)      # 베이스 가속도에 중력 주입
    F, N = [], []
    for i in range(6):
        R, p = Ts[i][:3, :3], Ts[i][:3, 3]
        Rt = R.T
        # 관절 회전은 프레임 i−1 의 z 축에서 일어난다
        w_j = omega + qd[i] * _Z
        a_j = alpha + qdd[i] * _Z + _cross(omega, qd[i] * _Z)
        acc = Rt @ (acc + _cross(a_j, p) + _cross(w_j, _cross(w_j, p)))
        omega, alpha = Rt @ w_j, Rt @ a_j           # 이제 프레임 i 기준
        c = COM[i]
        a_c = acc + _cross(alpha, c) + _cross(omega, _cross(omega, c))
        F.append(MASS[i] * a_c)
        N.append(INERTIA[i] @ alpha + _cross(omega, INERTIA[i] @ omega))

    # 후진 재귀 (Spong 표준 DH 형식). 관절 i 의 축은 **프레임 i−1 의 원점**을 지나므로,
    # 모멘트 팔을 두 개 구분해야 한다:
    #   r_prev = 프레임 i−1 원점 → 링크 i 무게중심 (프레임 i 표현) = R_iᵀ p_i + c_i
    #   r_cur  = 프레임 i   원점 → 링크 i 무게중심 = c_i
    # (Craig 의 수정 DH 형식을 그대로 쓰면 a_i ≠ 0 인 링크에서만 틀린다 — 실제로
    #  링크 2·3 의 토크만 어긋나는 증상으로 나타났다.)
    tau = np.zeros(6)
    f = np.zeros(3)
    n = np.zeros(3)
    for i in range(5, -1, -1):
        R_i, p_i = Ts[i][:3, :3], Ts[i][:3, 3]
        c_i = COM[i]
        r_prev = R_i.T @ p_i + c_i
        if i < 5:
            R_next = Ts[i + 1][:3, :3]
            f_next = R_next @ f
            n_next = R_next @ n
        else:
            f_next = np.zeros(3)
            n_next = np.zeros(3)
        f = f_next + F[i]
        n = n_next - _cross(f, r_prev) + _cross(f_next, c_i) + N[i]
        tau[i] = (R_i @ n) @ _Z                     # 관절 i 의 축 = 프레임 i−1 의 z
    return tau


def mass_matrix_rnea(q):
    """단위 가속도 열로 M(q) 조립 (중력·속도항 제외한 RNEA 6회)."""
    M = np.zeros((6, 6))
    e = np.zeros(6)
    for j in range(6):
        e[:] = 0.0
        e[j] = 1.0
        M[:, j] = rnea(q, np.zeros(6), e, gravity=False)
    return 0.5 * (M + M.T)


def forward_dynamics_fast(q, qd, tau):
    """순동역학(시뮬레이션용). M 은 라그랑주 조립(행렬 연산이라 빠름), 속도·중력
    바이어스는 RNEA 1회로 얻는다 — 수치미분 크리스토펠(=M 12회 평가)을 없애 ~5배 빠르다.
    결과는 forward_dynamics() 와 수치적으로 동일하다(tests 에서 확인)."""
    bias = rnea(q, qd, np.zeros(6))                 # C q̇ + g
    return np.linalg.solve(mass_matrix(q), tau - bias)


def energy(q, qd):
    """총 역학에너지(운동+위치) — 무토크·무마찰 적분에서 보존되어야 한다."""
    T = 0.5 * qd @ mass_matrix(q) @ qd
    Ts = fk_all(q)
    U = 0.0
    for i in range(6):
        p_ci = Ts[i][:3, 3] + Ts[i][:3, :3] @ COM[i]
        U -= MASS[i] * GRAVITY @ p_ci
    return float(T + U)
