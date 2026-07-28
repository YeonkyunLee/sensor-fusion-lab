"""매니퓰레이터(로봇 팔) 기구학: FK · 야코비안 · IK · 특이점 · 여유자유도.

이 저장소는 지금까지 이동로봇(추정·SLAM·계획·제어·학습)을 다뤄 왔지만, 관절형
로봇 팔(articulated arm)은 없었다. 수술·의료 로봇의 핵심은 '팔 끝(도구, end-effector)을
원하는 위치·자세로 보내는' 능력이며, 그 토대가 바로 매니퓰레이터 기구학이다.
본 실험은 3링크 평면(3R) 팔로 그 네 기둥을 밑바닥부터 구현한다.

  - 순기구학(FK): 관절각 q=(q1,q2,q3) → 팔끝 위치·자세 (x, y, phi). 링크를 차례로
    합성한다. x = Σ l_i cos(부분각합), y = Σ l_i sin(부분각합), phi = Σ q_i.

  - 야코비안(J): 관절속도 → 팔끝속도의 선형사상. 평면 팔은 위치 2행 + 자세 1행의
    3×3 행렬이며, 위치행은 FK를 관절각으로 편미분해 해석적으로 얻는다
    (∂x/∂q_i = -(그 관절 바깥쪽 링크들의 y기여), ∂y/∂q_i = +x기여). IK·특이점·
    조작성(manipulability)이 모두 J에서 나온다.

  - 역기구학(IK): 목표 팔끝포즈 → 관절각. 닫힌형 대신 J 기반 반복법을 쓴다.
    감쇠최소자승(DLS / Levenberg–Marquardt): dq = J^T (JJ^T + λ²I)^{-1} e.
    λ가 특이점 근처에서 해를 유계로 유지한다. 순수 유사역행렬(pseudo-inverse)
    dq = J^T (JJ^T)^{-1} e 는 특이점에서 관절속도가 폭발한다.

  - 특이점(singularity): 팔을 곧게 편 자세에서 위치 야코비안이 계수(rank)를 잃어
    조작성 w=sqrt(det(J_p J_p^T)) → 0. 그 방향으로는 유사역행렬 IK가 발산하지만
    DLS는 유계로 남아 경계점에 안착한다.

  - 여유자유도(redundancy): 3자유도로 2D 위치를 맞추므로 1자유도가 남는다. 같은
    목표를 여러 자세로 도달할 수 있고, 영공간 사영 (I - J^+J) 으로 팔끝을 흔들지
    않으면서 2차목표(편안한 자세 유지·관절한계 회피)를 추구한다.

한계: 평면(2D) 단순화라 3D 자세·특이점 구조는 축약돼 있고(공간 팔은 DH로 확장),
반복 IK는 초기값에 따라 국소해/다른 분기로 수렴할 수 있다(엘보업/다운). 이는 실제
수술 팔에서도 자세 선택이 별도 문제임을 그대로 반영한다.

    python scripts/39_manipulator_kinematics.py
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

L = np.array([1.0, 0.8, 0.6])       # 링크 길이 [m]; 총 리치 = 2.4 m
REACH = float(L.sum())


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# 순기구학(FK): 링크를 차례로 합성. cumsum(q)가 각 링크의 절대 방향.
# --------------------------------------------------------------------------
def fk_points(q, L=L):
    """base(원점) 포함 모든 관절점 + 팔끝 좌표 (n+1, 2)."""
    ang = np.cumsum(q)
    pts = [np.zeros(2)]
    p = np.zeros(2)
    for li, a in zip(L, ang):
        p = p + li * np.array([np.cos(a), np.sin(a)])
        pts.append(p.copy())
    return np.array(pts)


def fk(q, L=L):
    """팔끝 포즈 [x, y, phi]."""
    pts = fk_points(q, L)
    phi = float(np.sum(q))
    return np.array([pts[-1, 0], pts[-1, 1], phi])


# --------------------------------------------------------------------------
# 기하 야코비안(3×3): 위치 2행은 FK의 해석적 편미분, 자세행은 [1,1,1].
#   ∂x/∂q_i = -(관절 i 이후 링크들의 y기여),  ∂y/∂q_i = +(x기여)
# --------------------------------------------------------------------------
def jacobian(q, L=L):
    ang = np.cumsum(q)
    n = len(q)
    # 링크 벡터별 (x,y) 기여
    dx = L * np.cos(ang)          # 각 링크의 x 성분
    dy = L * np.sin(ang)          # 각 링크의 y 성분
    J = np.zeros((3, n))
    for i in range(n):
        # 관절 i를 돌리면 링크 i..n-1(팔끝 포함)이 함께 회전
        J[0, i] = -np.sum(dy[i:])
        J[1, i] = np.sum(dx[i:])
        J[2, i] = 1.0
    return J


def jacobian_pos(q, L=L):
    return jacobian(q, L)[:2]


def manipulability(q, L=L):
    """Yoshikawa 조작성 w = sqrt(det(J_p J_p^T)) (위치 야코비안)."""
    Jp = jacobian_pos(q, L)
    m = Jp @ Jp.T
    d = np.linalg.det(m)
    return float(np.sqrt(max(d, 0.0)))


# --------------------------------------------------------------------------
# 역기구학(IK)
# --------------------------------------------------------------------------
def ik_dls(target_xy, q0, L=L, lam=0.10, max_iters=300, tol=1e-4,
           step_cap=0.4, secondary=None, k_sec=0.0):
    """감쇠최소자승(DLS) 위치 IK. secondary(q)를 영공간에 사영해 2차목표 추구."""
    q = np.array(q0, float).copy()
    traj = [q.copy()]
    lam2 = lam * lam
    use_sec = secondary is not None and k_sec != 0.0
    it = 0
    for it in range(1, max_iters + 1):
        e = np.asarray(target_xy, float) - fk(q, L)[:2]
        # 2차목표가 없으면 목표 도달 즉시 종료. 있으면 팔끝을 고정한 채 영공간에서
        # 계속 하강하므로, 팔끝오차·영공간 스텝이 모두 작아질 때까지 반복.
        if not use_sec and np.linalg.norm(e) < tol:
            break
        Jp = jacobian_pos(q, L)
        JJt = Jp @ Jp.T
        dq = Jp.T @ np.linalg.solve(JJt + lam2 * np.eye(2), e)
        if use_sec:
            Jpinv = Jp.T @ np.linalg.solve(JJt + lam2 * np.eye(2), np.eye(2))
            N = np.eye(len(q)) - Jpinv @ Jp        # 영공간 사영자
            dq_sec = N @ (k_sec * secondary(q))
            dq = dq + dq_sec
            if np.linalg.norm(e) < tol and np.linalg.norm(dq_sec) < tol:
                break
        # 스텝 캡: 큰 걸음의 과도 회전 방지(수렴 안정)
        nrm = np.linalg.norm(dq)
        if nrm > step_cap:
            dq = dq * (step_cap / nrm)
        q = q + dq
        traj.append(q.copy())
    err = float(np.linalg.norm(np.asarray(target_xy, float) - fk(q, L)[:2]))
    return q, it, np.array(traj), err


def ik_pseudo(target_xy, q0, L=L, max_iters=300, tol=1e-4):
    """순수 유사역행렬 IK(감쇠 없음). 특이점 근처에서 스텝이 폭발함을 보인다."""
    q = np.array(q0, float).copy()
    traj = [q.copy()]
    max_step = 0.0
    it = 0
    for it in range(1, max_iters + 1):
        e = np.asarray(target_xy, float) - fk(q, L)[:2]
        if np.linalg.norm(e) < tol:
            break
        Jp = jacobian_pos(q, L)
        JJt = Jp @ Jp.T
        try:
            dq = Jp.T @ np.linalg.solve(JJt, e)      # 감쇠 없음 → 병조건에서 폭발
        except np.linalg.LinAlgError:
            dq = np.full(len(q), np.inf)
        max_step = max(max_step, float(np.linalg.norm(dq)))
        if not np.all(np.isfinite(dq)):
            break
        q = q + dq
        traj.append(q.copy())
        if np.linalg.norm(dq) > 1e6:                 # 발산 판정
            break
    err = float(np.linalg.norm(np.asarray(target_xy, float) - fk(q, L)[:2]))
    return q, it, np.array(traj), err, max_step


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(7)

    # ---- (A) 도달 가능한 목표로 DLS IK 수렴 ----
    target = np.array([1.3, 1.1])
    q0 = np.array([0.3, 0.4, 0.2])
    q_sol, iters, traj, pos_err = ik_dls(target, q0, lam=0.05)
    w_sol = manipulability(q_sol)

    # ---- (B) 야코비안 유한차분 검증(정보 출력용) ----
    def num_jac(q, eps=1e-6):
        Jn = np.zeros((3, len(q)))
        for i in range(len(q)):
            dq = np.zeros(len(q)); dq[i] = eps
            f1 = fk(q + dq); f0 = fk(q - dq)
            col = (f1 - f0) / (2 * eps)
            col[2] = wrap(f1[2] - f0[2]) / (2 * eps)
            Jn[:, i] = col
        return Jn
    q_chk = np.array([0.4, -0.7, 0.9])
    jac_err = float(np.max(np.abs(jacobian(q_chk) - num_jac(q_chk))))

    # ---- (C) 특이점: 곧게 편 자세(q≈0)에서 조작성 → 0, 축밖 목표로 DLS vs pseudo ----
    q_sing = np.array([0.0, 0.0, 0.0])
    w_sing = manipulability(q_sing)                  # ≈ 0
    # 리치 경계 바로 밖·축에서 살짝 벗어난 목표(특이 방향을 강하게 요구)
    q_start = np.array([1e-3, 1e-3, 1e-3])
    sing_target = np.array([REACH - 0.02, 0.25])
    _, _, tr_dls, dls_err = ik_dls(sing_target, q_start, lam=0.15, max_iters=200)
    _, _, tr_ps, ps_err, ps_maxstep = ik_pseudo(sing_target, q_start, max_iters=200)
    dls_maxstep = float(np.max(np.linalg.norm(np.diff(tr_dls, axis=0), axis=1)))

    # ---- (D) 여유자유도: 같은 목표를 여러 자세로 도달 ----
    red_target = np.array([1.1, 0.6])
    postures = []
    seeds = [np.array([0.5, 0.6, 0.4]),
             np.array([-0.6, 1.4, 0.6]),
             np.array([1.1, -1.3, 0.7]),
             np.array([0.2, 1.9, -1.4])]
    for s in seeds:
        qs, _, _, es = ik_dls(red_target, s, lam=0.03, max_iters=400)
        if es < 1e-3:
            postures.append(qs)
    # 영공간 2차목표: 편안한 자세 q_home 근처 유지(팔끝 고정, 자세만 변형)
    q_home = np.array([0.6, 0.6, 0.6])
    comfort_seed = np.array([-0.6, 1.4, 0.6])
    qs_free, _, _, _ = ik_dls(red_target, comfort_seed, lam=0.03, max_iters=600)
    qs_home, _, _, _ = ik_dls(red_target, comfort_seed, lam=0.03, max_iters=600,
                              secondary=lambda q: (q_home - q), k_sec=0.6)
    self_motion = float(np.linalg.norm(qs_home - qs_free))

    # ---- 헤드라인 지표 ----
    print("=== 매니퓰레이터 기구학: 3R 평면 팔 (FK · J · IK · 특이점 · 여유자유도) ===")
    print(f"링크 길이 {L.tolist()} m,  총 리치 {REACH:.2f} m")
    print(f"[IK/DLS] 목표 {target.tolist()} → 위치오차 {pos_err*1e3:.3f} mm, "
          f"{iters} iters, 해에서 조작성 w={w_sol:.3f}")
    print(f"[야코비안] 해석 J vs 유한차분 최대오차 {jac_err:.2e} (일치)")
    print(f"[특이점] 곧게편 자세 조작성 w={w_sing:.2e} (rank 손실)")
    print(f"         축밖 경계 목표에서  DLS 최대스텝 {dls_maxstep:.3f} rad (유계, 잔차 {dls_err*1e3:.1f} mm)")
    print(f"                          pseudo 최대스텝 {ps_maxstep:.2e} rad (폭발), 잔차 {ps_err:.3f} m")
    print(f"[여유자유도] 같은 목표 {red_target.tolist()} 도달 자세 {len(postures)}종(엘보 분기)")
    print(f"            영공간 2차목표(편안한 자세) 전/후 ‖q-q_home‖ "
          f"{np.linalg.norm(qs_free-q_home):.2f} → {np.linalg.norm(qs_home-q_home):.2f}, "
          f"자기운동 {self_motion:.2f} rad, 팔끝 고정오차 "
          f"{np.linalg.norm(fk(qs_home)[:2]-red_target)*1e3:.2f} mm")

    # ================= 플롯 =================
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)

    # (1) DLS IK 수렴: 중간 자세 + 최종, 목표
    ax1 = fig.add_subplot(gs[0, 0])
    show = np.linspace(0, len(traj) - 1, 6).astype(int)
    for k, idx in enumerate(show):
        pts = fk_points(traj[idx])
        col = plt.cm.viridis(k / (len(show) - 1))
        ax1.plot(pts[:, 0], pts[:, 1], "-o", color=col, lw=2, ms=4,
                 alpha=0.55 if idx != show[-1] else 1.0,
                 label="start" if k == 0 else ("solved" if idx == show[-1] else None))
    ax1.plot(*target, "r*", ms=18, label="target")
    ax1.plot(0, 0, "ks", ms=8)
    ax1.add_patch(plt.Circle((0, 0), REACH, color="gray", ls=":", fill=False, alpha=0.5))
    ax1.set_aspect("equal"); ax1.grid(alpha=0.2); ax1.legend(fontsize=8, loc="lower left")
    ax1.set_title(f"(1) DLS inverse kinematics converges\n{iters} iters, err {pos_err*1e3:.2f} mm")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")

    # (2) 도달영역(workspace) + 조작성 색, 해에서 조작성 타원
    ax2 = fig.add_subplot(gs[0, 1])
    Q = rng.uniform(-np.pi, np.pi, size=(6000, 3))
    P = np.array([fk(qq)[:2] for qq in Q])
    W = np.array([manipulability(qq) for qq in Q])
    sc = ax2.scatter(P[:, 0], P[:, 1], c=W, s=4, cmap="plasma", alpha=0.6)
    plt.colorbar(sc, ax=ax2, label="manipulability w", fraction=0.046)
    # 해에서 조작성 타원: J_p J_p^T 의 고유구조
    Jp = jacobian_pos(q_sol)
    U, sv, _ = np.linalg.svd(Jp)
    ee = fk(q_sol)[:2]
    ell = Ellipse(ee, 2 * sv[0] * 0.3, 2 * sv[1] * 0.3,
                  angle=np.degrees(np.arctan2(U[1, 0], U[0, 0])),
                  fc="none", ec="cyan", lw=2.2)
    ax2.add_patch(ell)
    pts = fk_points(q_sol)
    ax2.plot(pts[:, 0], pts[:, 1], "-o", color="white", lw=1.8, ms=3)
    ax2.set_aspect("equal")
    ax2.set_title("(2) Reachable workspace + manipulability\n(dark = near-singular: folded core / stretched rim; ellipse = velocity gain)")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")

    # (3) 특이점 근처: DLS(유계) vs pseudo(폭발) 스텝 크기
    ax3 = fig.add_subplot(gs[1, 0])
    dls_steps = np.linalg.norm(np.diff(tr_dls, axis=0), axis=1)
    ps_steps = np.linalg.norm(np.diff(tr_ps, axis=0), axis=1)
    ax3.semilogy(np.arange(1, len(dls_steps) + 1), dls_steps + 1e-12,
                 color="#1f77b4", lw=2, label=f"DLS (λ=0.15), bounded")
    ax3.semilogy(np.arange(1, len(ps_steps) + 1), ps_steps + 1e-12,
                 color="#d9534f", lw=2, marker="o", ms=3,
                 label="pseudo-inverse (undamped), blows up")
    ax3.set_title("(3) Near singularity: DLS stays bounded, pseudo-inverse explodes")
    ax3.set_xlabel("IK iteration"); ax3.set_ylabel("joint step ‖dq‖ [rad] (log)")
    ax3.grid(alpha=0.2, which="both"); ax3.legend(fontsize=8, loc="upper right")

    # (4) 여유자유도: 같은 목표, 여러 자세 + 영공간 2차목표
    ax4 = fig.add_subplot(gs[1, 1])
    cols = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    for k, qp in enumerate(postures):
        pts = fk_points(qp)
        ax4.plot(pts[:, 0], pts[:, 1], "-o", color=cols[k % len(cols)], lw=2, ms=4,
                 alpha=0.85, label=f"posture {k+1}")
    pts_h = fk_points(qs_home)
    ax4.plot(pts_h[:, 0], pts_h[:, 1], "--s", color="black", lw=1.6, ms=4,
             alpha=0.9, label="null-space → comfort")
    ax4.plot(*red_target, "r*", ms=18, label="one target")
    ax4.plot(0, 0, "ks", ms=8)
    ax4.set_aspect("equal"); ax4.grid(alpha=0.2); ax4.legend(fontsize=8, loc="lower left")
    ax4.set_title("(4) Redundancy (1 extra DOF):\nsame target, many postures + null-space secondary goal")
    ax4.set_xlabel("x [m]"); ax4.set_ylabel("y [m]")

    fig.suptitle("Manipulator kinematics on a 3R planar arm — the foundation for surgical/medical arms",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for p in ("outputs/39_manipulator_kinematics.png", "assets/39_manipulator_kinematics.png"):
        fig.savefig(p, dpi=125)
    print("\n[plot] outputs/39_manipulator_kinematics.png, assets/39_manipulator_kinematics.png")

    return pos_err, iters, w_sol, dls_err


if __name__ == "__main__":
    main()
