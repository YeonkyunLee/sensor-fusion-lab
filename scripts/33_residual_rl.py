"""잔차 강화학습(residual RL): 안전한 고전 제어기 위에 '보정만' 학습한다.

학습 정책을 현장에 그대로 올리기 어려운 이유는 안전성과 표본효율 때문이다. 처음부터
(from scratch) 신경망/무모델 정책으로 전부 배우게 하면, 학습 초기에 발산·전도 같은
위험한 행동을 거치고, 원하는 성능에 도달하기까지 막대한 상호작용이 필요하다. 반면
현장에는 이미 해석적으로 유도한 '충분히 안전하고 해석 가능한' 고전 제어기(LQR,
극배치, PD 등)가 있다. 문제는 이 제어기가 명목 모델에만 맞춰져 있어, 모델이 모르는
효과(정상상태 외란, 비선형 마찰, 질량 편차)가 실제 플랜트에 있으면 정상상태 오차나
성능 저하를 남긴다는 점이다.

잔차 RL은 이 둘을 결합한다: 제어입력을 u = u_base(state) + u_residual(state)로 쓰고,
고전 제어기 u_base는 그대로 두어 안전·안정성의 뼈대를 담당하며, 학습은 오직 작은
보정항 u_residual만 담당한다. 보정항은 0 근처에서 출발하므로(초기엔 base 그대로라
안전) 탐색 공간이 작고, base가 이미 대부분을 처리하니 '모델이 틀린 부분'만 메우면
된다 → 표본효율이 높고 학습 중에도 안전하다. 실제 로봇·자율주행에 배치되는 하이브리드
구조가 바로 이것이다.

이 실험은 그 이점을 숫자로 보인다. 플랜트는 카트-폴(도립진자)이며 ODE를 직접 구현하고
세미-임플리싯 오일러로 적분한다. 단, 실제 플랜트에는 base 제어기가 모르는 '정상상태
외란력'(미세한 바람/구동 바이어스)이 상수로 걸린다. base 제어기는 명목 선형화에 대한
LQR(연속시간 리카티 해)로, 명목 플랜트는 잘 안정화하지만 적분 작용이 없어 외란 아래에서
카트 위치에 정상상태 오차를 남긴다. 세 제어기를 '외란이 있는 실제 플랜트'에서 비교한다:
  (1) base 단독            — 고전 LQR. 폴은 세우지만 카트가 밀려 정상상태 오차.
  (2) from-scratch 정책    — base 없이 CEM으로 전부 학습. 동일 예산에서 느리고 불안정.
  (3) base + residual(하이브리드) — CEM으로 작은 보정만 학습. 정상상태 오차를 제거하고
      동일 예산에서 가장 빠르고 안정적으로 최고 성능에 도달.

미분 불가·시뮬레이터 기반 목적함수라 교차엔트로피법(CEM)으로 학습한다(torch/gym 없이
numpy만). 여러 시드에서 반복해 학습곡선의 신뢰도(시드 간 분산)까지 드러낸다.

정직한 한계: 잔차 RL은 '쓸 만한 base 제어기가 이미 존재'한다는 전제에 기댄다. 또한
from-scratch도 예산을 크게 늘리면 결국 따라잡을 수 있다 — 요점은 불가능성이 아니라
동일 예산에서의 표본효율과 학습 중 안전성이다.

    python scripts/33_residual_rl.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 물리 상수 / 과제 설정
# --------------------------------------------------------------------------
G = 9.8                 # 중력 [m/s^2]
DT = 0.02               # 적분 스텝 [s]
MAX_STEPS = 200         # 에피소드 길이 (= 4초)
THETA_LIM = 0.35        # 전도 판정 각도 [rad]
X_LIM = 2.4             # 트랙 이탈 판정 [m]
F_MAX = 15.0            # 총 구동력 포화 [N]
RES_MAX = 6.0           # 잔차 보정의 포화 [N] (base 대비 '작은' 보정임을 강제)

# 명목(모델) 세계: base 제어기가 아는 파라미터
M_NOM, m_NOM, l_NOM = 1.0, 0.1, 0.5

# 실제 플랜트의 '모델이 모르는' 정상상태 외란력 [N] (상수 바이어스/미풍)
F_DIST = 2.6

# 정상상태 오차를 평가할 후반 구간 비율
SS_TAIL = 0.25

# 학습·평가에 공통으로 쓰는 고정 초기상태 집합 (공통난수 → 깨끗한 비교)
EVAL_INITS = np.array([
    [0.00, 0.0,  0.10, 0.0],
    [0.00, 0.0, -0.08, 0.0],
    [0.30, 0.0,  0.05, 0.0],
])

# 상태-비용 가중 (regulation): 원점(x=0, 수직)으로 되돌리는 비용
Q_DIAG = np.array([1.0, 0.05, 5.0, 0.05])
FALL_PEN = 60.0         # 전도/이탈 시 남은 스텝당 페널티 (실패를 큰 비용으로)

# CEM 예산 (from-scratch와 residual이 '동일')
CEM_ITERS = 12
CEM_POP = 40
CEM_ELITE = 8
SEEDS = (0, 1, 2, 3)    # 시드 간 신뢰도 측정

# residual 정책: 작은 보정 → 좁은 초기 탐색분포, 평균 0(=시작은 base 그대로)
RES_STD0 = np.array([2.5, 1.5, 1.0, 4.0, 1.0])   # 특징 [1, x, xd, th, thd]
# from-scratch 정책: 전부 학습 → 넓은 초기 탐색분포(안정화 이득 규모를 덮어야 함)
SCR_STD0 = np.array([3.0, 4.0, 5.0, 30.0, 6.0])


# --------------------------------------------------------------------------
# 카트-폴 동역학 (Barto/Sutton 도립진자, 세미-임플리싯 오일러)
#   state = [x, x_dot, theta, theta_dot], theta=0 이 수직 상방
# --------------------------------------------------------------------------
def cartpole_step(s, force, M=M_NOM, m=m_NOM, l=l_NOM):
    x, xd, th, thd = s
    st, ct = np.sin(th), np.cos(th)
    tot = M + m
    temp = (force + m * l * thd * thd * st) / tot
    thdd = (G * st - ct * temp) / (l * (4.0 / 3.0 - m * ct * ct / tot))
    xdd = temp - m * l * thdd * ct / tot
    xd = xd + DT * xdd
    thd = thd + DT * thdd
    x = x + DT * xd
    th = th + DT * thd
    return np.array([x, xd, th, thd])


# --------------------------------------------------------------------------
# base 제어기: 명목 선형화에 대한 LQR (연속시간 리카티 해)
#   상태 [x, xd, th, thd], u = -K s.  명목 플랜트는 잘 안정화하지만
#   F_DIST(정상상태 외란)를 모르므로 적분작용이 없어 정상상태 오차를 남긴다.
# --------------------------------------------------------------------------
def lqr_gain():
    tot = M_NOM + m_NOM
    D = l_NOM * (4.0 / 3.0 - m_NOM / tot)      # 폴 관성 항
    # 명목 선형화 (th=0 근방): thdd = (g*th - F/tot)/D
    dthdd_dth = G / D
    dthdd_dF = -1.0 / (tot * D)
    dxdd_dth = -(m_NOM * l_NOM / tot) * (G / D)
    dxdd_dF = 1.0 / tot + (m_NOM * l_NOM / tot) * (1.0 / (tot * D))
    A = np.array([
        [0.0, 1.0, 0.0,       0.0],
        [0.0, 0.0, dxdd_dth,  0.0],
        [0.0, 0.0, 0.0,       1.0],
        [0.0, 0.0, dthdd_dth, 0.0],
    ])
    B = np.array([[0.0], [dxdd_dF], [0.0], [dthdd_dF]])
    Q = np.diag(Q_DIAG)
    R = np.array([[0.1]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)            # 1x4
    return K[0]


K_LQR = lqr_gain()


def base_action(s):
    return float(np.clip(-K_LQR @ s, -F_MAX, F_MAX))


# --------------------------------------------------------------------------
# 정책 특징 & 세 종류의 제어기(행동함수) 빌더
#   특징 phi(s) = [1, x, xd, th, thd]  (상수항 = 외란 상쇄용 바이어스)
# --------------------------------------------------------------------------
def phi(s):
    return np.array([1.0, s[0], s[1], s[2], s[3]])


def make_base():
    return base_action


def make_residual(w):
    def act(s):
        res = float(np.clip(w @ phi(s), -RES_MAX, RES_MAX))
        return float(np.clip(base_action(s) + res, -F_MAX, F_MAX))
    return act


def make_scratch(w):
    def act(s):
        return float(np.clip(w @ phi(s), -F_MAX, F_MAX))
    return act


# --------------------------------------------------------------------------
# 실제 플랜트 롤아웃: base가 모르는 상수 외란 F_DIST를 더해 적분
#   regulation 비용(원점 복귀) 합. 전도/이탈 시 남은 스텝을 크게 페널티.
# --------------------------------------------------------------------------
def _sim(act, s0):
    """한 에피소드. (cost, fell, xs, ths) 반환. fell=전도/이탈 여부."""
    s = np.array(s0, dtype=float)
    cost = 0.0
    fell = False
    xs, ths = [s[0]], [s[2]]
    for t in range(MAX_STEPS):
        u = act(s)
        s = cartpole_step(s, u + F_DIST)       # ← 모델이 모르는 외란
        xs.append(s[0]); ths.append(s[2])
        if abs(s[2]) > THETA_LIM or abs(s[0]) > X_LIM:
            cost += FALL_PEN * (MAX_STEPS - t)
            fell = True
            break
        cost += float(Q_DIAG @ (s * s))
    return cost, fell, np.array(xs), np.array(ths)


def rollout(act, s0, record=False):
    cost, _, xs, ths = _sim(act, s0)
    if record:
        return cost, xs, ths
    return cost


def mean_cost(act):
    """고정 초기상태 집합 평균 비용 (공통난수)."""
    return float(np.mean([_sim(act, s0)[0] for s0 in EVAL_INITS]))


def eval_cost_falls(act):
    """평균 비용과 전도한 롤아웃 수 (학습 중 안전성 집계용)."""
    tot_cost, falls = 0.0, 0
    for s0 in EVAL_INITS:
        c, fell, _, _ = _sim(act, s0)
        tot_cost += c
        falls += int(fell)
    return tot_cost / len(EVAL_INITS), falls


def steady_state_error(act):
    """대표 초기상태에서 후반 구간 평균 |x| (정상상태 카트 오차)."""
    _, xs, _ = rollout(act, EVAL_INITS[0], record=True)
    tail = max(1, int(len(xs) * SS_TAIL))
    return float(np.mean(np.abs(xs[-tail:])))


# --------------------------------------------------------------------------
# 교차엔트로피법(CEM): 비용 최소화. iter별 best-so-far 곡선 기록.
# --------------------------------------------------------------------------
def cem_minimize(make_act, std0, seed, iters=CEM_ITERS, pop=CEM_POP, elite=CEM_ELITE):
    rng = np.random.default_rng(seed)
    dim = std0.size
    mean = np.zeros(dim)                        # 평균 0 = residual은 base 그대로에서 출발
    std = std0.copy()
    best_w, best_c = mean.copy(), mean_cost(make_act(mean))
    curve = []
    total_roll, total_falls = 0, 0              # 학습 중 안전성 집계
    for _ in range(iters):
        W = rng.normal(mean, std, size=(pop, dim))
        W[0] = mean                             # 엘리트 유지(현 평균 포함)
        scores = np.empty(pop)
        for i in range(pop):
            scores[i], f = eval_cost_falls(make_act(W[i]))
            total_roll += len(EVAL_INITS)
            total_falls += f
        idx = np.argsort(scores)[:elite]        # 낮은 비용이 엘리트
        mean = W[idx].mean(0)
        std = W[idx].std(0) + 1e-3
        if scores[idx[0]] < best_c:
            best_c = float(scores[idx[0]])
            best_w = W[idx[0]].copy()
        curve.append(best_c)                    # best-so-far (동일 예산 비교)
    fall_frac = total_falls / max(total_roll, 1)
    return best_w, np.array(curve), fall_frac


# --------------------------------------------------------------------------
def main():
    base_cost = mean_cost(make_base())
    base_sse = steady_state_error(make_base())

    # 여러 시드에서 residual / scratch 학습 (동일 CEM 예산)
    res_curves, scr_curves = [], []
    res_final, scr_final = [], []
    res_falls, scr_falls = [], []
    res_w0, scr_w0 = None, None
    for sd in SEEDS:
        rw, rc, rf = cem_minimize(make_residual, RES_STD0, seed=sd)
        sw, sc, sf = cem_minimize(make_scratch, SCR_STD0, seed=sd)
        res_curves.append(rc); scr_curves.append(sc)
        res_final.append(rc[-1]); scr_final.append(sc[-1])
        res_falls.append(rf); scr_falls.append(sf)
        if sd == SEEDS[0]:
            res_w0, scr_w0 = rw, sw

    res_curves = np.array(res_curves)           # (S, iters)
    scr_curves = np.array(scr_curves)
    residual_cost = float(np.mean(res_final))
    scratch_cost = float(np.mean(scr_final))
    residual_std = float(np.std(res_final))
    scratch_std = float(np.std(scr_final))
    residual_fall = float(np.mean(res_falls))   # 학습 중 전도율(안전성)
    scratch_fall = float(np.mean(scr_falls))

    res_sse = steady_state_error(make_residual(res_w0))

    print("=== Residual RL: 안전한 고전 base + 학습된 보정 (cart-pole, 정상상태 외란) ===")
    print(f"base = LQR(연속 리카티),  K = {np.round(K_LQR, 2)}")
    print(f"실제 플랜트 외란 F_dist = {F_DIST} N (base가 모름),  잔차 포화 ±{RES_MAX} N")
    print(f"CEM 예산(동일): iters={CEM_ITERS}, pop={CEM_POP}, elite={CEM_ELITE}, seeds={list(SEEDS)}")
    print()
    print(f"[실제 플랜트 regulation 비용 ↓]")
    print(f"  (1) base 단독              {base_cost:9.1f}")
    print(f"  (2) from-scratch (동일예산) {scratch_cost:9.1f}  ± {scratch_std:.1f} (시드간)")
    print(f"  (3) base + residual        {residual_cost:9.1f}  ± {residual_std:.1f} (시드간)")
    print(f"  → residual은 base 대비 {base_cost / max(residual_cost,1e-9):.1f}× 낮고, "
          f"동일 예산 scratch 대비 {scratch_cost / max(residual_cost,1e-9):.1f}× 낮음")
    print()
    print(f"[정상상태 카트 오차 |x| (m) ↓]  base {base_sse:.3f}  →  base+residual {res_sse:.3f}")
    print(f"  → residual 보정이 외란을 상쇄해 정상상태 오차 {base_sse/max(res_sse,1e-9):.1f}× 감소")
    print()
    print(f"[학습 중 안전성: 롤아웃 전도율 ↓]  residual {residual_fall:.1%}  |  scratch {scratch_fall:.1%}")
    print(f"  → residual은 base가 이미 안정화하므로 탐색 내내 거의 넘어지지 않음; "
          f"scratch는 안정화 이득을 찾느라 위험 영역을 헤맴")
    # 정직: from-scratch는 예산을 늘리면 따라잡을 수 있음 — 요점은 표본효율/안전
    hit = np.argmax(res_curves.mean(0) <= residual_cost * 1.1)
    print(f"  (정직: residual 평균곡선은 ~{hit+1} iter만에 최종의 1.1배 이내 도달; "
          f"from-scratch는 동일예산 내 더 느리고 시드간 분산 크며, 예산을 크게 늘리면 결국 따라잡음)")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9], wspace=0.30)

    # (a) 카트 위치 궤적: base(정상상태 오차) vs base+residual(원점 복귀)
    ax0 = fig.add_subplot(gs[0, 0])
    _, xb, thb = rollout(make_base(), EVAL_INITS[0], record=True)
    _, xr, thr = rollout(make_residual(res_w0), EVAL_INITS[0], record=True)
    tb = np.arange(len(xb)) * DT
    tr = np.arange(len(xr)) * DT
    ax0.axhline(0.0, color="k", lw=0.8, ls=":")
    ax0.plot(tb, xb, color="#d9534f", lw=2, label="base only (LQR)")
    ax0.plot(tr, xr, color="#1f77b4", lw=2, label="base + residual")
    x_ss = float(xb[-1])                        # base의 (부호 있는) 정상상태 위치
    ax0.axhline(x_ss, color="#d9534f", lw=1, ls="--", alpha=0.6)
    ax0.annotate(f"steady-state error {base_sse:.2f} m",
                 xy=(tb[-1], x_ss), xytext=(tb[-1] * 0.28, x_ss + 0.06),
                 fontsize=8, color="#a02622")
    ax0.set_xlabel("time [s]")
    ax0.set_ylabel("cart position x [m]  (target 0)")
    ax0.set_title("Residual cancels the unmodeled disturbance\n"
                  "(base drifts; base+residual returns to 0)", fontsize=10)
    ax0.legend(fontsize=8, loc="lower right")
    ax0.grid(alpha=0.25)

    # (b) 학습곡선: cost vs CEM iter, scratch vs residual (시드 밴드)
    ax1 = fig.add_subplot(gs[0, 1])
    it = np.arange(1, CEM_ITERS + 1)
    for curves, col, lab in [(scr_curves, "#d9534f", "from-scratch"),
                             (res_curves, "#1f77b4", "base + residual")]:
        mu = curves.mean(0)
        ax1.plot(it, mu, "o-", color=col, lw=2, ms=4, label=lab)
        ax1.fill_between(it, curves.min(0), curves.max(0), color=col, alpha=0.15)
    ax1.axhline(base_cost, color="k", ls="--", lw=1, alpha=0.7)
    ax1.text(CEM_ITERS * 0.5, base_cost * 1.05, "base-only cost", fontsize=7.5)
    ax1.text(0.03, 0.03,
             f"training falls:  residual {residual_fall:.0%}  vs  scratch {scratch_fall:.0%}",
             transform=ax1.transAxes, fontsize=7.5, color="#333",
             bbox=dict(fc="white", ec="#ccc", alpha=0.8))
    ax1.set_yscale("log")
    ax1.set_xlabel("CEM iteration (equal budget)")
    ax1.set_ylabel("best-so-far regulation cost ↓ (log)")
    ax1.set_title("Same budget: residual learns faster & more reliably\n"
                  "(band = min–max across seeds)", fontsize=10)
    ax1.set_xticks(it)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.25, which="both")

    # (c) 최종 비용 막대: base / scratch / residual (동일 예산)
    ax2 = fig.add_subplot(gs[0, 2])
    labels = ["base\nonly", "from-\nscratch", "base+\nresidual"]
    vals = [base_cost, scratch_cost, residual_cost]
    errs = [0, scratch_std, residual_std]
    cols = ["#7f7f7f", "#d9534f", "#1f77b4"]
    ax2.bar(labels, vals, yerr=errs, color=cols, capsize=4, alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("final regulation cost ↓ (log)")
    ax2.set_title("Hybrid wins at equal budget", fontsize=10)
    for i, v in enumerate(vals):
        ax2.text(i, v * 1.12, f"{v:.0f}", ha="center", fontsize=8)
    ax2.grid(alpha=0.25, axis="y", which="both")

    fig.suptitle("Residual RL: keep the safe classical controller, learn only the "
                 "correction it gets wrong", fontsize=12, y=1.03)
    for p in ("outputs/33_residual_rl.png", "assets/33_residual_rl.png"):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    print("\n[plot] outputs/33_residual_rl.png, assets/33_residual_rl.png")

    return (base_cost, scratch_cost, residual_cost, base_sse, res_sse,
            scratch_fall, residual_fall)


if __name__ == "__main__":
    main()
