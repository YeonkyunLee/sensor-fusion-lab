"""모델기반 강화학습(MBRL): 데이터로 동역학 모델을 배우고, 그 모델 위에서 MPC로 계획한다.

무모델(model-free) 정책탐색(exp30~33의 CEM)은 시뮬레이터/실환경을 블랙박스로 두고
'행동→보상'만 보고 정책을 직접 흔들어 찾는다. 미분 불가 목적함수에 강건하지만, 매 후보
정책을 실제 롤아웃으로 평가해야 하므로 표본효율이 낮다 — 좋은 제어에 도달하기까지 막대한
환경 상호작용(transition)이 든다. 실제 로봇에서 상호작용은 비싸고 위험하다.

모델기반 RL은 발상을 뒤집는다. 환경에서 모은 전이 (s, a) -> s' 로 '동역학 모델' f_hat 을
먼저 지도학습으로 적합한 뒤, 학습된 모델 안에서 마음껏(=공짜로) 계획한다. 계획은 실환경을
건드리지 않으므로, 환경 상호작용 예산은 오직 '모델을 배우는 데'만 쓰인다. 이렇게 모은 소량의
데이터로 얻은 모델을 receding-horizon MPC(무작위 슈팅/CEM으로 행동열 탐색)에 넣어 첫 행동만
적용하고 매 스텝 재계획한다. 이것이 PETS·Dreamer 같은 현대 MBRL의 뼈대인 'learn a model,
then plan' 패러다임이다(여기서는 정직한 장난감 규모).

이 실험은 그 표본효율을 숫자로 보인다. 플랜트는 카트-폴(도립진자)이며 ODE를 직접 구현하고
세미-임플리싯 오일러로 적분한다. 에이전트는 진짜 동역학을 모른다. 모델 f_hat 은 랜덤푸리에
특징(random Fourier features) + 능형회귀(ridge)로 상태증분 s'-s 를 예측하는 numpy 회귀기다
(torch/gym 없이 numpy만). 데이터는 무작위 행동 롤아웃으로 초기수집하고, 이후 현재 모델의 MPC로
온-폴리시 전이를 덧붙여 재적합한다(Dyna/PETS식 데이터 병합). 세 방법을 '환경 전이 수'의 함수로
비교한다:
  (1) MBRL           — N개 전이로 모델을 배우고 그 모델로 MPC 계획. 적은 전이로 좋은 제어.
  (2) 무모델 baseline — 동일한 환경-상호작용 예산의 CEM 정책탐색(exp30식). 같은 성능에
                        훨씬 많은 전이가 필요.
  (3) 오라클 MPC     — 진짜 모델로 MPC. MBRL이 다가가야 할 상한선.

정직한 결과: MBRL은 무모델이 필요로 하는 것보다 훨씬 적은 전이로 오라클에 근접한다. 다만 학습된
모델에는 편향(model bias)이 있고 예측오차가 지평을 따라 누적(compounding error)되므로, MBRL은
오라클에 '근접'하되 완전히 도달하지 못하고 그 아래에서 정체한다. 또한 무모델도 예산을 크게
늘리면 결국 따라잡는다 — 요점은 불가능성이 아니라 동일 전이예산에서의 표본효율이다.

    python scripts/38_model_based_rl.py
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 물리 상수 / 과제 설정
# --------------------------------------------------------------------------
G = 9.8                 # 중력 [m/s^2]
DT = 0.02               # 적분 스텝 [s]
MAX_STEPS = 120         # 에피소드 길이 (= 2.4초)
THETA_LIM = 0.35        # 전도 판정 각도 [rad]
X_LIM = 2.4             # 트랙 이탈 판정 [m]
F_MAX = 15.0            # 구동력 포화 [N]

# 진짜(에이전트가 모르는) 플랜트 파라미터: 카트질량 M, 폴질량 m, 폴 반길이 l
M_TRUE, m_TRUE, l_TRUE = 1.0, 0.1, 0.5

# 성능(return) 스케일: r_t = exp(-((th/TH_S)^2 + (x/X_S)^2)) ∈ (0,1]
TH_S = 0.20             # 각도 스케일 [rad]
X_S = 0.80              # 위치 스케일 [m]

# 평가·수집 공통 초기상태 (전부 회복 가능한 기울기)
EVAL_INITS = np.array([
    [0.00, 0.0,  0.12, 0.0],
    [0.00, 0.0, -0.10, 0.0],
    [0.20, 0.0,  0.06, 0.0],
])

# MPC(무작위 슈팅+CEM 정제) 하이퍼파라미터 — 학습모델/오라클 공용 플래너
MPC_H = 25              # 예측 지평(스텝)
MPC_POP = 64            # 후보 행동열 수 (배치된 MBRL 플래너)
MPC_ITERS = 3           # CEM 정제 반복
MPC_ELITE = 10
# 오라클은 '진짜 동역학을 알고 + 충분히 세게 계획'하는 상한 → 더 강한 플래너
ORACLE_POP = 150
ORACLE_ITERS = 5
ORACLE_ELITE = 20
STAGE_WA = 0.0005       # 제어노력 가중
FALL_C = 6.0            # 모델 롤아웃 중 한계 이탈 시 스텝당 페널티
TERMINAL_W = 6.0        # 종단(terminal) 비용 가중 → farsighted MPC. 학습모델에선
#                        이 종단 스텝의 예측오차(누적)가 커 model bias 가 계획에 주입됨

# 랜덤푸리에 특징(RFF) 회귀 모델
RFF_D = 160             # 특징 수
RFF_LEN = 0.75          # 대역폭(표준화 입력 기준)
RFF_SEED = 7
RIDGE_LAM = 1e-3

# 환경-전이 예산 체크포인트 (MBRL 학습곡선 x축)
BUDGETS = [30, 60, 120, 240, 480]
COMPARE_BUDGET = 480    # MBRL vs 무모델 '동일 소예산' 비교 지점
SOLVE_FRAC = 0.90       # 오라클의 이 비율 이상이면 'solved'

# 무모델 CEM 정책탐색(baseline) — 동일 return 을 환경 롤아웃으로 직접 최적화
MF_POP = 30
MF_ELITE = 7
MF_STD0 = np.array([3.0, 3.0, 25.0, 5.0])
MF_BUDGET_MAX = 60000   # 무모델은 오라클 근접까지 이만큼(정직한 점근선)

SEED = 0

# RFF 파라미터는 한 번만 생성해 모든 모델이 공유(결정론적)
_rff_rng = np.random.default_rng(RFF_SEED)
RFF_OMEGA = _rff_rng.normal(0.0, 1.0, size=(5, RFF_D)) * RFF_LEN
RFF_B = _rff_rng.uniform(0.0, 2 * np.pi, size=RFF_D)

# 학습모델 예측상태를 물리적으로 타당한 범위로 클립(외삽 발산 방지)
_S_LO = np.array([-5.0, -15.0, -1.2, -15.0])
_S_HI = np.array([5.0, 15.0, 1.2, 15.0])


# --------------------------------------------------------------------------
# 카트-폴 동역학 (Barto/Sutton 도립진자, 세미-임플리싯 오일러)
#   state = [x, x_dot, theta, theta_dot], theta=0 이 수직 상방
#   배치판(S:(P,4), a:(P,)) — 진짜 플랜트이자 오라클 MPC의 모델
# --------------------------------------------------------------------------
def cartpole_step_batch(S, a, M=M_TRUE, m=m_TRUE, l=l_TRUE):
    x, xd, th, thd = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    st, ct = np.sin(th), np.cos(th)
    tot = M + m
    temp = (a + m * l * thd * thd * st) / tot
    thdd = (G * st - ct * temp) / (l * (4.0 / 3.0 - m * ct * ct / tot))
    xdd = temp - m * l * thdd * ct / tot
    xd = xd + DT * xdd
    thd = thd + DT * thdd
    x = x + DT * xd
    th = th + DT * thd
    return np.stack([x, xd, th, thd], axis=1)


def true_step(s, a):
    """진짜 플랜트 1스텝(스칼라) — 에이전트가 환경과 상호작용할 때 사용."""
    return cartpole_step_batch(s[None, :], np.array([a]))[0]


# --------------------------------------------------------------------------
# 학습 동역학 모델: RFF + ridge 로 상태증분 s'-s 를 예측 (numpy 적합)
#   특징 phi(z) = [1, z_norm, sqrt(2/D)cos(z_norm·Omega + b)],  z = [s, a]
# --------------------------------------------------------------------------
class DynamicsModel:
    def __init__(self, mean, std, Wout):
        self.mean = mean
        self.std = std
        self.Wout = Wout

    def _phi(self, Z):
        Zn = (Z - self.mean) / self.std
        rff = np.sqrt(2.0 / RFF_D) * np.cos(Zn @ RFF_OMEGA + RFF_B)
        ones = np.ones((Z.shape[0], 1))
        return np.concatenate([ones, Zn, rff], axis=1)

    def predict_delta(self, S, a):
        Z = np.concatenate([S, a[:, None]], axis=1)
        return self._phi(Z) @ self.Wout

    def step_batch(self, S, a):
        Snext = S + self.predict_delta(S, a)
        return np.clip(Snext, _S_LO, _S_HI)


def fit_model(Z, dS):
    """전이 (Z=[s,a] -> dS=s'-s) 로 RFF-ridge 모델 적합."""
    mean = Z.mean(0)
    std = Z.std(0) + 1e-6
    Zn = (Z - mean) / std
    rff = np.sqrt(2.0 / RFF_D) * np.cos(Zn @ RFF_OMEGA + RFF_B)
    Phi = np.concatenate([np.ones((Z.shape[0], 1)), Zn, rff], axis=1)
    A = Phi.T @ Phi + RIDGE_LAM * np.eye(Phi.shape[1])
    Wout = np.linalg.solve(A, Phi.T @ dS)
    return DynamicsModel(mean, std, Wout)


# --------------------------------------------------------------------------
# MPC: 무작위 슈팅 + CEM 정제로 H스텝 행동열을 model 위에서 최적화
#   stage cost = (th/TH_S)^2 + (x/X_S)^2 + 0.1(xd^2+thd^2) + WA a^2, 이탈 시 큰 페널티
#   첫 행동만 적용, 매 스텝 warm-start 로 재계획(receding horizon)
# --------------------------------------------------------------------------
def _rollout_cost(step_batch, s, A):
    """행동열 배치 A:(pop,H) 를 model(step_batch)로 굴려 누적비용 (pop,) 반환."""
    pop = A.shape[0]
    S = np.tile(s, (pop, 1)).astype(float)
    cost = np.zeros(pop)
    H = A.shape[1]
    for k in range(H):
        a = A[:, k]
        S = step_batch(S, a)
        th, x, xd, thd = S[:, 2], S[:, 0], S[:, 1], S[:, 3]
        fell = (np.abs(th) > THETA_LIM) | (np.abs(x) > X_LIM)
        stepc = ((th / TH_S) ** 2 + (x / X_S) ** 2
                 + 0.02 * xd * xd + 0.02 * thd * thd
                 + STAGE_WA * a * a + FALL_C * fell)
        cost += (TERMINAL_W if k == H - 1 else 1.0) * stepc
    return cost


def plan_action(step_batch, s, warm, rng, pop, iters, elite):
    """model 로 CEM 계획 → (첫 행동, 시프트한 warm-start 평균)."""
    mu = warm.copy()
    sig = np.full(MPC_H, 5.0)
    elite_mean = mu
    for _ in range(iters):
        A = np.clip(mu + sig * rng.standard_normal((pop, MPC_H)),
                    -F_MAX, F_MAX)
        A[0] = mu                                   # 현 평균 유지(엘리트 후보)
        cost = _rollout_cost(step_batch, s, A)
        idx = np.argsort(cost)[:elite]
        elite_mean = A[idx].mean(0)
        mu = elite_mean
        sig = A[idx].std(0) + 0.5
    a0 = float(np.clip(elite_mean[0], -F_MAX, F_MAX))
    warm_next = np.concatenate([elite_mean[1:], elite_mean[-1:]])
    return a0, warm_next


def make_mpc_controller(step_batch, seed=0,
                        pop=MPC_POP, iters=MPC_ITERS, elite=MPC_ELITE):
    rng = np.random.default_rng(seed)

    def controller(s, warm):
        return plan_action(step_batch, s, warm, rng, pop, iters, elite)
    return controller


# --------------------------------------------------------------------------
# 성능(return): 진짜 플랜트에서 MPC/정책을 굴려 평균 보상
#   r_t = exp(-stage_cost) — MPC 가 최소화하는 stage cost 와 '동일'하게 맞춰
#   오라클(진짜 모델 MPC)이 이 return 을 실제로 최대화하는 상한이 되도록 한다.
#   이탈 시 truncate(남은 스텝은 r=0). return ∈ [0,1].
# --------------------------------------------------------------------------
def stage_cost(s, a):
    return ((s[2] / TH_S) ** 2 + (s[0] / X_S) ** 2
            + 0.02 * s[1] ** 2 + 0.02 * s[3] ** 2 + STAGE_WA * a ** 2)


def rollout_return(controller, s0, count_steps=False):
    s = np.array(s0, dtype=float)
    warm = np.zeros(MPC_H)
    R = 0.0
    used = 0
    for _ in range(MAX_STEPS):
        a, warm = controller(s, warm)
        s = true_step(s, a)
        used += 1
        if abs(s[2]) > THETA_LIM or abs(s[0]) > X_LIM:
            break
        R += float(np.exp(-stage_cost(s, a)))
    ret = R / MAX_STEPS
    return (ret, used) if count_steps else ret


def eval_return(controller, inits=EVAL_INITS):
    return float(np.mean([rollout_return(controller, s0) for s0 in inits]))


# --------------------------------------------------------------------------
# 데이터 수집: 무작위 행동(초기) + 온-폴리시 MPC(이후). (s,a) -> (s'-s)
# --------------------------------------------------------------------------
def collect_random(n, rng):
    """상관된 무작위 행동으로 회복가능 기울기에서 롤아웃, n개 전이 수집."""
    Z, dS = [], []
    while len(Z) < n:
        s = np.array([rng.uniform(-0.3, 0.3), 0.0,
                      rng.uniform(-0.2, 0.2), 0.0])
        a_prev = 0.0
        for _ in range(MAX_STEPS):
            a = float(np.clip(0.7 * a_prev + rng.normal(0, 0.6 * F_MAX),
                              -F_MAX, F_MAX))
            a_prev = a
            s2 = true_step(s, a)
            Z.append(np.concatenate([s, [a]]))
            dS.append(s2 - s)
            s = s2
            if len(Z) >= n or abs(s[2]) > THETA_LIM or abs(s[0]) > X_LIM:
                break
    return Z[:n], dS[:n]


def collect_onpolicy(n, model, rng):
    """현재 모델의 MPC + 탐색잡음으로 온-폴리시 전이 수집(Dyna/PETS식 병합)."""
    ctrl = make_mpc_controller(model.step_batch, seed=int(rng.integers(1 << 30)))
    Z, dS = [], []
    i = 0
    while len(Z) < n:
        s = EVAL_INITS[i % len(EVAL_INITS)].copy()
        i += 1
        warm = np.zeros(MPC_H)
        for _ in range(MAX_STEPS):
            a, warm = ctrl(s, warm)
            a = float(np.clip(a + rng.normal(0, 2.5), -F_MAX, F_MAX))
            s2 = true_step(s, a)
            Z.append(np.concatenate([s, [a]]))
            dS.append(s2 - s)
            s = s2
            if len(Z) >= n or abs(s[2]) > THETA_LIM or abs(s[0]) > X_LIM:
                break
    return Z[:n], dS[:n]


# --------------------------------------------------------------------------
# MBRL 학습곡선: 예산 체크포인트마다 데이터 수집 → 모델 적합 → MPC 평가
# --------------------------------------------------------------------------
def mbrl_curve(budgets, seed=SEED):
    rng = np.random.default_rng(seed)
    Z, dS = [], []
    collected = 0
    model = None
    perf = []
    for budget in budgets:
        need = budget - collected
        if model is None:
            nz, nd = collect_random(need, rng)
        else:
            nz, nd = collect_onpolicy(need, model, rng)
        Z += nz
        dS += nd
        collected = budget
        model = fit_model(np.array(Z), np.array(dS))
        ctrl = make_mpc_controller(model.step_batch, seed=123)
        perf.append(eval_return(ctrl))
    return np.array(perf), model, np.array(Z), np.array(dS)


# --------------------------------------------------------------------------
# 무모델 CEM 정책탐색 baseline: 선형정책 a=clip(w·s), return 을 롤아웃으로 최적화
#   누적 환경전이 vs best-so-far return 곡선을 기록(동일 예산 비교용)
# --------------------------------------------------------------------------
def _linpolicy(w):
    def controller(s, warm):
        return float(np.clip(w @ s, -F_MAX, F_MAX)), warm
    return controller


def _policy_return_steps(w, inits):
    tot_r, tot_s = 0.0, 0
    ctrl = _linpolicy(w)
    for s0 in inits:
        r, used = rollout_return(ctrl, s0, count_steps=True)
        tot_r += r
        tot_s += used
    return tot_r / len(inits), tot_s


def modelfree_curve(seed=SEED, budget_max=MF_BUDGET_MAX):
    rng = np.random.default_rng(seed)
    mean = np.zeros(4)
    std = MF_STD0.copy()
    best_perf, _ = _policy_return_steps(mean, EVAL_INITS)
    best_w = mean.copy()
    trans = 0
    xs, ys = [trans], [best_perf]
    while trans < budget_max:
        W = rng.normal(mean, std, size=(MF_POP, 4))
        W[0] = mean
        scores = np.empty(MF_POP)
        for i in range(MF_POP):
            scores[i], used = _policy_return_steps(W[i], EVAL_INITS)
            trans += used
        idx = np.argsort(scores)[-MF_ELITE:]
        mean = W[idx].mean(0)
        std = W[idx].std(0) + 1e-3
        if scores[idx[-1]] > best_perf:
            best_perf = float(scores[idx[-1]])
            best_w = W[idx[-1]].copy()
        xs.append(trans)
        ys.append(best_perf)
    return np.array(xs), np.array(ys), best_w


def perf_at_budget(xs, ys, budget):
    """누적전이 xs 에서 예산 이하 최고 best-so-far return."""
    mask = xs <= budget
    return float(ys[mask][-1]) if mask.any() else float(ys[0])


# --------------------------------------------------------------------------
# 모델 예측정확도: 다중스텝 롤아웃에서 model vs 진짜 상태 오차 vs 지평
# --------------------------------------------------------------------------
def prediction_error_vs_horizon(model, rng, n_traj=40, H=40):
    errs = np.zeros(H)
    theta_true = theta_pred = None
    for j in range(n_traj):
        s = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.5, 0.5),
                      rng.uniform(-0.15, 0.15), rng.uniform(-0.5, 0.5)])
        acts = np.clip(rng.normal(0, 5.0, H), -F_MAX, F_MAX)
        st, sp = s.copy(), s.copy()
        tt, tp = [], []
        for k in range(H):
            st = true_step(st, acts[k])
            sp = model.step_batch(sp[None, :], np.array([acts[k]]))[0]
            errs[k] += np.linalg.norm(st - sp)
            tt.append(st[2])
            tp.append(sp[2])
        if j == 0:
            theta_true, theta_pred = np.array(tt), np.array(tp)
    return errs / n_traj, theta_true, theta_pred


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    # 1) 오라클 MPC (진짜 모델 + 강한 플래너) — 상한선
    oracle_ctrl = make_mpc_controller(cartpole_step_batch, seed=99,
                                      pop=ORACLE_POP, iters=ORACLE_ITERS,
                                      elite=ORACLE_ELITE)
    oracle_perf = eval_return(oracle_ctrl)

    # 2) MBRL: 예산별 모델학습 → MPC 계획
    mbrl_perf, model, Zdata, dSdata = mbrl_curve(BUDGETS, seed=SEED)

    # 3) 무모델 CEM 정책탐색 (동일 return 을 환경 롤아웃으로 최적화)
    mf_x, mf_y, _ = modelfree_curve(seed=SEED)

    # 동일 소예산(COMPARE_BUDGET) 비교
    ci = BUDGETS.index(COMPARE_BUDGET)
    mbrl_at = float(mbrl_perf[ci])
    mf_at = perf_at_budget(mf_x, mf_y, COMPARE_BUDGET)

    # MBRL 이 오라클의 SOLVE_FRAC 에 처음 도달하는 전이 수
    thr = SOLVE_FRAC * oracle_perf
    solved = [b for b, p in zip(BUDGETS, mbrl_perf) if p >= thr]
    mbrl_trans_to_solve = solved[0] if solved else None
    mbrl_best = float(mbrl_perf.max())

    # 무모델이 같은 문턱에 도달하는 전이 수(표본효율 대비)
    mf_reach = mf_x[mf_y >= thr]
    mf_trans_to_solve = int(mf_reach[0]) if mf_reach.size else None

    # 모델 예측정확도
    err_h, th_true, th_pred = prediction_error_vs_horizon(model, rng)

    print("=== Model-based RL: learn a dynamics model, plan with MPC (cart-pole) ===")
    print(f"모델 f_hat: RFF({RFF_D}) + ridge, 상태증분 예측 | 플래너: MPC CEM"
          f"(H={MPC_H}, pop={MPC_POP}, iters={MPC_ITERS})")
    print(f"환경-전이 예산 체크포인트: {BUDGETS}")
    print()
    print(f"[성능 return ↑ (진짜 플랜트, 평균 upright 보상, 이탈 시 truncate)]")
    print(f"  오라클 MPC (진짜 모델, 상한)     {oracle_perf:.3f}")
    for b, p in zip(BUDGETS, mbrl_perf):
        mark = "  <- solved" if p >= thr else ""
        print(f"  MBRL @ {b:5d} transitions        {p:.3f}   "
              f"({100*p/oracle_perf:.0f}% of oracle){mark}")
    print()
    print(f"[동일 소예산 {COMPARE_BUDGET} transitions 에서 표본효율 비교]")
    print(f"  MBRL       {mbrl_at:.3f}   ({100*mbrl_at/oracle_perf:.0f}% of oracle)")
    print(f"  무모델 CEM {mf_at:.3f}   ({100*mf_at/oracle_perf:.0f}% of oracle)")
    print(f"  → MBRL 이 동일 예산에서 무모델 대비 +{mbrl_at - mf_at:.3f} "
          f"({mbrl_at / max(mf_at, 1e-9):.1f}x)")
    if mbrl_trans_to_solve is not None:
        line = (f"  → MBRL 은 {mbrl_trans_to_solve} transitions 로 오라클의 "
                f"{100*SOLVE_FRAC:.0f}% 도달")
        if mf_trans_to_solve is not None:
            line += (f"; 무모델은 {mf_trans_to_solve} transitions 필요 "
                     f"({mf_trans_to_solve / mbrl_trans_to_solve:.0f}x 더 많음)")
        else:
            line += f"; 무모델은 예산 {MF_BUDGET_MAX} 내 미도달"
        print(line)
    print()
    print(f"[모델 예측정확도 (다중스텝 누적오차)]")
    print(f"  1-step L2 오차     {err_h[0]:.4f}")
    print(f"  {len(err_h)}-step L2 오차   {err_h[-1]:.4f}   "
          f"(지평 따라 누적 = model bias/compounding error)")
    print("  (정직: 학습모델은 편향이 있어 MPC 는 오라클에 '근접'하되 그 아래에서 정체; "
          "무모델도 예산을 크게 늘리면 결국 따라잡음)")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(15, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.30)

    # (a) 학습곡선: return vs #transitions (MBRL / 무모델 / 오라클)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axhline(oracle_perf, color="k", ls="--", lw=1.5,
                label=f"oracle MPC (true model) {oracle_perf:.2f}")
    ax0.axhline(thr, color="gray", ls=":", lw=1,
                label=f"{100*SOLVE_FRAC:.0f}% of oracle (solved)")
    ax0.plot(mf_x, mf_y, "-", color="#d9534f", lw=2, label="model-free CEM")
    ax0.plot(BUDGETS, mbrl_perf, "o-", color="#1f77b4", lw=2.2, ms=6,
             label="MBRL (learn model + MPC)")
    ax0.axvline(COMPARE_BUDGET, color="#1f77b4", ls=":", lw=1, alpha=0.5)
    ax0.annotate(f"same budget\n{COMPARE_BUDGET} transitions:\nMBRL {mbrl_at:.2f} "
                 f"vs MF {mf_at:.2f}", xy=(COMPARE_BUDGET, mbrl_at),
                 xytext=(COMPARE_BUDGET * 1.4, oracle_perf * 0.45),
                 fontsize=7.5, color="#1f4d80",
                 arrowprops=dict(arrowstyle="->", color="#1f4d80", lw=1))
    ax0.set_xscale("log")
    ax0.set_xlabel("environment transitions (log)")
    ax0.set_ylabel("task return ↑  (avg upright reward)")
    ax0.set_title("Sample efficiency: MBRL nears oracle with far fewer\n"
                  "transitions than model-free needs", fontsize=10)
    ax0.set_ylim(0, oracle_perf * 1.12)
    ax0.legend(fontsize=7.5, loc="lower right")
    ax0.grid(alpha=0.25, which="both")

    # (b) 모델 예측정확도: 다중스텝 누적 L2 오차 vs 지평
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(np.arange(1, len(err_h) + 1), err_h, "o-", color="#2ca02c",
             lw=2, ms=3)
    ax1.set_xlabel("prediction horizon [steps]")
    ax1.set_ylabel("mean L2 state error")
    ax1.set_title("Learned model accuracy\n(error compounds over horizon = model bias)",
                  fontsize=10)
    ax1.grid(alpha=0.25)

    # (c) 예측 궤적 검증: model vs 진짜 (theta over horizon)
    ax2 = fig.add_subplot(gs[0, 2])
    kk = np.arange(1, len(th_true) + 1)
    ax2.plot(kk, th_true, "k-", lw=2, label="true rollout")
    ax2.plot(kk, th_pred, "--", color="#2ca02c", lw=2, label="model prediction")
    ax2.axhline(THETA_LIM, color="gray", ls=":", lw=0.8)
    ax2.axhline(-THETA_LIM, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("horizon [steps]")
    ax2.set_ylabel("pole angle θ [rad]")
    ax2.set_title("Model tracks the true rollout\n(the model is what enables planning)",
                  fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.25)

    fig.suptitle("Model-based RL: learn a dynamics model from few transitions, "
                 "then plan with MPC — far more sample-efficient than model-free",
                 fontsize=12, y=1.03)
    for p in ("outputs/38_model_based_rl.png", "assets/38_model_based_rl.png"):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    print("\n[plot] outputs/38_model_based_rl.png, assets/38_model_based_rl.png")

    return mbrl_at, mf_at, oracle_perf, mbrl_trans_to_solve


if __name__ == "__main__":
    main()
