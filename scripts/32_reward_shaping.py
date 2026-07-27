"""보상 설계(reward shaping)가 RL의 성패를 가른다 — 그리고 reward hacking.

강화학습에서 '어려운 부분'은 대개 최적화기가 아니라 보상 함수 그 자체다. 똑같은 과제,
똑같은 최적화기(여기서는 CEM)라도 보상을 어떻게 설계하느냐에 따라 학습이 되기도 하고
전혀 안 되기도 한다. 이 실험은 그 사실을 데이터로 보인다.

과제는 고전적인 진자 흔들어 세우기(pendulum swing-up)다. 막대는 아래로 늘어져 시작하고,
목표는 위로 흔들어 올려 거꾸로(도립) 균형을 잡는 것이다. 토크가 제한돼 곧장 밀어 올릴 수
없으므로, 에이전트는 앞뒤로 흔들며 에너지를 펌핑해야 한다 — 바로 이 저구동(under-actuated)
성질 때문에 보상 설계가 결정적으로 중요해진다.

정책은 에너지 펌핑 + 상단 PD 균형을 섞은 소수 파라미터 컨트롤러이고, 동일한 CEM(교차
엔트로피법, numpy만)으로 최적화한다. 오직 '보상'만 바꿔 세 가지를 비교한다.

  1. Sparse(희소):   상단 근처에 있을 때만 +1, 아니면 0. → 신호가 거의 없어 CEM이 좋은
                     파라미터를 구분하지 못함 → 학습 실패/큰 분산.
  2. Shaped(잘 설계): 상단으로부터의 각오차^2 + 작은 각속도^2 + 작은 토크^2 에 대한 밀집
                     페널티. → 매 스텝 방향을 알려줘 흔들어 세우기+균형을 안정적으로 학습.
  3. Hacked(오설계):  "활기차게 움직여라"라는 선의의 보상으로 각속도^2을 보상. → 최적화기가
                     이를 악용해 균형은 안 잡고 영원히 빙빙 도는 퇴화 정책으로 수렴. 학습
                     보상은 매우 높지만 실제 과제(안정적 도립)는 전혀 풀지 못함 = reward hacking.

핵심은 '학습 보상과 무관한' 과제 성공 지표로 정직하게 평가하는 것이다: 에피소드 마지막
구간에서 상단 허용오차 안에 '그리고 저속으로' 머문 스텝 비율. 이 지표로 "shaped는 과제를
풀었고, sparse는 학습하지 못했으며, hacked는 보상은 높지만 실제 성공은 낮다(해킹됨)"를
숫자로 말할 수 있다. (토이 과제이며 요지는 정성적이다: 보상 설계가 결과를 바꾼다.)

    python scripts/32_reward_shaping.py
"""

from __future__ import annotations

import math

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 진자 물리 (직접 구현). theta는 상단(도립)으로부터의 각도: theta=0 위, theta=±pi 아래.
# gym Pendulum 스케일: theta_ddot = 3g/(2l) sin(theta) + 3/(m l^2) u.
#   상단(0)은 불안정 평형, 하단(pi)은 안정 평형 → 아래로 늘어져 시작.
# 토크 한계 u_max 때문에 수평에서 중력토크를 못 버팀 → 에너지 펌핑 필수(저구동).
# --------------------------------------------------------------------------
G, L, M = 10.0, 1.0, 1.0
U_MAX = 2.0                 # 토크 한계 (중력토크보다 작아 곧장 못 올림)
DT = 0.05
T = 200                     # 에피소드 길이(스텝) = 10 s
MAX_SPEED = 8.0
E_TOP = 3.0 * G / (2.0 * L)     # 상단 정지 시 에너지(위치에너지 기준 최대)

NP = 6                      # 정책 파라미터 수(일반 선형 상태피드백 가중치)

# 과제 성공 지표(학습 보상과 무관): 마지막 N스텝을 '상단 허용오차 & 저속'으로
TAIL = 60
TOL_ANG = 0.20             # rad (~11.5°)
TOL_VEL = 1.5              # rad/s (빠르게 스쳐 지나가는 건 성공 아님)


def angle_norm(x: float) -> float:
    """각도를 [-pi, pi]로 래핑."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def policy_torque(theta: float, thdot: float, p) -> float:
    """일반 선형 상태피드백 컨트롤러 — 스윙업 해법을 미리 넣지 '않는다'.

    상단 기준 각도 th의 특징벡터에 대한 선형결합만으로 토크를 낸다:
        f = [sin th, cos th, thdot, thdot·cos th, thdot·sin th, 1]
        u = clip(p · f, ±u_max)
    이 특징들은 원리상 균형(PD: -k1·sin th - k2·thdot)과 에너지 펌핑(thdot·cos th 류)을
    '표현할 수 있으나', 어느 것도 공짜로 주어지지 않는다 — 좋은 가중치를 '탐색'해 찾아야
    한다. 그래서 보상 설계가 결정적이다: 밀집 보상은 그 탐색을 안내하고, 희소 보상은 못
    한다. 세 보상 모두 '같은' 이 정책군을 같은 CEM으로 튜닝한다(보상만 다름).
    """
    th = angle_norm(theta)
    s, c = math.sin(th), math.cos(th)
    u = (p[0] * s + p[1] * c + p[2] * thdot
         + p[3] * thdot * c + p[4] * thdot * s + p[5])
    if u > U_MAX:
        u = U_MAX
    elif u < -U_MAX:
        u = -U_MAX
    return u


def rollout(p, reward_type: str, record: bool = False):
    """아래로 늘어진 상태에서 결정적으로 굴린다. 총 학습보상과(필요시) 궤적을 반환."""
    theta, thdot = math.pi, 0.0            # 정확히 아래로 늘어져 시작(결정적)
    total = 0.0
    ths = [theta] if record else None
    tds = [thdot] if record else None
    for _ in range(T):
        u = policy_torque(theta, thdot, p)
        th = angle_norm(theta)
        # --- 세 가지 보상 설계 (오직 여기만 다르다) ---
        if reward_type == "sparse":
            r = 1.0 if abs(th) < 0.1 else 0.0                    # 상단 근처만 +1
        elif reward_type == "shaped":
            r = -(th * th + 0.1 * thdot * thdot + 0.001 * u * u)  # 밀집 도립 비용
        elif reward_type == "hacked":
            r = thdot * thdot                                    # "활기차게" → 회전 보상
        else:
            raise ValueError(reward_type)
        total += r
        # --- 반음시적 오일러 적분 ---
        thddot = 3.0 * G / (2.0 * L) * math.sin(theta) + 3.0 / (M * L * L) * u
        thdot += thddot * DT
        if thdot > MAX_SPEED:
            thdot = MAX_SPEED
        elif thdot < -MAX_SPEED:
            thdot = -MAX_SPEED
        theta += thdot * DT
        if record:
            ths.append(theta)
            tds.append(thdot)
    if record:
        return total, np.array(ths), np.array(tds)
    return total


def true_success(ths: np.ndarray, tds: np.ndarray) -> float:
    """학습 보상과 무관한 과제 성공률: 마지막 TAIL 스텝 중 '상단 & 저속' 비율."""
    th = np.array([angle_norm(a) for a in ths[-TAIL:]])
    td = tds[-TAIL:]
    ok = (np.abs(th) < TOL_ANG) & (np.abs(td) < TOL_VEL)
    return float(np.mean(ok))


# --------------------------------------------------------------------------
# CEM (교차 엔트로피법) — numpy만. 세 보상에 대해 '완전히 동일한' 설정으로 실행.
# --------------------------------------------------------------------------
INIT_STD = np.full(NP, 2.0)
POP, ITERS, ELITE = 100, 30, 15


def cem(reward_type: str, seed: int):
    rng = np.random.default_rng(seed)
    mean = np.zeros(NP)
    std = INIT_STD.copy()
    for _ in range(ITERS):
        samples = mean + std * rng.standard_normal((POP, NP))
        rewards = np.array([rollout(s, reward_type) for s in samples])
        elite_idx = np.argsort(rewards)[-ELITE:]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3          # 붕괴 방지 하한
    return mean


SEEDS = (0, 1, 2, 3, 4)     # 여러 시드 평균: 특히 sparse의 큰 분산을 정직하게 반영
REWARDS = ("sparse", "shaped", "hacked")
COLORS = {"sparse": "#d9534f", "shaped": "#1f77b4", "hacked": "#6f42c1"}
LABELS = {"sparse": "Sparse (no signal)", "shaped": "Shaped (good)",
          "hacked": "Hacked (spins)"}


def main():
    # 각 보상 × 각 시드로 CEM 학습 → 최종 정책을 결정적으로 평가.
    results = {}       # reward -> dict(success_mean, rep_ths, rep_tds, train_reward, spin_reward)
    for rt in REWARDS:
        succs, reps = [], None
        for sd in SEEDS:
            p = cem(rt, sd)
            _, ths, tds = rollout(p, rt, record=True)
            succs.append(true_success(ths, tds))
            if sd == SEEDS[0]:
                reps = (ths, tds, p)
        ths0, tds0, p0 = reps
        # 대표 정책의 '자기 보상'과 회전량(각속도^2 합) 기록
        train_reward = rollout(p0, rt)
        spin_reward = float(np.sum(tds0[:-1] ** 2))
        results[rt] = dict(success=float(np.mean(succs)),
                           succ_all=succs, ths=ths0, tds=tds0,
                           train_reward=train_reward, spin_reward=spin_reward)

    sparse_s = results["sparse"]["success"]
    shaped_s = results["shaped"]["success"]
    hacked_s = results["hacked"]["success"]
    # reward hacking 정량화: hacked 정책이 낸 회전보상 / 이론상 최대 회전보상
    hacked_spin = results["hacked"]["spin_reward"]
    max_spin = MAX_SPEED ** 2 * T
    hacked_reward_frac = hacked_spin / max_spin              # 높음(회전으로 보상 채움)
    shaped_spin = results["shaped"]["spin_reward"]

    print("=== 보상 설계가 RL 학습을 좌우한다: 진자 흔들어 세우기 ===")
    print(f"동일 과제·동일 CEM(pop={POP}, iters={ITERS}, elite={ELITE}), 시드 {SEEDS}. 보상만 다름.")
    print(f"토크 한계 u_max={U_MAX} (중력 최대토크 {M*G*L:.0f} 보다 작음 → 에너지 펌핑 필수)")
    print(f"과제 성공 지표(학습보상과 무관): 마지막 {TAIL}스텝 중 |angle|<{TOL_ANG} & |speed|<{TOL_VEL} 비율\n")
    print(f"  1) Sparse  참성공률 {sparse_s:.2f}   시드별 {['%.2f'%s for s in results['sparse']['succ_all']]}  → 학습 실패")
    print(f"  2) Shaped  참성공률 {shaped_s:.2f}   시드별 {['%.2f'%s for s in results['shaped']['succ_all']]}  → 과제 해결")
    print(f"  3) Hacked  참성공률 {hacked_s:.2f}   시드별 {['%.2f'%s for s in results['hacked']['succ_all']]}  → 보상만 높음")
    print()
    print(f"reward hacking: hacked 정책의 회전보상 {hacked_spin:.0f} "
          f"(= 최대치의 {100*hacked_reward_frac:.0f}%), 그러나 참성공률은 {hacked_s:.2f}")
    print(f"  대조: shaped 정책은 도립을 유지해 회전보상 {shaped_spin:.0f}로 '낮지만' 참성공률 {shaped_s:.2f}")
    print(f"  → 높은 학습보상 ≠ 과제 해결. 잘못 설계된 보상은 퇴화 정책으로 악용된다.")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0])

    # (좌) 대표 궤적: 상단 높이 cos(theta). +1=도립, -1=아래.
    axT = fig.add_subplot(gs[:, 0])
    tvec = np.arange(T + 1) * DT
    for rt in REWARDS:
        h = np.cos(results[rt]["ths"])          # 상단 높이
        axT.plot(tvec, h, color=COLORS[rt], lw=1.9, label=LABELS[rt])
    axT.axhline(math.cos(TOL_ANG), color="gray", ls=":", lw=1)
    axT.text(tvec[-1], math.cos(TOL_ANG), "  upright band", va="center",
             fontsize=7, color="gray")
    axT.set_ylim(-1.15, 1.2)
    axT.set_xlabel("time [s]")
    axT.set_ylabel("uprightness  cos(theta)   (+1 = balanced up)")
    axT.set_title("Same task, same CEM — only the reward differs\n"
                  "shaped rises & holds at top; sparse flails; hacked spins forever")
    axT.legend(loc="lower right", fontsize=8)
    axT.grid(alpha=0.2)

    # (우상) 참 과제 성공률 막대
    axB = fig.add_subplot(gs[0, 1])
    xs = np.arange(3)
    vals = [sparse_s, shaped_s, hacked_s]
    bars = axB.bar(xs, vals, color=[COLORS[r] for r in REWARDS], alpha=0.9)
    for b, v in zip(bars, vals):
        axB.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", fontsize=9)
    axB.set_xticks(xs)
    axB.set_xticklabels(["sparse", "shaped", "hacked"], fontsize=8)
    axB.set_ylim(0, 1.1)
    axB.set_ylabel("true-task success")
    axB.set_title("True success (reward-independent)", fontsize=9)
    axB.grid(alpha=0.2, axis="y")

    # (우하) reward hacking: hacked의 학습보상(정규화)은 높지만 참성공률은 낮음
    axH = fig.add_subplot(gs[1, 1])
    hx = np.arange(2)
    hvals = [hacked_reward_frac, hacked_s]
    hbars = axH.bar(hx, hvals, color=["#6f42c1", "#b0b0b0"], alpha=0.9)
    for b, v in zip(hbars, hvals):
        axH.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", fontsize=9)
    axH.set_xticks(hx)
    axH.set_xticklabels(["training reward\n(norm.)", "true success"], fontsize=8)
    axH.set_ylim(0, 1.1)
    axH.set_title("Reward hacking (hacked policy):\nhigh reward, task NOT solved", fontsize=9)
    axH.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    for path in ("outputs/32_reward_shaping.png", "assets/32_reward_shaping.png"):
        fig.savefig(path, dpi=130)
    print("\n[plot] outputs/32_reward_shaping.png, assets/32_reward_shaping.png")

    return sparse_s, shaped_s, hacked_s, hacked_reward_frac


if __name__ == "__main__":
    main()
