"""휨 보상을 닫힌 루프로: 병목은 센서가 아니라 **제어 권한**이었다.

exp 48 은 경사면 바늘의 휨을 **180° 스핀 flip** 으로 상쇄했다 — 최적 시점은 삽입의 29.3%
(1−1/√2). 그런데 그건 **열린 루프**였다. 곡률을 미리 안다고 가정했고, VERIFICATION.md 는
H3 의 잔여 위험에 "bending compensation is open-loop; tissue inhomogeneity changes curvature.
Needs tip tracking" 이라고 적어 뒀다. 이 실험이 그 칸을 채운다.

--- 왜 균질 조직에서는 열린 루프로 충분했나 ---
곡률이 **상수**면 최적 flip 시점이 곡률 **크기와 무관**하다. 실제로 최적 조건은

    F(d_f) = F(L)/2,   F(d) ≡ ∫₀^d (L−u)·κ(u) du      ("누적 모멘트의 절반")

이고, κ가 상수면 d_f = L(1−1/√2) 로 κ가 소거된다. 그래서 exp 48 은 κ를 몰라도 됐다.
**층이 생기면 소거되지 않는다.** 바늘이 조직 경계를 넘으면(지방→근육, 피막 통과) 곡률이
κ1 → κ2 로 바뀌고, 그때 최적 시점은 **비율 r = κ2/κ1 에 의존한다.** r 은 환자마다 다르고
수술 전 영상으로는 알 수 없다 — 그래서 팁을 추적해 **삽입 중에 추정**해야 한다.

--- 이 실험의 핵심 긴장 ---
κ2 는 경계를 넘은 뒤에만 관측되고, 그 정보는 (S−s_b)² 로 자란다. 즉 **깊이 들어갈수록 추정이
좋아진다.** 그런데 r > 1(층2가 더 휘는 경우)이면 최적 flip 시점이 **앞으로 당겨진다** — 기다려서
잘 추정하고 나면 이미 지나쳐 있다. **추정이 쓸 만해질 때쯤 제어 권한이 사라진다.**
그 사이 어딘가에 최적 결정 시점이 있고, 그걸 재는 것이 이 실험이다.

그리고 exp 53 의 교훈이 그대로 시험대에 오른다 — **팁 센서도 잡음과 지연을 가진 센서**다.
"폐루프면 해결"이 아니라, 무엇이 실제로 이득을 만들었는지 **절제(ablation)** 로 갈라야 한다.

--- 미리 말해두는 결론 ---
1. 층이 생기면 exp 48 의 29.3% 가 깨진다: r=2.5 환자에서 팁 편차 1.15 mm(허용치 1.0 초과),
   그리고 **r>1 이면 최적 시점이 앞으로 당겨진다** — 기다려 추정하면 이미 지나친다.
2. 결정해야 할 순간(20.5 mm)에 σ(κ̂₂) ≈ 35/m 이다. 사전분포 표준편차 0.49/m 보다 훨씬 크다.
   **우도가 사전분포를 이기지 못한다.**
3. 그래서 **절제하면 측정의 기여가 거의 0 이다.** p90 은 열린 루프 0.98 → 0.50 mm 로 좋아지는데,
   그 실체는 정보가 아니라 ① 더 맞는 기본값(r=1 → r̄=1.45) ② flip 을 늦추는 타이밍 제약이다.
   측정 0회의 "사전지식만" 정책이 0.49 mm 로 **MAP 와 같다.** 절제 없이 봤으면 정보 덕이라
   오독했을 것이다. 5-DOF 센서(방향까지)를 줘도 0.47 mm — 간극의 6% 만 메운다.
4. 병목은 센싱이 아니라 **작동**이었다. duty cycling 으로 바꾸되 **재계획 1회면 flip 과 똑같다**
   (0.47 mm; 명령이 포화해 u=−1 이면 duty 가 곧 flip 이다). 재계획을 4회로 늘리면
   p90 **0.37 mm**, 중앙값 0.23 → 0.09 mm. 즉 이득은 '다르게 조작'이 아니라 **'다시 조작'** 이다 —
   정보가 늦었던 게 아니라 **쓸 기회가 한 번뿐이었다.**
5. 그리고 병목이 또 옮겨 간다: 참 κ 를 줘도 duty p90 이 0.36 mm 라, 추정 탓은 0.01 mm 뿐이고
   나머지는 재계획 이산화·포화·모델이다. 센서를 10배 좋게 해도 0.38 → 0.37 mm.
   **이제 돈을 쓸 곳은 센서가 아니다.**

    python scripts/54_closed_loop_needle.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

needle = import_module("48_flexible_needle")     # 곡률·통로 상수 재사용

L_INS = 70.0e-3            # 총 삽입 길이 [m] (exp 48 과 동일)
S_BOUND = 15.0e-3          # 조직 경계 깊이 [m] — 수술 전 영상에서 안다
KAPPA1 = 0.81              # 층 1 곡률 [1/m] — exp 48 이 보 해석으로 얻은 값
R_RANGE = (0.4, 2.5)       # 층 2/층 1 곡률비, 환자마다 미지
TIP_SIGMA = 0.3e-3         # 팁 위치 추적 잡음 σ [m] (전자기 트래커 규모)
THETA_SIGMA = np.deg2rad(0.5)   # 팁 방향 잡음 σ [rad] — 5-DOF EM 센서 사양 규모
TIP_LAG = 0.0              # 측정 지연을 삽입 거리로 환산 [m]
MEAS_STEP = 2.0e-3         # 팁 측정 간격 [m]
D_NOMINAL = (1.0 - 1.0 / np.sqrt(2)) * L_INS   # exp 48 의 열린 루프 시점 (29.3%)
TOL = 1.0e-3               # 팁 편차 허용치 [m]
CORRIDOR = 2.20e-3         # exp 48 의 강체 가정 통로 여유 [m]

N_GRID = 700               # 비선형 적분 격자 (κL≈0.06 rad 이라 이 정도면 충분)
SIGMA_SWEEP = (0.1e-3, 0.3e-3, 1.0e-3)
LAG_SWEEP = (0.0, 5.0e-3, 10.0e-3)


# --------------------------------------------------------------------------- #
# 1) 곡률 모델과 그 적분 (선형 모델 = 추정·계획용)
# --------------------------------------------------------------------------- #
def kappa_profile(s, k1, k2, s_b=S_BOUND):
    """두 층 곡률 κ(s). 경계 깊이는 알고, 두 값은 모른다."""
    return np.where(s < s_b, k1, k2)


def moment_F(d, k1, k2, s_b=S_BOUND, L=L_INS):
    """F(d) = ∫₀^d (L−u)κ(u) du — '누적 모멘트'. flip 조건 F(d_f)=F(L)/2 의 좌변."""
    d = np.atleast_1d(np.asarray(d, float))
    out = np.empty_like(d)
    for i, dd in enumerate(d):
        m = min(dd, s_b)
        out[i] = k1 * (L * m - m ** 2 / 2)
        if dd > s_b:
            out[i] += k2 * (L * (dd - s_b) - (dd ** 2 - s_b ** 2) / 2)
    return out


def plan_flip(k1, k2, d_now=0.0, s_b=S_BOUND, L=L_INS):
    """F(d_f) = F(L)/2 를 푸는 flip 깊이. 이미 지나쳤으면 **지금 즉시** 뒤집는다.

    F 가 구간별 이차식이라 **해석적으로** 푼다(처음엔 격자 보간으로 짰다가, 몬테카를로가
    2만 번 넘게 부르면서 그게 병목이 됐다). κ가 상수면 해가 정확히 L(1−1/√2) 로 떨어진다 —
    exp 48 의 29.3% 가 이 법칙의 특수해다."""
    F_b = k1 * (L * s_b - s_b ** 2 / 2)                    # F(s_b)
    F_L = F_b + k2 * (L * (L - s_b) - (L ** 2 - s_b ** 2) / 2)
    T = 0.5 * F_L
    if T <= F_b:                                            # 해가 층 1 안에 있다
        c = 2 * T / k1
    else:                                                   # 층 2 안에 있다
        c = 2 * (T - F_b) / k2 + 2 * L * s_b - s_b ** 2
    disc = max(L ** 2 - c, 0.0)
    return max(L - np.sqrt(disc), d_now)


def tip_lateral_linear(S, k1, k2, s_b=S_BOUND):
    """소각 근사에서 (flip 전) 삽입 깊이 S 일 때의 팁 횡변위.

    y(S) = ∫₀^S (S−u)κ(u) du = a₁(S)·κ₁ + a₂(S)·κ₂ — (k1,k2) 에 **선형**이라 추정이
    최소자승이 되고, 계수는 design_row 가 준다(적분을 돌 필요가 없다)."""
    a = design_row(S, s_b)
    return float(a[0] * k1 + a[1] * k2)


def design_row(S, s_b=S_BOUND):
    """팁 **위치** 측정 y(S) 의 (k1, k2) 계수 — flip 전(σ=+1) 구간에서 유효.

    k2 열이 (S−s_b)²/2 로 **0 에서 이차로** 자란다: 경계를 넘은 직후에는 사실상 관측 불가."""
    m = min(S, s_b)
    a1 = S * m - m ** 2 / 2
    a2 = (S - s_b) ** 2 / 2 if S > s_b else 0.0
    return np.array([a1, a2])


def design_row_theta(S, s_b=S_BOUND):
    """팁 **방향** 측정 θ(S) = ∫₀^S κ du 의 (k1, k2) 계수.

    5-DOF 전자기 트래커는 위치뿐 아니라 방향도 준다. 방향은 곡률의 **1차 적분**이라 k2 열이
    (S−s_b) 로 **선형** 증가한다 — 위치의 이차 증가보다 훨씬 일찍 정보가 생긴다.
    (추정에서 '상태에 가장 가까운 미분을 재라'는 규칙이 그대로 나온다.)"""
    return np.array([min(S, s_b), max(S - s_b, 0.0)])


# --------------------------------------------------------------------------- #
# 2) 진짜 바늘 (비선형 적분) — 추정기는 위의 선형 모델을 쓴다
# --------------------------------------------------------------------------- #
def simulate(k1, k2, u, s_b=S_BOUND, L=L_INS, n=N_GRID):
    """평면에서 θ' = u(s)·κ(s), x' = cosθ, y' = sinθ 를 적분한 실제 중심선.

    u(s) 는 **명령 조향**이다. exp 48 의 두 정책이 여기 특수해로 들어간다.
      - flip:  u = +1 → −1 (한 번 뒤집기; 권한이 그 순간 소진된다)
      - duty:  u ∈ [−1,1] 연속 (duty cycling 으로 유효 곡률을 비례 조절; 권한이 끝까지 남는다)
    u 는 스칼라(=flip 깊이)로도 배열로도 받는다.

    추정·계획은 소각 선형 모델을 쓰므로 여기에 **모델 오차**가 들어간다(작지만 0 은 아니다)."""
    s = np.linspace(0.0, L, n)
    ds = s[1] - s[0]
    u_arr = np.where(s < u, 1.0, -1.0) if np.isscalar(u) else np.asarray(u, float)
    theta = np.cumsum(u_arr * kappa_profile(s, k1, k2, s_b)) * ds
    theta -= theta[0]
    x = np.cumsum(np.cos(theta)) * ds
    y = np.cumsum(np.sin(theta)) * ds
    return s, x - x[0], y - y[0]


def tip_error(k1, k2, u, **kw):
    """계획한 직선 표적에서 팁이 벗어난 거리(횡방향)."""
    _, _, y = simulate(k1, k2, u, **kw)
    return abs(float(y[-1]))


def duty_schedule(rng, k1_true, k2_true, replan_depths, sigma=TIP_SIGMA, lag=TIP_LAG,
                  s_b=S_BOUND, L=L_INS, n=N_GRID, orient=True, known=False):
    """duty cycling 폐루프: 재계획 시점마다 추정하고 **남은 구간의 유효 곡률**을 다시 정한다.

    남은 구간에 상수 명령 u 를 쓴다고 하면 예측 팁 변위가
        y(L) = P + u·Q,  P = ∫₀^{d}(L−s)u(s)κ̂ ds,  Q = ∫_d^L (L−s)κ̂ ds
    이므로 u = −P/Q (|u|≤1 로 포화). flip 과 달리 **매번 다시 고칠 수 있다** — 늦게 온 정보도
    쓸 수 있다는 것이 이 정책의 전부다."""
    s = np.linspace(0.0, L, n)
    ds = s[1] - s[0]
    u = np.ones(n)
    for d in replan_depths:
        if known:                                   # 추정 오차를 뺀 상한(제어·모델 잔차만)
            p = np.array([k1_true, k2_true])
        else:
            p, _, _ = estimate(rng, k1_true, k2_true, d, sigma, lag, s_b=s_b,
                               prior=True, orient=orient)
        kh = kappa_profile(s, max(p[0], 1e-6), max(p[1], 1e-6), s_b)
        w = (L - s) * kh * ds
        past = s < d
        P = float(np.sum(w[past] * u[past]))
        Q = float(np.sum(w[~past]))
        u[~past] = np.clip(-P / Q, -1.0, 1.0) if abs(Q) > 1e-15 else 1.0
    return u


# --------------------------------------------------------------------------- #
# 3) 삽입 중 추정 — 팁 측정에서 (κ1, κ2) 최소자승
# --------------------------------------------------------------------------- #
R_PRIOR_MU = float(np.mean(R_RANGE))                        # 모집단 곡률비 평균
R_PRIOR_SD = (R_RANGE[1] - R_RANGE[0]) / np.sqrt(12.0)      # 균일분포의 표준편차


def estimate(rng, k1_true, k2_true, d_dec, sigma=TIP_SIGMA, lag=TIP_LAG,
             step=MEAS_STEP, s_b=S_BOUND, prior=False, orient=False,
             sigma_theta=THETA_SIGMA):
    """결정 시점 d_dec 까지 얻은 팁 측정으로 (κ1, κ2) 를 추정. 반환 (추정, 공분산, 측정 수).

    지연 lag 은 '지금 보이는 값이 lag 만큼 얕은 깊이의 것'으로 모델링한다 — 그만큼 정보가 준다.

    prior=True 면 **MAP**: 모집단 분포(r ~ U(0.4,2.5))를 사전분포로 넣어
        p̂ = (AᵀA/σ² + P⁻¹)⁻¹(Aᵀy/σ² + P⁻¹μ)
    를 푼다. 데이터가 약하면 공칭으로 수축하고 강하면 데이터를 따른다 — exp 51 의
    "사전지식 / 데이터 / 둘 다" 구조가 제어 루프 안에서 그대로 반복된다."""
    s_max = d_dec - lag
    mu = np.array([KAPPA1, KAPPA1 * R_PRIOR_MU])
    Pinv = np.diag([1.0 / (0.15 * KAPPA1) ** 2,             # κ1 은 exp 48 에서 잘 안다
                    1.0 / (KAPPA1 * R_PRIOR_SD) ** 2])
    if s_max <= step:
        return mu.copy(), None, 0                            # 정보 없음 → 사전평균
    S = np.arange(step, s_max + 1e-12, step)
    p_true = np.array([k1_true, k2_true])
    A = np.stack([design_row(s, s_b) for s in S])
    y = A @ p_true + rng.normal(0.0, sigma, len(S))           # flip 전이라 σ=+1
    W = np.full(len(S), 1.0 / sigma ** 2)
    if orient:                                                # 5-DOF 센서: 방향도 준다
        At = np.stack([design_row_theta(s, s_b) for s in S])
        A = np.vstack([A, At])
        y = np.concatenate([y, At @ p_true + rng.normal(0.0, sigma_theta, len(S))])
        W = np.concatenate([W, np.full(len(S), 1.0 / sigma_theta ** 2)])
    AtW = A.T * W
    if prior:
        H = AtW @ A + Pinv
        p = np.linalg.solve(H, AtW @ y + Pinv @ mu)
        return p, np.linalg.inv(H), len(S)
    ATA = AtW @ A
    if np.linalg.cond(ATA) > 1e14:                            # κ2 가 아직 관측되지 않았다
        k1h = float(AtW[0] @ y / max(AtW[0] @ A[:, 0], 1e-30))
        return np.array([k1h, KAPPA1]), None, len(S)
    return np.linalg.solve(ATA, AtW @ y), np.linalg.inv(ATA), len(S)


def run_patient(rng, r, d_dec, sigma=TIP_SIGMA, lag=TIP_LAG, mode="ls", s_b=S_BOUND):
    """한 환자 한 삽입. 정책 5종 — **이득의 출처를 분리하기 위한 절제(ablation)** 다.

      open      : 공칭 r=1 로 계획 (exp 48 의 29.3%)          ← 사전지식(틀린 것)만
      prior     : 모집단 평균 r̄ 로 계획, **측정 안 함**        ← 사전지식(맞는 것)만
      hold      : 그냥 d_dec 에서 뒤집는다, **추정 안 함**      ← 타이밍 제약만
      ls / map  : 팁 **위치** 측정으로 추정 → 재계획              ← 데이터(+사전지식)
      map5      : 위치 + **방향**(5-DOF 센서)                    ← 더 나은 관측
      oracle    : 참 r 로 계획                                  ← 상한

    prior·hold 가 있어야 "폐루프가 좋아졌다"를 **측정이 준 것**과 **더 나은 기본값·제약이 준 것**
    으로 나눌 수 있다. 이게 없으면 정보를 얻지 않고도 좋아진 것을 정보 덕이라 오독한다."""
    k1, k2 = KAPPA1, KAPPA1 * r
    if mode == "open":
        d_f = plan_flip(KAPPA1, KAPPA1, s_b=s_b)
    elif mode == "prior":
        d_f = plan_flip(KAPPA1, KAPPA1 * R_PRIOR_MU, s_b=s_b)
    elif mode == "hold":
        d_f = d_dec
    elif mode == "oracle":
        d_f = plan_flip(k1, k2, s_b=s_b)
    else:
        p, _, _ = estimate(rng, k1, k2, d_dec, sigma, lag, s_b=s_b,
                           prior=mode in ("map", "map5"), orient=(mode == "map5"))
        d_f = plan_flip(max(p[0], 1e-6), max(p[1], 1e-6), d_now=d_dec, s_b=s_b)
    return tip_error(k1, k2, d_f, s_b=s_b), d_f


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(seed=7, n_patients=300, quick=False):
    print("=== 54. 휨 보상 폐루프: 추정이 좋아질 때쯤 제어 권한이 사라진다 ===")
    print(f"바늘: exp 48 의 곡률 κ₁={KAPPA1:.2f}/m, 삽입 {L_INS*1e3:.0f} mm, "
          f"조직 경계 {S_BOUND*1e3:.0f} mm, 곡률비 r ∈ {R_RANGE}")
    print(f"팁 추적: σ={TIP_SIGMA*1e3:.1f} mm, 측정 간격 {MEAS_STEP*1e3:.0f} mm, "
          f"지연 {TIP_LAG*1e3:.0f} mm")

    # ---- 교차검증: 균질 조직이면 이 법칙이 exp 48 의 29.3% 와 같아야 한다 ----
    d_hom = plan_flip(KAPPA1, KAPPA1)
    print(f"\n[검증] 균질 조직(r=1)에서 F(d)=F(L)/2 해 = {d_hom/L_INS*100:.2f}% "
          f"(exp 48 의 해석해 1−1/√2 = {(1-1/np.sqrt(2))*100:.2f}%) — 일치.")
    print(f"        곡률을 2배로 해도 {plan_flip(2*KAPPA1, 2*KAPPA1)/L_INS*100:.2f}% "
          f"— 균질하면 최적 시점이 곡률과 무관하다(exp 48 이 κ를 몰라도 됐던 이유).")
    e_flip = tip_error(KAPPA1, KAPPA1, d_hom)
    e_noflip = tip_error(KAPPA1, KAPPA1, np.inf)
    print(f"        같은 조건 팁 편차: flip 없음 {e_noflip*1e3:.2f} mm → flip "
          f"{e_flip*1e3:.3f} mm (exp 48: 1.97 → 0.04 mm)")

    # ------------------------------------------------------------------ #
    # A. 층이 생기면 최적 시점이 조직에 의존한다
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print("[A] 조직 경계가 있으면 최적 flip 시점이 곡률비 r 에 의존한다")
    print("     r      최적 flip[%]   공칭(29.3%)로 갔을 때 팁 편차   최적으로 갔을 때")
    r_list = (0.4, 0.7, 1.0, 1.5, 2.0, 2.5)
    A_rows = []
    for r in r_list:
        k2 = KAPPA1 * r
        d_opt = plan_flip(KAPPA1, k2)
        e_open = tip_error(KAPPA1, k2, D_NOMINAL)
        e_opt = tip_error(KAPPA1, k2, d_opt)
        A_rows.append((r, d_opt, e_open, e_opt))
        print(f"    {r:4.1f}   {d_opt/L_INS*100:8.1f}      {e_open*1e3:12.2f} mm"
              f"            {e_opt*1e3:6.3f} mm")
    hard = max(A_rows, key=lambda z: z[2])
    print(f"  → r 이 1 에서 멀어지면 공칭 시점의 오차가 커진다(최악 r={hard[0]:.1f} 에서 "
          f"{hard[2]*1e3:.2f} mm, 허용치 {TOL*1e3:.1f} mm 초과).")
    print(f"     그리고 **r>1 이면 최적 시점이 앞으로 당겨진다**"
          f"({A_rows[0][1]/L_INS*100:.0f}% @r=0.4 → {A_rows[-1][1]/L_INS*100:.0f}% @r=2.5) — "
          f"기다려서 추정하면 이미 지나친다.")

    # ------------------------------------------------------------------ #
    # B. 관측성: κ2 는 경계를 넘은 뒤에만, 그것도 천천히 보인다
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print("[B] κ₂ 의 관측성 — 결정 시점을 늦출수록 추정이 좋아진다")
    print("     결정 시점[mm]   측정 수   cond(AᵀA)      σ(κ̂₂) [1/m]   남은 삽입[mm]")
    B_rows = []
    for d_dec in (18e-3, 20e-3, 25e-3, 30e-3, 40e-3, 55e-3):
        S = np.arange(MEAS_STEP, d_dec + 1e-12, MEAS_STEP)
        A = np.stack([design_row(s) for s in S])
        ATA = A.T @ A
        c = float(np.linalg.cond(ATA))
        sg = float(np.sqrt(TIP_SIGMA ** 2 * np.linalg.inv(ATA)[1, 1])) if c < 1e14 else np.inf
        B_rows.append((d_dec, len(S), c, sg))
        print(f"     {d_dec*1e3:9.0f}   {len(S):7d}   {c:9.1e}   "
              + (f"{sg:11.3f}" if np.isfinite(sg) else "        inf")
              + f"     {(L_INS-d_dec)*1e3:9.0f}")
    print(f"  → κ₂ 열이 (S−s_b)²/2 로 자라서, 경계({S_BOUND*1e3:.0f} mm) 직후에는 "
          f"σ(κ̂₂) 가 κ₁={KAPPA1:.2f} 보다 크다(= 정보 없음).")
    print(f"     공칭 flip 시점({D_NOMINAL*1e3:.1f} mm)에서도 σ(κ̂₂)="
          f"{[b[3] for b in B_rows if abs(b[0]-20e-3)<1e-9][0]:.2f}/m — "
          f"**결정해야 할 순간에 아직 모른다.**")

    # ------------------------------------------------------------------ #
    # C. 핵심 긴장: 결정 시점 스윕
    # ------------------------------------------------------------------ #
    print("-" * 100)
    n_pat = max(n_patients // 3, 100) if quick else n_patients
    print(f"[C] 결정 시점을 언제로? — 환자 {n_pat}명 (r ~ U{R_RANGE}), 팁 σ={TIP_SIGMA*1e3:.1f} mm")
    d_decs = np.array([18, 20, 22, 25, 28, 32, 38, 45, 55]) * 1e-3
    rs = np.random.default_rng(seed).uniform(*R_RANGE, n_pat)
    e_open = np.array([run_patient(None, r, 0, mode="open")[0] for r in rs])
    e_orc = np.array([run_patient(None, r, 0, mode="oracle")[0] for r in rs])
    e_pri = np.array([run_patient(None, r, 0, mode="prior")[0] for r in rs])
    C = {}
    for mode in ("hold", "ls", "map", "map5"):
        med, p90, bad = [], [], []
        for d_dec in d_decs:
            e = np.array([run_patient(np.random.default_rng([seed, i]), r, d_dec, mode=mode)[0]
                          for i, r in enumerate(rs)])
            med.append(float(np.median(e)))
            p90.append(float(np.percentile(e, 90)))
            bad.append(float(np.mean(e > TOL)))
        C[mode] = dict(med=np.array(med), p90=np.array(p90), bad=np.array(bad))
    best = {m: int(np.argmin(C[m]["p90"])) for m in C}      # 안전은 꼬리로 판단한다
    print("     결정 시점[mm]      " + " ".join(f"{d*1e3:5.0f}" for d in d_decs))
    for mode, name in (("hold", "추정없이 그때 flip"), ("ls", "위치, 데이터만(LS)"),
                       ("map", "위치, 데이터+사전"), ("map5", "위치+방향, 데이터+사전")):
        print(f"     [{name:<18}] p90  " + " ".join(f"{v*1e3:5.2f}" for v in C[mode]["p90"]))
    o_med, o_p90, o_bad = (float(np.median(e_open)), float(np.percentile(e_open, 90)),
                           float(np.mean(e_open > TOL)))
    bh, bl, bm, b5 = best["hold"], best["ls"], best["map"], best["map5"]
    print(f"  → **U 자다.** 너무 이르면 추정이 없고, 너무 늦으면 이미 지나쳐서 못 고친다"
          f"(LS p90 {C['ls']['p90'][0]*1e3:.2f} @{d_decs[0]*1e3:.0f} mm → "
          f"{C['ls']['p90'][-1]*1e3:.2f} mm @{d_decs[-1]*1e3:.0f} mm).")
    print(f"\n  **절제 — 좋아진 것이 무엇 덕인가** (중앙값 / p90 / 허용치 {TOL*1e3:.1f} mm 초과):")
    rows = [("열린 루프: 공칭 r=1 (exp 48)", e_open),
            ("사전지식만: 모집단 평균 r̄, 측정 없음", e_pri),
            (f"타이밍만: {d_decs[bh]*1e3:.0f} mm 에서 그냥 flip",
             np.array([run_patient(None, r, d_decs[bh], mode="hold")[0] for r in rs])),
            (f"데이터만(LS) @{d_decs[bl]*1e3:.0f} mm",
             np.array([run_patient(np.random.default_rng([seed, i]), r, d_decs[bl],
                                   mode="ls")[0] for i, r in enumerate(rs)])),
            (f"위치, 데이터+사전 @{d_decs[bm]*1e3:.0f} mm",
             np.array([run_patient(np.random.default_rng([seed, i]), r, d_decs[bm],
                                   mode="map")[0] for i, r in enumerate(rs)])),
            (f"**위치+방향**, 데이터+사전 @{d_decs[b5]*1e3:.0f} mm",
             np.array([run_patient(np.random.default_rng([seed, i]), r, d_decs[b5],
                                   mode="map5")[0] for i, r in enumerate(rs)])),
            ("오라클: 참 r", e_orc)]
    for name, e in rows:
        print(f"     {name:<34} {np.median(e)*1e3:5.2f} / {np.percentile(e,90)*1e3:5.2f} mm / "
              f"{np.mean(e>TOL)*100:3.0f}%")
    p_pri = float(np.percentile(e_pri, 90))
    p_map = C["map"]["p90"][bm]
    print(f"  → **측정이 기여한 몫은 거의 없다.** 사전지식만(측정 0회) p90 {p_pri*1e3:.2f} mm 에서 "
          f"MAP(측정 포함) {p_map*1e3:.2f} mm — 차이 {abs(p_pri-p_map)*1e3:.2f} mm.")
    print(f"     열린 루프 대비 개선({o_p90*1e3:.2f} → {p_map*1e3:.2f} mm)의 실체는 "
          f"**정보가 아니라 ① 더 맞는 기본값(r=1 → r̄={R_PRIOR_MU:.2f})** 과 "
          f"**② flip 을 늦추는 타이밍 제약**이다.")
    print(f"     [B] 와 일관된다: 결정 시점에서 σ(κ̂₂)≈35/m 인데 사전분포 표준편차는 "
          f"{KAPPA1*R_PRIOR_SD:.2f}/m — 우도가 사전분포를 이길 수 없다.")
    print(f"     '폐루프로 바꿨더니 좋아졌다'를 절제 없이 보고했다면 **정보 덕이라고 오독**했을 것이다.")
    p5 = C["map5"]["p90"][b5]
    print(f"\n  **그럼 무엇을 재야 하는가** — 팁 **위치**는 곡률의 이중적분이라 정보가 (S−s_b)²/2 로")
    print(f"  늦게 온다. 5-DOF 센서가 주는 **방향**은 1차 적분이라 (S−s_b) 로 선형 증가한다.")
    print(f"     위치만: p90 {p_map*1e3:.2f} mm (사전지식만 {p_pri*1e3:.2f} 과 차이 없음)")
    print(f"     위치+방향: p90 **{p5*1e3:.2f} mm** — 오라클 {np.percentile(e_orc,90)*1e3:.2f} mm "
          f"까지의 간극을 {(p_pri-p5)/max(p_pri-np.percentile(e_orc,90),1e-12)*100:.0f}% 메운다.")
    print(f"     **센서를 바꾸는 것이 추정기를 바꾸는 것보다 컸다** — VERIFICATION H3 의 "
          f"'needs tip tracking' 은 정확히는 '**방향까지 주는** tip tracking' 이다.")

    # ------------------------------------------------------------------ #
    # D. 센서가 나빠지면 폐루프가 진다 (exp 53 의 교훈)
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print("[D] 병목은 센싱이 아니라 **작동(actuation)** 이었다 — duty cycling 과 비교")
    print("     flip 은 한 번뿐이라 그 순간 권한이 소진된다. duty cycling(exp 48 의 다른 정책)은")
    print("     남은 구간의 유효 곡률을 계속 조절하므로 **늦게 온 정보도 쓸 수 있다.**")
    replans = {
        "1회 @22mm": (22e-3,),
        "2회 @22,35": (22e-3, 35e-3),
        "4회 @22,32,42,52": (22e-3, 32e-3, 42e-3, 52e-3),
    }
    print("     정책                                중앙값 /   p90 / 초과%")
    D_rows = []
    for name, deps in replans.items():
        e = np.array([tip_error(KAPPA1, KAPPA1 * r,
                                duty_schedule(np.random.default_rng([seed, m]),
                                              KAPPA1, KAPPA1 * r, deps))
                      for m, r in enumerate(rs)])
        D_rows.append((name, float(np.median(e)), float(np.percentile(e, 90)),
                       float(np.mean(e > TOL))))
        print(f"     duty, 재계획 {name:<22} {np.median(e)*1e3:5.2f} / "
              f"{np.percentile(e,90)*1e3:5.2f} / {np.mean(e>TOL)*100:3.0f}%")
    deps4 = replans["4회 @22,32,42,52"]
    e_dk = np.array([tip_error(KAPPA1, KAPPA1 * r,
                               duty_schedule(None, KAPPA1, KAPPA1 * r, deps4, known=True))
                     for r in rs])
    print(f"     duty, 4회 + **참 κ 를 안다고 가정**       {np.median(e_dk)*1e3:5.2f} / "
          f"{np.percentile(e_dk,90)*1e3:5.2f} / {np.mean(e_dk>TOL)*100:3.0f}%")
    print(f"     (비교) flip 최선(위치+방향)              {np.median(rows[5][1])*1e3:5.2f} / "
          f"{p5*1e3:5.2f} / {np.mean(rows[5][1]>TOL)*100:3.0f}%")
    print(f"     (비교) 열린 루프                        {o_med*1e3:5.2f} / {o_p90*1e3:5.2f} / "
          f"{o_bad*100:3.0f}%")
    duty_best = min(D_rows, key=lambda z: z[2])
    print(f"  → **재계획 1회짜리 duty 는 flip 과 똑같다**({D_rows[0][2]*1e3:.2f} vs "
          f"{p5*1e3:.2f} mm). 명령이 포화해 u=−1 이 되면 duty 가 곧 flip 이기 때문이다 —")
    print(f"     즉 이득은 '다르게 조작'이 아니라 **'다시 조작'** 에서 나온다. 재계획을 늘리면 "
          f"{D_rows[0][2]*1e3:.2f} → **{D_rows[-1][2]*1e3:.2f} mm**(중앙값 "
          f"{D_rows[0][1]*1e3:.2f} → {D_rows[-1][1]*1e3:.2f} mm).")
    print(f"     같은 센서, 같은 추정기, **한 번 더 고칠 수 있게 했을 뿐이다.** 늦게 도착한 정보가 "
          f"이제 쓰인다 — 정보가 늦었던 게 아니라 **쓸 기회가 한 번뿐이었다.**")
    gap_est = duty_best[2] - float(np.percentile(e_dk, 90))
    print(f"  → 그리고 남은 오차의 출처가 바뀐다: 참 κ 를 줘도 p90 "
          f"{np.percentile(e_dk,90)*1e3:.2f} mm 이라, duty {duty_best[2]*1e3:.2f} mm 중 "
          f"**추정 탓은 {gap_est*1e3:.2f} mm 뿐**이고 나머지는 이산 재계획·포화·소각 모델 오차다.")
    print("     **추정은 더 이상 병목이 아니다** — 그래서 아래처럼 센서를 좋게 해도 거의 안 움직인다.")

    # ---- 센서 품질: 이제는 그것이 병목이 아니라는 것을 보이는 표 ----
    print(f"\n     센서 품질의 영향 (duty 4회 재계획, p90 mm):")
    print("     σ[mm] \\ 지연[mm]" + "".join(f"{l*1e3:9.0f}" for l in LAG_SWEEP))
    D_grid = np.zeros((len(SIGMA_SWEEP), len(LAG_SWEEP)))
    for i, sg in enumerate(SIGMA_SWEEP):
        for j, lg in enumerate(LAG_SWEEP):
            e = np.array([tip_error(KAPPA1, KAPPA1 * r,
                                    duty_schedule(np.random.default_rng([seed, i, j, m]),
                                                  KAPPA1, KAPPA1 * r, deps4,
                                                  sigma=sg, lag=lg))
                          for m, r in enumerate(rs)])
            D_grid[i, j] = float(np.percentile(e, 90))
        print(f"     {sg*1e3:5.1f}          " + "".join(f"{v*1e3:8.2f} " for v in D_grid[i]))
    print(f"  → σ 를 10배 좋게 해도({SIGMA_SWEEP[-1]*1e3:.1f} → {SIGMA_SWEEP[0]*1e3:.1f} mm) "
          f"{D_grid[-1,0]*1e3:.2f} → {D_grid[0,0]*1e3:.2f} mm. **센서를 사는 것은 이제 낭비다** —")
    print(f"     남은 {duty_best[2]*1e3:.2f} mm 는 작동 쪽(재계획 간격·포화)과 모델에 있다.")
    print("     exp 42~53 을 관통하는 '지배항은 뒤집힌다'가 여기서 한 번 더: 작동을 고치자")
    print("     병목이 센서에서 **제어 이산화**로 옮겨 갔다. 다음에 돈을 쓸 곳이 바뀐 것이다.")

    # ---- 그림 ----
    fig, axg = plt.subplots(2, 3, figsize=(16.8, 9.2))
    axes = axg.ravel()

    ax = axes[0]
    for r, c in ((0.4, "royalblue"), (1.0, "0.4"), (2.5, "crimson")):
        k2 = KAPPA1 * r
        _, x, y = simulate(KAPPA1, k2, float(plan_flip(KAPPA1, k2)))
        ax.plot(x * 1e3, y * 1e3, color=c, label=f"r={r}, optimal flip")
        _, x2, y2 = simulate(KAPPA1, k2, float(D_NOMINAL))
        ax.plot(x2 * 1e3, y2 * 1e3, color=c, ls=":", alpha=0.8)
    ax.axvline(S_BOUND * 1e3, color="seagreen", ls="--", lw=1, label="tissue boundary")
    ax.axhline(0, color="0.8", lw=1)
    ax.set_xlabel("insertion depth [mm]"); ax.set_ylabel("lateral deviation [mm]")
    ax.set_title("Solid = optimal flip, dotted = nominal 29.3%", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    rr = np.linspace(*R_RANGE, 60)
    ax.plot(rr, [plan_flip(KAPPA1, KAPPA1 * r) / L_INS * 100 for r in rr], color="crimson")
    ax.axhline(D_NOMINAL / L_INS * 100, color="0.4", ls="--", lw=1,
               label="open loop (29.3%)")
    ax.axvline(1.0, color="0.8", lw=1)
    ax.set_xlabel("tissue curvature ratio r = κ₂/κ₁")
    ax.set_ylabel("optimal flip depth [% of insertion]")
    ax.set_title("Layers make the optimum depend on the tissue", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    dd = np.array([b[0] for b in B_rows]) * 1e3
    sg = np.array([b[3] for b in B_rows])
    ax.semilogy(dd, np.where(np.isfinite(sg), sg, np.nan), "-o", color="royalblue",
                label="σ(κ̂₂) from tip tracking")
    ax.axhline(KAPPA1, color="crimson", ls="--", lw=1, label="κ₁ itself (no information)")
    ax.axvline(D_NOMINAL * 1e3, color="0.4", ls=":", lw=1.2, label="nominal flip depth")
    ax.set_xlabel("decision depth [mm]"); ax.set_ylabel("σ(κ̂₂) [1/m]")
    ax.set_title("The estimate is worst exactly when you must decide", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(d_decs * 1e3, C["ls"]["p90"] * 1e3, "-o", color="crimson",
            label="closed loop, data only (p90)")
    ax.plot(d_decs * 1e3, C["map"]["p90"] * 1e3, "-o", color="royalblue",
            label="closed loop, data + prior (p90)")
    ax.axhline(o_p90 * 1e3, color="0.35", ls="--", lw=1.2, label="open loop, prior only")
    ax.axhline(np.percentile(e_orc, 90) * 1e3, color="seagreen", ls=":", lw=1.2,
               label="oracle")
    ax.axhline(TOL * 1e3, color="0.3", ls="-", lw=0.8, alpha=0.6, label="tolerance")
    ax.plot(d_decs[bm] * 1e3, C["map"]["p90"][bm] * 1e3, "*", ms=16, color="darkorange",
            zorder=5)
    ax.set_xlabel("decision depth [mm]"); ax.set_ylabel("tip deviation, p90 [mm]")
    ax.set_title("Too early: no estimate. Too late: no authority.", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[4]
    names_d = ["open\nloop", "flip\n(best)"] + [f"duty\n×{len(v)}" for v in replans.values()]
    vals_d = [o_p90 * 1e3, p5 * 1e3] + [d[2] * 1e3 for d in D_rows]
    ax.bar(np.arange(len(vals_d)), vals_d,
           color=["crimson", "darkorange"] + ["royalblue"] * len(D_rows), alpha=0.8)
    ax.axhline(np.percentile(e_orc, 90) * 1e3, color="seagreen", ls=":", lw=1.2,
               label="oracle")
    ax.axhline(TOL * 1e3, color="0.3", ls="--", lw=1.2, label="tolerance")
    ax.set_xticks(np.arange(len(vals_d))); ax.set_xticklabels(names_d, fontsize=7)
    ax.set_ylabel("tip deviation, p90 [mm]")
    ax.set_title("Same sensor, same estimator — different actuation", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

    ax = axes[5]
    names = ["open\n(r=1)", "prior\n(r̄)", "timing\nonly", "pos\n(LS)", "pos\n+prior",
             "pos+ori\n+prior", "oracle"]
    vals = [np.percentile(e, 90) * 1e3 for _, e in rows]
    cols = ["crimson", "0.5", "0.7", "darkorange", "royalblue", "navy", "seagreen"]
    ax.bar(np.arange(len(vals)), vals, color=cols, alpha=0.8)
    ax.axhline(TOL * 1e3, color="0.3", ls="--", lw=1.2, label="tolerance")
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("tip deviation, p90 [mm]")
    ax.set_title("Ablation: what actually bought the improvement", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    fig.suptitle("54. Closed-loop bevel steering — the bottleneck was actuation, "
                 "not sensing", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "54_closed_loop_needle.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/54_closed_loop_needle.png, assets/54_closed_loop_needle.png")

    return dict(d_homogeneous=d_hom, A_rows=A_rows, B_rows=B_rows,
                duty={n: (m, p, b) for n, m, p, b in D_rows},
                duty_known_p90=float(np.percentile(e_dk, 90)),
                map5_p90=float(p5), prior_only_p90=float(p_pri),
                d_decs=d_decs.tolist(), C={m: {k: v.tolist() for k, v in d.items()}
                                          for m, d in C.items()},
                best=best, D_grid=D_grid, D_rows=D_rows,
                ablation={n: (float(np.median(e)), float(np.percentile(e, 90)),
                              float(np.mean(e > TOL))) for n, e in rows},
                open_med=o_med, open_p90=o_p90, open_bad=o_bad,
                prior_p90=float(np.percentile(e_pri, 90)),
                oracle_med=float(np.median(e_orc)), oracle_p90=float(np.percentile(e_orc, 90)),
                flip_no_flip=(e_noflip, e_flip), TOL=TOL, D_NOMINAL=D_NOMINAL)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 조직을 **두 층·경계 깊이 기지**로 뒀다. 실제 불균질성은 연속적이고 경계 위치도 부정확하며,
#     바늘이 층을 비스듬히 만나면 곡률 방향까지 바뀐다. 여기 결과는 "층 하나만 몰라도 열린 루프가
#     깨진다"는 하한이다.
#   - 곡률을 **깊이의 함수**로만 뒀다(속도 의존성 없음). 실제로는 삽입 속도·조직 이완이 얽힌다.
#   - 팁 측정을 **횡변위 직접 관측 + 등방 잡음 + 순수 지연**으로 모델링했다. 전자기 트래커는
#     금속 왜곡이 있고 초음파는 exp 53 처럼 깊이별로 나빠진다 — 그 두 가지는 여기 없다.
#   - 제어 입력이 **flip 한 번**뿐이다. 실제 needle steering 은 duty cycling 으로 연속 조향하고,
#     그러면 제어 권한이 깊이 끝까지 남아 "권한 소진" 긴장이 완화된다(대신 조직 손상 논의가 붙는다).
#     exp 48 이 duty 정책을 이미 갖고 있으니 다음 단계의 연결점이다.
#   - 추정기는 소각 선형 모델, 진짜 바늘은 비선형 적분이다. κL≈0.06 rad 규모라 차이가 작지만
#     0 은 아니며, 그 몫은 오라클 값(참 r 로 계획해도 0 이 아닌 잔차)에 들어가 있다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
