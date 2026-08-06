"""측정 설계: 무엇을, 얼마나 정확히 재야 이 결정이 갈리는가.

exp 59 는 "붙들 것인가 후퇴할 것인가"를 물어 놓고 **결정을 못 하고 끝냈다.** 답이 조직의
**미끄러짐 한계(F_slip)** 에 달려 있는데 그 값이 실측이 아니라 자릿수만 맞춘 추정이었기 때문이다.
그래서 남은 것이 제어가 아니라 **측정**으로 넘어갔고, 나는 "이제 그 값을 재러 가자"고 적었다.

측정을 하러 가기 전에 하는 일이 있다. **그 측정이 결정을 가르기는 하는지**를 먼저 계산하는 것이다.
이 실험이 그것이고, 결과는 내 계획에 대한 반증이었다.

  A. **결정 지도** — (F_slip × 환자 움직임) 평면에서 붙들기와 후퇴 중 무엇이 이기는가.
     두 축(조직에 얹는 힘 / 맹행 거리)의 **교환비는 임상이 정하는 값**이라 스윕한다.
  B. **정보의 값** — 그래서 F_slip 을 알면 결정이 바뀌는가. → **그럴듯한 범위에서는 안 바뀐다.**
     결정을 뒤집는 것은 조직 값이 아니라 **임상 교환비**다. 내가 재려던 것이 구속변수가 아니었다.
     덤으로 **exp 59 가 적어둔 뒤집힘 조건이 틀렸다**는 것도 나왔다 — 미끄러짐 한계가 크면 후퇴가
     이긴다고 썼는데 반대다. **후퇴도 같은 파악 항을 거슬러 끈다**(끌리는 힘이 정확히 F_slip).
  C. **그래도 재야 한다면: 식별 가능성** — 평범한 삽입 로그로 F_slip 을 알 수 있는가.
     → 없다. 삽입 중에는 조직이 **계속 미끄러지므로** F_slip 이 절삭력과 상수로 합쳐진다.
     회귀 상수항이 **F_cut + F_slip 과 소수점까지 같다** — 완전 교락이다.
  D. **여기(excitation) 사양** — 멈춰서 조직이 움직이게 하면 갈라지는데, **움직임이 충분할 때만**이다.
     양쪽이 다 미끄러지려면 왕복 진폭이 **2·F_slip / K_grip** 을 넘어야 하는데 참값을 모르니 그
     진폭을 미리 못 고른다. **진폭 사다리**로 푼다 — 값이 더 안 자라면 거기가 참값이다.

그리고 A~D 가 하나로 묶인다. 파악 항은 **K_grip × 상대 운동**에서 포화하고, 그 위로는 F_slip 이
시스템에 아무 영향도 주지 않는다. 그런데 측정을 막는 천장도 **정확히 같은 곱**이다 —
**같은 양이 위해와 관측 가능성을 동시에 자른다.** 그 위의 F_slip 은 해롭지도 않고 보이지도 않는다.
못 재는 이유와 안 중요한 이유가 같은 것이다.

exp 46 이 "모델 확장과 여기 설계는 한 수"였는데, 여기서는 **조직 파라미터에서 같은 일**이 벌어졌다.
산출물은 숫자가 아니라 **프로토콜과 그 사양, 그리고 그 측정을 지금 할 필요가 없다는 판단**이다.

    python scripts/60_measure_to_decide.py

한계·트레이드오프
  - 결정 지도는 exp 59 의 시나리오(연집 손실 채널·정지 정책·1 mm 여유) 하나에서 그린 것이다.
    다른 시술·다른 여유에서 경계가 어디로 갈지는 이 실험이 말하지 않는다.
  - '조직에 얹는 힘'을 정지 구간 안의 **힘 진폭**으로 쟀다. 처음에는 정지 시점 대비 증분을 썼는데
    **정지가 걸린 위상에 민감**해서(호흡 마루에서 멈추면 작게 나온다) 환자 움직임이 클수록 부하가
    작아지는 비단조가 나왔다. 다만 진폭도 최댓값 계열이라, 위해가 **누적·지속시간**이면 결론이
    달라질 수 있다 — **B 의 정보의 값은 이 지표 선택 위에 있다.**
  - 식별은 **이 모델이 참**이라는 전제 위에 있다. 실제 조직은 점탄성·이완이 있어 dwell 중 힘이
    시간에 따라 떨어진다(exp 53 이 남긴 미결 항목). 그러면 평탄부가 평탄하지 않다.
  - K_grip 은 결정에 거의 영향이 없지만 **프로토콜 설계에는 결정적**이다(필요 진폭 ∝ 1/K_grip).
    이 이중성 때문에 "결정에 안 중요한 파라미터"를 "안 재도 되는 파라미터"로 읽으면 안 된다.
  - 힘센서 잡음만 넣었다. 실제로는 도구 관성·마찰이 팁 힘 추정에 섞이고, 그쪽이 대개 더 크다.
  - 여기 진폭 20~60 mm 는 **식별용 수치일 뿐 시술 중에 할 동작이 아니다.** 실제로는 팬텀·적출
    조직에서 별도로 재야 하고, 이 실험이 정하는 것은 그 벤치 시험의 사양이다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
jc = import_module("56_jittery_channel")
bc = import_module("57_bursty_channel")
s59 = import_module("59_what_is_safe_state")

DT = jc.DT
X_SURFACE = jc.X_SURFACE
BREATH_HZ = s59.BREATH_HZ
K_GRIP0, F_SLIP0 = s59.K_GRIP, s59.F_SLIP
N_SEEDS = 6

# 교환비: 조직에 1 N 을 덜 얹기 위해 맹행 몇 mm 를 낼 용의가 있는가.
# **임상이 정하는 값**이라 하나로 못 박지 않고 스윕한다(exp 58 의 '선언된 여유'와 같은 구조).
TRADES = (1.0, 5.0, 20.0, 100.0)
FSS = (0.4, 0.8, 1.6, 3.2, 6.4)
BRS = (2.0, 5.0, 10.0)


class TunableTissue(s59.GrippingTissue):
    """exp 59 의 파악 모델에 **절삭 기저 배율**을 더한 것. 결정 지도를 그리려면 두 축이 필요하다.

    관통 후 절삭+마찰 항 전체에 배율을 건다("이 조직이 얼마나 저항하는가"). 파악 항은 따로 둔다.
    배율 1.0 이면 exp 59 의 GrippingTissue 와 완전히 같다(테스트로 고정).
    """

    def __init__(self, k_grip=K_GRIP0, f_slip=F_SLIP0, cut_scale=1.0):
        super().__init__(k_grip=k_grip, f_slip=f_slip)
        self.cut_scale = cut_scale

    def force(self, x):
        base = jc.tele.Tissue.force(self, x)
        if not self.punctured:
            self.anchor = x
            return base
        base = base * self.cut_scale
        if self.anchor is None:
            self.anchor = x
        f_el = self.k * (x - self.anchor)
        if abs(f_el) > self.f_slip:
            self.anchor = x - np.sign(f_el) * self.f_slip / self.k
            f_el = np.sign(f_el) * self.f_slip
        return base - f_el


def policy(f_slip, breath_mm, cut_scale=1.0, retract=False, seeds=N_SEEDS):
    """exp 59 의 조건에서 정지 정책 하나를 돌려 (조직에 얹은 몫[N], 맹행[mm])을 낸다."""
    load, blind = [], []
    for s in range(seeds):
        r = bc.run("tdpa", seed=s, tail_ms=s59.TAIL_MS, loss=0.10,
                   burst_len=s59.BURST_MS, estop=True, resume_ms=60.0, blind_mm=1.0,
                   breath_mm=breath_mm, breath_hz=BREATH_HZ,
                   retract_mm=(s59.RETRACT_MM if retract else 0.0),
                   tissue_obj=TunableTissue(f_slip=f_slip, cut_scale=cut_scale))
        if not r["diverged"]:
            load.append(r["f_e_held_swing"])
            blind.append(r["blind_max_mm"])
    return (float(np.median(load)) if load else np.nan,
            float(np.median(blind)) if blind else np.nan)


def flip_trade(lh, bh, lr, br):
    """후퇴가 이기기 시작하는 교환비 [mm/N]. 아낀 힘이 0 이하면 어떤 교환비에서도 안 이긴다."""
    saved, cost = lh - lr, br - bh
    if not np.isfinite(saved) or not np.isfinite(cost) or saved <= 0:
        return np.inf
    if cost <= 0:
        return 0.0
    return cost / saved


# --------------------------------------------------------------------------- #
# C·D 를 위한 식별
# --------------------------------------------------------------------------- #
def insertion_log(f_slip, k_grip=K_GRIP0, cut_scale=1.0, dwell=False,
                  breath_mm=5.0, exc_mm=0.0, noise_N=0.0, seed=0, depth_mm=60.0):
    """삽입 로그. dwell=True 면 중간에 멈춘다. exc_mm>0 이면 **도구를 일부러 왕복**시킨다.

    힘은 **부호 있는 저항력**(양수 = 조직이 삽입을 막는 방향)으로 낸다. 파악 항이 절삭 기저보다
    크면 실제로 부호가 뒤집히므로 절댓값을 쓰면 그 구간이 접혀 버린다.
    왕복 진폭은 침투 깊이보다 작아야 한다 — 안 그러면 **도구가 조직 밖으로 나가** 힘이 0 이 된다.

    반환: (상대 침투 d[m], 저항력[N], dwell 마스크)
    """
    rng = np.random.default_rng(1000 + seed)
    ts = TunableTissue(k_grip=k_grip, f_slip=f_slip, cut_scale=cut_scale)
    x, xs, fs, mask = X_SURFACE, [], [], []
    n_push = 4000
    for _ in range(n_push):                        # 등속 삽입
        x += depth_mm * 1e-3 / n_push
        xs.append(x - X_SURFACE); fs.append(-ts.force(x)); mask.append(False)
    if dwell:
        for k in range(6000):
            t = k * DT
            surf = breath_mm * 1e-3 * np.sin(2 * np.pi * BREATH_HZ * t)
            # 호흡과 **다른 주파수**여야 한다. 같은 주파수면 둘이 상쇄돼 상대 운동이 0 이 되고,
            # 여기를 키웠는데 신호가 사라지는 일이 벌어진다(실제로 밟았던 함정).
            exc = exc_mm * 1e-3 * np.sin(2 * np.pi * 0.11 * t)
            xr = x + exc - surf                    # 조직 기준 상대 위치
            xs.append(xr - X_SURFACE); fs.append(-ts.force(xr)); mask.append(True)
    f = np.asarray(fs)
    if noise_N:
        f = f + rng.normal(0.0, noise_N, f.shape)
    return np.asarray(xs), f, np.asarray(mask)


def fit_from_insertion(d, f):
    """삽입 구간만으로 맞춘다: F ≈ a + b·d.

    **a 에 절삭력과 F_slip 이 합쳐져 들어간다**(분리 불가). 하지만 **b 는 마찰 기울기 MU 로
    깨끗하게 나온다** — 삽입 중에는 파악 항이 상수라 기울기에 안 섞이기 때문이다.
    그 b 를 dwell 추정의 보정에 쓴다: 못 얻는 것과 얻는 것을 갈라 쓰는 것이다.

    **관통 전 구간을 넣으면 안 된다** — 비선형 강성(K1·d + K2·d²)이 회귀를 지배해서 a·b 가 둘 다
    무의미해진다. 깊은 쪽 절반만 쓴다.
    """
    sel = d > 0.5 * d.max()
    dd, ff = d[sel], f[sel]
    A = np.stack([np.ones_like(dd), dd], axis=1)
    coef, *_ = np.linalg.lstsq(A, ff, rcond=None)
    return float(coef[0]), float(coef[1])


def _slope(u, g):
    if u.size < 20 or np.ptp(u) < 1e-9:
        return np.nan
    uu, gg = u - u.mean(), g - g.mean()
    return float(abs(np.linalg.lstsq(uu[:, None], gg, rcond=None)[0][0]))


def fit_from_dwell(d, f, mask, mu=0.0):
    """dwell 구간에서 F_slip·K_grip 을 뽑는다.

    멈춘 동안 상대 침투가 왕복하면 힘이 히스테리시스 고리를 그린다 — 붙어 있는 구간의 기울기가
    K_grip, 양쪽이 다 미끄러졌을 때 반진폭이 F_slip 이다. **다 미끄러지지 않았으면 하한이다.**

    mu: 삽입 구간에서 얻은 **마찰 기울기**. 침투가 왕복하면 마찰항도 같이 움직여 고리를 기울이므로
        빼 준다(안 빼면 F_slip 이 부풀고 기울기 판정도 흐려진다).

    반환: (F_slip 추정 또는 하한, K_grip 추정)
    """
    dd, ff = d[mask], f[mask]
    if dd.size < 200:
        return np.nan, np.nan
    u, g = dd - dd[0], ff - ff[0]
    g = g - mu * u                                   # 깊이 의존 마찰을 제거
    amp = float(np.ptp(g))
    if amp < 1e-9:
        return 0.0, np.nan
    c = 0.5 * (g.max() + g.min())
    mid = np.abs(g - c) < 0.30 * amp                 # 붙어 있는 구간
    return amp / 2.0, _slope(u[mid], g[mid])


LADDER = (0.0, 5.0, 10.0, 20.0, 40.0, 60.0)


def identify_ladder(f_slip, amps=LADDER, tol=0.05, **kw):
    """**진폭 사다리** — 추정이 더 안 자랄 때까지 왕복 진폭을 키운다.

    한 번의 고리 모양으로 '포화했나'를 판정하려 했지만, 큰 왕복 위에 호흡이 겹치면 평탄부에도
    되붙음 리플이 생겨 모양 판정이 실패한다(실제로 밟았다). **두 진폭에서 같은 값이 나오는가**로
    묻는 쪽이 견고하고, 그대로 프로토콜이 된다 — **참값을 몰라도 수렴을 알 수 있다.**

    반환: (추정, 수렴한 진폭[mm], 수렴했는가, 사다리 값들)
    """
    d0, f0, _ = insertion_log(f_slip, dwell=False, **kw)
    _, mu = fit_from_insertion(d0, f0)
    ests = []
    for a in amps:
        d, f, m = insertion_log(f_slip, dwell=True, exc_mm=a, **kw)
        ests.append(fit_from_dwell(d, f, m, mu=mu)[0])
    for i in range(1, len(ests)):
        if ests[i] <= ests[i - 1] * (1.0 + tol) and ests[i] > 1e-6:
            return ests[i], amps[i], True, ests
    return ests[-1], amps[-1], False, ests


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    fss = (0.8, 3.2) if quick else FSS
    brs = (2.0, 10.0) if quick else BRS
    print("=== 60. 측정 설계: 무엇을, 얼마나 정확히 재야 이 결정이 갈리는가 ===")
    print("exp 59 는 붙들기/후퇴를 결정하지 못하고 '이제 조직 값을 재러 가자'로 끝났다.")
    print("가기 전에 **그 측정이 결정을 가르는지**부터 계산한다. 결과는 그 계획에 대한 반증이었다.")

    # ---------------- A. 결정 지도 ----------------
    print("-" * 100)
    print("[A] 결정 지도 — (미끄러짐 한계 × 환자 움직임)에서 무엇이 이기는가")
    print("    교환비 w = 조직에 1 N 을 덜 얹으려고 낼 용의가 있는 맹행 [mm/N]. **임상이 정한다.**")
    A = {}
    print(f"{'F_slip[N]':>9s} | {'움직임[mm]':>9s} | {'붙들기 힘/맹행':>18s} | {'후퇴 힘/맹행':>18s} | "
          f"{'뒤집히는 w':>10s} | " + " | ".join(f"{'w=' + str(int(w)):>7s}" for w in TRADES))
    for fs_ in fss:
        for br in brs:
            lh, bh = policy(fs_, br, retract=False, seeds=seeds)
            lr, brr = policy(fs_, br, retract=True, seeds=seeds)
            w_star = flip_trade(lh, bh, lr, brr)
            A[(fs_, br)] = (lh, bh, lr, brr, w_star)
            cells = ["후퇴" if w >= w_star else "붙들기" for w in TRADES]
            ws = "안 뒤집힘" if not np.isfinite(w_star) else f"{w_star:.0f}"
            print(f"{fs_:9.1f} | {br:9.0f} | {lh:8.2f} N /{bh:7.2f} mm | "
                  f"{lr:8.2f} N /{brr:7.2f} mm | {ws:>10s} | "
                  + " | ".join(f"{c:>7s}" for c in cells))
    finite = [v[4] for v in A.values() if np.isfinite(v[4])]
    n_never = sum(1 for v in A.values() if not np.isfinite(v[4]))
    if finite:
        print(f"  후퇴가 이기려면 교환비가 최소 {min(finite):.0f} mm/N 은 되어야 하고, "
              f"{n_never}/{len(A)} 조합은 **힘까지 더 나쁘다**(어떤 교환비에서도 안 이긴다).")
    print("  후퇴는 **정보 없이 하는 움직임**이라 맹행 대가가 크고 조직과 무관하게 거의 일정한데,")
    print("  아끼는 힘은 파악 항이 포화해서 유계다. **비대칭이 구조적이다.**")

    # ---------------- B. 정보의 값 ----------------
    print("-" * 100)
    print("[B] 그래서 F_slip 을 재면 결정이 바뀌는가 — **정보의 값을 먼저 계산한다**")
    print(f"{'교환비 w':>9s} | {'F_slip 별 결정':>52s} | {'F_slip 이 결정을 바꾸는가':>24s}")
    B = {}
    br_ref = brs[len(brs) // 2]
    for w in TRADES:
        dec = ["후퇴" if w >= A[(f, br_ref)][4] else "붙들기" for f in fss]
        flips = len(set(dec)) > 1
        B[w] = (dec, flips)
        print(f"{w:9.0f} | {'  '.join(f'{f}N:{d}' for f, d in zip(fss, dec)):>52s} | "
              f"{('바뀐다' if flips else '**안 바뀐다**'):>24s}")
    hi, mid_ = A[(fss[-1], br_ref)], A[(fss[-2], br_ref)]
    print("  낮은 교환비(맹행이 비싼 쪽)에서는 **F_slip 을 아무리 정확히 알아도 결정이 같다.**")
    print("  높은 쪽에서 일부 셀이 뒤집히지만 그때조차 힘 차이는 스윙 3~4 N 대비 0.1~0.6 N 이라")
    print("  주변적이다. 결정을 실제로 뒤집는 것은 조직 값이 아니라 **교환비**이고, 그건 측정이")
    print("  아니라 **임상에서 받아야 하는 선언**이다(exp 58 '선언된 여유', exp 57 '허용 지연').")
    print("  → **내가 재러 가려던 것이 구속변수가 아니었다.** 측정 비용을 쓰기 전에 정보의 값을")
    print("    계산하는 것이 순서다.")
    print()
    print("  그리고 **exp 59 가 적어둔 뒤집힘 조건이 틀렸다.** 거기서는 '미끄러짐 한계가 크면 후퇴가")
    print("  이긴다'고 썼는데 실제로는 반대다 — 큰 쪽일수록 후퇴가 **더** 나쁘다. 후퇴 자체가 파악")
    print("  항을 거슬러 끌기 때문이다(끌리는 힘이 정확히 F_slip). exp 59 는 붙들기 쪽만 보고")
    print("  '파악 몫이 유계'라고 추론했는데, 후퇴도 같은 항을 문다는 걸 빠뜨렸다.")
    print()
    print(f"  결정적으로 **F_slip {fss[-2]} N 과 {fss[-1]} N 의 결과가 소수점까지 같다**"
          f"({mid_[0]:.2f}/{mid_[1]:.2f} vs {hi[0]:.2f}/{hi[1]:.2f}).")
    print("  파악 항이 **K_grip × 상대 운동**에서 포화하기 때문이고, 그 위로는 F_slip 이 시스템에")
    print("  **아무 영향도 주지 않는다.** 그런데 C·D 에서 측정을 막는 천장이 **정확히 같은 곱**이다.")
    print("  → **같은 K_grip × 상대운동 이 위해와 관측 가능성을 동시에 자른다.** 그 위의 F_slip 은")
    print("    해롭지도 않고 보이지도 않는다 — 못 재는 이유와 안 중요한 이유가 같은 것이다.")

    # ---------------- C. 식별 가능성 ----------------
    print("-" * 100)
    print("[C] 그래도 재야 할 날이 온다면 — **평범한 삽입 로그로는 안 된다**")
    print(f"{'참 F_slip[N]':>12s} | {'삽입 상수항[N]':>15s} | {'참 F_cut+F_slip':>16s} | "
          f"{'삽입 기울기[N/m]':>16s} | {'F_slip 분리':>12s}")
    C = {}
    for fs_ in fss:
        d0, f0, _ = insertion_log(fs_, dwell=False)
        a0, mu0 = fit_from_insertion(d0, f0)
        C[fs_] = (a0, mu0)
        print(f"{fs_:12.1f} | {a0:15.3f} | {jc.tele.F_CUT + fs_:16.3f} | {mu0:16.1f} | "
              f"{'불가(교락)':>12s}")
    print("  삽입 중에는 조직이 **계속 미끄러지므로** 파악 항이 부호 고정 상수로 들어간다 —")
    print(f"  상수항이 **F_cut + F_slip 과 소수점까지 같다.** 완전 교락이라 회귀를 아무리 잘해도")
    print("  안 갈라진다. 절삭력을 따로 알지 못하면 F_slip 은 삽입 로그에 **없는 정보**다.")
    mus = [C[k][1] for k in C]
    print(f"  다만 **기울기는 깨끗하다** — 마찰 계수가 {min(mus):.1f}~{max(mus):.1f} N/m"
          f"(참값 {jc.tele.MU:.0f})로 F_slip 과 무관하게 나온다.")
    print("  그래서 삽입 로그는 버리는 게 아니라 **dwell 추정의 보정값을 준다**(깊이 의존 마찰 제거).")
    print("  **못 얻는 것과 얻는 것을 갈라 쓰는 것**이 식별 설계다.")

    # ---------------- D. 여기 사양 ----------------
    print("-" * 100)
    print("[D] 그러면 멈춰서 흔든다 — 단 **충분히 흔들 때만** 나온다")
    print(f"    양쪽이 다 미끄러지려면 왕복 진폭 > 2·F_slip/K_grip 이어야 한다"
          f"(K_grip={K_GRIP0:.0f} N/m). **참값을 모르니 이 값을 미리 못 고른다.**")
    print(f"    그래서 **진폭 사다리**를 올린다 — 추정이 더 안 자라면 거기가 참값이다.")
    ladder = LADDER if not quick else (0.0, 5.0, 20.0, 60.0)
    print(f"{'참 F_slip[N]':>12s} | {'필요 진폭[mm]':>12s} | "
          + " | ".join(f"{str(int(a)) + 'mm':>7s}" for a in ladder)
          + f" | {'수렴 진폭':>9s} | {'최종 오차':>9s}")
    D = {}
    for fs_ in fss:
        est, amp_c, conv, vals = identify_ladder(fs_, amps=ladder)
        D[fs_] = (est, amp_c, conv, vals)
        print(f"{fs_:12.1f} | {2.0 * fs_ / K_GRIP0 * 1e3:12.1f} | "
              + " | ".join(f"{v:7.2f}" for v in vals)
              + f" | {amp_c:8.0f}mm | {abs(est - fs_) / fs_ * 100:8.1f}%")
    cap = K_GRIP0 * (2 * 5.0e-3) / 2.0
    print(f"  **호흡만 쓰면 참값과 무관하게 {cap:.2f} N 에서 잘린다** — 그게 우연이 아니라")
    print(f"  K_grip × (호흡 진폭 p-p) / 2 = {K_GRIP0:.0f} × 0.010 / 2 이기 때문이다. 큰 조직에서는")
    print("  '재 봤더니 1.5 N' 이 나오는데 그건 조직이 아니라 **호흡 진폭을 잰 것**이다.")
    print("  수렴 진폭이 예측한 2·F_slip/K_grip 을 사다리 한 칸 안에서 따라간다.")
    print("  → **참값을 몰라도 수렴은 알 수 있다.** 한 번의 고리 모양으로 판정하려다 실패했는데")
    print("    (큰 왕복 위에 호흡이 겹치면 평탄부에 되붙음 리플이 생긴다), **두 진폭에서 같은 값이")
    print("    나오는가**로 바꾸니 견고해졌고 그대로 프로토콜이 됐다.")
    print("  → exp 46 의 '모델 확장과 여기 설계는 한 수'가 조직 파라미터에서 그대로 재현됐다.")

    print("-" * 100)
    print("[D2] 힘센서 잡음 — **사다리를 올라간 뒤에만** 따질 값이다")
    noise_fss = (0.4, 0.8)
    print(f"{'센서 잡음[N]':>11s} | " + " | ".join(f"{'F_slip=' + str(f) + ' N':>16s}"
                                                  for f in noise_fss))
    D2 = {}
    for noise in ([0.0, 0.05] if quick else [0.0, 0.01, 0.02, 0.05, 0.10]):
        cells = []
        for fs_ in noise_fss:
            errs = []
            mu0 = C[fs_][1] if fs_ in C else jc.tele.MU
            for s in range(max(seeds, 3)):
                d, f, m = insertion_log(fs_, dwell=True, exc_mm=20.0,
                                        noise_N=noise, seed=s)
                est, _ = fit_from_dwell(d, f, m, mu=mu0)
                errs.append((est - fs_) / fs_ * 100)
            e = float(np.median(errs))
            D2[(noise, fs_)] = e
            cells.append(f"{e:+6.1f}%")
        print(f"{noise:11.2f} | " + " | ".join(f"{c:>16s}" for c in cells))
    print("  반진폭을 최댓값-최솟값 차로 잡으면 잡음이 **양끝만 바깥으로 밀어내** 부호가 한쪽인")
    print("  과대추정 편향이 된다(잡음이 커질수록 단조 증가). 작은 F_slip 일수록 신호 대비 잡음이")
    print("  커서 심하다. 분위수를 쓰면 완화되지만 그러면 평탄부를 깎아 **과소**추정으로 넘어간다 —")
    print("  잡음 처리에서 편향의 부호를 고르는 문제이지 없애는 문제가 아니다.")

    # ---------------- E ----------------
    print("-" * 100)
    print("[E] 산출물 — 숫자가 아니라 **판단과 프로토콜**")
    print("  0. **위해와 관측 가능성이 같은 양에서 잘린다** — K_grip × 상대 운동. 그 위의 F_slip 은")
    print("     해롭지도 않고 보이지도 않는다. 못 재는 이유와 안 중요한 이유가 같다.")
    print("  1. **지금은 이 측정을 할 이유가 없다.** 그럴듯한 교환비 범위에서 결정이 같다.")
    print("     먼저 받아야 할 것은 조직 값이 아니라 **임상 교환비(mm/N)** 다.")
    print("  1b. 그리고 **exp 59 의 뒤집힘 조건을 정정한다** — 미끄러짐 한계가 클수록 후퇴가 이기는")
    print("     게 아니라 **더 나빠진다**. 후퇴도 같은 파악 항을 거슬러 끌기 때문이다.")
    print("  2. 재게 되면: **평범한 삽입 로그로는 안 된다.** 상수항이 F_cut + F_slip 과 소수점까지")
    print("     같다 — 완전 교락이다. 프로토콜에 **dwell** 이 있고 그 동안 상대 운동이 있어야 한다.")
    print("     (삽입 구간은 버리지 않는다 — **마찰 기울기**를 줘서 dwell 추정을 보정한다.)")
    print("  3. 그 dwell 의 **왕복 진폭 > 2·F_slip/K_grip**. 참값을 모르니 미리 못 고르는데,")
    print("     **진폭 사다리를 올려 값이 안 자라면 거기가 참값**이다. 참값을 몰라도 수렴은 안다.")
    print(f"  4. **호흡만 믿으면 안 된다.** 참값과 무관하게 {cap:.2f} N 에서 잘리고, 그 숫자는 조직이")
    print("     아니라 K_grip × 호흡 진폭이다. 큰 조직일수록 그럴듯한 오답이 나온다.")
    print("  5. K_grip 은 결정에는 거의 무관하지만 **프로토콜 사양을 정한다**(필요 진폭 ∝ 1/K_grip).")
    print("     '결정에 안 중요한 값'을 '안 재도 되는 값'으로 읽으면 안 되는 자리다.")
    print("  6. 그리고 B 의 결론은 **'조직에 얹는 힘'을 최댓값 증분으로 잰 선택 위에 있다.**")
    print("     위해가 누적·지속시간이면 정보의 값도 달라진다 — **지표가 정보의 값을 정한다.**")

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    for br, c in zip(brs, ("0.6", "tab:blue", "crimson")):
        ax.plot(fss, [A[(f, br)][0] - A[(f, br)][2] for f in fss], "-o", color=c,
                label=f"patient motion {br:.0f} mm")
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("slip limit F_slip [N]")
    ax.set_ylabel("tissue load saved by retracting [N]")
    ax.set_title("What retraction buys (it also costs ~6-10 mm blind)", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for br, c in zip(brs, ("0.6", "tab:blue", "crimson")):
        ax.plot(fss, [min(A[(f, br)][4], 1e4) for f in fss], "-o", color=c,
                label=f"patient motion {br:.0f} mm")
    ax.axhspan(0.5, 20.0, color="seagreen", alpha=0.12)
    ax.text(fss[0], 3.0, "plausible clinical range", fontsize=7, color="seagreen")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("slip limit F_slip [N]")
    ax.set_ylabel("exchange rate at which retracting wins [mm/N]")
    ax.set_title("Every tissue lands above the plausible range", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[0, 2]
    d0, f0, m0 = insertion_log(1.6, dwell=True, exc_mm=20.0)
    ax.plot(np.asarray(d0)[~m0] * 1e3, f0[~m0], lw=0.9, color="0.6", label="insertion")
    ax.plot(np.asarray(d0)[m0] * 1e3, f0[m0], lw=0.9, color="crimson", label="dwell")
    ax.set_xlabel("relative penetration [mm]"); ax.set_ylabel("axial tissue force [N]")
    ax.set_title("The hysteresis loop only opens when you stop", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ks = sorted(C)
    ax.plot(ks, [C[k][0] for k in ks], "-o", color="0.55",
            label="insertion only: fitted constant")
    ax.plot(ks, [jc.tele.F_CUT + k for k in ks], ":", color="0.3", lw=1.2,
            label="F_cut + F_slip (exactly the constant)")
    ax.plot(ks, [D[k][0] for k in ks], "-^", color="crimson",
            label="dwell + amplitude ladder")
    ax.plot(ks, ks, "--", color="seagreen", lw=1, label="true F_slip")
    ax.set_xlabel("true F_slip [N]"); ax.set_ylabel("estimate [N]")
    ax.set_title("Insertion gives their sum; only a dwell separates them", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1, 1]
    for k, c in zip(ks, plt.cm.viridis(np.linspace(0.1, 0.9, len(ks)))):
        ax.plot(ladder, D[k][3], "-o", ms=4, color=c, label=f"F_slip = {k} N")
        ax.axhline(k, color=c, ls=":", lw=0.8)
    ax.axhline(cap, color="crimson", ls="--", lw=1.2)
    ax.text(ladder[1], cap * 1.06, "breathing-only ceiling = K_grip x A / 2",
            fontsize=7, color="crimson")
    ax.set_xlabel("commanded dwell excursion [mm]"); ax.set_ylabel("estimate [N]")
    ax.set_title("Climb the ladder until it stops growing", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["What this experiment delivers", "",
             "1. DO NOT run the measurement yet.",
             "   Across every plausible exchange rate",
             "   the decision is the same. What is",
             "   missing is a clinical number (mm per N),",
             "   not a tissue number.", "",
             "2. If it is ever needed: a normal insertion",
             "   cannot give it. The fitted constant IS",
             "   F_cut + F_slip to three decimals - fully",
             "   confounded. The log needs a DWELL.",
             "   (Keep the insertion anyway: its slope is",
             "    a clean friction estimate that corrects",
             "    the dwell fit.)", "",
             "3. The dwell excursion must exceed",
             "   2 * F_slip / K_grip - which you cannot",
             "   pick in advance. Climb an amplitude",
             "   ladder until the estimate stops growing.",
             "   Convergence is knowable without truth.", "",
             "4. Do not trust breathing alone. It caps at",
             "   K_grip * A / 2 for ANY tissue, so a stiff",
             "   tissue returns a plausible wrong number:",
             "   the breathing amplitude, not the tissue.",
             "", "5. All of this rests on scoring harm as a",
             "   peak. Score it as dose and the value of",
             "   the measurement changes."]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8,
            family="monospace")

    fig.suptitle("60. Designing the measurement - and finding it is not the one worth "
                 "making yet", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "60_measure_to_decide.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/60_measure_to_decide.png, assets/60_measure_to_decide.png")

    return dict(A=A, B=B, C=C, D=D, D2=D2)


if __name__ == "__main__":
    main()
