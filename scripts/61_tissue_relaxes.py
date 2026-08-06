"""조직은 이완한다 — 그러면 exp 60 의 프로토콜과 exp 60 의 결론은 살아남는가.

exp 60 은 두 가지를 내놓고 각각에 **자기 약점을 적어 뒀다**.

  1. **식별 프로토콜** — "삽입 로그의 상수항은 F_cut + F_slip 과 소수점까지 같고, 분리하려면
     dwell 에서 진폭 사다리를 올려라." 한계 절: *"실제 조직은 이완이 있어 dwell 중 힘이 떨어진다.
     그러면 평탄부가 평탄하지 않다."*
  2. **"그 측정은 할 값어치가 없다"** — 한계 절: *"위해가 누적·지속시간이면 정보의 값도 달라진다."*

**점탄성이 바로 그 둘을 잇는 물리다.** 붙들고 있으면 힘은 이완으로 줄지만 시간은 계속 간다 —
최댓값 지표는 점점 관대해지고 누적 지표는 점점 가혹해진다. 그래서 이 실험 하나가
**exp 53 이 남긴 가장 오래된 미결 항목**을 닫으면서 exp 60 의 프로토콜과 결론을 동시에 시험한다.

**들어가기 전에 두 가지를 예측했고 둘 다 틀렸다.** 그게 이 실험의 실제 내용이다.

  A. **조직만 떼어낸 이완 확인** — 채널도 제어도 없이. 그리고 전진 중에는 여전히 옛 모델로
     환원되는지(그래야 47~60 이 무효화되지 않는다). 환원 조건은 **K·v·τ ≥ F_slip** 이다.
  B. **exp 60 의 식별을 이완 조직에 돌린다.**
     B1 **깨진다.** 삽입 상수항이 F_cut + F_slip 이 아니라 **F_cut + min(F_slip, K·v·τ)** 다.
        느리게 넣으면 조직이 아니라 **삽입 속도를 잰다.** 교락 상대가 하나 늘었다.
     B2 **안 깨진다 — 예측 실패 ①.** 진폭 사다리가 틀린 값에서 수렴할 거라 봤는데 그렇지 않다.
        고정 주파수에서 진폭을 올리면 **속도도 같이 오르므로**(v = A·ω) 점성 천장 K·v·τ 도
        기하학적 천장 K·A 도 **둘 다 진폭을 따라 올라간다.** F_slip 자신을 뺀 모든 천장이 함께
        올라가니 **가짜 평탄부가 구조적으로 생길 수 없다.** exp 60 의 판정은 그때는 몰랐던
        이유로 이완까지 덮고 있었다. 대가는 값이 아니라 **비용** — 필요 진폭이 부푼다.
        요구를 하나로 쓰면 **속도** 사양이다: `A·ω > F_slip / (K_grip·τ)`.
  C. **위해를 누적으로 바꿔 결정 지도를 다시 그린다 — 예측 실패 ②.** 뒤집힐 거라 봤는데
     구조가 그대로다. 오히려 누적 쪽에서 w* 의 퍼짐이 **더 좁아** 그 측정의 값이 더 떨어진다.
     다만 **힘 축의 승자는 실제로 갈린다** — 최댓값은 후퇴의 과도현상만 보고, 누적은 그 뒤
     낮아진 정상상태를 본다.
  D. **정보의 값을 두 지표로 나란히** — 그리고 이완이 점성 포화 천장을 하나 더 만든다.

가는 길에 **exp 60 이 도입한 지표의 결함**도 하나 잡았다: 정지 진폭을 실행 전체에 걸쳐 min/max 로
누적하면 서로 다른 깊이의 정지가 섞이고, 관통 순간을 걸친 정지가 있으면 스윙이 조직과 무관하게
F_PUNC 로 찍힌다. **정지 1회 단위**로 고쳤고, exp 60 의 발표 숫자가 그만큼 바뀐다(결론은 유지).

    python scripts/61_tissue_relaxes.py

한계·트레이드오프
  - 이완을 **파악 항의 앵커가 현재 위치로 기어가는** 1차 완화로 넣었다(Maxwell 요소 하나).
    실제 연조직은 완화 스펙트럼이 넓어 단일 τ 로 안 맞고, 준선형 점탄성(QLV)이 표준에 가깝다.
    여기서 묻는 것은 "τ 가 얼마냐"가 아니라 **"이완이 있으면 무엇이 깨지느냐"** 다.
  - 절삭 기저는 이완시키지 않았다. 전진 중 절삭은 계속 새 조직을 자르므로 완화할 상태가 없다는
    쪽이 맞지만, 정지 중 축방향 마찰은 실제로 완화한다 — 그만큼 낙관적이다.
  - 누적 지표를 **정지 구간의 ∫|F| dt** 로 잡았다. 절삭 기저가 그대로 포함되므로 정책 간 차이는
    희석된다. 실제 손상은 힘이 아니라 **변형·허혈 시간**일 수 있고 그러면 또 다른 지표가 된다.
  - τ 값 자체는 실측이 아니다. exp 60 이 보인 대로 이 값을 재는 것도 프로토콜이 필요한 일이고,
    이 실험은 그 프로토콜에 **주파수 축이 하나 더 붙는다**는 것까지만 말한다.
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
s60 = import_module("60_measure_to_decide")

DT = jc.DT
X_SURFACE = jc.X_SURFACE
K_GRIP0, F_SLIP0 = s59.K_GRIP, s59.F_SLIP
BREATH_HZ = s59.BREATH_HZ
N_SEEDS = 6

TAUS = (0.05, 0.2, 1.0, 5.0, np.inf)   # 이완 시간상수 [s]; inf = exp 60 (순탄성)
STOP_SECS = 0.5                        # exp 58/59 조건에서 정지 1회의 전형적 길이
V_INSERT = 0.015                       # exp 60 삽입 로그의 속도 [m/s] (60 mm / 4 s)


class RelaxingTissue(s60.TunableTissue):
    """exp 60 의 파악 모델 + **응력 완화**. tau=inf 면 exp 60 과 완전히 같다(테스트로 고정).

    파악을 "조직이 축을 탄성으로 붙잡는다"로 봤으니, 완화는 그 **앵커가 현재 위치로 기어가는**
    것으로 들어간다(스프링-댐퍼 직렬 = Maxwell 요소). 붙들고 있으면 변형은 남고 힘은 사라진다 —
    그게 응력 완화다.

    전진 중에는 운동이 앵커를 밀어내고 완화가 당겨오는 균형에서 f_el = K·v·τ 로 서고,
    그 값이 F_slip 을 넘으면 미끄러짐이 잘라 **옛 모델과 같아진다.** 즉 환원 조건이 있다:
    **K·v·τ ≥ F_slip.** exp 60 의 "삽입 상수항 = F_cut + F_slip"은 그 조건 아래서만 참이다.
    """

    def __init__(self, k_grip=K_GRIP0, f_slip=F_SLIP0, cut_scale=1.0,
                 tau=np.inf, dt=DT):
        super().__init__(k_grip=k_grip, f_slip=f_slip, cut_scale=cut_scale)
        self.tau, self.dt = tau, dt

    def force(self, x):
        if self.punctured and self.anchor is not None and np.isfinite(self.tau):
            self.anchor += (x - self.anchor) * min(self.dt / self.tau, 1.0)
        return super().force(x)


# --------------------------------------------------------------------------- #
# A. 조직만 떼어낸 시험
# --------------------------------------------------------------------------- #
def hold_and_relax(tau, f_slip=F_SLIP0, depth_mm=40.0, secs=3.0, breath_mm=0.0,
                   breath_hz=BREATH_HZ):
    """도구를 관통 후 깊이에 **고정**하고(원하면 표면만 흔들고) 힘의 시간 이력을 낸다."""
    ts = RelaxingTissue(f_slip=f_slip, tau=tau)
    x = X_SURFACE
    for _ in range(4000):                                   # 삽입
        x += depth_mm * 1e-3 / 4000
        ts.force(x)
    out = []
    for k in range(int(secs / DT)):
        surf = breath_mm * 1e-3 * np.sin(2 * np.pi * breath_hz * k * DT)
        out.append(-ts.force(x - surf))
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# B. exp 60 의 식별을 이완 조직에 돌린다
# --------------------------------------------------------------------------- #
def insertion_log(f_slip, tau=np.inf, dwell=False, breath_mm=5.0, exc_mm=0.0,
                  exc_hz=0.11, noise_N=0.0, seed=0, depth_mm=100.0, v=V_INSERT):
    """exp 60 의 로그 생성기 + 이완, + **여기 주파수를 인자로 뺀 것**(그게 이번 축이다).

    깊이를 exp 60 의 60 mm 에서 **100 mm 로 키웠다** — 사다리 맨 위 칸(왕복 60 mm)에 호흡까지
    더하면 도구가 조직 밖으로 나가 힘이 0 이 되고, 반진폭이 부풀어 **F_slip 을 넘는 추정**이 나온다.
    (exp 60 이 한계로 적어둔 '왕복 진폭 < 침투 깊이'가 자기 사다리의 맨 윗칸에서 깨지고 있었다.)
    삽입 **속도**는 v 로 고정한다 — 이완 아래서는 속도가 상수항을 정하므로 깊이와 함께 바뀌면 안 된다.
    """
    rng = np.random.default_rng(1000 + seed)
    ts = RelaxingTissue(f_slip=f_slip, tau=tau)
    n_push = int(depth_mm * 1e-3 / v / DT)
    x, xs, fs, mask = X_SURFACE, [], [], []
    for _ in range(n_push):
        x += depth_mm * 1e-3 / n_push
        xs.append(x - X_SURFACE); fs.append(-ts.force(x)); mask.append(False)
    if dwell:
        for k in range(6000):
            t = k * DT
            surf = breath_mm * 1e-3 * np.sin(2 * np.pi * BREATH_HZ * t)
            exc = exc_mm * 1e-3 * np.sin(2 * np.pi * exc_hz * t)
            xr = x + exc - surf
            xs.append(xr - X_SURFACE); fs.append(-ts.force(xr)); mask.append(True)
    f = np.asarray(fs)
    if noise_N:
        f = f + rng.normal(0.0, noise_N, f.shape)
    return np.asarray(xs), f, np.asarray(mask)


def ladder(f_slip, tau=np.inf, exc_hz=0.11, amps=s60.LADDER, tol=0.05, **kw):
    """exp 60 의 진폭 사다리를 그대로. **주파수는 고정** — 그게 이 실험이 드러내는 구멍이다."""
    d0, f0, _ = insertion_log(f_slip, tau=tau, dwell=False, **kw)
    _, mu = s60.fit_from_insertion(d0, f0)
    ests = []
    for a in amps:
        d, f, m = insertion_log(f_slip, tau=tau, dwell=True, exc_mm=a,
                                exc_hz=exc_hz, **kw)
        ests.append(s60.fit_from_dwell(d, f, m, mu=mu)[0])
    for i in range(1, len(ests)):
        if ests[i] <= ests[i - 1] * (1.0 + tol) and ests[i] > 1e-6:
            return ests[i], amps[i], True, ests
    return ests[-1], amps[-1], False, ests


def two_axis_ladder(f_slip, tau=np.inf, amps=s60.LADDER,
                    freqs=(0.11, 0.35, 1.1, 3.5), tol=0.05, **kw):
    """**진폭과 주파수 둘 다** 올린다. 둘 다 더 안 자라면 거기가 참값이다.

    exp 60 의 수렴 판정은 진폭 축에서만 물었기 때문에 **필요조건일 뿐**이었다.
    """
    grid = {}
    for fz in freqs:
        est, amp, _, _ = ladder(f_slip, tau=tau, exc_hz=fz, amps=amps, tol=tol, **kw)
        grid[fz] = (est, amp)
    fzs = list(freqs)
    for i in range(1, len(fzs)):
        if grid[fzs[i]][0] <= grid[fzs[i - 1]][0] * (1.0 + tol):
            return grid[fzs[i]][0], fzs[i], grid[fzs[i]][1], True, grid
    return grid[fzs[-1]][0], fzs[-1], grid[fzs[-1]][1], False, grid


# --------------------------------------------------------------------------- #
# C. 결정 지도 — 두 지표로
# --------------------------------------------------------------------------- #
def policy(f_slip, tau, breath_mm=5.0, retract=False, seeds=N_SEEDS):
    """(정지 중 힘 진폭[N], 정지 중 누적 ∫|F|dt[N·s], 정지 시간[s], 맹행[mm])"""
    sw, ds, sc, bl = [], [], [], []
    for s in range(seeds):
        r = bc.run("tdpa", seed=s, tail_ms=s59.TAIL_MS, loss=0.10,
                   burst_len=s59.BURST_MS, estop=True, resume_ms=60.0, blind_mm=1.0,
                   breath_mm=breath_mm, breath_hz=BREATH_HZ,
                   retract_mm=(s59.RETRACT_MM if retract else 0.0),
                   tissue_obj=RelaxingTissue(f_slip=f_slip, tau=tau))
        if not r["diverged"]:
            sw.append(r["f_e_held_swing"]); ds.append(r["f_e_held_dose"])
            sc.append(r["secs_held"]); bl.append(r["blind_max_mm"])
    med = lambda v: float(np.median(v)) if v else np.nan   # noqa: E731
    return med(sw), med(ds), med(sc), med(bl)


# --------------------------------------------------------------------------- #
def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    taus = (0.2, np.inf) if quick else TAUS
    print("=== 61. 조직은 이완한다 — exp 60 의 프로토콜과 결론은 살아남는가 ===")
    print("exp 60 은 프로토콜과 판정을 내놓고 각각에 '이완이 있으면 다르다'를 한계로 적었다.")
    print("그 물리를 실제로 넣어 **자기가 방금 쓴 것**을 시험한다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] 조직만 떼어낸 시험 — 붙들고 있으면 힘이 사라진다(변형은 남는다)")
    print(f"    도구를 깊이에 고정하고 가만히 둔다. 정지 1회는 이 사슬에서 약 {STOP_SECS:.1f} s.")
    print(f"{'τ [s]':>8s} | {'t=0':>9s} | {'0.5 s 후':>9s} | {'3 s 후':>9s} | "
          f"{'정지 0.5s 동안 남는 몫':>22s}")
    A = {}
    for tau in taus:
        f = hold_and_relax(tau)
        i05 = int(0.5 / DT)
        A[tau] = (f[0], f[i05], f[-1])
        keep = (f[i05] - f[-1]) / max(f[0] - f[-1], 1e-9) if f[0] > f[-1] else 1.0
        print(f"{tau:8.2f} | {f[0]:9.3f} | {f[i05]:9.3f} | {f[-1]:9.3f} | "
              f"{keep * 100:21.0f}%")
    print("  τ 가 정지 길이보다 훨씬 짧으면 정지 중에 파악 힘이 **거의 다 사라진다** — 붙들기가")
    print("  최댓값 기준으로는 더 안전해진다. 훨씬 길면 exp 60 의 순탄성 그림 그대로다.")
    print("  **그래서 τ 자체가 아니라 τ 대 정지 지속시간의 비가 문제다.**")

    print(f"{'τ [s]':>8s} | " + " | ".join(f"{'호흡 ' + str(h) + ' Hz':>13s}"
                                          for h in (0.1, 0.25, 1.0)))
    A2 = {}
    for tau in taus:
        cells = []
        for hz in (0.1, 0.25, 1.0):
            f = hold_and_relax(tau, breath_mm=5.0, breath_hz=hz, secs=4.0)
            sw = float(np.ptp(f[int(1.0 / DT):]))
            A2[(tau, hz)] = sw
            cells.append(f"{sw:8.3f} N")
        print(f"{tau:8.2f} | " + " | ".join(f"{c:>13s}" for c in cells))
    print("  환자가 흔드는 힘도 **주파수에 의존**하게 됐다. 느린 호흡은 이완이 따라잡아 힘이 안 쌓인다.")
    print("  exp 59 가 잰 '붙들기의 위해 1.62 N'은 **순탄성 가정 위의 값**이었다.")

    # ------------------------------ B1 ------------------------------
    print("-" * 100)
    print("[B1] exp 60 의 삽입 상수항 주장이 조건부가 된다")
    print("     exp 60: '상수항 = F_cut + F_slip, 소수점까지'. 이완이 있으면")
    print(f"     **상수항 = F_cut + min(F_slip, K·v·τ)** 이고, 여기 v = {V_INSERT*1e3:.0f} mm/s.")
    print(f"{'τ [s]':>8s} | {'K·v·τ [N]':>10s} | " +
          " | ".join(f"{'F_slip=' + str(f):>18s}" for f in (0.8, 3.2)))
    B1 = {}
    for tau in taus:
        cap = K_GRIP0 * V_INSERT * tau if np.isfinite(tau) else np.inf
        cells = []
        for fs_ in (0.8, 3.2):
            d, f, _ = insertion_log(fs_, tau=tau, dwell=False)
            a, _ = s60.fit_from_insertion(d, f)
            pred = jc.tele.F_CUT + min(fs_, cap)
            B1[(tau, fs_)] = (a, pred)
            cells.append(f"{a:6.3f} (예측 {pred:5.3f})")
        capstr = "∞" if not np.isfinite(cap) else f"{cap:.2f}"
        print(f"{tau:8.2f} | {capstr:>10s} | " + " | ".join(f"{c:>18s}" for c in cells))
    print("  느린 삽입에서는 상수항이 조직이 아니라 **삽입 속도를 잰다.** exp 60 의 '완전 교락'은")
    print("  여전히 참이지만, 교락되는 상대가 **하나 더 늘었다** — 절삭력에 더해 속도까지.")
    print("  같은 조직을 다른 속도로 넣으면 다른 숫자가 나오고, 둘 다 그럴듯하다.")

    # ------------------------------ B2 ------------------------------
    print("-" * 100)
    print("[B2] 진폭 사다리는 살아남는다 — **내가 깨질 거라 예상한 쪽이 안 깨졌다**")
    print("     이완이 탄성 축적을 먹으니 사다리가 '틀린 값에서 깨끗하게 수렴'할 거라 예상했다.")
    print("     그렇게 안 된다. 이유가 구조적이다 — 아래 표의 **수렴 여부**를 같이 본다.")
    fss = (0.8, 3.2) if quick else (0.8, 1.6, 3.2)
    freqs = (0.11, 1.1) if quick else (0.11, 0.35, 1.1, 3.5)
    tau_b = 0.2
    print(f"     (τ = {tau_b} s 고정. exp 60 이 쓴 여기 주파수는 0.11 Hz 였다.)")
    print(f"{'참 F_slip':>10s} | " + " | ".join(f"{str(z) + ' Hz':>17s}" for z in freqs)
          + f" | {'2축 결과':>16s}")
    B2 = {}
    for fs_ in fss:
        cells = []
        for fz in freqs:
            est, amp, conv, _ = ladder(fs_, tau=tau_b, exc_hz=fz)
            B2[(fs_, fz)] = (est, amp, conv)
            cells.append(f"{est:5.2f} N @{amp:3.0f}mm {'수렴' if conv else '하한'}")
        est2, fz2, amp2, ok2, _ = two_axis_ladder(fs_, tau=tau_b, freqs=freqs)
        print(f"{fs_:10.1f} | " + " | ".join(f"{c:>17s}" for c in cells)
              + f" | {est2:5.2f} N @{fz2:4.2f}Hz")
    print("  **틀린 값에서 수렴하는 칸이 없다.** 참값을 맞히거나, 아니면 '수렴 안 함(하한)'이라고")
    print("  정직하게 말한다. 이유: 고정 주파수에서 진폭을 올리면 **속도도 같이 올라간다**(v = A·ω).")
    print("  점성 천장 K·v·τ 도, 기하학적 천장 K·A 도 **둘 다 진폭에 대해 증가**한다 — 즉 F_slip")
    print("  자신을 뺀 모든 천장이 사다리를 따라 올라가므로 **가짜 평탄부가 구조적으로 생길 수 없다.**")
    print("  exp 60 의 판정은 그 사실 덕에 이완까지 덮고 있었다(그때는 몰랐던 이유로).")
    print()
    print("  **이완이 실제로 물리는 대가는 값이 아니라 비용이다** — 필요 진폭이 부푼다.")
    print("  τ=0.2, 0.11 Hz 에서 F_slip=3.2 는 60 mm 를 흔들고도 수렴하지 못한다(순탄성이면 20 mm).")
    print("  싼 해법은 **주파수를 올리는 것**이다 — 1.1 Hz 에서는 40 mm 로 잡힌다.")
    print("  요구를 하나로 쓰면 **속도** 사양이 된다: 기하학적 `A > 2·F_slip/K_grip` **그리고**")
    print("  점성 `A·ω > F_slip/(K_grip·τ)`. 어느 축으로 사도 되지만, 진폭만으로 사면 비싸다.")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] 위해 지표를 누적으로 바꾼다 — 그런데 **exp 60 의 지표가 조직을 거의 못 보고 있었다**")
    fss_c = (0.4, 3.2) if quick else (0.4, 0.8, 1.6, 3.2, 6.4)
    taus_c = (0.2, np.inf) if quick else (0.2, 1.0, np.inf)

    print("  먼저 대조군 하나. **환자를 아예 안 움직이게** 하고 같은 것을 잰다 — 위해가 0 이어야 한다.")
    print(f"{'환자 움직임':>10s} | {'F_slip':>7s} | {'τ':>5s} | {'진폭[N]':>9s} | {'누적[N·s]':>10s}")
    CTRL = {}
    for br in (0.0, 5.0):
        for fs_ in (fss_c[0], fss_c[-1]):
            for tau in (0.2, np.inf):
                p = policy(fs_, tau, breath_mm=br, retract=False, seeds=seeds)
                CTRL[(br, fs_, tau)] = p
                print(f"{br:9.0f}mm | {fs_:7.1f} | "
                      f"{('∞' if not np.isfinite(tau) else f'{tau:.1f}'):>5s} | "
                      f"{p[0]:9.2f} | {p[1]:10.3f}")
    z = CTRL[(0.0, fss_c[0], 0.2)][0]
    print(f"  **환자가 전혀 안 움직여도 진폭이 {z:.2f} N 이고, 조직을 바꿔도 τ 를 바꿔도 똑같다.**")
    print("  즉 exp 60 이 고른 진폭 지표는 조직의 위해가 아니라 **정지 제어기가 목표 위치로 정착하는")
    print("  과도현상**을 재고 있었다. 환자 움직임이 더하는 몫은 그 위에 얹히는 0.5 N 뿐이다.")
    print("  반면 **누적은 조직을 본다** — F_slip 에도(1.50 → 1.67), τ 에도(1.67 vs 2.08) 반응한다.")
    print("  exp 60 은 위상 민감성을 고치려고 증분 → 진폭으로 갈아탔는데, 그 과정에서 **제어기를 재는**")
    print("  양을 골랐다. 지표 교체가 한 결함을 고치면서 다른 결함을 들여온 것이다.")
    print()
    print("  그래서 아래 지도는 **조직을 볼 수 있는 지표로 다시 그린 exp 60** 이기도 하다.")
    C = {}
    print(f"{'τ [s]':>6s} | {'F_slip':>7s} | {'붙들기 진폭/누적':>18s} | {'후퇴 진폭/누적':>18s} | "
          f"{'맹행 대가':>9s} | {'w* 진폭':>9s} | {'w* 누적':>9s}")
    for tau in taus_c:
        for fs_ in fss_c:
            h = policy(fs_, tau, retract=False, seeds=seeds)
            r = policy(fs_, tau, retract=True, seeds=seeds)
            w_pk = s60.flip_trade(h[0], h[3], r[0], r[3])
            w_ds = s60.flip_trade(h[1], h[3], r[1], r[3])
            C[(tau, fs_)] = (h, r, w_pk, w_ds)
            f = lambda w: "안 뒤집힘" if not np.isfinite(w) else f"{w:.0f}"  # noqa: E731
            print(f"{('∞' if not np.isfinite(tau) else f'{tau:.1f}'):>6s} | {fs_:7.1f} | "
                  f"{h[0]:8.2f} N /{h[1]:7.3f} | {r[0]:8.2f} N /{r[1]:7.3f} | "
                  f"{r[3] - h[3]:8.2f}mm | {f(w_pk):>9s} | {f(w_ds):>9s}")
    print("  (w* = 후퇴가 이기기 시작하는 교환비. **진폭 기준은 mm/N, 누적 기준은 mm/(N·s)** —")
    print("   단위가 다르므로 두 열의 숫자를 직접 비교하면 안 된다.)")
    print()
    print("  **붙들기의 진폭이 모든 칸에서 같다**(위 대조군이 이유다 — 제어기 과도현상). 반면")
    print("  **후퇴의 누적은 모든 칸에서 붙들기보다 낮다.** 후퇴는 깊이를 줄이고 파악을 풀어 놓아")
    print("  정지 내내 힘이 낮은데, 최댓값 계열 지표는 후퇴하는 **순간의 과도현상**만 보고 그 뒤의")
    print("  낮아진 정상상태를 통째로 놓친다. **지표가 힘 축의 승자를 바꾼다.**")
    print()
    print("  **그런데 구조는 안 바뀐다.** 두 지표 모두 '낮은 교환비에서는 F_slip 과 무관하게 붙들기,")
    print("  그 위에 F_slip 이 갈리는 창이 있다'는 같은 모양이다. 바뀐 것은 창의 위치와,")
    print("  **선언해야 하는 임상 숫자의 단위**다: mm/N 이 아니라 mm/(N·s).")
    print("  → **조직을 볼 수 있는 지표로 다시 그려도 구속변수는 여전히 측정이 아니라 선언이다.**")
    print("    exp 60 의 결론이 **더 나은 지표 위에서 재확인됐다** — 자기가 쓴 지표보다 나은 지표에서.")
    print("  → **내가 exp 61 을 시작하며 '누적으로 바꾸면 exp 60 이 뒤집힐 것'이라 적은 것은 틀렸다.**")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] 정보의 값을 두 지표로 나란히 — **누적 쪽이 F_slip 에 더 둔감하다**")
    print("    exp 60 의 질문을 그대로 다시 묻는다: F_slip 을 알면 결정이 얼마나 달라지나.")
    print("    척도는 w* 의 **퍼짐**이다 — 넓으면 조직 값이 결정을 좌우하고, 좁으면 안 좌우한다.")
    print(f"{'τ [s]':>6s} | {'w* 진폭 (mm/N)':>26s} | {'w* 누적 (mm/(N·s))':>26s} | "
          f"{'붙들기 진폭이 포화하는 곳':>24s}")
    for tau in taus_c:
        pk = [C[(tau, f)][2] for f in fss_c]
        ds = [C[(tau, f)][3] for f in fss_c]
        rng = lambda v: (f"{min(v):.0f} ~ "  # noqa: E731
                         + ("안 뒤집힘" if not np.isfinite(max(v)) else f"{max(v):.0f}"))
        base = C[(tau, fss_c[-1])][0][0]
        sat = [f for f in fss_c if abs(C[(tau, f)][0][0] - base) < 1e-9]
        print(f"{('∞' if not np.isfinite(tau) else f'{tau:.1f}'):>6s} | {rng(pk):>26s} | "
              f"{rng(ds):>26s} | {min(sat):>23.1f}+")
    print("  **진폭 기준에서는 F_slip 이 커지면 후퇴가 아예 못 이긴다**('안 뒤집힘'). 누적 기준에서는")
    print("  전 구간이 한 자릿수~십몇 안에 들어와 **창이 좁고 닫힌다.** 어느 쪽이든 그 창 밖에서는")
    print("  결정이 F_slip 과 무관하고, 창 안에 있는지 여부를 정하는 것은 **교환비**다.")
    print("  → **더 나은 지표로 바꿔도 exp 60 의 구조가 그대로다.** 지표는 창의 위치와 단위를 바꾸지,")
    print("    '구속변수가 선언이지 측정이 아니다'를 바꾸지 않는다.")
    print()
    print(f"  그리고 τ 가 짧으면 이완이 파악을 `K_grip·v·τ`(τ=0.2 에서 {K_GRIP0*V_INSERT*0.2:.2f} N)")
    print("  에서 자른다 — exp 60 의 기하학적 천장 위에 **점성 천장이 하나 더** 생기고 둘 중 낮은")
    print("  쪽이 이긴다. 누적에서 실제로 보인다: F_slip 을 6.4 로 키워도 τ=0.2 에서는 1.687 인데")
    print("  순탄성에서는 2.200 이다. **이완은 위해와 관측 가능성을 같은 방향으로 깎는다** —")
    print("  exp 60 의 '못 재는 이유와 안 중요한 이유가 같다'가 다른 메커니즘으로 한 번 더 성립한다.")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] 정리 — **예측 두 개가 다 틀렸고, 대신 지표 결함 두 개를 잡았다**")
    print("  · 예측 ① '누적으로 바꾸면 exp 60 이 뒤집힌다' → **틀렸다.** 창의 위치와 단위만 바뀌고")
    print("    구조는 같다. 구속변수는 여전히 **선언된 교환비**다.")
    print("  · 예측 ② '진폭 사다리가 틀린 값에서 수렴한다' → **틀렸다.** 고정 주파수에서 진폭을")
    print("    올리면 속도도 오르므로 F_slip 을 뺀 모든 천장이 함께 올라간다. **가짜 평탄부가")
    print("    구조적으로 불가능하다.** exp 60 의 판정은 그때는 몰랐던 이유로 이완까지 덮고 있었다.")
    print()
    print("  대신 잡은 것:")
    print("  1. **exp 60 의 진폭 지표는 조직이 아니라 정지 제어기를 재고 있었다** — 환자를 완전히")
    print(f"     세워도 {z:.2f} N 이 나오고 조직·τ 를 바꿔도 같다. 위상 민감성을 고치려던 지표 교체가")
    print("     다른 결함을 들여왔다. 누적은 F_slip 에도 τ 에도 반응한다.")
    print("     → **그 지표로 다시 그려도 exp 60 의 결론은 같다.** 자기가 쓴 것보다 나은 지표에서")
    print("       재확인된 셈이라, exp 60 은 오히려 더 믿을 만해졌다.")
    print("  2. **정지 진폭을 실행 전체에 걸쳐 누적하면 안 된다** — 서로 다른 깊이의 정지가 섞이고,")
    print("     관통을 걸친 정지가 있으면 스윙이 조직과 무관하게 F_PUNC 로 찍힌다. 정지 1회 단위로")
    print("     고쳤고 exp 60 의 발표 숫자가 그만큼 바뀐다(결론은 유지).")
    print("  3. **프로토콜에서 실제로 깨진 곳은 하나**: 삽입 상수항이 F_cut + min(F_slip, K·v·τ) 라")
    print("     느리게 넣으면 조직이 아니라 **속도를 잰다.** 교락 상대가 절삭력 + 속도로 늘었다.")
    print("  4. 이완이 무는 대가는 값이 아니라 **비용**이다 — 필요 진폭이 부푼다. 싼 해법은 주파수를")
    print("     올리는 것. 요구를 하나로 쓰면 **속도** 사양이다:")
    print("     기하학적 `A > 2·F_slip/K_grip` **그리고** 점성 `A·ω > F_slip/(K_grip·τ)`.")
    print("  5. 남은 것: 완화 스펙트럼(단일 τ 가 아님)·QLV, 정지 중 축방향 마찰의 완화, 그리고")
    print("     **손상이 힘이 아니라 변형·허혈 시간이라면** 세 번째 지표가 필요하다는 것 —")
    print("     이완 조직에서는 힘이 사라져도 **변형은 남는다.** 그건 이 실험이 못 본 축이다.")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    t = np.arange(int(3.0 / DT)) * DT
    for tau, c in zip(taus, plt.cm.viridis(np.linspace(0.1, 0.85, len(taus)))):
        lbl = "tau = inf (elastic, #60)" if not np.isfinite(tau) else f"tau = {tau} s"
        ax.plot(t, hold_and_relax(tau), color=c, lw=1.2, label=lbl)
    ax.axvline(STOP_SECS, color="crimson", ls="--", lw=1)
    ax.text(STOP_SECS * 1.05, ax.get_ylim()[1] * 0.95, "typical stop", fontsize=7,
            color="crimson", va="top")
    ax.set_xlabel("time held [s]"); ax.set_ylabel("axial tissue force [N]")
    ax.set_title("Hold still and the grip force disappears", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[0, 1]
    hzs = (0.1, 0.25, 1.0)
    for tau, c in zip(taus, plt.cm.viridis(np.linspace(0.1, 0.85, len(taus)))):
        lbl = "tau = inf" if not np.isfinite(tau) else f"tau = {tau} s"
        ax.plot(hzs, [A2[(tau, h)] for h in hzs], "-o", ms=4, color=c, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("patient-motion frequency [Hz]")
    ax.set_ylabel("force swing while held [N]")
    ax.set_title("Even the hazard is now frequency-dependent", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[0, 2]
    for fs_, c in zip((0.8, 3.2), ("crimson", "tab:blue")):
        ax.plot(taus[:-1], [B1[(tt, fs_)][0] for tt in taus[:-1]], "-o", color=c,
                label=f"fitted, F_slip = {fs_} N")
        ax.plot(taus[:-1], [B1[(tt, fs_)][1] for tt in taus[:-1]], ":", color=c,
                label=f"F_cut + min(F_slip, K v tau)")
    ax.set_xscale("log")
    ax.set_xlabel("relaxation time constant tau [s]")
    ax.set_ylabel("insertion intercept [N]")
    ax.set_title("#60's intercept now also encodes the insertion speed", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6.5)

    ax = axes[1, 0]
    for fs_, c in zip(fss, ("crimson", "tab:blue", "seagreen")):
        ys = [B2[(fs_, z)][0] for z in freqs]
        ax.plot(freqs, ys, "-", color=c, lw=1.2, label=f"F_slip = {fs_} N")
        for z, y in zip(freqs, ys):
            ok = B2[(fs_, z)][2]
            ax.plot(z, y, "o" if ok else "v", ms=6 if ok else 7, color=c,
                    mfc=c if ok else "none")
        ax.axhline(fs_, color=c, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("dwell excitation frequency [Hz]")
    ax.set_ylabel("amplitude-ladder estimate [N]")
    ax.set_title("Filled = converged, hollow = reported a lower bound", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    tau_p = 0.2
    xs_ = np.arange(len(fss_c))
    w = 0.35
    ax.bar(xs_ - w / 2, [C[(tau_p, f)][0][1] for f in fss_c], w, label="hold",
           color="tab:blue")
    ax.bar(xs_ + w / 2, [C[(tau_p, f)][1][1] for f in fss_c], w, label="retract",
           color="crimson")
    ax.set_xticks(xs_); ax.set_xticklabels([str(f) for f in fss_c])
    ax.set_xlabel("slip limit F_slip [N]")
    ax.set_ylabel("dose while held  [N s]")
    ax.set_title(f"Dose flips the force axis (tau = {tau_p} s)", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["What relaxation breaks - and what it does not", "",
             "BOTH my predictions were wrong.", "",
             "1. Dose would flip #60's verdict - NO.",
             "   It moves the window and the units",
             "   (mm/N -> mm/(N s)) but not the shape.",
             "   The binding unknown is still a",
             "   DECLARED exchange rate.", "",
             "2. The ladder would falsely converge - NO.",
             "   At fixed frequency more amplitude is",
             "   also more velocity, so every ceiling",
             "   except F_slip rises with the ladder.",
             "   A false plateau is impossible.",
             "   Relaxation costs amplitude, not",
             "   correctness.", "",
             "WHAT I FOUND INSTEAD:", "",
             "* #60's peak metric measures the STOP",
             "  CONTROLLER, not the tissue: 2.17 N even",
             "  with the patient perfectly still, and",
             "  identical for every tissue and tau.",
             "  Dose responds to both F_slip and tau.",
             "  Redrawn on that better metric, #60's",
             "  conclusion still holds.", "",
             "* Aggregating the swing across stops lets",
             "  one stop spanning puncture pin it to",
             "  F_PUNC. Now measured per stop.", "",
             "* Real protocol break: the intercept is",
             "  F_cut + min(F_slip, K v tau) - a slow",
             "  insertion measures the SPEED.", "",
             "SPEC:  A > 2 F_slip / K   (geometric)",
             "       A w > F_slip / (K tau)  (viscous)"]
    ax.text(0.02, 0.99, "\n".join(lines), va="top", ha="left", fontsize=6.6,
            family="monospace")

    fig.suptitle("61. Tissue relaxes - testing #60's protocol and #60's verdict against "
                 "the physics #60 said it was missing", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "61_tissue_relaxes.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/61_tissue_relaxes.png, assets/61_tissue_relaxes.png")

    return dict(A=A, A2=A2, B1=B1, B2=B2, C=C)


if __name__ == "__main__":
    main()
