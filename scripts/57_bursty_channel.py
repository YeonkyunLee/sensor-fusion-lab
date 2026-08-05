"""원격조작 III: 연집 손실과 긴 지연 꼬리 — exp 56 의 상쇄가 기대고 있던 대칭성을 뺀다.

exp 56 은 지터·손실을 넣어도 파동 채널이 수동으로 남는 것을 보고 그 이유를 이렇게 적었다:
지연이 **늘면** 굶어서 같은 파동을 두 번 꺼내고(에너지 생성), **줄면** 낡은 패킷을 버린다(소멸).
평균 0 인 지터에서는 두 몫이 상쇄된다. 그리고 한계 절에 **그 논리가 지터의 대칭성에 기대고
있다**고 적어 뒀다. 이 실험은 그 가정을 뺀다.

실제 통신망은 두 가지가 다르다.
  · **지연 분포가 한쪽으로 길다**(late tail). 평균은 작아도 드물게 크게 늦는 패킷이 있다.
  · **손실이 연집(burst)한다.** 같은 손실률이라도 한 번에 여러 개를 잃는다.

exp 56 의 플랜트·제어·장부를 그대로 쓰고 **채널만** 갈아 끼운다(56 이 exp 50 에 한 것과 같은 방식).

--- 공정한 비교를 위한 장치 ---
꼬리를 붙이면 **평균 지연도 같이 오른다.** 그대로 비교하면 '연집성의 효과'가 아니라 '늘어난
지연의 효과'를 재게 된다 — exp 55 에서 표면장·심부장이 어긋나 '대응의 효과'가 아니라 '장
불일치'를 재던 것과 같은 함정이다. 그래서 **평균 총지연을 맞춘다**: 꼬리의 평균만큼 공칭
지연을 깎아서, 모든 조건이 같은 평균 편도 지연을 갖게 한 뒤 비교한다.
연집도 같은 원칙으로 **평균 손실률을 고정**하고 평균 연집 길이만 바꾼다(길이 1 = exp 56).

--- 미리 밝히는 결론 (예측이 틀렸다) ---
exp 56 을 마치면서 "이 대칭성을 깨면 그 결론이 다시 좁아질 것"이라고 적었다. **틀렸다.**
연집과 긴 꼬리를 넣어도 exp 56 의 결론은 대체로 버틴다. 그런데 **버티는 이유가 채널이 아니라
플랜트**이고, 그 사실이 예측보다 값이 나갔다.

1. **붙들고 있는 것은 자기 제한적이다.** 낡은 명령에도 정지 평형이 있어서 팔이 거기로 수렴한다.
   60 ms 를 넘겨 붙들면 도구는 초반의 1/3 속도로 기어간다(≈20 → 5~7 ㎛/스텝). 그래서 평균
   손실률을 고정한 채 연집 길이를 80 배로 늘려도 **총 맹행 거리는 오히려 줄고**(17.5 → 6.6 mm)
   한 사건의 크기는 포화한다.
2. **그 자기 제한을 만드는 항이 exp 56 이 '증명 밖의 결함'으로 지목한 그 항이다.** 표류 보정
   λ(x_m − x_s) 는 낡았지만 유계인 setpoint 를 향한 위치 servo = **브레이크**다. 그래서 예산으로
   그 항까지 함께 죄는 '당연한 처방'은 오히려 맹행을 늘린다(3.36 → 4.17 mm). exp 56 의 H24 는
   절반만 맞았다 — **같은 항이 보증을 깨면서 동시에 안전 기능을 하고 있다.**
3. **정작 무너진 것은 버퍼 설계다.** 파레토 꼬리에는 최댓값이 없으므로 exp 56 의 "45 ms 사면 끝"
   이 성립하지 않고, 재생 기한을 두는 순간 **없던 손실이 만들어진다**(p50 버퍼에서 41%).
   작게 잡으면 모든 축에서 지고, 크게 잡으면 편도 328 ms 를 상시로 낸다.
4. **지표를 못 고르면 아무것도 안 보인다.** E_min 은 선로 위 에너지 저수지에 가려 네 조건을
   정렬조차 못 하고, 최대 낙폭은 정렬하지만 **지연이 다른 조건끼리는 비교가 안 된다**(선로가
   길면 떠 있는 에너지가 커진다). 평균으로 채점하면 연집성은 아예 보이지 않는다.

    python scripts/57_bursty_channel.py
"""

from __future__ import annotations

import sys
from functools import partial
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
jc = import_module("56_jittery_channel")

DT = jc.DT
STEPS = jc.STEPS
DELAY_MS = jc.DELAY_MS          # 평균 편도 지연을 여기에 맞춘다(공칭이 아니라 **평균**)
X_TARGET = jc.X_TARGET
LAM_50, LAM_TASK = jc.LAM_50, jc.LAM_TASK
N_SEEDS = 6

TAIL_ALPHA = 1.8         # 파레토 꼬리 지수. α>1 이면 평균은 유한하고 α<2 면 **분산이 무한**하다
                         # — 네트워크 지연의 표준적인 모형이고, "최댓값이 없다"는 성질이 C 절의
                         #   버퍼 설계를 분위수 선택 문제로 바꾼다.
TAIL_MS = 6.0            # 꼬리 스케일 [ms] (파레토 x_m)
BURST_LEN = 40           # 평균 연집 길이 [샘플] = 40 ms 동안 통째로 끊김


# --------------------------------------------------------------------------- #
# 연집 손실 + 긴 지연 꼬리 채널
# --------------------------------------------------------------------------- #
class BurstyChannel(jc.Channel):
    """exp 56 채널의 send() 만 바꾼다. 수신기 정책(최신 채택/낡은 폐기)은 **그대로** 둔다 —
    바꾸는 것을 하나로 유지해야 원인을 귀속할 수 있다.

    · 지연 = 공칭 + 파레토 꼬리(항상 ≥ 0, 최댓값 없음)
    · 손실 = Gilbert-Elliott 2상태. good 에서는 안 잃고 bad 에서는 다 잃는다.
      bad→good 확률 1/L 로 평균 연집 길이 L, good→bad 확률을 맞춰 **정상상태 손실률 = loss**.
      L=1 이면 독립 베르누이(= exp 56)로 되돌아간다.
    """

    def __init__(self, rng, delay_ms=DELAY_MS, jitter_ms=0.0, loss=0.0, zero=0.0,
                 tail_ms=0.0, tail_alpha=TAIL_ALPHA, burst_len=1.0):
        super().__init__(rng, delay_ms, jitter_ms, loss, zero)
        self.tail_n = tail_ms * 1e-3 / DT          # 꼬리 스케일 [샘플]
        self.alpha = tail_alpha
        self.bad = False
        self.p_bg = 1.0 / max(burst_len, 1.0)      # bad → good
        self.p_gb = (0.0 if loss <= 0 else
                     min(self.p_bg * loss / max(1.0 - loss, 1e-9), 1.0))
        self.n_burst = 0                           # 연집 구간 진입 횟수

    def _extra(self):
        """파레토 꼬리 지연 [샘플]. numpy 의 pareto 는 x_m=1 기준이라 (1+p)·x_m 이 표본이다."""
        if self.tail_n <= 0:
            return 0
        return int(round(self.tail_n * (1.0 + self.rng.pareto(self.alpha))))

    def send(self, k, payload):
        self.n_sent += 1
        # ---- 연집 손실 상태 갱신 ----
        if self.loss > 0.0:
            if self.bad:
                if self.rng.random() < self.p_bg:
                    self.bad = False
            elif self.rng.random() < self.p_gb:
                self.bad = True
                self.n_burst += 1
            if self.bad:
                self.n_lost += 1
                return
        # ---- 지연: 균등 지터 + 파레토 꼬리 ----
        d = self.n0
        if self.nj:
            d += int(self.rng.integers(-self.nj, self.nj + 1))
        d += self._extra()
        self.q.append((k + max(d, 1), k, payload))


def tail_mean_ms(tail_ms, alpha=TAIL_ALPHA):
    """파레토(x_m, α) 의 평균 = α·x_m/(α−1). 평균 총지연을 맞추는 데 쓴다."""
    return 0.0 if tail_ms <= 0 else tail_ms * alpha / (alpha - 1.0)


def tail_quantile_ms(tail_ms, q, alpha=TAIL_ALPHA):
    """분위수 = x_m·(1−q)^(−1/α). C 절에서 버퍼를 분위수로 고를 때 쓴다."""
    return 0.0 if tail_ms <= 0 else tail_ms * (1.0 - q) ** (-1.0 / alpha)


def run(mode="zoh", seed=0, tail_ms=0.0, burst_len=1.0, loss=0.0, jitter_ms=0.0,
        match_mean=True, **kw):
    """exp 56 의 run() 을 그대로 쓰고 채널만 끼운다.

    match_mean=True 면 꼬리의 평균만큼 **공칭 지연을 깎아** 평균 총지연을 DELAY_MS 로 맞춘다.
    이걸 안 하면 '꼬리의 효과'와 '지연이 늘어난 효과'가 섞인다.
    """
    extra = tail_mean_ms(tail_ms)
    nominal = DELAY_MS - extra if match_mean else DELAY_MS
    clipped = nominal < 1.0
    chan = partial(BurstyChannel, tail_ms=tail_ms, burst_len=burst_len)
    r = jc.run(mode=mode, seed=seed, jitter_ms=jitter_ms, loss=loss,
               delay_ms=max(nominal, 1.0), chan=chan, **kw)
    r["tail_ms"] = tail_ms
    r["burst_len"] = burst_len
    r["mean_delay_ms"] = max(nominal, 1.0) + extra
    r["mean_clipped"] = clipped
    a, b = r["chans"]
    # 방향별 최대 굶음. 참고용이고, 아래 **결합** 값이 실제로 제어에 영향하는 값이다 —
    # 양방향이 **둘 다** 새로 도착해야 한 스텝이 온전하므로 결합 구간이 훨씬 길다.
    r["max_starve_dir_ms"] = max(a.max_starve_run, b.max_starve_run) * DT * 1e3
    r["n_burst"] = a.n_burst + b.n_burst
    r["late_frac"] = (a.n_late + b.n_late) / max(a.n_sent + b.n_sent, 1)
    # 감쇠기가 사실상 0 으로 죄고 있는 구간 = **통신이 끊긴 것과 같은 상태**(E 절)
    beta = r["log"]["beta"]
    r["mute_frac"] = float(np.mean(beta < 0.1)) if len(beta) else np.nan
    # **모르는 채로 간 거리**: 새 표본이 없던 스텝에서 도구가 움직인 총 거리. 연집 손실의 안전
    # 비용이 여기서 나온다 — 평균 손실률이 같아도 뭉쳐 있으면 한 번에 멀리 간다.
    st = r["log"]["starved"]
    xs = r["log"]["xs"]
    # 시작 과도를 뺀다. 첫 수신 전에는 양방향이 모두 미도착이라 st=1 인데 그건 사건이 아니라
    # 공칭 지연이다 — 안 빼면 최대 굶음 구간이 어느 조건에서나 '공칭 지연 = 50 ms'로 깔린다.
    first = int(np.argmin(st)) if len(st) and st[0] > 0 else 0
    if first > 0:
        st = st[first:]
        xs = xs[first:]
    if len(xs) > 1:
        step = np.abs(np.diff(xs)) * 1e3
        blind = step * st[1:]
        r["blind_mm"] = float(blind.sum())
        # 한 번의 굶음 구간에서 간 최대 거리(구간별 누적의 최댓값)와, **구간 안에서의 나이별**
        # 이동량. 뒤쪽이 앞쪽보다 작으면 붙들고 있는 것이 **자기 제한적**이라는 뜻이다 —
        # 낡은 명령이 정지 평형을 가지므로 팔이 거기로 수렴해 멈춘다.
        cur, best = 0.0, 0.0
        age = np.zeros_like(step)
        a = 0
        for i, (s, d) in enumerate(zip(st[1:], step)):
            cur = cur + d if s else 0.0
            best = max(best, cur)
            a = a + 1 if s else 0
            age[i] = a
        r["blind_max_mm"] = float(best)
        r["max_starve_ms"] = float(age.max() * DT * 1e3)
        early, late = (age >= 1) & (age <= 5), age >= 61
        r["hold_early_um"] = float(step[early].mean() * 1e3) if early.any() else np.nan
        r["hold_late_um"] = float(step[late].mean() * 1e3) if late.any() else np.nan
    else:
        r["blind_mm"] = r["blind_max_mm"] = r["max_starve_ms"] = np.nan
        r["hold_early_um"] = r["hold_late_um"] = np.nan
    return r


def sweep(mode="zoh", seeds=N_SEEDS, **kw):
    rs = [run(mode=mode, seed=s, **kw) for s in range(seeds)]
    ok = [r for r in rs if not r["diverged"]]

    def med(key, default=np.nan):
        # 조건에 따라 표본이 아예 없는 지표가 있다(예: 60 스텝 넘는 홀드가 안 생기는 짧은 연집).
        # 일부만 없으면 나머지로 중앙값을 내고, 전부 없으면 그냥 nan 을 돌려준다.
        if not ok:
            return default
        v = np.asarray([r[key] for r in ok], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else np.nan

    return dict(runs=rs, n=len(rs), n_stable=sum(r["stable"] for r in rs),
                e_min=(float(np.min([r["e_min"] for r in ok])) if ok else -np.inf),
                e_drawdown=(float(np.max([r["e_drawdown"] for r in ok]))
                            if ok else np.inf),
                blind_mm=med("blind_mm"), blind_max_mm=med("blind_max_mm"),
                hold_early_um=med("hold_early_um"), hold_late_um=med("hold_late_um"),
                osc_mm=med("osc_mm", np.inf), pos_err_mm=med("pos_err_mm", np.inf),
                final_depth_mm=med("final_depth_mm"), force_err_N=med("force_err_N"),
                att_duty=float(np.mean([r["att_duty"] for r in rs])),
                mute_frac=float(np.mean([r["mute_frac"] for r in rs])),
                starve_frac=float(np.mean([r["starve_frac"] for r in rs])),
                max_starve_ms=(float(np.max([r["max_starve_ms"] for r in ok]))
                               if ok else np.nan),
                late_frac=float(np.mean([r["late_frac"] for r in rs])),
                mean_delay_ms=float(np.mean([r["mean_delay_ms"] for r in rs])),
                n_div=sum(r["diverged"] for r in rs))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    print("=== 57. 연집 손실과 긴 지연 꼬리: exp 56 의 상쇄가 기대던 대칭성을 뺀다 ===")
    print(f"exp 56 의 플랜트·제어·장부 그대로, 채널만 교체. **평균** 편도 지연을 "
          f"{DELAY_MS:.0f} ms 로 맞춰 비교한다.")
    print(f"꼬리 = 파레토(α={TAIL_ALPHA}, x_m={TAIL_MS:.0f} ms) — α<2 라 **분산이 무한**하고 "
          f"최댓값이 없다. 평균 {tail_mean_ms(TAIL_MS):.1f} ms, "
          f"p99 {tail_quantile_ms(TAIL_MS, 0.99):.0f} ms, "
          f"p99.9 {tail_quantile_ms(TAIL_MS, 0.999):.0f} ms.")

    # ---------------- A. 무엇이 상쇄를 깨는가 ----------------
    print("-" * 96)
    print("[A] 분해 — 평균 지연과 평균 손실률을 고정하고 **모양만** 바꾼다 (완주 이득 λ=24)")
    print(f"{'조건':>30s} | {'E_min[mJ]':>10s} | {'최대 낙폭[mJ]':>12s} | "
          f"{'최대 굶음[ms]':>12s} | {'모르고 간 거리[mm]':>16s} | {'진동[mm]':>8s}")
    A = {}
    cases = [
        ("exp 56 조건 (균등±20, 독립 10%)", dict(jitter_ms=20.0, loss=0.10, burst_len=1.0)),
        ("긴 꼬리만 (손실 없음)", dict(tail_ms=TAIL_MS)),
        ("연집만 (손실 10%, L=40)", dict(loss=0.10, burst_len=BURST_LEN)),
        ("긴 꼬리 + 연집", dict(tail_ms=TAIL_MS, loss=0.10, burst_len=BURST_LEN)),
    ]
    for lbl, kw in cases:
        s = sweep("zoh", seeds=seeds, **kw)
        A[lbl] = s
        print(f"{lbl:>30s} | {s['e_min']*1e3:10.3f} | {s['e_drawdown']*1e3:12.3f} | "
              f"{s['max_starve_ms']:12.0f} | {s['blind_max_mm']:16.2f} | "
              f"{s['osc_mm']:8.3f}")
    print("  같은 평균 지연·같은 평균 손실률인데 **모양**만 바꿨다. 그런데 지표마다 순위가 다르다.")
    print("  · **E_min** 은 순위를 못 만든다 — 선로에 떠 있는 에너지 저수지에 가려진다.")
    print("  · **최대 굶음·맹행 거리**는 exp 56 조건이 가장 크다. 뜻밖이 아니다: ±20 ms 지터는")
    print("    방향별로 이미 80% 를 굶기고(exp 56), 한 스텝이 온전하려면 **양방향이 다** 와야")
    print("    하므로 결합 구간이 길어진다. 즉 이 두 지표는 '암전 길이'가 아니라 '지터의 촘촘한")
    print("    구멍'을 세고 있다.")
    print("  · **최대 낙폭**만이 네 조건을 심각도 순으로 정렬한다 "
          f"({A[cases[0][0]]['e_drawdown']*1e3:.1f} → "
          f"{A[cases[3][0]]['e_drawdown']*1e3:.1f} mJ) — 어느 구간에서든 채널이 순수하게 "
          "꺼내 쓴 양의 최대치라서 한 사건의 크기를 보여준다.")
    print("  → exp 52·53 에서 최댓값↔RMS 를 두고 겪은 문제가 또 나왔다. **무엇이 나빠졌는지를**")
    print("  **말하려면 먼저 그 나쁨을 볼 수 있는 지표를 골라야 한다.**")

    # ---------------- B. 연집 길이 스윕 ----------------
    print("-" * 96)
    print("[B] 평균 손실률을 10% 로 고정하고 **평균 연집 길이만** 늘린다 (L=1 이 exp 56)")
    print(f"{'평균 연집 L':>10s} | {'최대 굶음[ms]':>12s} | {'E_min[mJ]':>10s} | "
          f"{'최대 낙폭[mJ]':>12s} | {'모르고 간 거리[mm]':>16s} | {'총 맹행[mm]':>11s}")
    B = {}
    for L in ([1, 20, 80] if quick else [1, 5, 10, 20, 40, 80]):
        s = sweep("zoh", seeds=seeds, loss=0.10, burst_len=float(L))
        B[L] = s
        print(f"{L:10d} | {s['max_starve_ms']:12.0f} | {s['e_min']*1e3:10.3f} | "
              f"{s['e_drawdown']*1e3:12.3f} | {s['blind_max_mm']:16.2f} | "
              f"{s['blind_mm']:11.1f}")
    lo_, hi_ = min(B), max(B)
    print("  **평균 손실률은 내내 10% 다.** 달라진 건 그 10% 가 어떻게 뭉쳐 있는지뿐이다.")
    print("  정직하게: **총량 지표에는 추세가 없다.** 재생되는 스텝 수가 손실률로 정해져 있어서")
    print(f"  E_min·최대 낙폭이 오르내리기만 하고, 총 맹행 거리는 오히려 **줄어든다**"
          f"({B[lo_]['blind_mm']:.1f} → {B[hi_]['blind_mm']:.1f} mm).")
    print(f"  단조로 늘어나는 것은 최대 굶음 구간뿐이다({B[lo_]['max_starve_ms']:.0f} → "
          f"{B[hi_]['max_starve_ms']:.0f} ms) — 정의상 그렇다.")
    print()
    print("  총 맹행이 왜 줄어드나: **붙들고 있는 것이 자기 제한적**이다. 낡은 명령에도 정지 평형이")
    print("  있어서 팔이 거기로 수렴해 멈춘다. 굶은 구간 **안에서의 나이별** 스텝 이동량:")
    print(f"{'평균 연집 L':>10s} | {'홀드 1~5 스텝':>13s} | {'홀드 61 스텝 이후':>16s}")
    for L in sorted(B):
        s = B[L]
        e, l = s["hold_early_um"], s["hold_late_um"]
        print(f"{L:10d} | {e:12.1f}㎛ | "
              f"{(f'{l:.1f}㎛' if np.isfinite(l) else '—(그만큼 긴 홀드 없음)'):>16s}")
    deep = [L for L in sorted(B) if np.isfinite(B[L]["hold_late_um"])]
    if deep:
        Ld = deep[-1]
        ratio = B[Ld]["hold_early_um"] / max(B[Ld]["hold_late_um"], 1e-9)
        print(f"  → 60 ms 를 넘겨 붙들고 있으면 도구는 초반의 **1/{ratio:.1f} 속도**로 기어간다"
              f"(L={Ld}).")
    print("  연집이 길어져도 한 사건의 크기가 **포화한다**. exp 56 의 결론이 여기서도 버틴 이유는")
    print("  채널이 아니라 **플랜트** 다 — 그리고 그걸 만드는 항이 무엇인지가 E 절의 결과다.")
    print("  다만 지표 선택은 여전히 중요하다: **평균으로 채점하면 연집성은 아예 안 보인다.**")
    print("  (주의: L 이 커지면 4 초 실행에 연집이 몇 번밖에 안 들어간다 — L=80 이면 5~6 회다.")
    print("   그래서 이 표의 큰 L 쪽은 사건 수가 적어 실현 손실률과 최댓값이 시드마다 흔들린다.)")

    # ---------------- C. 버퍼 설계가 분위수 선택이 된다 ----------------
    print("-" * 96)
    print("[C] 긴 꼬리에서는 버퍼에 '충분한 크기'가 없다 — 분위수를 고르는 문제가 된다")
    print("    (exp 56 의 균등 지터는 최댓값이 있어서 그만큼 사면 끝이었다. 여기선 최댓값이 없다.)")
    print("    (에너지 지표는 여기서 쓰지 않는다 — 선로가 길어지면 떠 있는 에너지 자체가 커져서")
    print("     **지연이 다른 조건끼리는 비교가 안 된다.** 오염되지 않은 축만 본다.)")
    print(f"{'버퍼 분위수':>10s} | {'버퍼[ms]':>8s} | {'실효 편도[ms]':>12s} | {'늦어 버림':>9s} | "
          f"{'진동[mm]':>8s} | {'도달깊이[mm]':>11s} | {'맹행[mm]':>9s} | {'수동?':>5s}")
    C = {}
    qs = [0.0, 0.9, 0.99] if quick else [0.0, 0.5, 0.9, 0.99, 0.999]
    for q in qs:
        bm = tail_quantile_ms(TAIL_MS, q) if q > 0 else 0.0
        s = sweep("zoh", seeds=seeds, tail_ms=TAIL_MS, loss=0.05,
                  burst_len=BURST_LEN, buf_ms=bm)
        C[q] = (bm, s)
        print(f"{(f'p{q*100:g}' if q > 0 else '없음'):>10s} | {bm:8.0f} | "
              f"{s['mean_delay_ms']+bm:12.0f} | {s['late_frac']*100:8.1f}% | "
              f"{s['osc_mm']:8.3f} | {s['final_depth_mm']:11.1f} | "
              f"{s['blind_max_mm']:9.2f} | "
              f"{'예' if s['e_min'] >= -1e-9 else '아니오':>5s}")
    q_lo, q_hi = sorted(C)[1], max(C)
    print("  **이게 이 실험에서 실제로 무너진 것이다.** exp 56 의 균등 지터는 최댓값이 있어서")
    print("  '45 ms 사면 끝'이었다. 파레토 꼬리에는 최댓값이 없으니 버퍼 크기가 **분위수 선택**이 된다.")
    print("  그리고 재생 기한을 두는 순간 **없던 손실이 만들어진다** — 버퍼가 없으면 늦게 온 패킷도")
    print("  그냥 쓰면 되지만, 기한을 두면 그건 버려야 하는 패킷이 된다.")
    print(f"  · 작게 잡으면 **모든 축에서 진다**: p{q_lo*100:g} 버퍼는 지연을 "
          f"{C[q_lo][0]:.0f} ms 더 내면서 늦은 패킷 "
          f"{C[q_lo][1]['late_frac']*100:.0f}% 를 손실로 바꾸고(망 자체의 손실은 5%),")
    print(f"    그 대가로 얻은 것이 없다(진동 {C[0.0][1]['osc_mm']:.2f} → "
          f"{C[q_lo][1]['osc_mm']:.2f} mm). 꼬리를 지연으로 바꾸려던 것이 꼬리를 손실로 바꿨다.")
    print(f"  · 크게 잡으면 늦은 패킷은 사라지지만(p{q_hi*100:g} 에서 "
          f"{C[q_hi][1]['late_frac']*100:.1f}%) 편도 지연이 "
          f"{C[q_hi][1]['mean_delay_ms']+C[q_hi][0]:.0f} ms 가 되고, exp 56 이 보인 대로 그 지연이")
    print(f"    보증 밖 위치 루프를 때려 진동이 {C[0.0][1]['osc_mm']:.2f} → "
          f"{C[q_hi][1]['osc_mm']:.2f} mm 로 커진다.")
    print("  → **꼬리가 있는 망에는 '충분한 버퍼'가 없다.** 지연↔손실↔진동 세 축의 교환비를 고르는")
    print("  문제이고, 그 선택은 시스템 안에서 안 나온다 — 임상이 허용하는 지연이 정해 줘야 한다.")

    # ---------------- D. exp 56 의 'TDPA 는 거의 공짜' 를 다시 잰다 ----------------
    print("-" * 96)
    print("[D] exp 56 은 에너지 예산이 감쇠기 5.7% 가동으로 거의 공짜라고 했다 — 연집에서 다시 잰다")
    print(f"{'조건':>26s} | {'정책':>14s} | {'E_min[mJ]':>10s} | {'가동률':>7s} | "
          f"{'묵음 구간':>8s} | {'도달깊이[mm]':>11s} | {'진동[mm]':>8s}")
    D = {}
    conds = [("독립 손실 10% (exp 56)", dict(jitter_ms=20.0, loss=0.10, burst_len=1.0)),
             ("연집 손실 10%, L=40", dict(tail_ms=TAIL_MS, loss=0.10,
                                          burst_len=BURST_LEN))]
    for clbl, ckw in conds:
        for plbl, m in (("ZOH", "zoh"), ("0 채움", "zero"), ("에너지 예산", "tdpa")):
            s = sweep(m, seeds=seeds, **ckw)
            D[(clbl, m)] = s
            print(f"{clbl if m == 'zoh' else '':>26s} | {plbl:>14s} | "
                  f"{s['e_min']*1e3:10.3f} | {s['att_duty']*100:6.1f}% | "
                  f"{s['mute_frac']*100:7.1f}% | {s['final_depth_mm']:11.1f} | "
                  f"{s['osc_mm']:8.3f}")
    print("  묵음 구간 = 감쇠기가 파동을 90% 이상 죄고 있는 스텝 비율 = **사실상 0 채움 상태**.")
    print("  예산은 연집에서 마른다. 그게 결함이 아니라 정의대로 동작하는 것이다 — 보내온 에너지가")
    print("  없으면 꺼낼 것도 없다. 대신 exp 56 의 '거의 공짜' 라는 값은 독립 손실에서만 참이다.")

    # ---------------- E. 그러면 예산 고갈을 '정지 신호'로 쓴다 ----------------
    print("-" * 96)
    print("[E] 보증을 깨는 그 항이, 끊긴 동안 도구를 붙잡고 있던 항이었다")
    print("    예산은 파동만 죈다. 그러면 끊긴 동안 도구를 세우는 것은 무엇인가 — 시험해 봤다.")
    print(f"{'평균 연집[ms]':>12s} | {'ZOH':>9s} | {'예산':>9s} | {'예산+λ게이트':>12s} | "
          f"{'묵음 구간':>9s} | {'게이트 진동[mm]':>14s}")
    print(f"{'':>12s} | " + " | ".join(f"{c:>9s}" for c in
                                       ("한 구간 맹행[mm]", "한 구간", "한 구간"))
          + f" | {'':>9s} | {'':>14s}")
    E = {}
    for Lms in ([20, 80] if quick else [10, 20, 40, 80, 160]):
        kw = dict(tail_ms=TAIL_MS, loss=0.10, burst_len=float(Lms))
        z = sweep("zoh", seeds=seeds, **kw)
        t = sweep("tdpa", seeds=seeds, **kw)
        g = sweep("tdpa", seeds=seeds, lam_gate=True, **kw)
        E[Lms] = (z, t, g)
        print(f"{Lms:12d} | {z['blind_max_mm']:9.2f} | {t['blind_max_mm']:9.2f} | "
              f"{g['blind_max_mm']:12.2f} | {t['mute_frac']*100:8.1f}% | "
              f"{g['osc_mm']:14.3f}")
    print("  **예산만으로는 도구가 안 선다.** 예산은 파동 채널의 유출만 막고, 팔을 실제로 미는")
    print("  표류 보정 λ(x_m − x_s) 는 그 장부 밖이라 낡은 위치를 향해 계속 servo 한다.")
    print()
    print("  그래서 당연한 처방을 시험했다 — **λ 도 같은 β 로 함께 죄기.** 결과는 더 나빠진다.")
    print("  이유를 보면 당연하다: λ(x_m − x_s) 는 낡았지만 **유계인 setpoint 를 향한 위치 servo**,")
    print("  즉 **브레이크**다. 그걸 떼면 붙잡아 주던 것이 없어진다. B 절에서 본 자기 제한(홀드가")
    print("  길어지면 도구가 기어간다)을 만들던 것이 바로 이 항이다.")
    print()
    print("  → exp 56 은 이 항을 '증명이 덮지 않는 결함'으로 지목했다. 절반만 맞았다.")
    print("  **같은 항이 수동성 보증을 깨면서 동시에 통신이 끊긴 동안 도구를 붙잡고 있다.**")
    print("  그러니 '보증 밖 항을 없애라'가 아니라 **그 항이 하고 있는 안전 기능을 먼저 대체하라**가")
    print("  옳은 순서다. 지금 이 구성에서 '끊기면 멈춘다'는 별도의 기능으로 넣어야 한다 —")
    print("  예산 고갈은 그 **판정 시점**을 임계값 없이 주지만(물리에서 오는 신호), 정지 자체를")
    print("  집행하지는 않는다. exp 9 의 '모를 땐 멈춘다'가 여기서는 아직 미해결로 남는다.")

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    for lbl, kw, c in [("uniform jitter, independent loss (exp 56)",
                        dict(jitter_ms=20.0, loss=0.10, burst_len=1.0), "seagreen"),
                       ("heavy tail only", dict(tail_ms=TAIL_MS), "tab:orange"),
                       ("bursty loss only", dict(loss=0.10, burst_len=BURST_LEN),
                        "tab:blue"),
                       ("heavy tail + bursty loss",
                        dict(tail_ms=TAIL_MS, loss=0.10, burst_len=BURST_LEN),
                        "crimson")]:
        lg = run("zoh", seed=0, **kw)["log"]
        ax.plot(lg["t"], lg["e_ch"] * 1e3, color=c, lw=1.3, label=lbl)
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("wave-channel stored energy [mJ]")
    ax.set_title("Same mean delay and loss rate — only the shape differs", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[0, 1]
    Ls = sorted(B)
    ax.plot(Ls, [B[L]["blind_mm"] for L in Ls], "-o", color="0.55",
            label="total blind travel [mm] (flat)")
    ax.plot(Ls, [B[L]["blind_max_mm"] for L in Ls], "-s", color="crimson",
            label="worst single episode [mm]")
    ax2 = ax.twinx()
    ax2.plot(Ls, [B[L]["max_starve_ms"] for L in Ls], "-^", color="tab:blue")
    ax2.set_ylabel("longest starved run [ms]", color="tab:blue", fontsize=9)
    ax.set_xlabel("mean burst length [samples]  (loss rate fixed at 10%)")
    ax.set_ylabel("blind tool travel [mm]")
    ax.set_title("Averaging hides clumping; harm comes from one episode", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="center left")

    ax = axes[0, 2]
    qq = sorted(C)
    lat = [C[q][1]["mean_delay_ms"] + C[q][0] for q in qq]
    ax.plot(lat, [C[q][1]["late_frac"] * 100 for q in qq], "-o", color="crimson",
            label="delay turned into loss [%]")
    ax2 = ax.twinx()
    ax2.plot(lat, [C[q][1]["osc_mm"] for q in qq], "-^", color="tab:blue",
             label="settled oscillation [mm]")
    for q, x in zip(qq, lat):
        ax.annotate("none" if q == 0 else f"p{q*100:g}",
                    (x, C[q][1]["late_frac"] * 100), fontsize=7,
                    xytext=(3, 4), textcoords="offset points")
    ax.set_xlabel("effective one-way delay [ms]  (playout buffer size)")
    ax.set_ylabel("late drops [%]", color="crimson")
    ax2.set_ylabel("oscillation [mm]", color="tab:blue", fontsize=9)
    ax.set_title("No buffer is big enough when the tail has no maximum", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper center")

    ax = axes[1, 0]
    names = ["ZOH", "zero-fill", "energy budget"]
    x = np.arange(3)
    ind = [D[(conds[0][0], m)] for m in ("zoh", "zero", "tdpa")]
    bur = [D[(conds[1][0], m)] for m in ("zoh", "zero", "tdpa")]
    ax.bar(x - 0.2, [s["final_depth_mm"] for s in ind], 0.4, color="0.6",
           label="independent loss (exp 56)")
    ax.bar(x + 0.2, [s["final_depth_mm"] for s in bur], 0.4, color="crimson",
           label="bursty loss, L=40")
    ax.axhline(X_TARGET * 1e3, color="0.3", ls="--", lw=1, label="target")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("depth reached [mm]")
    ax.set_title("What bursts cost the task", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar(x - 0.2, [s["att_duty"] * 100 for s in ind], 0.4, color="0.6",
           label="attenuator duty, independent")
    ax.bar(x + 0.2, [s["att_duty"] * 100 for s in bur], 0.4, color="crimson",
           label="attenuator duty, bursty")
    ax.plot(x + 0.2, [s["mute_frac"] * 100 for s in bur], "k^", ms=7,
            label="of which effectively muted")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("[% of steps]")
    ax.set_title("exp 56's \"nearly free\" was true of independent loss", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

    ax = axes[1, 2]
    Es = sorted(E)
    ax.plot(Es, [E[L][0]["blind_max_mm"] for L in Es], "-s", color="crimson",
            label="hold last")
    ax.plot(Es, [E[L][1]["blind_max_mm"] for L in Es], "-o", color="tab:blue",
            label="energy budget")
    ax.plot(Es, [E[L][2]["blind_max_mm"] for L in Es], "-^", color="darkorange",
            label="budget + gating the drift term")
    ax.set_xlabel("mean burst length [ms]")
    ax.set_ylabel("worst blind tool travel [mm]")
    ax.set_title("The term that breaks the proof is the brake", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("57. Bursty loss and a heavy late tail — removing the symmetry exp 56's "
                 "cancellation leaned on", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "57_bursty_channel.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/57_bursty_channel.png, assets/57_bursty_channel.png")

    return dict(A=A, B=B, C=C, D=D, E=E)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 자기 제한(B 절)은 **이 플랜트의 성질**이다. 팔이 무겁고 감쇠가 크고 조직이 저항하기 때문에
#     낡은 명령에 정지 평형이 있다. 관성이 작고 마찰이 적은 축(exp 47 이 발산했던 바늘 스핀 같은)
#     이면 같은 홀드가 자기 제한적이지 않을 수 있다. **이 결론을 축 전체로 일반화하면 안 된다.**
#   - 그래서 '끊기면 멈춘다'는 여전히 필요하다. 여기서 확인한 것은 예산 고갈이 그 **판정 시점**을
#     임계값 없이 준다는 것까지이고, 정지를 **집행**하는 기능은 넣지 않았다(미해결로 남긴다).
#   - Gilbert-Elliott 는 2상태 기하분포 연집이다. 실제 무선·WAN 손실은 상태가 더 많고 자기상관이
#     길다(자기유사 트래픽). 파레토도 꼬리 지수 하나로 요약한 모형이다.
#   - **연집이 길면 표본이 적다.** 정상상태 손실률은 설계상 10% 로 고정돼 있지만, 4 초(4000 스텝)
#     안에 L=80 짜리 연집은 5~6 회밖에 들어가지 않아서 **실현** 손실률이 7~13% 로 흔들린다.
#     B 절의 큰 L 쪽 숫자와 '자기 제한 배율'(시드에 따라 1.7~3.3배)을 그만큼 넓게 읽어야 한다.
#     사건 수를 늘리려면 실행을 길게 해야 하는데, 그러면 술자 궤적이 한 번의 삽입을 넘어간다.
#   - 양방향 채널을 **독립**으로 뒀다. 실제로는 같은 경로를 공유해 혼잡이 동시에 온다 —
#     그러면 결합 굶음이 여기 결과보다 길어진다.
#   - 평균 총지연을 맞추려고 공칭 지연을 깎았다. 꼬리가 아주 두꺼우면 공칭이 1 샘플 아래로 내려가
#     고정되므로(mean_clipped) 그 구간은 평균이 맞지 않는다 — 현재 설정에서는 발생하지 않는다.
#   - 에너지 지표(E_min·최대 낙폭)는 같은 평균 지연 안에서만 비교했다. C 절이 그 이유다.
#   - exp 56 과 마찬가지로 술자는 계획 궤적을 따르고 끊김에 반응해 속도를 줄이지 않는다.
#     실제 술자는 화면이 멈추면 손을 멈춘다 — 맹행 거리는 그만큼 보수적인 상한이다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
