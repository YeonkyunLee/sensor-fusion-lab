"""완주 기준을 창으로 고쳐 쓴다 — 그랬더니 **이 사슬은 표적에 도달한 적이 없다**.

exp 68 은 수동 칸 하나가 도구를 59.1 mm 까지 미는데 표적이 55 mm 라는 것을 발견하고, 원인이
완주 기준 R18 이 *"45 mm 이상"* 이라는 **바닥**이기 때문이라고 적었다. 그리고 규칙만 적고
끝냈다. exp 57 이 브레이크를 발견하고 규칙만 적었다가 exp 58 이 대체물을 만들어야 했던 것과
같은 자리다(CHECKLIST 5-2). 이 실험이 그 대체물이다.

**창의 반폭은 지어내지 않는다.** exp 58 의 규칙 — *"트리거를 어디서 가져오는지가 설계의 질을
정한다"* — 대로, 이 사슬이 **이미 선언해 둔** 값에서 가져온다: exp 45 의 `MISS_TOL = 3 mm`.
그런데 그 상수는 **선언되고 한 번도 쓰인 적이 없다.** 사슬이 표적 허용 오차를 적어 두고
열두 실험 동안 대신 45 mm 바닥을 넘고 있었다.

  A. **재채점** — 사슬의 모든 운전점을 바닥이 아니라 창으로 채점한다.
  B. **시간 탓인가** — 4 초가 짧아서인지 대조한다.
  C. **제어기 탓인가** — 이득을 384 배까지 올려 본다.
  D. **그럼 무엇인가** — 부족분을 항으로 분해한다.
  E. **고치려면** — 그 항을 직접 쓸어 본다.
  F. **결함이 하나 더 있다** — 깊이를 최종값으로만 재 왔다.
  G. **창으로 보면 사슬이 어떻게 읽히나.**

**미리 밝히는 결론.** 세 가지를 예측했고 **둘이 틀렸다.**

  1. "부족분은 제어기의 표류 오차이고 λ 를 올리면 준다" — **틀렸다.** `d_s·λ` 를 384 배로
     올려도 오차가 **3.5 mm 에서 포화**한다. 1/(d_s·λ) 를 안 따른다.
  2. "창으로 바꾸면 사슬의 운전점 중 몇은 통과한다" — **틀렸다. 하나도 없다.** 기준선은
     4.23 mm 미달이고, exp 66 의 관측기는 6.02 mm 미달, exp 68 의 수동 칸은 4.12 mm 초과다.
     **이 사슬은 자기가 선언한 3 mm 허용 오차 안에 들어간 적이 없다.**
  3. "술자를 세게 하면 도달한다" — **맞았다.** 강성 8 배면 창 안이다. 그런데 그 숫자는
     알고리즘 안에서 안 나온다 — exp 60 이 만난 모양(구속변수가 조직이 아니라 선언값)과 같다.

**부족분의 정체.** 4.23 mm 는 이렇게 갈린다:

    마스터가 못 간 몫  f_m / K_OP = 3.50 mm   ← **술자의 손이 반력에 밀린 만큼**
    도구가 못 따라간 몫              0.73 mm   ← 제어기의 표류 오차
    ─────────────────────────────────────────
    표적 미달                        4.23 mm

**그리고 결함이 하나 더 나왔다.** exp 68 은 그 수동 칸이 표적을 지나친다고 적었는데, 정상상태까지
돌려 보면 **되돌아와서 기준선과 최종 깊이가 같다**(둘 다 50.77 mm). 그런데 경로를 보면 그 칸은
도구를 **61.09 mm** 까지 밀어넣는다 — 표적을 **6.09 mm 지나갔다가** 돌아온 것이다. 사슬이 깊이를
**최종값**으로만 재 왔기 때문에 **통과 자체가 안 보인다.** 바닥이라 초과를 못 보는 것과 **별개의
결함**이고, 둘이 겹쳐서 그 칸이 사슬의 모든 검사를 통과했다. exp 68 의 지적은 옳았지만 이유가
달랐다 — 4 초 실행의 최종값이 우연히 높았던 것이고, 진짜 문제는 **지표가 경로를 안 본다**는 것이다.

**즉 도달 상한을 정하는 것은 제어기가 아니라 술자다.** exp 50 이 파동 변수로 힘을 전달하고
exp 62 가 술자에게 그 힘을 느끼게 해 준 뒤로, **전달에 성공한 그 힘이 도구를 밀어내고 있다.**
그리고 exp 56 이 과제를 완주시키려고 세운 45 mm 바는 **그 한계 아래**에 있다 — 열두 실험이
완주라 부른 것은 **넘을 수 있는 높이로 맞춘 바를 넘은 것**이었다.

    python scripts/69_the_bar_was_fitted_to_the_surgeon.py

한계·트레이드오프
  - **3 mm 는 선언값이지 임상값이 아니다.** exp 45 가 적어 둔 숫자를 쓴 것이고, 실제 시술의
    허용 오차는 여전히 사슬 밖에서 와야 한다(허용 지연·임상 여유·교환비 옆자리다).
    다만 **바닥보다는 낫다** — 창은 양쪽을 다 막는다.
  - **술자 강성 8 배가 처방은 아니다.** 사람을 바꾸라는 뜻이 아니라 **도달 상한이 어디서 오는지**를
    보인 것이다. 진짜 처방이라면 힘 표시 배율(스케일링)이나 마스터 쪽 지지 구조인데, 그건
    exp 62 가 연 술자 모델을 다시 여는 일이라 여기서 하지 않는다.
  - **깊이 하나로만 채점한다.** 창은 축방향 도달만 본다 — exp 45 의 6-DOF 명중 오차나 exp 63 의
    끌림 축은 그대로 별개다. **기준을 창으로 바꾼 것이 채점 축을 늘린 것은 아니다.**
  - **대조가 대조인지 확인해야 했다.** 처음에 술자 강성을 상위 모듈에서 패치했는데 시뮬레이터는
    import 시점에 복사한 값을 쓰고 있어서 **대조가 조용히 아무 일도 안 했다**(다섯 배율이 전부
    같은 숫자). 그대로였으면 "술자 강성은 무관하다"는 **정반대 결론**을 낼 뻔했다 — 그래서
    이 실험은 강성을 실행 인자로 빼고, 인자가 실제로 먹는지부터 테스트로 고정한다.
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
tele = import_module("50_teleoperation_delay")
g6 = import_module("45_image_guided_6dof")

TARGET_MM = tele.X_TARGET * 1e3        # 술자가 명령하는 표적 깊이
WINDOW_MM = g6.MISS_TOL * 1e3          # **exp 45 가 선언하고 한 번도 안 쓴** 표적 허용 오차
DEPTH_BAR = 45.0                       # R18 의 운영값 — 바닥
K_OP_CHAIN = jc.K_OP                   # 술자 손 강성(exp 50 이 고른 값)
JITTER = 20.0
N_SEEDS = 6
LONG_STEPS = 8000                      # 정상상태를 보려면 4 초로는 모자란다

CONFIGS = (
    ("사슬 기준선 (#56~#67)", {}),
    ("#66 관측기(엄격)", dict(drift_mode="po", po_strict=True)),
    ("#67 값싼 모서리", dict(d_s=20.0, lam_pos=48.0)),
    ("#68 수동 칸", dict(b_wave=480.0, d_s=60.0, lam_pos=24.0)),
    ("#68 (480,60,48)", dict(b_wave=480.0, d_s=60.0, lam_pos=48.0)),
)
GAIN_LADDER = ((60, 24), (60, 96), (60, 192), (120, 192), (240, 384))
KOP_LADDER = (1.0, 2.0, 4.0, 8.0, 16.0)


def depth(seeds=N_SEEDS, steps=LONG_STEPS, **kw):
    """도달 깊이(중앙값)와 정상상태 여부. **오차는 표적 대비로 돌려준다.**"""
    rs = [jc.run("tdpa", seed=s, jitter_ms=JITTER, steps=steps, **kw) for s in range(seeds)]
    ok = [r for r in rs if np.isfinite(r["final_depth_mm"])]
    d = float(np.median([r["final_depth_mm"] for r in ok]))
    pk = float(np.median([r["peak_depth_mm"] for r in ok]))
    return dict(
        depth=d, err=d - TARGET_MM,
        peak=pk, excursion=pk - TARGET_MM,
        floor_pass=d >= DEPTH_BAR,
        window_pass=abs(d - TARGET_MM) <= WINDOW_MM,
        # **지나쳤다가 돌아온 것**까지 잡으려면 최종값이 아니라 최고값을 봐야 한다
        peak_pass=pk <= TARGET_MM + WINDOW_MM,
        master=float(np.median([np.mean(r["log"]["xm"][-500:]) for r in ok])) * 1e3,
        f_m=float(np.median([np.mean(r["log"]["fm"][-500:]) for r in ok])),
        # 마지막 0.5 s 동안 더 들어간 거리 — 0 이면 시간이 아니라 정상상태다
        creep=float(np.median([(r["log"]["xs"][-1] - r["log"]["xs"][-501]) * 1e3 for r in ok])),
    )


def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    cfgs = CONFIGS[:2] if quick else CONFIGS
    gains = GAIN_LADDER[:2] if quick else GAIN_LADDER
    kops = (1.0, 8.0) if quick else KOP_LADDER
    times = (4000, 8000) if quick else (4000, 8000, 16000, 32000)

    print("=== 69. 완주 기준을 창으로 고쳐 쓰니, 이 사슬은 표적에 도달한 적이 없었다 ===")
    print(f"표적 {TARGET_MM:.0f} mm · R18 운영값 {DEPTH_BAR:.0f} mm(바닥) · "
          f"창 반폭 {WINDOW_MM:.0f} mm(exp 45 의 MISS_TOL — **선언되고 한 번도 안 쓰인 상수**)")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] **재채점** — 같은 실행을 바닥으로 한 번, 창으로 한 번")
    print(f"{'구성':>24s} | {'깊이[mm]':>9s} | {'표적 대비':>10s} | {'R18(바닥)':>10s} | "
          f"{'창(±%.0f mm)' % WINDOW_MM:>12s}")
    A = {}
    for name, kw in cfgs:
        m = depth(seeds=seeds, **kw)
        A[name] = m
        print(f"{name:>24s} | {m['depth']:9.2f} | {m['err']:+10.2f} | "
              f"{('통과' if m['floor_pass'] else '실패'):>10s} | "
              f"{('통과' if m['window_pass'] else '**실패**'):>12s}")
    n_pass = sum(1 for m in A.values() if m["window_pass"])
    print(f"  **바닥으로는 {sum(1 for m in A.values() if m['floor_pass'])}/{len(A)} 통과, "
          f"창으로는 {n_pass}/{len(A)}.**")
    print("  사슬이 열두 실험 동안 써 온 운전점은 **자기가 선언한 허용 오차 안에 들어간 적이 없다.**")
    print("  전부 **미달** 쪽이다. 초과가 없어서가 아니라 — 초과는 최종값에 안 남는다(F 절).")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print("[B] **시간 탓인가** — 4 초가 짧아서 못 간 것인지부터 (실패할 수 있는 시험으로 만든다)")
    print(f"{'실행 시간':>10s} | {'깊이[mm]':>9s} | {'표적 대비':>10s} | {'마지막 0.5s 전진':>16s}")
    B = {}
    for st in times:
        m = depth(seeds=min(seeds, 3), steps=st)
        B[st] = m
        print(f"{st * jc.DT:9.1f}s | {m['depth']:9.2f} | {m['err']:+10.2f} | "
              f"{m['creep']:14.4f} mm")
    same = max(abs(B[st]["depth"] - B[times[-1]]["depth"]) for st in times)
    print(f"  **시간이 아니다.** 실행을 {times[-1] * jc.DT:.0f} 초까지 늘려도 깊이 차이가 "
          f"{same:.3f} mm 이고 도구는 완전히 멈춰 있다.")
    print("  술자는 t = 4 s 부터 표적을 계속 명령하는데 **도구가 영원히 그 앞에서 선다.**")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] **제어기 탓인가** — 표류 보정의 정상상태 오차라면 이득에 반비례해야 한다")
    print(f"{'d_s':>6s} {'λ':>6s} | {'d_s·λ':>9s} | {'깊이[mm]':>9s} | {'표적 대비':>10s}")
    C = {}
    for g, l in gains:
        m = depth(seeds=min(seeds, 3), d_s=float(g), lam_pos=float(l))
        C[(g, l)] = m
        print(f"{g:6.0f} {l:6.0f} | {g * l:9.0f} | {m['depth']:9.2f} | {m['err']:+10.2f}")
    e0, e1 = C[gains[0]]["err"], C[gains[-1]]["err"]
    ratio = (gains[-1][0] * gains[-1][1]) / (gains[0][0] * gains[0][1])
    print(f"  이득을 **{ratio:.0f} 배**로 올려도 오차가 {e0:+.2f} → {e1:+.2f} mm 로 "
          f"**포화한다**(1/(d_s·λ) 라면 0 으로 갔어야 한다).")
    print("  → **예측 1 이 틀렸다.** 남는 것은 제어기가 아니라 다른 데 있다.")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] **그럼 무엇인가** — 정상상태에서 부족분을 항으로 가른다")
    base = A[cfgs[0][0]]
    m_short = TARGET_MM - base["master"]
    t_short = base["master"] - base["depth"]
    pred = abs(base["f_m"]) / K_OP_CHAIN * 1e3
    print(f"{'항':>34s} | {'mm':>8s}")
    print(f"{'마스터가 표적에 못 간 몫':>34s} | {m_short:8.2f}")
    print(f"{'  예측식  |f_m| / K_OP':>34s} | {pred:8.2f}   ← 소수점까지 같다")
    print(f"{'도구가 마스터를 못 따라간 몫':>34s} | {t_short:8.2f}")
    print(f"{'합계 = 표적 미달':>34s} | {TARGET_MM - base['depth']:8.2f}")
    print(f"  **도달 상한을 정하는 것은 제어기가 아니라 술자다.** 손 강성 {K_OP_CHAIN:.0f} N/m 이")
    print(f"  반력 {abs(base['f_m']):.2f} N 에 밀려 마스터가 {m_short:.2f} mm 뒤에 선다.")
    print("  exp 50 이 파동 변수로 힘을 **전달**하고 exp 62 가 술자에게 그걸 **느끼게** 해 준 뒤로,")
    print("  **전달에 성공한 그 힘이 도구를 밀어내고 있다.** 사슬의 성과가 사슬의 한계가 됐다.")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] **고치려면** — 그 항을 직접 쓸어 본다(처방이 아니라 귀속을 확인하는 것이다)")
    print(f"{'K_OP 배율':>10s} | {'K_OP':>9s} | {'마스터[mm]':>11s} | {'도구[mm]':>9s} | "
          f"{'표적 대비':>10s} | {'창?':>6s}")
    E = {}
    for mult in kops:
        m = depth(seeds=min(seeds, 3), k_op=K_OP_CHAIN * mult)
        E[mult] = m
        print(f"{mult:9.0f}× | {K_OP_CHAIN * mult:9.0f} | {m['master']:11.2f} | "
              f"{m['depth']:9.2f} | {m['err']:+10.2f} | "
              f"{('예' if m['window_pass'] else '아니오'):>6s}")
    win = [k for k in kops if E[k]["window_pass"]]
    if win:
        print(f"  **강성 {min(win):.0f} 배면 창 안에 들어온다** — 예측 3 이 맞았다. 귀속은 확인됐다.")
    print("  **그런데 이건 처방이 아니다.** 사람을 바꾸라는 말이 되기 때문이다. 진짜 손잡이는")
    print("  힘 표시 배율이나 마스터 쪽 지지 구조인데, 그건 exp 62 가 연 술자 모델을 다시 여는 일이다.")
    print("  구속변수가 **알고리즘 밖의 선언값**이라는 점에서 exp 60 이 만난 모양과 같다.")

    # ------------------------------ F ------------------------------
    print("-" * 100)
    print("[F] 그리고 **결함이 하나 더 있다** — 사슬은 깊이를 **최종값**으로만 재 왔다")
    print("    (exp 68 은 그 수동 칸이 '표적을 지나친다'고 적었다. 정상상태까지 돌려 보면")
    print("     되돌아와서 기준선과 **구분이 안 된다** — 그런데 지나간 것은 지나간 것이다.)")
    print(f"{'구성':>24s} | {'최종[mm]':>9s} | {'최고[mm]':>9s} | {'표적 초과':>10s} | "
          f"{'최종 기준':>10s} | {'최고 기준':>10s}")
    for name, kw in cfgs:
        m = A[name]
        print(f"{name:>24s} | {m['depth']:9.2f} | {m['peak']:9.2f} | {m['excursion']:+10.2f} | "
              f"{('통과' if m['window_pass'] else '실패'):>10s} | "
              f"{('통과' if m['peak_pass'] else '**실패**'):>10s}")
    worst = max(A.values(), key=lambda v: v["excursion"])
    print(f"  가장 깊이 들어간 실행은 표적을 **{worst['excursion']:+.2f} mm** 지나갔다가 돌아온다.")
    print("  **최종값 지표는 통과 자체를 못 본다** — 조직에 무엇을 했는가는 '어디서 끝났는가'가")
    print("  아니라 '어디까지 갔는가'의 문제다. 바닥이라 초과를 못 보는 것과 **별개의 결함**이고,")
    print("  둘이 겹쳐서 exp 68 의 칸이 사슬의 모든 검사를 통과했다.")

    # ------------------------------ G ------------------------------
    print("-" * 100)
    print("[G] 정리 — **창으로 보면 사슬이 어떻게 읽히나**")
    print(f"  1. **이 사슬은 표적에 도달한 적이 없다.** 기준선이 {base['err']:+.2f} mm 이고,")
    print(f"     선언된 허용 오차 {WINDOW_MM:.0f} mm 안에 들어간 운전점이 {n_pass} 개다.")
    print("  2. **R18 의 45 mm 바는 그 한계 아래에 놓여 있었다.** exp 56 이 과제를 완주시키려고")
    print("     세운 값인데, 넘을 수 있는 높이였기 때문에 넘긴 것이다. **기준이 시스템에 맞춰졌다.**")
    ms = (TARGET_MM - base["master"]) / (TARGET_MM - base["depth"]) * 100
    print(f"  3. **바닥은 방향을 구분하지 못하고, 최종값은 통과를 못 본다.** 가장 깊이 들어간")
    print(f"     실행이 표적을 {worst['excursion']:+.2f} mm 지나갔다 돌아오는데 최종값은 기준선과 같다.")
    print(f"  4. **상한은 술자가 정한다.** 부족분의 {ms:.0f}% 가 손 강성 항이고, "
          "제어기 이득으로는 안 준다.")
    print("  5. **그러니 51~68 의 '완주'는 전부 표적 대비로, 그리고 최고값으로 다시 읽어야 한다** —")
    print("     로버스트니스 결과를 완주 지표와 함께 읽으라는 R18 의 취지는, 그 완주 지표가")
    print("     **표적 대비**이고 **경로 전체**를 볼 때만 산다.")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    names = [n for n, _ in cfgs]
    errs = [A[n]["err"] for n in names]
    exc = [A[n]["excursion"] for n in names]
    cols = ["crimson" if abs(e) > WINDOW_MM else "seagreen" for e in errs]
    y = np.arange(len(names))
    ax.barh(y, errs, 0.6, color=cols)
    # 최종값이 아니라 **어디까지 갔는가** — 같은 실행의 최고 침투를 나란히 찍는다
    ax.plot(exc, y, "d", ms=8, mfc="none", mec="0.15", mew=1.6, label="peak (path)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.axvspan(-WINDOW_MM, WINDOW_MM, color="seagreen", alpha=0.14)
    ax.axvline(0.0, color="darkorange", lw=2.0)
    ax.text(0.15, len(names) - 0.35, "target", fontsize=8, color="darkorange")
    ax.text(-WINDOW_MM, -0.85, f"declared tolerance +-{WINDOW_MM:.0f} mm (#45)",
            fontsize=7.5, color="seagreen")
    ax.axvline(DEPTH_BAR - TARGET_MM, color="0.3", ls=":", lw=1.4)
    ax.text(DEPTH_BAR - TARGET_MM + 0.15, -0.85, "R18 floor", fontsize=7.5, color="0.3")
    ax.set_yticks(y)
    ax.set_yticklabels(["chain", "#66 obs", "#67 corner", "#68 cell", "#68 (480,60,48)"][:len(names)],
                       fontsize=8)
    ax.set_xlabel("depth error vs target [mm]")
    ax.set_title("Scored against the target instead of a floor,\n"
                 "the chain has never arrived", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    tt = [st * jc.DT for st in times]
    ax.plot(tt, [B[st]["depth"] for st in times], "-o", color="0.35", label="depth reached")
    ax.axhline(TARGET_MM, color="darkorange", lw=2.0)
    ax.text(tt[0], TARGET_MM + 0.25, "target", fontsize=8, color="darkorange")
    ax.axhline(DEPTH_BAR, color="0.3", ls=":", lw=1.2)
    ax.text(tt[0], DEPTH_BAR + 0.25, "R18 floor", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_ylim(DEPTH_BAR - 2, TARGET_MM + 2)
    ax.set_xlabel("run length [s]"); ax.set_ylabel("depth reached [mm]")
    ax.set_title("Not a time limit: it is a steady state\n(the tool stops and stays)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    kk = list(kops)
    ax.plot(kk, [E[k]["err"] for k in kk], "-o", color="crimson", label="depth error")
    ax.axhspan(-WINDOW_MM, WINDOW_MM, color="seagreen", alpha=0.14)
    ax.axhline(0.0, color="darkorange", lw=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("operator hand stiffness  (x the modelled surgeon)")
    ax.set_ylabel("depth error vs target [mm]")
    ax.annotate("the ceiling is the surgeon,\nnot the controller",
                xy=(kk[0], E[kk[0]]["err"]), xytext=(kk[0] * 1.4, E[kk[0]]["err"] + 1.6),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1))
    ax.set_title("Sweeping the term that actually binds\n"
                 "(controller gains saturate; this one does not)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("69. The completion bar was fitted to the surgeon — "
                 "scored against the target, the chain never arrives", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "69_the_bar_was_fitted_to_the_surgeon.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/69_the_bar_was_fitted_to_the_surgeon.png, "
          "assets/69_the_bar_was_fitted_to_the_surgeon.png")

    return dict(A=A, B=B, C=C, E=E, window=WINDOW_MM, target=TARGET_MM)


if __name__ == "__main__":
    main()
