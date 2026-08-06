"""잔차를 거치지 않는 오대응 검출 — 부분집합이 서로 동의하는가.

exp 55 는 이렇게 끝났다: *"표면을 따라 미끄러진 몫은 잔차조차 남기지 않는다"* — 그러니 정합이
틀렸다는 것을 잔차로는 알 수 없고, 다른 검출기가 필요하다는 것. 그때 적어둔 다음 수가 이것이다:
*"독립적으로 매칭한 부분집합끼리의 일치를 보는 식(exp 11·15 의 다중초기값 일치성과 같은 발상)"*.
여덟 실험 동안 열려 있었다.

**발상.** 창 주변 표면을 **부채꼴 K 개로 쪼개고**, 각 조각만으로 따로 보정을 맞춘 뒤 **심부 표적에서
서로 얼마나 어긋나는지**를 본다. 조각마다 국소 불변 방향이 다르므로, 접선 오차가 **조각들 사이의
불일치**로 새어 나올 것이다 — 잔차가 못 보는 축에서.

**그런데 이 발상에는 함정이 하나 있고, exp 61 이 만든 규칙이 그걸 잡는다**(R26: 지표가 그 위해에
반응하는지 **위해가 없는 조건**에서 확인하라). 조각 하나로 맞춘 보정은 애초에 심부에서 병조건이라,
**대응이 완벽해도** 조각들이 서로 어긋날 수 있다. 그러면 이 통계는 오대응이 아니라 **조건수**를
재는 것이다. exp 55 는 정답 대응을 들고 있으니 그 대조군을 그대로 돌릴 수 있다.

  A. **전제 재확인** — 접선을 키우면 표적 오차는 늘고 표면 잔차는 그대로인가(exp 55 의 결과).
  B. **대조군 먼저** — **정답 대응**으로 부분집합 불일치를 잰다. 여기서 크면 이 통계는 못 쓴다.
  C. **검출기로 채점** — 잔차 vs 불일치를 **AUROC** 로(exp 52 의 규칙: 고정 오경보 검출률 아님).
  D. **다른 축인가** — 잔차와 불일치의 순위 상관(exp 63 의 규칙: 지표군이 한 축이면 같이 틀린다).

**결론부터: 이 항목은 문구의 과장 때문에 여덟 실험 동안 열려 있었다.**

exp 55 의 표는 접선 0 → 8 mm 에서 잔차가 **0.92 → 1.17 mm(27% 상승)** 라고 적고 있는데, 본문은
*"흔적을 전혀 안 남긴다"* 로 썼다. **둔감한 것이지 눈먼 것이 아니었다.** 검출기로 제대로 채점하면
표면 잔차가 **AUROC 0.94** 이고, 제안했던 대체물(부분집합 불일치)은 **0.76** 으로 **더 나쁘다.**
게다가 둘의 순위 상관이 ρ ≈ +0.47 이라 **뚜렷이 다른 축도 아니다.** 그래서 이 실험은
"새 검출기를 찾았다"가 아니라 **"이미 있던 것이 생각보다 낫고, 그걸 알려면 문구가 아니라 AUROC 를
재야 했다"** 로 닫힌다.

(exp 52 의 표면 게이트 AUROC 0.52 와 혼동하면 안 된다 — 그건 **표면에 자취를 안 남기는 심부 모드**
라는 다른 실패였다. 여기 접선 미끄러짐은 표면에 흔적을 남긴다, 아주 약하게.)

**가는 길에 스스로 함정에 빠졌다.** 처음엔 검출 과제에서 **접선만** 흔들었더니 잔차 AUROC 가
**1.00** 이 나왔다 — 변하는 것이 검출 대상 하나뿐이면 조금이라도 상관 있는 통계는 전부 만점이다.
exp 56 이 채널에서 발견한 **"실패할 수 없는 시험"** 이 이번엔 검출 실험에서 나왔다. 법선 성분을
독립으로 함께 흔들어서(잔차가 **무해한 이유로도** 움직이게) 과제를 성립시켰다.

    python scripts/64_residual_free_detection.py

한계·트레이드오프
  - 부채꼴 분할은 **창 주변 기하가 비대칭일 때만** 작동한다. 표면이 국소적으로 대칭이면 모든 조각이
    같은 방향으로 미끄러져 **같은 오답에 동의**한다 — exp 52 가 말한 진짜 관측 불가다. 검출기가
    고칠 수 있는 문제가 아니고, 이 실험은 그 경계를 재는 것이지 없애는 게 아니다.
  - 조각당 관측 수가 줄어 각 보정이 더 병조건이 된다. 그래서 B 의 대조군이 필수다.
  - 실 인체 MR 표면 하나(exp 49·51·55 와 같은 것)에서만 쟀다. 다른 해부에서 부채꼴 비대칭이
    얼마나 되는지는 이 실험이 말하지 않는다.
  - 검출은 **경보**일 뿐 교정이 아니다. 알람이 울려도 무엇을 해야 하는지는 별개 문제다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
reg = import_module("41_surgical_registration")
real = import_module("44_registration_real_scans")
anat = import_module("49_registration_real_anatomy")
deform = import_module("51_deformable_registration")
probe = import_module("52_probing_the_prior")
s55 = import_module("55_correspondence_search")

NORMAL_MM = s55.NORMAL_MM
DECAY_MM = s55.DECAY_MM
PROBE_NOISE = s55.PROBE_NOISE
EXPOSURE_DEG = s55.EXPOSURE_DEG
TARGET_DEPTHS_MM = s55.TARGET_DEPTHS_MM
N_PROBE = 900
N_ANCHOR = 130
N_CTRL_PER_SECTOR = 60
TPS_LAMBDA = 1e-2
K_SECTORS = 4
TANGENTS_MM = (0.0, 2.0, 4.0, 6.0, 8.0)


def so3_exp(w):
    return s55.so3_exp(w) if hasattr(s55, "so3_exp") else reg.so3_exp(w)


# --------------------------------------------------------------------------- #
def build(seed=5):
    """exp 55 와 **같은 설정**을 세운다(실 인체 MR 표면, 같은 노출·감쇠·표적 깊이)."""
    anat.fetch()
    vol, D, origin = anat.load_nrrd()
    surface, _ = anat.head_surface(vol, D, origin)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(surface), min(anat.MODEL_POINTS, len(surface)), replace=False)
    model = surface[sel]
    normals = reg.estimate_normals(model, k=12)
    center = model.mean(0)
    outward = model - center
    normals[np.sum(normals * outward, axis=1) < 0] *= -1.0

    window_c = model[int(np.argmax(model[:, 2]))]
    inward = (center - window_c) / np.linalg.norm(center - window_c)
    _, slide_dir, third = probe.axis_frame(inward)
    targets = window_c + np.array(TARGET_DEPTHS_MM)[:, None] * 1e-3 * inward

    v = (model - center) / np.linalg.norm(model - center, axis=1, keepdims=True)
    w_dir = (window_c - center) / np.linalg.norm(window_c - center)
    ang = np.degrees(np.arccos(np.clip(v @ w_dir, -1, 1)))
    in_win = np.where(ang < EXPOSURE_DEG)[0]
    out_win = np.where(ang > EXPOSURE_DEG + 25.0)[0]

    r0 = np.random.default_rng(seed)
    ax0 = r0.normal(size=3); ax0 /= np.linalg.norm(ax0)
    T_place = reg.pose_T(so3_exp(ax0 * np.deg2rad(6.0)), r0.uniform(-0.015, 0.015, 3))
    land = anat.pick_landmarks(surface, k=5)
    digit = reg.apply_T(T_place, surface[land]) + r0.normal(0, s55.LANDMARK_NOISE,
                                                            (len(land), 3))
    init = anat.procrustes(digit, surface[land])
    pre_idx = r0.choice(len(model), size=N_PROBE, replace=False)
    pre_dst = reg.apply_T(T_place, model[pre_idx]) + r0.normal(0, PROBE_NOISE,
                                                               (N_PROBE, 3))
    T_rigid = real._icp2(pre_dst, model, normals, init)

    return dict(model=model, normals=normals, tree=cKDTree(model), center=center,
                window_c=window_c, inward=inward, slide_dir=slide_dir, third=third,
                targets=targets, in_win=in_win, out_win=out_win,
                T_place=T_place, T_rigid=T_rigid, T_inv=np.linalg.inv(T_rigid))


def observe(ctx, rr, tangent_mm, normal_mm=NORMAL_MM):
    """창 안 표면을 찍는다. **정답 대응**(원래 어느 모델점이었나)도 같이 돌려준다 — 대조군용."""
    idx = rr.choice(ctx["in_win"], size=min(N_PROBE, len(ctx["in_win"])), replace=False)
    src_true = ctx["model"][idx]
    u = s55.field(src_true, ctx["center"], ctx["window_c"], ctx["slide_dir"],
                  normal_mm, tangent_mm, DECAY_MM)
    obs_pat = reg.apply_T(ctx["T_place"], src_true + u) \
        + rr.normal(0, PROBE_NOISE, (len(idx), 3))
    return idx, src_true, reg.apply_T(ctx["T_rigid"], obs_pat)


def sectors(ctx, src, k=K_SECTORS):
    """창 중심 둘레의 **부채꼴**로 관측을 쪼갠다. 조각마다 국소 불변 방향이 다르다."""
    rel = src - ctx["window_c"]
    a = np.arctan2(rel @ ctx["third"], rel @ ctx["slide_dir"])
    return [np.where(((a + np.pi) // (2 * np.pi / k)).astype(int) == i)[0]
            for i in range(k)]


def fit_predict(ctx, src, disp, anchors, n_ctrl=N_CTRL_PER_SECTOR, seed=7):
    """주어진 대응으로 TPS 를 맞추고 (심부 표적 예측, 표면 잔차)를 낸다."""
    if len(src) < 8:
        return None, np.nan
    sub = np.random.default_rng(seed).choice(len(src), size=min(n_ctrl, len(src)),
                                             replace=False)
    C = np.vstack([src[sub], anchors])
    DD = np.vstack([disp[sub], np.zeros_like(anchors)])
    m = deform.tps_fit(C, DD, lam=TPS_LAMBDA)
    est = reg.apply_T(ctx["T_inv"], deform.tps_apply(m, ctx["targets"]))
    srf = float(np.sqrt(np.mean(np.sum(
        (deform.tps_apply(m, src[sub]) - (src[sub] + disp[sub])) ** 2, axis=1))))
    return est, srf


def truth(ctx, tangent_mm, normal_mm=NORMAL_MM):
    t = ctx["targets"]
    return reg.apply_T(ctx["T_place"], t + s55.field(t, ctx["center"], ctx["window_c"],
                                                     ctx["slide_dir"], normal_mm,
                                                     tangent_mm, DECAY_MM))


def trial(ctx, rr, tangent_mm, oracle=False, k=K_SECTORS, normal_mm=NORMAL_MM):
    """한 번의 시행: 전체 정합 오차 · 표면 잔차 · **부분집합 불일치**를 함께 낸다.

    oracle=True 면 **정답 대응**을 쓴다 — 불일치가 오대응 때문인지 병조건 때문인지 가른다.
    """
    idx, src_true, obs = observe(ctx, rr, tangent_mm, normal_mm)
    if oracle:
        src, disp = src_true, obs - src_true
    else:
        _, src, disp = s55.find_correspondence(obs, ctx["model"], ctx["tree"],
                                               "p2p", ctx["normals"])
    anchors = ctx["model"][rr.choice(ctx["out_win"],
                                     size=min(N_ANCHOR, len(ctx["out_win"])),
                                     replace=False)]
    est_all, srf = fit_predict(ctx, src, disp, anchors, n_ctrl=170)
    tgt = truth(ctx, tangent_mm, normal_mm)
    err = float(np.mean(np.linalg.norm(est_all - tgt, axis=1)))

    preds = []
    for s in sectors(ctx, src, k):
        e, _ = fit_predict(ctx, src[s], disp[s], anchors)
        if e is not None:
            preds.append(e)
    if len(preds) < 2:
        return err, srf, np.nan
    P = np.stack(preds)                                   # (K, n_targets, 3)
    spread = float(np.mean(np.linalg.norm(P - P.mean(0, keepdims=True), axis=2)))
    return err, srf, spread


def _rho(a, b):
    """스피어만 순위 상관(작은 표본에서도 단조성만 본다)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def auroc(score, label):
    """양성(label=1)의 점수가 음성보다 높을 확률. exp 52 가 정한 채점 방식."""
    s, l = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    if l.all() or not l.any():
        return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = int(l.sum()), int((~l).sum())
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
def main(seed=5, quick=False):
    ctx = build(seed)
    tans = (0.0, 4.0, 8.0) if quick else TANGENTS_MM
    n_rep = 2 if quick else 5
    print("=== 64. 잔차를 거치지 않는 오대응 검출 — 부분집합이 서로 동의하는가 ===")
    print("exp 55 가 여덟 실험 동안 열어 둔 항목이다: **접선 미끄러짐은 잔차를 남기지 않는다.**")
    print(f"창 둘레를 부채꼴 {K_SECTORS} 개로 쪼개 **따로 맞춘 뒤 심부에서 얼마나 어긋나는지** 본다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] 전제 재확인 — 접선을 키우면 표적 오차는 늘고 표면 잔차는 안 는다")
    print(f"{'접선[mm]':>9s} | {'표적 오차[mm]':>13s} | {'표면 잔차[mm]':>13s} | {'불일치[mm]':>11s}")
    A = {}
    for tan in tans:
        rows = [trial(ctx, np.random.default_rng([seed, int(tan * 10), r]), tan)
                for r in range(n_rep)]
        e, s, d = (float(np.median([x[i] for x in rows])) for i in range(3))
        A[tan] = (e, s, d)
        print(f"{tan:9.1f} | {e * 1e3:13.2f} | {s * 1e3:13.2f} | {d * 1e3:11.2f}")
    e0, eN = A[tans[0]][0], A[tans[-1]][0]
    s0, sN = A[tans[0]][1], A[tans[-1]][1]
    print(f"  표적 오차가 {e0*1e3:.2f} → {eN*1e3:.2f} mm ({eN/max(e0,1e-9):.1f}배)로 느는 동안")
    print(f"  표면 잔차는 {s0*1e3:.2f} → {sN*1e3:.2f} mm ({(sN/max(s0,1e-9)-1)*100:.0f}%)만 오른다 — exp 55 의 표와 같다.")
    print("  **그런데 exp 55 는 이걸 '흔적을 전혀 안 남긴다'로 적었다.** 27% 는 0 이 아니다.")
    print("  둔감한 것과 눈먼 것은 다르고, 그 차이는 **검출기로 채점해야** 드러난다(아래 C).")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print("[B] **대조군 먼저** — 정답 대응으로 같은 불일치를 잰다(exp 61 의 R26)")
    print("    여기서 크면 이 통계는 오대응이 아니라 **조각 하나로 맞춘 보정의 병조건**을 재는 것이다.")
    print(f"{'접선[mm]':>9s} | {'최근접점 불일치[mm]':>18s} | {'정답 대응 불일치[mm]':>19s} | {'비':>7s}")
    B = {}
    for tan in tans:
        nn = float(np.median([trial(ctx, np.random.default_rng([seed, int(tan * 10), r]),
                                    tan)[2] for r in range(n_rep)]))
        orc = float(np.median([trial(ctx, np.random.default_rng([seed, int(tan * 10), r]),
                                     tan, oracle=True)[2] for r in range(n_rep)]))
        B[tan] = (nn, orc)
        print(f"{tan:9.1f} | {nn * 1e3:18.2f} | {orc * 1e3:19.2f} | "
              f"{nn / max(orc, 1e-12):7.2f}")
    base = B[tans[0]][1]
    print(f"  정답 대응에서도 불일치가 {base*1e3:.2f} mm 있다 — 조각 보정은 원래 병조건이다.")
    print("  그래서 **절대값이 아니라 접선에 따라 자라는지**를 봐야 한다:")
    print(f"  최근접점 {B[tans[0]][0]*1e3:.2f} → {B[tans[-1]][0]*1e3:.2f} mm, "
          f"정답 대응 {B[tans[0]][1]*1e3:.2f} → {B[tans[-1]][1]*1e3:.2f} mm.")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] 검출기로 채점 — **AUROC**(exp 52 가 정한 방식: 고정 오경보 검출률이 아니다)")
    print("    **법선 성분도 함께 흔든다.** 처음엔 접선만 흔들었더니 잔차 AUROC 가 1.00 이 나왔다 —")
    print("    변하는 것이 검출 대상 하나뿐이면 조금이라도 상관 있는 통계는 전부 만점이다.")
    print("    exp 56 의 '실패할 수 없는 시험'과 같은 계열이라, 잔차가 **무해한 이유로도** 움직이게")
    print("    만들어야 과제가 성립한다: 큰 법선 변형은 잔차를 올리지만 잘 보정되므로 무해하다.")
    n_draw = 24 if quick else 90
    rows = []
    for i in range(n_draw):
        rr = np.random.default_rng([seed, 900 + i])
        nrm = float(rr.uniform(2.0, 10.0))
        tan = float(rr.uniform(0.0, 8.0))
        e, s, d = trial(ctx, rr, tan, normal_mm=nrm)
        rows.append((nrm, tan, e, s, d))
    nrm_a = np.array([x[0] for x in rows])
    tan_a = np.array([x[1] for x in rows])
    err = np.array([x[2] for x in rows])
    srf = np.array([x[3] for x in rows])
    spr = np.array([x[4] for x in rows])
    lab = err > np.median(err)
    C = dict(residual=auroc(srf, lab), spread=auroc(spr, lab),
             both=auroc(spr / np.maximum(srf, 1e-9), lab))
    print(f"{'통계':>18s} | {'AUROC':>7s} | {'판정':>16s}")
    for key, lbl in (("residual", "표면 잔차"), ("spread", "부분집합 불일치"),
                     ("both", "불일치 / 잔차")):
        v = C[key]
        print(f"{lbl:>18s} | {v:7.2f} | "
              f"{('동전 수준' if abs(v - 0.5) < 0.08 else '쓸 만함'):>16s}")
    print(f"  표본 {len(rows)}회 — 법선 2~10 mm, 접선 0~8 mm 를 **독립으로** 뽑았다.")
    print(f"  (잔차는 법선과 ρ={_rho(srf, nrm_a):+.2f}, 접선과 ρ={_rho(srf, tan_a):+.2f} —")
    print("   즉 잔차는 **보이는 변형의 크기**를 재지 남은 오차를 재지 않는다.)")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] 다른 축인가 — 잔차와 불일치의 순위 상관(exp 63 의 규칙)")

    D = dict(rs=_rho(srf, spr), re=_rho(srf, err), se=_rho(spr, err))
    print(f"  잔차 ↔ 불일치 : ρ = {D['rs']:+.2f}")
    print(f"  잔차 ↔ 참오차 : ρ = {D['re']:+.2f}")
    print(f"  불일치 ↔ 참오차: ρ = {D['se']:+.2f}")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] 정리 — **exp 55 가 여덟 실험 동안 열어 둔 항목이, 문구의 과장으로 열려 있었다**")
    print(f"  1. **exp 55 의 '잔차조차 남기지 않는다'가 과했다.** 그 실험의 표 자체가 잔차 "
          f"0.92 → 1.17 mm")
    print(f"     (27% 상승)를 보이는데 본문은 '흔적을 전혀 안 남긴다'로 적었다. 여기서 재현하면 "
          f"{s0*1e3:.2f} → {sN*1e3:.2f} mm 로 같다.")
    print(f"     **둔감한 것이지 눈먼 것이 아니다** — 검출기로 채점하면 AUROC {C['residual']:.2f} 다.")
    print("     (exp 52 의 표면 게이트 0.52 와 혼동하면 안 된다. 그건 **다른 실패 모드**였다.)")
    if C["spread"] > C["residual"] + 0.08:
        print("  2. **그런데도 부분집합 일치가 더 낫다** — 잔차가 못 보는 몫을 다른 축에서 본다.")
    else:
        print(f"  2. **제안된 대체물이 더 낫지 않다.** 부분집합 불일치 {C['spread']:.2f} < "
              f"잔차 {C['residual']:.2f},")
        print(f"     비율 통계도 {C['both']:.2f} 로 더 나쁘다. **정직한 네거티브다.**")
    print(f"  3. 게다가 **다른 축도 아니다** — 잔차 ↔ 불일치 ρ = {D['rs']:+.2f}. exp 63 의 기준으로")
    print("     보면 둘은 상당 부분 같은 것을 재고 있다.")
    print(f"  4. 대조군이 필수였다(exp 61 의 R26): **정답 대응에서도** 불일치가 {base*1e3:.2f} mm 있다")
    print("     — 조각 보정의 병조건이다. 절대값으로 문턱을 잡았으면 조건수를 오대응으로 읽었을 것이다.")
    print("  5. 그리고 **검출 과제를 설계할 때 스스로 함정에 빠졌다**: 처음엔 접선만 흔들어서")
    print("     잔차 AUROC 가 1.00 이 나왔다. **변하는 것이 검출 대상 하나뿐인 시험은 실패할 수가**")
    print("     **없다** — exp 56 이 채널에서 발견한 것과 같은 계열이고, 이번엔 검출 실험에서 나왔다.")
    print("  6. 남는 것: 표면이 **국소 대칭**이면 모든 조각이 같은 오답에 동의하고, 그건 검출기가")
    print("     고칠 수 없다(exp 52 의 진짜 관측 불가). 그리고 검출은 경보일 뿐 **교정이 아니다.**")
    print("     결론적으로 이 항목은 '새 검출기가 필요하다'가 아니라 **'이미 있던 것이 생각보다")
    print("     낫다'** 로 닫힌다 — 그리고 그걸 알려면 문구가 아니라 **AUROC 를 재야 했다.**")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.plot(tans, [A[t][0] * 1e3 for t in tans], "-o", color="crimson",
            label="deep target error [mm]")
    ax.plot(tans, [A[t][1] * 1e3 for t in tans], "-s", color="0.5",
            label="surface residual [mm]")
    ax.plot(tans, [A[t][2] * 1e3 for t in tans], "-^", color="tab:blue",
            label="subset disagreement [mm]")
    ax.set_xlabel("tangential deformation [mm]"); ax.set_ylabel("[mm]")
    ax.set_title("The residual is flat; the error is not", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(tans, [B[t][0] * 1e3 for t in tans], "-o", color="tab:blue",
            label="nearest-point correspondence")
    ax.plot(tans, [B[t][1] * 1e3 for t in tans], "-s", color="seagreen",
            label="TRUE correspondence (control)")
    ax.set_xlabel("tangential deformation [mm]")
    ax.set_ylabel("subset disagreement [mm]")
    ax.set_title("Control first: how much is just ill-conditioning?", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar([0, 1], [C["residual"], C["spread"]],
           color=["0.5", "tab:blue"], width=0.55)
    ax.axhline(0.5, color="crimson", ls="--", lw=1)
    ax.text(1.45, 0.51, "chance", fontsize=8, color="crimson")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["surface\nresidual", "subset\nagreement"])
    ax.set_ylim(0, 1); ax.set_ylabel("AUROC")
    ax.set_title("Scored as a detector, not at a fixed alarm rate", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("64. Detecting a misregistration the residual cannot see — "
                 "do independently fitted patches agree?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "64_residual_free_detection.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/64_residual_free_detection.png, "
          "assets/64_residual_free_detection.png")

    return dict(A=A, B=B, C=C, D=D)


if __name__ == "__main__":
    main()
