"""가정을 관측으로 검증하기: 표면이 볼 수 없는 것은 표면 알고리즘으로 못 고친다.

실험 51 은 좁은 개두창에서 **사전지식 하나**("창 밖 두피는 두개골이 잡고 있으니 변위 0")가
정교한 보간기보다 크게 이긴다는 것을 보였다(3.31 → 0.60 mm). 그런데 같은 실험이 곧바로 구멍을
드러냈다 — **그 사전지식은 정작 그것이 적용되는 데이터로 검증되지 않는다.**

여기서는 그 구멍을 정면으로 친다. 표면에서 **원리적으로 보이지 않는** 변형 모드를 하나 넣고,
수술 중 초음파(iUS)로 개두창 아래 몇 지점의 변위를 재는 관측을 추가한다.

--- 환자 두 종류 ---
  - **표면설명형**: exp 51 과 같은 변형. 심부의 움직임이 표면 움직임의 매끄러운 연장이다.
  - **심부 모드**: 거기에 깊이 45 mm 근처에 국한된 **횡방향** 성분(뇌실 허탈·심부 조직 이완처럼
    표면과 다른 방향으로 미끄러지는 모드)을 더한다. 표면에 남기는 자취는 평균 0.03 mm —
    프로브 잡음 1.0 mm 아래라 **관측 자체가 불가능**하다. 표적에서는 2~6 mm다.

--- 답하려는 세 질문 ---
1. 표면만으로 그 모드를 복원할 수 있는가? (없다 — 사전지식이 있든 없든)
2. 심부 관측 몇 개면 **교정**되는가?
3. 심부 관측 몇 개면 **가정이 틀렸다는 걸 알 수 있는가**? — 그리고 둘의 값이 같은가?

--- 미리 말해두는 결론 ---
1. 심부 모드는 창 안 표면에 평균 0.03 mm 자취만 남긴다(프로브 잡음 1.0 mm 아래). 표적에서는
   2~6 mm다. 표면 데이터로는 **사전지식이 있든 없든 3.5 / 3.3 mm에 갇힌다** — 알고리즘 문제가
   아니라 **관측성** 문제다. exp 4·37 의 IMU 바이어스, exp 43·46 의 스틱션과 같은 얼굴이다.
2. 그런데 **표면 잔차는 두 환자에서 똑같다**(1.74 vs 1.74 mm, 심부 오차는 5.5배 차이).
   지금 가진 유일한 지표로 게이트를 만들면 AUROC 0.52 — 동전 던지기다.
3. 심부 관측 1개면 판별 AUROC 0.52 → **0.81**, 2개 0.85, 3개 0.90. 표적 근처에 두면 아무 데나
   두는 것(0.71)보다 낫지만 **차이는 완만하다** — 여기 심부 모드가 넓어서 원뿔 어디서나 어느
   정도 보이기 때문이다. 같은 관측을 **교정**에 쓰면 3.10 → 2.49 → 2.22 mm(아직 허용치 밖).
   **첫 관측 몇 개는 고치는 것보다 아는 것을 더 많이 산다.**
4. 그래도 게이트 적응이 '항상 4개 받기'를 이기지는 못했다. 관측이 싸면 그냥 다 받는 게 낫고,
   게이트의 값은 관측이 비쌀 때와 **고칠 수 없을 때 경고를 낼 수 있다**는 데 있다.

    python scripts/52_probing_the_prior.py
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

from sensor_fusion.se3 import so3_exp  # noqa: E402

reg = import_module("41_surgical_registration")
real = import_module("44_registration_real_scans")
anat = import_module("49_registration_real_anatomy")
deform = import_module("51_deformable_registration")     # tps_fit / tps_apply 재사용

SAG_MM = 12.0             # exp 51 과 같은 개두창 침하
BULGE_MM = 5.0
DECAY_MM = 45.0
RETRACT_MM = 22.0
DEEP_MM = 6.0             # 심부 모드 크기
DEEP_DEPTH_MM = 45.0      # 심부 모드 중심 깊이
DEEP_SIGMA_MM = 25.0      # 심부 모드 폭 — 표면까지 닿지 않을 만큼 좁다

PROBE_NOISE = 1.0e-3      # 표면 디지타이징 잡음 σ [m]
US_NOISE = 1.5e-3         # 수술 중 초음파 변위 관측 잡음 σ [m] — 표면보다 나쁘다
US_DEPTH_RANGE = (15e-3, 85e-3)
US_CONE_DEG = 20.0        # 개두창 아래 원뿔 안에서만 볼 수 있다
N_PROBE = 900
N_TPS_CTRL = 170
N_TPS_ANCHOR = 130
TPS_LAMBDA = 1e-2         # exp 51 의 λ 스윕에서 이 노출(45°)의 최적값
LANDMARK_NOISE = 2.5e-3
EXPOSURE_DEG = 45.0
TARGET_DEPTHS_MM = (20, 35, 50, 70)
MISS_TOL = 2.0e-3         # 심부 표적 허용오차 [m]
GATE_TOL = 4.5e-3         # 검산 관측 잔차 게이트 [m] ≈ 3σ_us

N_SWEEP = (0, 1, 2, 4, 8, 16, 32)
METHODS = ("prior", "data", "both")
LABEL = {"prior": "surface + prior", "data": "surface + depth data",
         "both": "surface + prior + depth", "adaptive": "gated"}
COLOR = {"prior": "darkorange", "data": "crimson", "both": "seagreen"}


# --------------------------------------------------------------------------- #
# 변형장 — exp 51 의 장 + 표면에서 안 보이는 심부 모드
# --------------------------------------------------------------------------- #
def deformation(pts, window_c, sag_dir, inward, lateral, deep_mm=0.0):
    """수술 중 변위 u(x).

    앞의 두 항은 exp 51 과 같다(개두창 중력 침하 + 견인). 세 번째 항이 이번 실험의 핵심 —
    깊이 DEEP_DEPTH_MM 근처에 국한된 **횡방향** 성분이다. 가우시안 폭이 좁아서 표면에 거의
    자취를 남기지 않으므로, 표면 데이터로는 그 존재조차 알 수 없다. 임상적으로는 뇌실 허탈이나
    심부 조직 이완처럼 표면 침하와 **다른 방향·다른 깊이**로 일어나는 모드에 해당한다."""
    d = np.linalg.norm(pts - window_c, axis=1)
    u = (SAG_MM * 1e-3) * np.exp(-d / (DECAY_MM * 1e-3))[:, None] * sag_dir[None, :]
    u += (BULGE_MM * 1e-3) * np.exp(-(d / (RETRACT_MM * 1e-3)) ** 2)[:, None] * inward[None, :]
    if deep_mm:
        c = window_c + DEEP_DEPTH_MM * 1e-3 * inward
        dd = np.linalg.norm(pts - c, axis=1)
        u = u + (deep_mm * 1e-3) * np.exp(
            -(dd / (DEEP_SIGMA_MM * 1e-3)) ** 2)[:, None] * lateral[None, :]
    return u


def axis_frame(inward):
    """진입축 기준 정규직교 기저 (축, 횡1, 횡2)."""
    a = inward / np.linalg.norm(inward)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, tmp); e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def cone_points(rng, n, window_c, inward, depth_range=US_DEPTH_RANGE, cone_deg=US_CONE_DEG):
    """개두창 아래 원뿔 안의 점들 = 초음파가 볼 수 있는 심부 위치."""
    a, e1, e2 = axis_frame(inward)
    depth = rng.uniform(*depth_range, n)
    ang = np.deg2rad(cone_deg) * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    lat = depth * np.tan(ang)
    return (window_c + depth[:, None] * a[None, :]
            + lat[:, None] * (np.cos(phi)[:, None] * e1[None, :]
                              + np.sin(phi)[:, None] * e2[None, :]))


def inside_head(pts, mask, D, origin):
    vi = np.rint((pts - origin) @ np.linalg.inv(D)).astype(int)
    ok = np.all((vi >= 0) & (vi < np.array(mask.shape)[::-1]), axis=1)
    out = np.zeros(len(pts), bool)
    out[ok] = mask[vi[ok, 2], vi[ok, 1], vi[ok, 0]]
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(seed=5, n_trials=300, quick=False):
    anat.fetch()
    vol, D, origin = anat.load_nrrd()
    surface, mask = anat.head_surface(vol, D, origin)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(surface), min(anat.MODEL_POINTS, len(surface)), replace=False)
    model = surface[sel]
    normals = reg.estimate_normals(model, k=12)

    center = model.mean(0)
    window_c = model[int(np.argmax(model[:, 2]))]
    sag_dir = np.array([0.0, 0.0, -1.0])
    inward = (center - window_c) / np.linalg.norm(center - window_c)
    _, lateral, _ = axis_frame(inward)
    depths = np.array(TARGET_DEPTHS_MM)
    targets = window_c + depths[:, None] * 1e-3 * inward

    print("=== 52. 가정을 관측으로 검증하기: 반박은 싸고 교정은 비싸다 ===")
    print(f"데이터: exp 49·51 과 같은 실 인체 MR 표면 {len(model)}점, 노출 {EXPOSURE_DEG:.0f}°")
    print(f"새 관측: 수술 중 초음파 심부 변위 (σ {US_NOISE*1e3:.1f} mm, 깊이 "
          f"{US_DEPTH_RANGE[0]*1e3:.0f}~{US_DEPTH_RANGE[1]*1e3:.0f} mm, 원뿔 ±{US_CONE_DEG:.0f}°)")

    # ---- 심부 모드가 정말 표면에서 안 보이는지 먼저 확인 ----
    v = (model - center) / np.linalg.norm(model - center, axis=1, keepdims=True)
    w_dir = (window_c - center) / np.linalg.norm(window_c - center)
    ang = np.degrees(np.arccos(np.clip(v @ w_dir, -1, 1)))
    in_win = np.where(ang < EXPOSURE_DEG)[0]
    out_win = np.where(ang > EXPOSURE_DEG + 25.0)[0]

    def u_of(p, deep_mm):
        return deformation(p, window_c, sag_dir, inward, lateral, deep_mm)

    trace = np.linalg.norm(u_of(model[in_win], DEEP_MM) - u_of(model[in_win], 0.0), axis=1)
    at_tgt = np.linalg.norm(u_of(targets, DEEP_MM) - u_of(targets, 0.0), axis=1)
    print(f"심부 모드({DEEP_MM:.0f} mm, 깊이 {DEEP_DEPTH_MM:.0f} mm, 횡방향)가 "
          f"**창 안 표면에 남기는 자취: 평균 {trace.mean()*1e3:.2f} / 최대 {trace.max()*1e3:.2f} mm**")
    print(f"  프로브 잡음 σ {PROBE_NOISE*1e3:.1f} mm 아래다 — 표면에서는 원리적으로 관측 불가.")
    print("  같은 모드가 표적에서는 " + " ".join(
        f"{d:.0f}mm:{x*1e3:.2f}" for d, x in zip(depths, at_tgt)) + " mm")

    # ---- 개두 전 강체 정합 (변형과 무관하므로 한 번만) ----
    r0 = np.random.default_rng(seed)
    ax0 = r0.normal(size=3); ax0 /= np.linalg.norm(ax0)
    T_place = reg.pose_T(so3_exp(ax0 * np.deg2rad(6.0)), r0.uniform(-0.015, 0.015, 3))
    land = anat.pick_landmarks(surface, k=5)
    digit = reg.apply_T(T_place, surface[land]) + r0.normal(0, LANDMARK_NOISE, (len(land), 3))
    init = anat.procrustes(digit, surface[land])
    pre_idx = r0.choice(len(model), size=N_PROBE, replace=False)
    pre_dst = reg.apply_T(T_place, model[pre_idx]) + r0.normal(0, PROBE_NOISE, (N_PROBE, 3))
    T_rigid = real._icp2(pre_dst, model, normals, init)
    T_inv = np.linalg.inv(T_rigid)
    rigid_only = float(np.mean(np.linalg.norm(
        reg.apply_T(T_inv, targets) - reg.apply_T(T_place, targets), axis=1)))
    print(f"개두 전 강체 정합: 변형 없을 때 표적 오차 {rigid_only*1e3:.3f} mm "
          f"(exp 51 과 동일) — 이후 오차는 전부 변형 몫")
    chk = cone_points(np.random.default_rng(1), 400, window_c, inward)
    print(f"초음파 원뿔 표본이 머리 안에 있는 비율 {inside_head(chk, mask, D, origin).mean()*100:.0f}%"
          f" · 창 안 표면 {len(in_win)/len(model)*100:.0f}%")

    def make_case(rr, deep_mm, n_us, n_hold=1, hold_at="random"):
        """한 환자·한 세션의 관측 일습.

        hold_at="random" 이면 검산 관측을 초음파 원뿔 안 아무 데나, "target" 이면 **계획 표적
        근처**(6 mm 오차로 조준)에 둔다. exp 49 에서 표면 검증점을 어디에 찍느냐가 검출력을
        갈랐던 것과 같은 질문을 심부에서 다시 묻는 것이다."""
        idx = rr.choice(in_win, size=min(N_PROBE, len(in_win)), replace=False)
        src = model[idx]
        dst = reg.apply_T(T_place, src + u_of(src, deep_mm)) \
            + rr.normal(0, PROBE_NOISE, (len(idx), 3))
        disp = reg.apply_T(T_rigid, dst) - src              # 영상좌표에서 본 잔여 변위
        anchors = model[rr.choice(out_win, size=min(N_TPS_ANCHOR, len(out_win)), replace=False)]
        us = cone_points(rr, n_us, window_c, inward)        # 피팅용: 원뿔 스윕
        if hold_at == "target":
            j = rr.integers(len(targets), size=n_hold)
            hold = targets[j] + rr.normal(0, 6e-3, (n_hold, 3))
        else:
            hold = cone_points(rr, n_hold, window_c, inward)
        return dict(src=src, disp=disp, anchors=anchors,
                    us=us, us_disp=u_of(us, deep_mm) + rr.normal(0, US_NOISE, (len(us), 3)),
                    hold=hold, hold_disp=u_of(hold, deep_mm) + rr.normal(0, US_NOISE, (n_hold, 3)),
                    tgt_true=reg.apply_T(T_place, targets + u_of(targets, deep_mm)))

    def fit(case, method, lam=TPS_LAMBDA, lam_us=TPS_LAMBDA, sub_seed=7):
        """세 가지 정보 조합. 보간기는 셋 다 같은 TPS 이고 λ 도 같다 — 정보만 다르다.

        (λ 를 제어점별 벡터로 준다: exp 51 의 tps_fit 은 K + diag(λ) 를 풀므로 잡음이 큰
         초음파 관측에 다른 정규화를 줄 수 있다.)"""
        rr = np.random.default_rng(sub_seed)
        sub = rr.choice(len(case["src"]), size=min(N_TPS_CTRL, len(case["src"])), replace=False)
        ctrl, dd, lm = [case["src"][sub]], [case["disp"][sub]], [np.full(len(sub), lam)]
        if method in ("prior", "both"):
            ctrl.append(case["anchors"]); dd.append(np.zeros_like(case["anchors"]))
            lm.append(np.full(len(case["anchors"]), lam))
        if method in ("data", "both") and len(case["us"]):
            ctrl.append(case["us"]); dd.append(case["us_disp"])
            lm.append(np.full(len(case["us"]), lam_us))
        return deform.tps_fit(np.vstack(ctrl), np.vstack(dd), lam=np.concatenate(lm))

    def score(case, m):
        est = reg.apply_T(T_inv, deform.tps_apply(m, targets))
        return float(np.mean(np.linalg.norm(est - case["tgt_true"], axis=1)))

    def gate(case, m):
        """검산: 피팅에 쓰지 않은 심부 관측에서 예측 변위 vs 실측 변위 잔차(RMS).

        (처음엔 최댓값을 썼는데 점을 늘릴수록 안전 환자 쪽 꼬리도 같이 올라가 검출률이
         비단조가 됐다 — 최댓값은 잡음을 모으고 RMS 는 평균한다.)"""
        if not len(case["hold"]):
            return 0.0
        pred = deform.tps_apply(m, case["hold"]) - case["hold"]
        return float(np.sqrt(np.mean(np.sum((pred - case["hold_disp"]) ** 2, axis=1))))

    def surf_res(case, m):
        w = deform.tps_apply(m, case["src"]) - (case["src"] + case["disp"])
        return float(np.sqrt(np.mean(np.sum(w ** 2, axis=1))))

    # ------------------------------------------------------------------ #
    # A. 표면만으로는 복원할 수 없다 / 심부 관측은 몇 개면 되는가
    # ------------------------------------------------------------------ #
    print("-" * 100)
    n_reps = 3 if quick else 5              # 관측 배치가 무작위라 시드 평균이 필요하다
    print(f"[A] 심부 관측 개수 스윕 (심부 표적 오차 mm, 시드 {n_reps}개 평균)")
    n_list = N_SWEEP[:-1] if quick else N_SWEEP
    sweep = {}
    for tag, deep in (("표면설명형", 0.0), ("심부 모드", DEEP_MM)):
        rows = {k: [] for k in METHODS}
        for n_us in n_list:
            acc = {k: [] for k in METHODS}
            for rep in range(n_reps):
                c = make_case(np.random.default_rng([100 + n_us, rep]), deep, n_us)
                for k in METHODS:
                    acc[k].append(score(c, fit(c, k)))
            for k in METHODS:
                rows[k].append(float(np.mean(acc[k])))
        sweep[tag] = rows
        print(f"  [{tag}]  " + "관측  " + " ".join(f"{n:5d}" for n in n_list))
        for k in METHODS:
            print(f"    {LABEL[k]:<26}" + " ".join(f"{v*1e3:5.2f}" for v in rows[k]))

    ok, bad = sweep["표면설명형"], sweep["심부 모드"]
    print(f"  → 표면설명형에서는 사전지식만으로 {ok['prior'][0]*1e3:.2f} mm. 관측은 잡음만 "
          f"들여와서 적을 때 오히려 손해다({ok['both'][1]*1e3:.2f} mm @1개).")
    n_fix = next((n for n, v in zip(n_list, bad["both"]) if v < MISS_TOL), None)
    print(f"  → 심부 모드에서는 사전지식이든 아니든 표면만으로는 {bad['prior'][0]*1e3:.2f} / "
          f"{bad['data'][0]*1e3:.2f} mm에 갇힌다. **모드가 표면에 없으므로 알고리즘 문제가 아니다.**")
    print(f"     관측을 넣으면 내려오지만 느리다: " +
          " ".join(f"{n}개 {v*1e3:.2f}" for n, v in zip(n_list, bad["both"]) if n) +
          f" mm → 허용치 {MISS_TOL*1e3:.0f} mm 도달에 **{n_fix}개**"
          if n_fix else " — 32개로도 허용치 미달")

    # ------------------------------------------------------------------ #
    # B. 표면 지표는 그 사실을 전혀 모른다
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print("[B] 그런데 표면에서 볼 수 있는 지표는 두 환자를 구별하지 못한다 (사전지식만, 관측 0개)")
    surf = {}
    for tag, deep in (("표면설명형", 0.0), ("심부 모드", DEEP_MM)):
        c = make_case(np.random.default_rng(100), deep, 0)
        m = fit(c, "prior")
        surf[tag] = (surf_res(c, m), score(c, m))
        print(f"     {tag:<10} 표면 잔차 {surf[tag][0]*1e3:5.2f} mm  │  "
              f"심부 오차 {surf[tag][1]*1e3:5.2f} mm")
    rs = surf["심부 모드"][0] / max(surf["표면설명형"][0], 1e-12)
    rt = surf["심부 모드"][1] / max(surf["표면설명형"][1], 1e-12)
    print(f"  → 표면 잔차는 {rs:.2f}배인데 심부 오차는 {rt:.1f}배다. **볼 수 있는 지표에 신호가 "
          f"없다** — exp 44·49 의 게이트 실패가 이번엔 정합 품질이 아니라 **모델 가정** 층위에서 "
          f"반복된다.")

    # ------------------------------------------------------------------ #
    # C. 반박은 싸다 — 검산 관측 하나
    # ------------------------------------------------------------------ #
    print("-" * 100)
    # 표본이 작으면 AUROC 순위가 흔들린다(초기엔 60명으로 돌렸다가 배치 효과의 부호가
    # 시행 수에 따라 뒤집혔다). 200명 아래로는 내리지 않는다.
    trials = max(n_trials * 2 // 3, 200) if quick else n_trials
    print(f"[C] 관측 예산을 **반박에 쓸까 교정에 쓸까** — 환자 {trials}명 몬테카를로")
    print("     환자마다 계획 표적 하나(깊이 무작위), 절반은 심부 모드. 오차는 그 표적에서 잰다")

    def score_at(case, m, j):
        est = reg.apply_T(T_inv, deform.tps_apply(m, targets[j:j + 1]))[0]
        return float(np.linalg.norm(est - case["tgt_true"][j]))

    K_HOLD = (1, 2, 3)          # 검산에 쓰는 관측 수
    N_FIX = (1, 2, 4, 8)        # 교정에 쓰는 관측 수
    rec = []
    for i in range(trials):
        rr = np.random.default_rng([seed, i])
        has_deep = bool(i % 2)
        deep = float(abs(rr.normal(0.0, 0.4))) if not has_deep else float(rr.uniform(4.0, 9.0))
        j = int(rr.integers(len(targets)))
        row = dict(has_deep=has_deep, deep=deep, j=j)
        for k in K_HOLD:        # 반박: 표적 근처 검산점 k개(피팅에는 쓰지 않음)
            c = make_case(np.random.default_rng([seed, i, 900 + k]), deep, 0,
                          n_hold=k, hold_at="target")
            c["hold"] = targets[j] + np.random.default_rng([seed, i, k]).normal(0, 6e-3, (k, 3))
            c["hold_disp"] = u_of(c["hold"], deep) + \
                np.random.default_rng([seed, i, 7, k]).normal(0, US_NOISE, (k, 3))
            m = fit(c, "prior")
            row[f"gate{k}"] = gate(c, m)
            row["prior"] = score_at(c, m, j)
            row["surf"] = surf_res(c, m)        # 표면 잔차 = 관측 없이 볼 수 있는 유일한 지표
        c_rand = make_case(np.random.default_rng([seed, i, 42]), deep, 0, hold_at="random")
        row["gate_rand"] = gate(c_rand, fit(c_rand, "prior"))
        for n in N_FIX:         # 교정: 같은 관측을 피팅에 넣으면
            c = make_case(np.random.default_rng([seed, i, n]), deep, n)
            row[f"fix{n}"] = score_at(c, fit(c, "both"), j)
        rec.append(row)

    Ep = np.array([x["prior"] for x in rec])
    Efix = {n: np.array([x[f"fix{n}"] for x in rec]) for n in N_FIX}
    E1, E4 = Efix[1], Efix[4]
    deep_mm = np.array([x["deep"] for x in rec])
    hasd = np.array([x["has_deep"] for x in rec])
    unsafe = Ep > MISS_TOL
    print(f"  사전지식만 썼을 때 unsafe(>{MISS_TOL*1e3:.0f} mm): 전체 {100*unsafe.mean():.0f}% "
          f"(심부 모드 환자 {100*unsafe[hasd].mean():.0f}% / 표면설명형 "
          f"{100*unsafe[~hasd].mean():.0f}%)")

    def roc(G, fa_target=0.10):
        """게이트 성능. AUROC 를 주 지표로 쓴다 — 순위 기반이라 임계값 선택에 흔들리지 않고,
        고정 오경보에서의 검출률은 표본 60~120명 규모에서 분위수 추정 잡음이 너무 크다
        (실제로 처음엔 검출률만 봤다가 시행 수에 따라 순위가 뒤집혔다)."""
        if not (~unsafe).any() or not unsafe.any():
            return float("nan"), float("nan"), float("nan")
        pos, neg = G[unsafe], G[~unsafe]
        auroc = float((np.greater.outer(pos, neg).mean()
                       + 0.5 * np.equal.outer(pos, neg).mean()))
        thr = float(np.quantile(neg, 1.0 - fa_target))
        return auroc, 100.0 * float(np.mean(pos > thr)), thr

    gates = {}
    print("  **게이트 비교** (AUROC = 위반 환자를 안전 환자보다 높게 매길 확률, 0.5 = 무정보)")
    S = np.array([x["surf"] for x in rec])
    a_surf, d_surf, _ = roc(S)
    print(f"    표면 잔차(관측 0개, 지금 가진 유일한 지표) → AUROC {a_surf:.2f} "
          f"(검출 {d_surf:.0f}% @오경보 10%) = 사실상 무정보")
    for k in K_HOLD:
        G = np.array([x[f"gate{k}"] for x in rec])
        au, d10, thr = roc(G)
        gates[k] = dict(G=G, auroc=au, det=d10, thr=thr)
        print(f"    표적 근처 심부 관측 {k}개 → AUROC {au:.2f} (검출 {d10:3.0f}%, "
              f"임계 {thr*1e3:4.1f} mm)")
    Gr = np.array([x["gate_rand"] for x in rec])
    a_rand, d_rand, _ = roc(Gr)
    gap = gates[1]["auroc"] - a_rand
    print(f"    같은 1개를 원뿔 안 **아무 곳**에 두면 AUROC {a_rand:.2f} "
          f"(표적 근처 {gates[1]['auroc']:.2f}, 차이 {gap:+.2f})")
    print("      → exp 49 의 '어디를 검산하느냐'가 심부에서도 나타나지만 **차이는 완만하다**. "
          "여기 심부 모드가")
    print("        폭 넓은 가우시안이라 원뿔 아무 데서나 어느 정도 보이기 때문이다. 더 국소적인 "
          "위반이라면 격차가 커진다.")
    print(f"    게이트의 상한은 관측 잡음이 정한다(σ {US_NOISE*1e3:.1f} mm > 허용치 "
          f"{MISS_TOL*1e3:.0f} mm). 더 나은 통계량으로 넘을 수 있는 벽이 아니라 모달리티 문제다.")

    print("  **관측 하나가 사는 것: 지식과 정확도** (심부 모드 환자 기준)")
    print("     관측 수   가정 위반 판별(AUROC)      표적 오차 중앙값")
    print(f"     {0:^7d}   {a_surf:^19.2f}      {np.median(Ep[hasd])*1e3:5.2f} mm  "
          f"← 표면만. 오차도 크고, 크다는 것조차 모른다")
    for n in N_FIX:
        a = f"{gates[n]['auroc']:.2f}" if n in gates else "—"
        e = np.median(Efix[n][hasd]) * 1e3
        print(f"     {n:^7d}   {a:^19}      {e:5.2f} mm"
              + ("  ← 허용치 밖" if e > MISS_TOL * 1e3 else ""))
    print(f"    → 첫 관측 하나가 판별력을 {a_surf:.2f} → {gates[1]['auroc']:.2f} 로 올리는 동안 "
          f"오차는 {np.median(Ep[hasd])*1e3:.2f} → {np.median(Efix[1][hasd])*1e3:.2f} mm 로만 "
          f"내려간다(여전히 허용치 밖).")
    print("       관측의 첫 몇 개는 **고치는 것보다 아는 것**을 더 많이 산다. 둘 다 필요하지만,")
    print("       모른 채 계획을 집행하는 것과 알고 추가 영상을 받는 것의 차이가 더 크다.")

    fired = gates[2]["G"] > gates[2]["thr"]
    Eg = np.where(fired, E4, Ep)                # 게이트가 울리면 관측 4개를 더 받아 재적합
    print("  정책별 표적 오차 (중앙값 / 90퍼센타일 / unsafe 비율):")
    for name, E in (("항상 사전지식(관측 0)", Ep), ("항상 관측 4개", E4),
                    ("**게이트 적응(2→+4)**", Eg)):
        print(f"    {name:<24} {np.median(E)*1e3:6.2f} / {np.percentile(E,90)*1e3:6.2f} mm / "
              f"{100*np.mean(E>MISS_TOL):4.0f}%")
    print("  → 게이트 적응이 '항상 4개'를 **이기지 못한다**. 관측이 싸면 그냥 다 받는 게 낫다.")
    print("     게이트의 값은 관측이 비쌀 때(추가 스캔 = 수술 시간), 그리고 무엇보다 **고칠 수")
    print("     없을 때 경고를 낼 수 있다**는 데 있다 — 조용한 실패를 알려진 실패로 바꾸는 것.")
    print("  관통하는 것은 이 저장소의 첫 번째 교훈이다: **관측되지 않는 것은 추정되지 않는다.**")
    print("  IMU 바이어스(4·37)도, 스틱션(43·46)도, 여기 심부 모드도 답은 같다 — 알고리즘을")
    print("  바꾸는 게 아니라 **그 상태를 흔드는 관측을 설계**하는 것.")
    det, fa, G = gates[2]["det"], 10.0, gates[2]["G"]

    # ---- 그림 ----
    fig, axg = plt.subplots(2, 3, figsize=(16.8, 9.2))
    axes = axg.ravel()

    for ax, (tag, rows), ttl in zip(
            axes[:2], sweep.items(),
            ("Surface-explained shift: depth data adds noise",
             "Deep mode: the surface cannot see it at all")):
        for k in METHODS:
            ax.plot(n_list, np.array(rows[k]) * 1e3, "-o", color=COLOR[k], label=LABEL[k])
        ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1, label="tolerance")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("sub-surface (iUS) observations used in the fit")
        ax.set_ylabel("deep-target error [mm]")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[2]
    x = np.arange(2)
    ax.bar(x - 0.18, [surf["표면설명형"][0] * 1e3, surf["심부 모드"][0] * 1e3], 0.36,
           color="0.6", label="surface residual (measurable)")
    ax.bar(x + 0.18, [surf["표면설명형"][1] * 1e3, surf["심부 모드"][1] * 1e3], 0.36,
           color="crimson", label="deep-target error (matters)")
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["surface-explained", "deep mode"])
    ax.set_ylabel("[mm]")
    ax.set_title("The measurable quantity is blind to it", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axes[3]
    ax.scatter(G[~hasd] * 1e3, Ep[~hasd] * 1e3, s=26, color="seagreen",
               label="surface-explained", alpha=0.85)
    ax.scatter(G[hasd] * 1e3, Ep[hasd] * 1e3, s=26, color="crimson", marker="s",
               label="deep mode", alpha=0.85)
    ax.axvline(gates[2]["thr"] * 1e3, color="0.3", ls="--", lw=1)
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("held-out depth residual [mm]   (2 points near the target)")
    ax.set_ylabel("target error [mm]")
    ax.set_title(f"A depth check sees it: AUROC {gates[2]['auroc']:.2f} "
                 f"(surface residual {a_surf:.2f})", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[4]
    ax.plot([0] + list(K_HOLD), [a_surf] + [gates[k]["auroc"] for k in K_HOLD], "-o",
            color="royalblue", label="spent on checking (AUROC)")
    ax.axhline(0.5, color="royalblue", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("sub-surface observations spent")
    ax.set_ylabel("violation AUROC", color="royalblue")
    ax.set_ylim(0.4, 1.02); ax.set_xticks([0] + list(N_FIX))
    ax.tick_params(axis="y", labelcolor="royalblue")
    ax2 = ax.twinx()
    ax2.plot([0] + list(N_FIX),
             [np.median(Ep[hasd]) * 1e3] + [np.median(Efix[n][hasd]) * 1e3 for n in N_FIX],
             "-s", color="crimson", label="spent on correcting (error mm)")
    ax2.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1)
    ax2.set_ylabel("target error [mm]", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.set_title("Same budget: what it buys you — knowing vs fixing", fontsize=10)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    ax = axes[5]
    ax.scatter(deep_mm, Ep * 1e3, s=26, color="darkorange", label="prior only", alpha=0.85)
    ax.scatter(deep_mm, Eg * 1e3, s=26, color="royalblue", marker="^",
               label="gated (2 → +4)", alpha=0.85)
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("hidden deep-mode amplitude [mm]")
    ax.set_ylabel("deep-target error [mm]")
    ax.set_title("How wrong the assumption is, and what it costs", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("52. Probing the prior — one sub-surface observation refutes what "
                 "the whole surface cannot", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "52_probing_the_prior.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/52_probing_the_prior.png, assets/52_probing_the_prior.png")

    return dict(sweep=sweep, n_list=list(n_list), surf=surf, rec=rec, det=det, fa=fa,
                gates={k: {kk: vv for kk, vv in g.items() if kk != "G"}
                       for k, g in gates.items()},
                auroc_surface=a_surf, auroc_random1=a_rand, auroc_target1=gates[1]["auroc"],
                det_surface=d_surf, det_random1=d_rand, det_target1=gates[1]["det"],
                median_fix1_deep=float(np.median(Efix[1][hasd])), MISS_TOL=MISS_TOL,
                Ep=Ep, E1=E1, E4=E4, Eg=Eg, has_deep=hasd, n_fix=n_fix,
                surface_trace_mm=float(trace.mean() * 1e3),
                target_mode_mm=float(at_tgt.mean() * 1e3),
                rigid_only_mm=rigid_only * 1e3)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 초음파 관측을 **정답 변위 + 등방 가우시안 잡음**으로 모델링했다. 실제 iUS 는 깊이에 따라
#     영상 품질이 나빠지고, 특징 대응 자체가 실패하며(오대응), 프로브가 조직을 누르면
#     **관측 행위가 변형을 바꾼다**. 숫자는 낙관적이다.
#   - 심부 모드를 **가우시안 하나**로 만들었다. 실제 위반은 형태가 다양하고, 원뿔 밖에서
#     일어나면 이 관측 배치로는 반박조차 못 한다 — "검산점을 어디에 둘 것인가"는 exp 49 의
#     표면 검증점과 같은 미해결 문제로 남는다.
#   - 게이트 임계(4.5 mm ≈ 3σ_us)는 이 잡음 수준에서 고른 값이다. 임계 선택은 오경보(추가
#     스캔·시간)와 미검출의 임상적 교환이며 여기서 정할 수 있는 문제가 아니다.
#   - 변형장·대응은 exp 51 과 같이 합성이다. 방법 간 상대 비교로 읽어야 한다.
#   - 게이트가 울린 뒤의 처방(관측 4개로 재적합)은 오차를 **줄이지 완치하지 않는다**. 실제
#     시스템이라면 "심부 계획을 신뢰하지 말라"는 경고와 추가 영상 획득이 정답에 가깝다.
#     실제로 이 실험에서 게이트 적응은 '항상 4개'를 이기지 못했다 — 관측이 싸면 게이트가 아니라
#     그냥 다 받는 게 낫다는 뜻이고, 그렇게 적었다.
#   - 검출률이 60%대에서 멈추는 것은 **모달리티 잡음이 임상 허용치보다 크기** 때문이다. 더 나은
#     통계량으로 넘을 수 있는 벽이 아니다(그래서 여기서 알고리즘을 더 만들지 않았다).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
