"""실 스캔 데이터로 정합 파이프라인 검증 — Stanford Bunny (합성 아닌 실측 3D 스캔).

exp 41·42의 정합·신뢰도 게이트는 전부 **합성 해부 표면** 위에서 만들어졌다. 합성
표면은 곡률이 매끄럽고, 잡음이 등방 가우시안이며, 구멍이 없다 — 즉 알고리즘에게
유리하게 생겼다. 이 실험은 같은 파이프라인을 **실제 레이저 스캐너로 찍은 점군**에
그대로 올려, 결론이 살아남는지 확인한다. exp 14(합성 SLAM → 표준 g2o 벤치마크)와
같은 역할의 '실데이터 앵커'다.

데이터: **Stanford 3D Scanning Repository 의 Bunny**(Turk & Levoy, 1994). 서로 다른
시점에서 찍은 실제 range scan(bun000, bun045)과, 데이터셋이 함께 제공하는 정렬
정보(bun.conf)를 쓴다. 재배포하지 않기 위해 실행 시 내려받아 `data_cache/`(git 제외)에
둔다.

  - **수술 전 모델** ← bun000 (조밀 스캔, 전역 좌표계)
  - **수술 중 프로브 점군** ← bun045 를 conf 정렬로 같은 좌표계에 올린 뒤, 일부만
    샘플링(부분 커버리지) + 프로브 잡음 + outlier 를 얹고, **미지의 SE(3)로 이동**
  - 정합기는 exp 41 의 점-대-평면 ICP 를 그대로 재사용(코드 변경 없음)

실 스캔이 합성과 다른 점(그래서 검증할 가치가 있는 점):
  - 시점이 다른 두 스캔은 **부분 겹침**만 있고, 서로 안 보이는 면·구멍이 있다.
  - 잡음이 등방이 아니다(시선 방향으로 더 크고, 경사면에서 커진다).
  - 표면 곡률이 균일하지 않다(귀·다리처럼 특징적인 부위와 평평한 몸통이 공존).

정직한 기준선: bun.conf 정렬 자체가 추정치다. conf 로 정렬한 두 스캔의 최근접점
중앙값(≈0.3 mm)이 이 데이터가 가진 **바닥 오차**이며, 우리 TRE 는 그 위에서 읽어야
한다. "정답"이 완벽하지 않다는 사실을 숨기지 않는 것이 실데이터 검증의 조건이다.

    python scripts/44_registration_real_scans.py
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensor_fusion.se3 import hat, so3_exp  # noqa: E402

reg = import_module("41_surgical_registration")     # 점-대-평면 ICP 자산 재사용

BUNNY_URL = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
CACHE = Path("data_cache")
ARCHIVE = CACHE / "bunny.tar.gz"

PROBE_NOISE = 3e-4          # 프로브 잡음 σ [m] (0.3 mm)
N_PROBE = 800               # 수술 중 디지타이징 점 수
N_OUTLIERS = 12
MODEL_SUBSAMPLE = 8000      # 모델 점군 서브샘플(속도)
K_SIGMA = 3.0
DISAGREE_TOL = 2e-3         # 다중초기값 불일치 허용치 [m] (2 mm)
INLIER_MIN = 0.80           # 대응 게이트를 통과해야 하는 최소 점 비율
MISS_TOL = 3e-3             # 표적 허용오차 [m] (3 mm) — unsafe 판정


# --------------------------------------------------------------------------- #
# 데이터: 내려받기 · PLY/conf 파싱
# --------------------------------------------------------------------------- #
def fetch(url=BUNNY_URL, dest=ARCHIVE):
    """없으면 내려받는다. 사내망 프록시 CA는 truststore로 우회한다."""
    if dest.exists():
        return dest
    dest.parent.mkdir(exist_ok=True)
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass
    urllib.request.urlretrieve(url, dest)
    return dest


def _read_ply(fobj):
    """Stanford range scan(ASCII PLY)의 정점 좌표만 (N,3) 으로 읽는다."""
    n = 0
    while True:
        line = fobj.readline().decode("ascii").strip()
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line == "end_header":
            break
    pts = np.empty((n, 3))
    for i in range(n):
        p = fobj.readline().split()
        pts[i] = (float(p[0]), float(p[1]), float(p[2]))
    return pts


def _quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def load_bunny(archive=ARCHIVE, scans=("bun000", "bun045")):
    """스캔들을 bun.conf 의 전역 정렬을 적용해 같은 좌표계로 반환.

    conf 규약(경험적으로 확인): global = Rᵀ·local + t. (R@p+t 로 쓰면 29 mm 어긋난다 —
    데이터셋 관례는 문서보다 데이터로 확인하는 편이 안전하다.)"""
    out = {}
    with tarfile.open(archive) as t:
        conf = t.extractfile("bunny/data/bun.conf").read().decode().splitlines()
        poses = {}
        for line in conf:
            p = line.split()
            if p and p[0] == "bmesh":
                name = p[1].replace(".ply", "")
                tv = np.array(list(map(float, p[2:5])))
                R = _quat_to_R(*map(float, p[5:9]))
                poses[name] = (R, tv)
        for s in scans:
            pts = _read_ply(t.extractfile(f"bunny/data/{s}.ply"))
            R, tv = poses[s]
            out[s] = pts @ R + tv          # = Rᵀ p + t
    return out


# --------------------------------------------------------------------------- #
# 수술 시나리오 합성(기하는 실측, 프로빙 과정만 모델링)
# --------------------------------------------------------------------------- #
def make_probe(rng, scan, T_perturb, n=N_PROBE, noise=PROBE_NOISE,
               n_out=N_OUTLIERS, patch=None):
    """실 스캔에서 프로브 점군을 만든다.

    patch=(center_idx, frac) 를 주면 표면의 국소 구획만 찍은 것처럼 공간적으로 연속된
    부분만 샘플링한다(부분 커버리지 재현)."""
    idx = np.arange(len(scan))
    if patch is not None:
        c, frac = patch
        d = np.linalg.norm(scan - scan[c], axis=1)
        keep = np.argsort(d)[:max(int(frac * len(scan)), n)]
        idx = keep
    sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
    P = scan[sel] + rng.normal(0, noise, (len(sel), 3))
    if n_out:
        span = P.max(0) - P.min(0)
        P = np.vstack([P, P.mean(0) + rng.uniform(-1, 1, (n_out, 3)) * span * 0.7])
    return reg.apply_T(T_perturb, P)


def random_perturb(rng, rot_deg=12.0, trans_mm=15.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = so3_exp(axis * np.deg2rad(rng.uniform(-rot_deg, rot_deg)))
    t = rng.uniform(-trans_mm, trans_mm, 3) * 1e-3
    return reg.pose_T(R, t)


def targets_inside(model, k=4, seed=0):
    """모델 내부의 임상 표적 대용 점 — 표면에서 떨어져 있어 TRE 외삽 성질을 드러낸다."""
    rng = np.random.default_rng(seed)
    c = model.mean(0)
    sel = rng.choice(len(model), k, replace=False)
    return c + 0.45 * (model[sel] - c)


# --------------------------------------------------------------------------- #
# 정합 + 신뢰도 지표(exp 42 의 두 갈래 게이트를 3D 로)
# --------------------------------------------------------------------------- #
def information_matrix(T, src, model_tree, model, normals, max_corr=4e-3):
    """최종 대응에서 A = JᵀJ (6×6, [ω, ν] 순서)와 잔차 RMS 를 다시 계산한다."""
    q = reg.apply_T(T, src)
    dist, idx = model_tree.query(q)
    m = dist < max_corr
    inlier_frac = float(m.mean())          # 겹침(overlap) 비율 = 세 번째 신뢰도 신호
    if m.sum() < 20:
        return None, np.inf, inlier_frac
    qm, mm, nm = q[m], model[idx[m]], normals[idx[m]]
    J = np.hstack([np.cross(qm, nm), nm])
    r = np.einsum("ij,ij->i", nm, mm - qm)
    return J.T @ J, float(np.sqrt(np.mean(r ** 2))), inlier_frac


def target_sigma(Cov, p):
    """표적점 위치 불확실도(최대 주축 1σ). δp = [−[p]×, I]·ξ."""
    Jp = np.hstack([-hat(np.asarray(p, float)), np.eye(3)])
    return float(np.sqrt(max(np.linalg.eigvalsh(Jp @ Cov @ Jp.T)[-1], 0.0)))


MULTISTART_ROTS = ((np.array([0, 0, 1.0]), np.pi), (np.array([1.0, 0, 0]), np.pi),
                   (np.array([0, 1.0, 0]), np.pi), (np.array([0, 0, 1.0]), np.pi / 2))


def _icp2(probe, model, normals, init):
    """조대→정밀 2단계 ICP (exp 42 와 같은 절차)."""
    T0, _, _, _ = reg.point_to_plane_icp(probe, model, dst_normals=normals,
                                         init=init, max_iter=40, max_corr_dist=8e-3)
    return reg.point_to_plane_icp(probe, model, dst_normals=normals,
                                  init=T0, max_iter=40, max_corr_dist=3e-3)[0]


def register(probe, model, normals, tree, verify_target, multistart=True):
    """정합 + 세 가지 신뢰도 신호. 반환 (T_reg, fre, σ_target, disagree, inlier_frac).

    신호 1) σ_target — 정보행렬 기반 조건화(exp 42 에서 합성 데이터에 통했던 것)
    신호 2) disagree — 다중초기값 해 불일치(잘못된 basin 탐지)
    신호 3) inlier_frac — 대응 게이트를 통과한 점 비율(겹침/정합 품질의 고전 지표)"""
    init = reg.centroid_init(probe, model)
    T_reg = _icp2(probe, model, normals, init)
    A, rms, inlier = information_matrix(T_reg, probe, tree, model, normals)
    if A is None or not np.isfinite(rms):
        return T_reg, np.inf, np.inf, np.inf, inlier
    sigma = max(rms, PROBE_NOISE)
    try:
        Cov = sigma ** 2 * np.linalg.inv(A + 1e-12 * np.eye(6))
        sig_t = target_sigma(Cov, verify_target)
    except np.linalg.LinAlgError:
        sig_t = np.inf

    disagree = 0.0
    if multistart:
        best_p = reg.apply_T(T_reg, verify_target[None, :])[0]
        c = model.mean(0)
        for axis, ang in MULTISTART_ROTS:
            R = so3_exp(axis * ang)
            T_r = _icp2(probe, model, normals, reg.pose_T(R, c - R @ c) @ init)
            _, rms_r, _ = information_matrix(T_r, probe, tree, model, normals)
            if not np.isfinite(rms_r) or rms_r > 1.15 * max(rms, PROBE_NOISE):
                continue
            p_r = reg.apply_T(T_r, verify_target[None, :])[0]
            disagree = max(disagree, float(np.linalg.norm(p_r - best_p)))
    return T_reg, rms, sig_t, disagree, inlier


def tre_of(T_reg, T_perturb, targets):
    """이상적으로 T_reg∘T_perturb = I. 표적별 잔여 변위가 TRE."""
    comp = T_reg @ T_perturb
    return np.linalg.norm(reg.apply_T(comp, targets) - targets, axis=1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(n_trials=40, quick=False):
    if quick:
        n_trials = 8
    fetch()
    scans = load_bunny()
    full0, full45 = scans["bun000"], scans["bun045"]

    rng = np.random.default_rng(0)
    sel = rng.choice(len(full0), min(MODEL_SUBSAMPLE, len(full0)), replace=False)
    model = full0[sel]
    normals = reg.estimate_normals(model, k=12)
    tree = cKDTree(model)

    # 데이터 자체의 바닥 오차: conf 정렬된 두 실 스캔의 최근접점 거리.
    # (서브샘플이 아니라 **전체 모델 점군**에 대해 재야 한다 — 성긴 모델에 재면
    #  샘플 간격이 섞여 들어가 바닥이 과대평가된다.)
    d_conf, _ = cKDTree(full0).query(full45)
    floor = float(np.median(d_conf))

    targets = targets_inside(model)
    vt = targets[0]

    print("=== 44. 실 스캔 정합 검증 — Stanford Bunny (합성 아닌 실측 점군) ===")
    print(f"데이터: Stanford 3D Scanning Repository, bun000 {len(full0)}점 / "
          f"bun045 {len(full45)}점 (모델은 {len(model)}점으로 서브샘플)")
    print(f"[바닥 오차] conf 정렬된 두 실 스캔의 최근접점 중앙값 {floor*1e3:.3f} mm "
          "← 데이터셋의 '정답'도 이 정도 오차를 가진다")

    # ---- 단일 시나리오 ----
    rng = np.random.default_rng(3)
    T_pert = random_perturb(rng)
    probe = make_probe(rng, full45, T_pert)
    T_reg, fre, sig_t, disagree, inlier = register(probe, model, normals, tree, vt)
    tres = tre_of(T_reg, T_pert, targets)
    comp = T_reg @ T_pert
    rot_err = np.rad2deg(np.arccos(np.clip((np.trace(comp[:3, :3]) - 1) / 2, -1, 1)))
    print("-" * 78)
    pert_rot = np.rad2deg(np.arccos(np.clip((np.trace(T_pert[:3, :3]) - 1) / 2, -1, 1)))
    print(f"[단일 정합] 프로브 {len(probe)}점(잡음 {PROBE_NOISE*1e3:.1f} mm, "
          f"outlier {N_OUTLIERS}) | 미지 변환: 회전 {pert_rot:.1f}°, 병진 "
          f"{np.linalg.norm(T_pert[:3,3])*1e3:.1f} mm")
    print(f"  복원 오차 : 회전 {rot_err:.3f}°, 병진 "
          f"{np.linalg.norm(comp[:3,3])*1e3:.3f} mm "
          "(가한 변환을 되돌린 정확도 — conf 바닥과는 별개 측정)")
    print(f"  FRE {fre*1e3:.3f} mm | TRE 평균 {tres.mean()*1e3:.3f} mm "
          f"(최대 {tres.max()*1e3:.3f}) | σ_target {sig_t*1e3:.3f} mm | "
          f"불일치 {disagree*1e3:.3f} mm | 겹침 {inlier*100:.0f}%")

    # ---- 프로브 점수 스윕: 실 스캔에서도 커버리지가 정확도를 만드는가 ----
    counts = [150, 300, 800, 2000]
    sweep = []
    for n in counts:
        vals = []
        for s in range(3):
            r = np.random.default_rng(100 + s)
            Tp = random_perturb(r)
            pr = make_probe(r, full45, Tp, n=n)
            Tr = register(pr, model, normals, tree, vt, multistart=False)[0]
            vals.append(float(np.mean(tre_of(Tr, Tp, targets))))
        sweep.append(float(np.median(vals)))
        print(f"  프로브 {n:5d}점 → TRE 중앙값 {sweep[-1]*1e3:7.3f} mm")

    # ---- 몬테카를로: exp 42 의 신뢰도 게이트가 실 데이터에서도 작동하는가 ----
    print("-" * 78)
    print(f"[게이트 전이 시험 {n_trials}회] 부분 커버리지·무작위 자세로 실 스캔 정합")
    res = []
    for i in range(n_trials):
        r = np.random.default_rng([7, i])
        Tp = random_perturb(r)
        # 절반은 국소 구획만 찍는 나쁜 커버리지
        patch = (int(r.integers(len(full45))), float(r.uniform(0.05, 0.25))) \
            if i % 2 == 0 else None
        pr = make_probe(r, full45, Tp, patch=patch)
        Tr, fre_i, sig_i, dis_i, inl_i = register(pr, model, normals, tree, vt)
        tre_i = float(np.mean(tre_of(Tr, Tp, targets)))
        res.append(dict(tre=tre_i, fre=fre_i, sigma=sig_i, disagree=dis_i,
                        inlier=inl_i, unsafe=tre_i > MISS_TOL))
    tre_a = np.array([r["tre"] for r in res])
    fre_a = np.array([r["fre"] for r in res])
    sig_a = np.array([r["sigma"] for r in res])
    dis_a = np.array([r["disagree"] for r in res])
    inl_a = np.array([r["inlier"] for r in res])
    unsafe = np.array([r["unsafe"] for r in res])

    gate_s = K_SIGMA * sig_a > MISS_TOL              # 신호 1: 조건화
    gate_d = dis_a > DISAGREE_TOL                    # 신호 2: basin 일치성
    gate_i = inl_a < INLIER_MIN                      # 신호 3: 겹침 비율
    gate_all = gate_s | gate_d | gate_i

    def summarize(tag, gate):
        det = 100 * np.sum(unsafe & gate) / max(unsafe.sum(), 1)
        fa = 100 * np.sum(~unsafe & gate) / max((~unsafe).sum(), 1)
        exec_mask = ~gate
        ex = 100 * np.sum(unsafe & exec_mask) / max(exec_mask.sum(), 1)
        print(f"  {tag:26s} 검출 {det:5.1f}% | 오경보 {fa:5.1f}% | "
              f"집행분 unsafe {ex:5.1f}% ({int(np.sum(unsafe & exec_mask))}/"
              f"{int(exec_mask.sum())})")
        return det, fa, ex

    print(f"  naive(항상 집행)           unsafe {100*unsafe.mean():.1f}% "
          f"({int(unsafe.sum())}/{n_trials}), TRE 중앙값 {np.median(tre_a)*1e3:.2f} mm")
    det_s, fa_s, ex_s = summarize("① k·σ (조건화)", gate_s)
    det_d, fa_d, ex_d = summarize("② 다중초기값 일치성", gate_d)
    det_i, fa_i, ex_i = summarize("③ 겹침 비율", gate_i)
    det_sd, fa_sd, ex_sd = summarize("①+②+③", gate_all)

    # FRE≠TRE 가 실 데이터에서도 성립하는가
    ok = np.isfinite(fre_a) & np.isfinite(tre_a)
    corr = float(np.corrcoef(fre_a[ok], tre_a[ok])[0, 1]) if ok.sum() > 2 else np.nan
    bad_lowfre = int(np.sum(unsafe & (fre_a < 2 * PROBE_NOISE)))
    print(f"  FRE-TRE 상관 {corr:.2f} (실 스캔에서는 대체로 함께 움직인다) — 다만 FRE가 "
          f"잡음바닥({2*PROBE_NOISE*1e3:.1f} mm) 수준인데 unsafe 인 경우가 {bad_lowfre}건. "
          "FRE는 유용한 1차 선별이지 TRE의 보증서가 아니다")

    # ---- 그림 ----
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    ax.scatter(model[:, 0], model[:, 1], s=1, color="0.6", label="pre-op model (bun000)")
    ax.scatter(probe[:, 0], probe[:, 1], s=4, color="crimson", alpha=0.6,
               label="probe, unknown pose")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("Real laser scans (Stanford Bunny), xy view", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    aligned = reg.apply_T(T_reg, probe)
    ax.scatter(model[:, 0], model[:, 2], s=1, color="0.6")
    ax.scatter(aligned[:, 0], aligned[:, 2], s=4, color="tab:blue", alpha=0.7,
               label="probe after ICP")
    ax.scatter(targets[:, 0], targets[:, 2], marker="*", s=120, color="k",
               label="targets (interior)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"After registration: FRE {fre*1e3:.2f} mm, TRE {tres.mean()*1e3:.2f} mm",
                 fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.plot(counts, np.array(sweep) * 1e3, "-o", color="tab:green")
    ax.axhline(floor * 1e3, color="0.5", ls="--", lw=1,
               label=f"dataset self-alignment ({floor*1e3:.2f} mm, different quantity)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("probed points"); ax.set_ylabel("TRE [mm]")
    ax.set_title("More surface, better registration (real scans)", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(np.clip(fre_a, 1e-5, 1) * 1e3, np.clip(tre_a, 1e-5, 1) * 1e3, s=18,
               c=np.where(unsafe, "crimson", "tab:blue"), alpha=0.75)
    ax.axhline(MISS_TOL * 1e3, color="0.4", ls=":", label="miss tolerance")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FRE [mm]"); ax.set_ylabel("TRE [mm]")
    ax.set_title(f"FRE screens, but does not certify, TRE (r={corr:.2f})", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.scatter(np.clip(sig_a, 1e-5, 10) * 1e3, np.clip(tre_a, 1e-5, 1) * 1e3, s=18,
               c=np.where(unsafe, "crimson", "tab:blue"), alpha=0.75)
    ax.axvline(MISS_TOL * 1e3 / K_SIGMA, color="seagreen", ls="--",
               label=f"abort gate (tol/{K_SIGMA:.0f})")
    ax.axhline(MISS_TOL * 1e3, color="0.4", ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted σ_target [mm]"); ax.set_ylabel("TRE [mm]")
    ax.set_title("Does the exp-42 covariance gate transfer?", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[1, 2]
    vals = [100 * unsafe.mean(), ex_s, ex_d, ex_i, ex_sd]
    ax.bar(["naive", "① k·σ", "② consistency", "③ overlap", "①+②+③"], vals,
           color=["crimson", "tab:orange", "tab:blue", "tab:purple", "seagreen"])
    ax.tick_params(axis="x", labelsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("unsafe among executed [%]")
    ax.set_title("Reliability gate on real scans", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("44. Registration validated on real laser scans — Stanford Bunny "
                 "(exp 41/42 pipeline, unmodified)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "44_registration_real_scans.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/44_registration_real_scans.png, "
          "assets/44_registration_real_scans.png")

    return dict(floor=floor, fre=fre, tre=float(tres.mean()), sigma=sig_t,
                sweep=sweep, counts=counts, unsafe_rate=float(unsafe.mean()),
                det_sigma=det_s, det_disagree=det_d, det_inlier=det_i, det_all=det_sd,
                exec_sigma=ex_s, exec_disagree=ex_d, exec_inlier=ex_i, exec_all=ex_sd,
                corr=corr, tre_trials=tre_a, unsafe=unsafe)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - Bunny 는 해부 구조가 아니다. 검증하는 것은 '수술'이 아니라 **실측 점군의 성질**
#     (부분 겹침, 비등방 잡음, 구멍, 불균일 곡률)에서 파이프라인이 버티는가이다.
#   - bun.conf 정렬은 정답이 아니라 추정치다(스캔 간 최근접 중앙값 ≈0.3 mm). 그보다
#     작은 차이를 주장하지 않는다.
#   - 프로빙 과정(샘플링·잡음·outlier)은 여전히 모델링이다. 실제 광학 트래커의 오차
#     구조(시선 방향 이방성·프로브 팁 캘리브레이션)는 더 복잡하다.
#   - 데이터는 재배포하지 않는다(data_cache/ 는 git 제외). 실행 시 원본에서 받는다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="시행 수를 줄여 빠르게")
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    main(n_trials=a.trials, quick=a.quick)
