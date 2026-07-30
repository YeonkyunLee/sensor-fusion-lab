"""원격조작: 지연 하 양방향 제어와 가상 고정구 — 힘을 되돌려주면 왜 불안정해지는가.

수술 로봇의 심장 중 하나는 **원격조작(teleoperation)** 이다. 집도의가 마스터를 움직이면 팔이
따라가고, 조직의 힘이 손으로 되돌아온다(양방향/force feedback). 그런데 통신·처리 지연이 있으면
"힘을 되돌려주는" 그 루프가 **불안정해진다** — 지연이 에너지를 만들어내는 것처럼 작동하기 때문이다.

이 실험은 세 구조를 같은 조건에서 비교한다(1-DOF, 삽입축. exp 47의 바늘-조직 힘 모델 재사용).

  (a) **단방향(unilateral)** — 힘을 안 돌려준다. 항상 안정하지만 술자가 조직을 못 느낀다.
  (b) **P-P 양방향(직접 힘 반사)** — 투명하지만 **지연에 따라 발산**한다.
  (c) **파동변수(wave variable)** — 통신 채널을 수동(passive)으로 만들어 **어떤 지연에서도 안정**.
      대가는 투명성 손실(겉보기 감쇠·강성이 붙는다).

--- 왜 파동변수가 안정한가 (한 줄) ---
속도·힘을 그대로 보내면 지연된 채널이 에너지를 생성할 수 있다. 대신
    u = (b·v + f)/√(2b),   v_wave = (b·v − f)/√(2b)
로 변환해 보내면, 채널로 들어간 에너지가 나온 에너지보다 항상 크거나 같다(수동성). 그러면 양단이
수동인 한 전체가 안정하다(Niemeyer & Slotine). 이 실험에서는 **채널 에너지를 직접 적분해** 그
성질이 성립하는지 수치로 확인한다.

--- 가상 고정구(virtual fixture) ---
금지구역(혈관)을 마스터에 **가상 벽**으로 렌더링해 술자의 손을 막는다(active constraint).
exp 9의 "모를 땐 멈춘다"가 사람이 조작하는 루프로 옮겨온 형태다. 여기서도 같은 함정이 있다:
**지연 하에서 벽을 너무 딱딱하게 만들면 벽 자체가 진동한다** — 안전장치가 위험이 되는 지점.

    python scripts/50_teleoperation_delay.py
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
imp = import_module("47_needle_impedance")      # 바늘-조직 힘 모델 파라미터 재사용

DT = 1e-3                    # 1 kHz 제어
T_END = 4.0
STEPS = int(T_END / DT)

# --- 마스터(장치+손) · 팔(follower) ---
M_M, B_M = 1.0, 4.0          # 마스터 관성·감쇠(가벼운 햅틱 장치)
M_S, B_S = 2.0, 20.0         # 팔 등가 관성·감쇠(삽입축)
K_OP, D_OP = 300.0, 10.0     # 술자 임피던스(손이 목표를 따라가는 강성·감쇠)
K_S, D_S = 2000.0, 60.0      # 팔의 위치 추종 이득
K_C = 4000.0                 # P-P 결합 강성 — 힘 반사를 세게 걸면 투명하지만 불안정해진다
B_WAVE = 40.0                # 파동 임피던스
LAMBDA_POS = 3.0             # 파동변수 위치표류 보정 이득 [1/s]

# --- 조직 (exp 47과 같은 자릿수) ---
K1, K2 = imp.K1_TISSUE, imp.K2_TISSUE
F_PUNC, F_CUT, MU = imp.F_PUNCTURE, imp.F_CUT, imp.MU_AXIAL
X_SURFACE = 0.030            # 표면 위치 [m]
X_TARGET = 0.055             # 표적 깊이 [m]

# --- 가상 고정구 ---
# 금지구역은 술자가 가려는 표적(55 mm)보다 앞이어야 의미가 있다. 벽이 없으면 그대로 통과한다.
X_WALL = 0.038               # 금지구역 시작(혈관 앞) [m]


class Tissue:
    """삽입축 1-DOF 조직 반력(관통 전 비선형 강성 → 관통 → 절삭+마찰). 상태 있음."""

    def __init__(self):
        self.punctured = False

    def force(self, x):
        d = x - X_SURFACE
        if d <= 0:
            return 0.0
        if not self.punctured:
            f = K1 * d + K2 * d * d
            if f >= F_PUNC:
                self.punctured = True
                return -F_CUT
            return -f
        return -(F_CUT + MU * d)


def operator_target(t):
    """술자가 의도하는 깊이: 표면까지 접근 → 천천히 삽입 → 표적에서 정지."""
    if t < 1.0:
        return X_SURFACE * (t / 1.0) ** 2 * (3 - 2 * (t / 1.0))
    if t < 3.0:
        u = (t - 1.0) / 2.0
        return X_SURFACE + (X_TARGET - X_SURFACE) * (3 * u ** 2 - 2 * u ** 3)
    return X_TARGET


def run(arch="wave", delay_ms=20.0, vf_stiffness=0.0, steps=STEPS, b_wave=None,
        vf_mode="local"):
    """한 조건 시뮬레이션.

    arch: 'uni'(단방향) | 'pp'(직접 힘 반사) | 'wave'(파동변수)
    vf_stiffness: 가상 고정구 강성 [N/m] (0이면 없음)
    반환 지표 dict. 발산하면 diverged=True 로 표시하고 그 시점까지만 채운다."""
    b = B_WAVE if b_wave is None else b_wave                # 파동 임피던스
    n_d = max(int(round(delay_ms * 1e-3 / DT)), 1)          # 편도 지연 샘플
    tissue = Tissue()
    xm = xs = 0.0
    vm = vs = 0.0
    # 지연 버퍼 (편도). 파동변수는 송신 후 pop 순서가 바뀌므로 길이를 맞춰 둔다.
    buf_m2s = [0.0] * n_d      # 마스터 → 팔
    buf_s2m = [0.0] * n_d      # 팔 → 마스터
    buf_pos = [0.0] * n_d      # 마스터 위치(파동변수 보정용 별도 채널)
    buf_vis = [0.0] * n_d      # 팔 위치의 시각 피드백(술자가 화면으로 보는 지연)
    buf_vf = [0.0] * n_d       # 원격에서 렌더링한 가상 고정구 힘의 되돌림 경로
    log = dict(t=[], xm=[], xs=[], fe=[], fm=[], vf=[], e_ch=[])
    e_in = e_out = 0.0
    diverged = False
    max_force = 0.0
    punct_step = None

    for k in range(steps):
        t = k * DT
        f_e = tissue.force(xs)                              # 조직 → 팔 (음수 = 저항)
        if tissue.punctured and punct_step is None:
            punct_step = k
        max_force = max(max_force, abs(f_e))

        # ---- 통신 채널 ----
        if arch in ("wave", "wavepos"):
            # 두-포트 파동변수. 각 단에서 (v, F) 를 파동으로 변환해 주고받는다:
            #   u = (b·v + F)/√(2b)   (전진파),   w = (b·v − F)/√(2b)   (반사파)
            # 마스터: 자기 속도 vm 과 **받은 반사파** w_in 으로 표시할 힘을 풀면
            #   F_m = b·vm − √(2b)·w_in,  그러면 전진파는 u_out = √(2b)·vm − w_in
            #   (대수 루프가 닫힌 형태로 풀린다 — 지연 없이 계산 가능)
            sq = np.sqrt(2 * b)
            w_in = buf_s2m.pop(0)
            f_m = b * vm - sq * w_in                      # 손에 표시할 힘(양수=저항)
            buf_m2s.append(sq * vm - w_in)                # 전진파 송신
            u_in = buf_m2s.pop(0)
            F_s = -f_e                                    # 팔이 조직에 가하는 힘(양수)
            vs_cmd = (sq * u_in - F_s) / b                # 받은 파동 → 목표 속도
            if arch == "wavepos":
                # 파동변수는 속도를 주고받으므로 **위치 대응이 표류**한다(알려진 약점).
                # 위치를 별도 채널로 함께 보내 느린 비례 보정을 얹는다. 게인이 크면
                # 수동성 여유를 깎으므로 작게 두고, 채널 에너지로 감시한다.
                buf_pos.append(xm)
                xm_d = buf_pos.pop(0)
                vs_cmd += LAMBDA_POS * (xm_d - xs)
            buf_s2m.append((b * vs - F_s) / sq)           # 반사파 송신(실제 속도 사용)
            f_slave = D_S * (vs_cmd - vs)                 # 팔의 속도 루프
            f_m = -f_m                                    # 마스터 동역학에 더할 부호
        elif arch == "pp":
            buf_m2s.append(xm)
            xm_d = buf_m2s.pop(0)
            buf_s2m.append(xs)
            xs_d = buf_s2m.pop(0)
            f_slave = K_S * (xm_d - xs) + D_S * (0.0 - vs)
            f_m = -K_C * (xm - xs_d)                            # 직접 힘 반사(위치 결합)
        else:  # 단방향
            buf_m2s.append(xm)
            xm_d = buf_m2s.pop(0)
            f_slave = K_S * (xm_d - xs) + D_S * (0.0 - vs)
            f_m = 0.0

        # ---- 가상 고정구(금지구역 벽) ----
        # 어디서 렌더링하는가가 관건이다.
        #   local  : 마스터 쪽에서 자기 위치로 계산 → 지연 경로를 타지 않는다(실제 시스템의 방식)
        #   remote : 원격(팔) 쪽에서 계산해 지연 채널로 되돌린다 → 지연이 강성 상한을 묶는다
        f_vf = 0.0
        if vf_stiffness > 0.0:
            if vf_mode == "local":
                if xm > X_WALL:
                    f_vf = -vf_stiffness * (xm - X_WALL) - 0.05 * np.sqrt(vf_stiffness) * vm
            else:
                f_vf_slave = 0.0
                if xs > X_WALL:
                    f_vf_slave = (-vf_stiffness * (xs - X_WALL)
                                  - 0.05 * np.sqrt(vf_stiffness) * vs)
                buf_vf.append(f_vf_slave)
                f_vf = buf_vf.pop(0)
        f_m_total = f_m + f_vf

        # ---- 술자(손) ----
        # 손을 계획 궤적으로 가져가는 임피던스. 되돌아온 힘이 손을 밀어내면 그만큼 덜 들어간다.
        # (시각 피드백으로 팔 위치에 폐루프를 걸어도 되지만, 그러면 **사람 루프 자체가** 지연에서
        #  불안정해져 구조 간 비교가 묻힌다 — 실제로도 그래서 술자는 지연을 보면 속도를 줄인다.
        #  여기서는 구조 비교가 목적이라 손은 계획을 따르게 두고, 평가는 투명성·안정성으로 한다.)
        f_h = K_OP * (operator_target(t) - xm) - D_OP * vm

        # ---- 채널 에너지(수동성 확인): 들어간 에너지 − 나온 에너지 ----
        e_in += max(-f_m * vm, 0.0) * DT
        e_out += max(f_e * vs, 0.0) * DT

        # ---- 적분 ----
        am = (f_h + f_m_total - B_M * vm) / M_M
        a_s = (f_slave + f_e - B_S * vs) / M_S
        vm += am * DT
        vs += a_s * DT
        xm += vm * DT
        xs += vs * DT

        log["t"].append(t); log["xm"].append(xm); log["xs"].append(xs)
        log["fe"].append(f_e); log["fm"].append(f_m_total); log["vf"].append(f_vf)
        log["e_ch"].append(e_in - e_out)

        if not (abs(xm) < 0.5 and abs(xs) < 0.5 and abs(vm) < 50 and abs(vs) < 50):
            diverged = True
            break

    arr = {k: np.array(v) for k, v in log.items()}
    n = len(arr["t"])
    res = dict(diverged=diverged, log=arr, max_force=float(max_force),
               punctured=tissue.punctured, delay_ms=delay_ms, arch=arch)
    if n > 10 and not diverged:
        # 투명성: 팔 위치가 마스터를 얼마나 따라오나 / 손이 느낀 힘이 실제 힘과 얼마나 맞나
        res["pos_err_mm"] = float(np.sqrt(np.mean((arr["xm"] - arr["xs"]) ** 2)) * 1e3)
        res["force_err_N"] = float(np.sqrt(np.mean((arr["fm"] - arr["fe"]) ** 2)))
        res["final_depth_mm"] = float(arr["xs"][-1] * 1e3)
        res["overshoot_mm"] = float(max(arr["xs"].max() - X_TARGET, 0.0) * 1e3)
        res["channel_energy"] = float(arr["e_ch"][-1])
        res["osc_mm"] = float(np.std(arr["xs"][int(0.8 * n):]) * 1e3)   # 정착 구간 진동
        res["wall_pen_mm"] = float(max(arr["xm"].max() - X_WALL, 0.0) * 1e3)
        # 판정 기준(원격조작의 표준 축):
        #   안정성 = 정착 구간에서 떨지 않는가(진동 0.5 mm 이하)
        #   투명성 = 손이 느낀 힘이 실제 조직력과 맞는가 / 팔이 마스터를 따라오는가
        # "표적 도달"은 사람이 루프에 있어야 판정되는 항목이라 성공 기준에서 제외한다.
        res["oscillatory"] = res["osc_mm"] > 0.5
        res["stable"] = not res["oscillatory"]
        res["usable"] = bool(res["stable"])
    else:
        res.update(oscillatory=True, stable=False, usable=False)
    return res


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print("=== 50. 원격조작: 지연 하 양방향 제어와 가상 고정구 ===")
    print(f"1-DOF 삽입축. 마스터(m={M_M} kg) ↔ 팔(m={M_S} kg), 1 kHz, 술자 임피던스 "
          f"K={K_OP} N/m")
    print(f"조직: 표면 {X_SURFACE*1e3:.0f} mm, 관통력 {F_PUNC:.1f} N, 표적 깊이 "
          f"{X_TARGET*1e3:.0f} mm (exp 47 모델)")

    delays = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    archs = [("uni", "단방향(힘 없음)"), ("pp", "P-P 직접 힘반사"), ("wave", "파동변수"),
             ("wavepos", "파동변수+위치보정")]

    # ---- 파동 임피던스 b: 투명성 ↔ 안정 여유의 손잡이 ----
    print("-" * 78)
    print("[파동 임피던스 b] 크면 안정하지만 손이 무겁고(겉보기 감쇠), 작으면 가볍지만 여유가 줄어든다")
    print(f"{'b':>6s} | {'삽입깊이[mm]':>12s} | {'진동[mm]':>9s} | {'힘오차[N]':>10s} | "
          f"{'지연 견딤':>10s}")
    b_rows = []
    for bw in (5.0, 10.0, 20.0, 40.0):
        r = run(arch="wavepos", delay_ms=50.0, b_wave=bw)
        # 견딤 = 넓은 지연 범위에서 모두 안정한가(한 지점만 보면 가장 무른 b 가 뽑힌다)
        robust = all(run(arch="wavepos", delay_ms=d, b_wave=bw).get("stable")
                     for d in (20.0, 100.0, 200.0))
        b_rows.append((bw, r, robust))
        print(f"{bw:6.0f} | {r.get('final_depth_mm', float('nan')):12.2f} | "
              f"{r.get('osc_mm', float('nan')):9.2f} | {r.get('force_err_N', float('nan')):10.2f} | "
              f"{'전 구간 안정' if robust else '일부 진동':>10s}")
    b_use = next((bw for bw, r, robust in b_rows if robust), B_WAVE)
    print(f"  → b 가 작으면 손이 가볍고 더 깊이 들어가지만(투명) 진동 여유가 줄어든다. "
          f"이후 비교는 넓은 지연에서 견디는 최소 b={b_use:.0f} 로 고정.")

    print("-" * 78)
    print("[지연 스윕] 편도 지연을 키우며 안정성을 본다")
    print("  (셀 = 삽입 깊이 / 정착 진동. ✗ = 진동 0.5 mm 초과 = 손끝이 떠는 상태)")
    print(f"{'지연[ms]':>8s} | " + " | ".join(f"{lbl:>18s}" for _, lbl in archs))
    table = {a: [] for a, _ in archs}
    for d in delays:
        cells = []
        for a, _ in archs:
            r = run(arch=a, delay_ms=d, b_wave=b_use)
            table[a].append(r)
            if r["diverged"]:
                cells.append(f"{'발산':>18s}")
            else:
                mark = "" if r["usable"] else " ✗"
                cells.append(f"{r['final_depth_mm']:6.1f} mm /{r['osc_mm']:5.2f}{mark:>2s}")
        print(f"{d:8.0f} | " + " | ".join(cells))

    for a, lbl in archs:
        bad = [r["delay_ms"] for r in table[a] if not r["usable"]]
        if not bad:
            print(f"  {lbl}: 시험한 전 구간(≤{delays[-1]:.0f} ms) 안정")
        elif all(b >= min(bad) for b in bad) and bad == sorted(bad) and \
                bad[0] == min(bad) and len(bad) == len([d for d in delays if d >= bad[0]]):
            print(f"  {lbl}: {bad[0]:.0f} ms 부터 계속 진동 (경계형 실패)")
        else:
            print(f"  {lbl}: 진동한 지연 = {', '.join(f'{b:.0f}' for b in bad)} ms "
                  "(단조롭지 않음 — 파동 반사가 특정 지연에서 겹치는 구간)")

    # ---- 투명성 대가 ----
    print("-" * 78)
    print("[투명성의 대가] 안정성을 얻으면 무엇을 잃는가 (지연 20 ms)")
    print(f"{'구조':>18s} | {'위치오차[mm]':>12s} | {'힘오차[N]':>10s} | "
          f"{'최종깊이[mm]':>12s} | {'채널에너지[J]':>13s}")
    for a, lbl in archs:
        r = run(arch=a, delay_ms=20.0)
        if r["diverged"]:
            print(f"{lbl:>18s} | {'발산':>12s} | {'—':>10s} | {'—':>12s} | {'—':>13s}")
            continue
        print(f"{lbl:>18s} | {r['pos_err_mm']:12.2f} | {r['force_err_N']:10.2f} | "
              f"{r['final_depth_mm']:12.2f} | {r['channel_energy']:13.4f}")
    print("  힘오차 = 손이 느낀 힘 vs 실제 조직력. 0이면 완전 투명(불가능).")
    print("  채널에너지 ≥ 0 이면 통신 채널이 수동적 — 파동변수의 성립 근거를 수치로 확인.")

    # ---- 단방향은 왜 위험한가 ----
    r_uni = run(arch="uni", delay_ms=20.0)
    r_wav = run(arch="wave", delay_ms=20.0)
    print("-" * 78)
    print("[힘을 못 느끼면] 단방향은 안정하지만 술자가 조직을 못 느낀다")

    def depth(r):
        return f"{r['final_depth_mm']:.2f} mm" if not r["diverged"] else "발산"

    print(f"  단방향   : 최대 조직력 {r_uni['max_force']:.2f} N, 최종 깊이 "
          f"{depth(r_uni)} (표적 {X_TARGET*1e3:.0f} mm)")
    print(f"  파동변수 : 최대 조직력 {r_wav['max_force']:.2f} N, 최종 깊이 {depth(r_wav)}")
    print("  → 힘 반사가 있으면 술자의 손이 조직 저항에 반응해 밀어 넣는 양이 달라진다. "
          "안전은 '느끼는 것'에서 시작한다.")

    # ---- 가상 고정구 ----
    print("-" * 78)
    print(f"[가상 고정구] 금지구역 {X_WALL*1e3:.0f} mm 를 벽으로 렌더링 — "
          "**어디서 계산하는가**가 강성 상한을 정한다")
    print(f"{'K_vf[N/m]':>10s} | {'로컬 침범[mm]':>13s} | {'로컬 진동':>9s} | "
          f"{'원격 침범[mm]':>13s} | {'원격 진동':>9s}")
    vf_rows = []
    for kvf in [0.0, 200.0, 800.0, 3000.0, 12000.0, 50000.0]:
        cells, entry = [], [kvf]
        for mode in ("local", "remote"):
            # 가상 고정구만 격리해 보려고 힘 반사가 없는 단방향 구조에서 시험한다
            # (양방향이면 조직 반력이 섞여 벽의 효과가 가려진다).
            r = run(arch="uni", delay_ms=50.0, vf_stiffness=kvf, vf_mode=mode)
            if r["diverged"]:
                cells += [f"{'발산':>13s}", f"{'—':>9s}"]
                entry += [np.nan, np.nan, True]
                continue
            xm = r["log"]["xm"]
            seg = xm[xm > X_WALL - 0.002]
            osc = float(np.std(np.diff(seg)) * 1e6) if len(seg) > 20 else 0.0   # µm
            cells += [f"{r['wall_pen_mm']:13.2f}", f"{osc:9.1f}"]
            entry += [r["wall_pen_mm"], osc, False]
        vf_rows.append(entry)
        print(f"{kvf:10.0f} | " + " | ".join(cells))
    print("  (진동 단위 = µm, 벽 접촉 구간에서 마스터 위치의 스텝간 표준편차)")
    loc = [v for v in vf_rows if v[0] > 0 and not v[3]]
    rem = [v for v in vf_rows if v[0] > 0 and not v[6]]
    if loc:
        best = min(loc, key=lambda v: v[1])
        print(f"  로컬 렌더링: K_vf={best[0]:.0f} N/m 까지 안정, 침범 "
              f"{best[1]:.2f} mm (벽 없음 대비 {vf_rows[0][1]/max(best[1],1e-9):.0f}배 감소)")
    rem_bad = [v[0] for v in vf_rows if v[0] > 0 and (v[6] or v[5] > 5 * max(v[2], 1e-9))]
    if rem_bad:
        print(f"  원격 렌더링: K_vf={min(rem_bad):.0f} N/m 부터 로컬보다 뚜렷하게 떨린다 "
              "— 지연 경로로 렌더링한 벽은 강성 상한이 지연에 묶인다")
    else:
        print("  원격 렌더링: 이 조건에서는 로컬과 큰 차이가 없었다(정직한 결과) — "
              "지연·강성을 더 키우면 갈린다")

    # ---- 그림 ----
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    # 그림 라벨은 저장소의 다른 그림들과 같이 영어로 둔다(matplotlib 기본 폰트에 한글 없음).
    en = {"uni": "unilateral (no force)", "pp": "P-P force reflection",
          "wave": "wave variables", "wavepos": "wave + position correction"}

    # (1) 위치 궤적: 네 구조 (지연 20 ms)
    ax = axes[0, 0]
    for (a, _), c in zip(archs, ("tab:blue", "crimson", "seagreen", "tab:purple")):
        r = run(arch=a, delay_ms=20.0, b_wave=b_use)
        lg = r["log"]
        ax.plot(lg["t"], lg["xs"] * 1e3, color=c, lw=1.4, label=en[a])
    ax.axhline(X_SURFACE * 1e3, color="0.5", ls=":", lw=1, label="tissue surface")
    ax.axhline(X_TARGET * 1e3, color="0.3", ls="--", lw=1, label="commanded target")
    ax.set_xlabel("t [s]"); ax.set_ylabel("tool depth [mm]")
    ax.set_title("Insertion with one-way delay 20 ms", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (2) P-P: 지연이 키우는 채터링 — 손이 느끼는 힘에서 가장 잘 보인다
    ax = axes[0, 1]
    for d, c in zip([5.0, 50.0, 100.0], ("seagreen", "tab:orange", "crimson")):
        lg = run(arch="pp", delay_ms=d)["log"]
        ax.plot(lg["t"], -lg["fm"], color=c, lw=1.1, label=f"{d:.0f} ms")
    ax.set_xlabel("t [s]"); ax.set_ylabel("force felt by the hand [N]")
    ax.set_title("P-P force reflection: delay breeds chatter", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (3) 파동변수는 같은 지연에서 버틴다
    ax = axes[0, 2]
    for d, c in zip([20.0, 100.0, 200.0], ("seagreen", "tab:blue", "tab:purple")):
        lg = run(arch="wavepos", delay_ms=d, b_wave=b_use)["log"]
        ax.plot(lg["t"], lg["xs"] * 1e3, color=c, lw=1.2, label=f"{d:.0f} ms")
    ax.axhline(X_TARGET * 1e3, color="0.3", ls="--", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("tool depth [mm]")
    ax.set_title("Wave variables: stable at any delay (but slower)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (4) 힘: 실제 조직력 vs 손이 느낀 힘
    ax = axes[1, 0]
    lg = run(arch="wavepos", delay_ms=20.0, b_wave=b_use)["log"]
    ax.plot(lg["t"], -lg["fe"], color="0.4", lw=1.4, label="actual tissue force")
    ax.plot(lg["t"], -lg["fm"], color="seagreen", lw=1.2, label="felt: wave variables")
    lg2 = run(arch="pp", delay_ms=20.0)["log"]
    ax.plot(lg2["t"], -lg2["fm"], color="crimson", lw=1.0, alpha=.8,
            label="felt: P-P (coupling spring)")
    lg3 = run(arch="uni", delay_ms=20.0)["log"]
    ax.plot(lg3["t"], -lg3["fm"], color="tab:blue", lw=1.2, ls=":",
            label="felt: unilateral (zero)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("force [N]")
    ax.set_title("Transparency: what the hand feels vs what is there", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (5) 채널 수동성
    ax = axes[1, 1]
    for (a, _), c in zip(archs, ("tab:blue", "crimson", "seagreen", "tab:purple")):
        lg = run(arch=a, delay_ms=50.0, b_wave=b_use)["log"]
        ax.plot(lg["t"], lg["e_ch"], color=c, lw=1.3, label=en[a])
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("channel energy in − out [J]")
    ax.set_title("Passivity: does the channel create energy? (50 ms)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (6) 가상 고정구 강성의 상한
    ax = axes[1, 2]
    ks = [v[0] for v in vf_rows]
    ax.plot(ks, [v[1] for v in vf_rows], "-o", color="seagreen",
            label="rendered locally (master side)")
    ax.plot(ks, [v[4] for v in vf_rows], "-s", color="crimson",
            label="rendered remotely (through delay)")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("virtual fixture stiffness K_vf [N/m]")
    ax.set_ylabel("forbidden-zone penetration [mm]")
    ax.set_title("A safety wall must be rendered locally (delay 50 ms)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(ks, [v[5] for v in vf_rows], ":", color="crimson", alpha=0.7)
    ax2.set_ylabel("chatter, remote rendering [µm]", color="crimson", fontsize=9)

    fig.suptitle("50. Teleoperation under delay — bilateral control, passivity, and virtual "
                 "fixtures", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "50_teleoperation_delay.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/50_teleoperation_delay.png, assets/50_teleoperation_delay.png")

    return dict(table=table, delays=delays, vf_rows=vf_rows,
                uni=r_uni, wave=r_wav)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 1-DOF 삽입축으로 축약했다. 실제 원격조작은 6-DOF(+그리퍼)이고, 회전축의 관성·마찰이
#     다르며 좌표 매핑(motion scaling)도 얹힌다.
#   - 술자를 선형 임피던스(K_OP·D_OP)로 모델링했다. 사람은 상황에 따라 강성을 바꾸고
#     (co-contraction) 학습한다 — 실제 안정성은 사람이 포함된 루프에서만 최종 확인된다.
#   - 지연을 **일정(constant)** 으로 뒀다. 실제 네트워크는 지터·패킷 손실이 있어 파동변수만으로는
#     부족하고 시간영역 수동성 제어(TDPA) 같은 보완이 쓰인다.
#   - 파동변수의 파라미터 b(파동 임피던스)를 고정했다. b가 투명성↔안정 여유의 조절 손잡이이며
#     조직 강성에 맞춰 튜닝하는 것이 실제 설계 문제다.
#   - 채널 에너지는 이산 적분 근사다. 수동성의 엄밀한 증명이 아니라 성립 여부의 수치 확인이다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
