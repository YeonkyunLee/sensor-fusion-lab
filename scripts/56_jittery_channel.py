"""원격조작 II: 지터·패킷 손실 채널 — 상수 지연 가정을 걷어내면 무엇이 남는가.

exp 50 은 편도 지연을 **상수**로 두고 파동변수가 200 ms 까지 안정하다는 걸 보였다. 그 안정성의
근거인 수동성 증명은 "지연이 일정하다"를 전제로 깔고 있다. 실제 통신망은 지터가 있고 패킷을 잃는다.

이 실험은 exp 50 의 플랜트(마스터·팔·조직·술자 임피던스)를 그대로 가져와 **채널만** 바꾼다:
공칭 지연 위에 지터를 얹고, 패킷을 잃고, 재정렬을 허용하고, 송신률을 낮춘다.

--- 무엇을 재는가 ---
수동성은 **주장이 걸려 있는 블록에서** 재야 한다. 파동 채널의 저장 에너지는 파동 좌표에서 정확히
쓸 수 있다:

    E_ch(t) = ∫₀ᵗ ½[ u_send² + w_send² − u_recv² − w_recv² ] dτ

상수 지연·무손실이면 이 값은 "지금 선로 위에 떠 있는 에너지"라 항상 ≥ 0 이다(그게 증명이다).
음수가 되면 채널이 없던 에너지를 만들어낸 것이다. exp 50 이 그렸던 시스템 전체 에너지 수지는
팔의 국소 감쇠 같은 **항상-소산 항**을 품고 있어서 이 위반을 볼 수 없다 — 그 대조도 같이 낸다.

--- 미리 밝히는 결론 ---
**가정을 걷어냈는데 처음에는 아무 일도 일어나지 않았다.** exp 50 의 설정 그대로 ±40 ms 지터와
40% 손실을 넣어도 채널은 사실상 수동이었고 진동도 늘지 않았다. 송신률을 1 kHz → 50 Hz 로
떨어뜨려도 마찬가지였다. 이 링크는 과제의 정보량 대비 과표본이고, 지터가 만드는 굶음(에너지 생성)은
같은 지터가 만드는 낡은 패킷 폐기(에너지 소멸)와 상쇄된다.

그런데 그 설정은 **과제를 완주하지 못한다** — 조직 반력 아래 정상상태 오차가 남아 도구가 표적
55 mm 가 아니라 34.8 mm 에서 멈춘다. 채널이 거의 여기되지 않는 상태로 시험을 통과한 것이다.
**실패할 수 없는 시험은 시험이 아니다.** 표류 보정 이득을 올려 완주시키자 같은 지터가 만드는
에너지가 세 자릿수로 커졌다.

그리고 그 자리는 구조적으로 예견돼 있었다. 표류 보정 λ(x_m − x_s) 는 **파동 변환 밖의 항**이라
수동성 증명이 애초에 덮은 적이 없다. **증명은 내내 참이었고, 일을 하는 부분을 덮고 있지 않았다.**

거기서부터가 설계다. 표준 처방인 디지터 버퍼는 수동성을 되찾아 주지만 그 추가 지연이 바로 그
보증 밖 위치 루프를 때려 진동을 0.21 → 1.60 mm 로 키운다(증명을 사고 성능을 팔았다).
**에너지 예산(TDPA)** 은 지연 없이 수동성을 회복하면서 진동도 가장 낮다 — 버퍼는 지연을 상시로
내고 예산은 사건이 있을 때만 낸다. 마지막으로, 그 예산을 **누적값으로** 보내면 손실 내성이 공짜로
따라온다(증분을 보내면 잃은 패킷의 몫이 영영 사라져 감쇠기가 상시 켜진다). 손실 내성이 알고리즘이
아니라 **보내는 양의 성질**에서 나오는 자리다.

    python scripts/56_jittery_channel.py
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
tele = import_module("50_teleoperation_delay")      # 플랜트를 통째로 재사용

DT = tele.DT
T_END = 4.0
STEPS = int(T_END / DT)

M_M, B_M = tele.M_M, tele.B_M
M_S, B_S = tele.M_S, tele.B_S
K_OP, D_OP = tele.K_OP, tele.D_OP
K_S, D_S = tele.K_S, tele.D_S
K_C = tele.K_C
LAM_50 = tele.LAMBDA_POS          # exp 50 이 쓴 위치 보정 이득 [1/s]
# exp 50 의 이득으로는 조직 반력 아래에서 정상상태 오차가 |f_e|/(D_S·λ) ≈ 13 mm 남아 도구가
# 표적(55 mm)에 닿지도 못한다. 그러면 채널이 거의 여기되지 않아 **가정을 걷어내도 아무 일이
# 없다** — 이 실험의 A 절이 그 상태다. 과제를 완주시키는 이득으로 올려야 시험이 시작된다.
LAM_TASK = 24.0
X_WALL, X_TARGET, X_SURFACE = tele.X_WALL, tele.X_TARGET, tele.X_SURFACE

DELAY_MS = 50.0          # 공칭 편도 지연 — exp 50 에서 파동변수가 여유 있게 버티던 지점
B_WAVE = 10.0            # exp 50 이 고른 최소 파동 임피던스
B_WAVE_BIG = 60.0        # "그냥 보수적으로 키우면 되지 않나" 기준선
N_SEEDS = 6
# 예산과 소비는 같은 항들의 합이라 이상적으로는 정확히 맞는다. 그런데 예산은 **전송된 스냅샷**이라
# 큰 두 수의 뺄셈으로 여유를 구하게 되고, 거기서 나는 반올림 오차(~1e-16 J)에 감쇠기가 매 스텝
# 걸린다 — 처음에 그렇게 짰다가 "아무 문제 없는데 가동률 73%"가 나와서 잡았다.
ATT_EPS = 1e-12
# 통신 상실 정지의 국소 위치 유지 이득(exp 58). 팔의 국소 감쇠 D_S 와 같은 자릿수로 두어 별도의
# 튜닝 손잡이가 되지 않게 한다 — 정지 자체가 새로운 자유도가 되면 "임계값 없는 판정"의 의미가 없다.
K_HOLD, D_HOLD = 4000.0, 120.0
# 적응형 술자(exp 62): 학습이 내부 시계를 늦출 수 있는 하한과, 아무 일 없을 때의 회복 속도.
OP_RATE_MIN, OP_RATE_UP = 0.15, 0.20

MODES = ("zoh", "zero", "tdpa", "bigb", "pp")


# --------------------------------------------------------------------------- #
# 지터·손실 채널
# --------------------------------------------------------------------------- #
class Channel:
    """단방향 패킷 채널. 보낸 것을 **도착 시각과 함께** 큐에 넣는다.

    매 스텝 도착분 중 **가장 최근에 보낸 것**만 채택하고 그보다 오래된 것은 버린다 — 재정렬된
    낡은 패킷을 쓰지 않는 실제 UDP 수신기의 동작이다. 이 수신기에서는 지연이 줄어드는 쪽 지터는
    낡은 패킷을 버리게 만들 뿐(소산)이고, 채널을 능동으로 만드는 경로는 오직 **굶는 스텝**이다.
    지터도 손실도 결국 같은 사건을 만든다.
    """

    def __init__(self, rng, delay_ms=DELAY_MS, jitter_ms=0.0, loss=0.0, zero=0.0):
        self.rng = rng
        self.n0 = max(int(round(delay_ms * 1e-3 / DT)), 1)
        self.nj = int(round(jitter_ms * 1e-3 / DT))
        self.loss = loss
        self.zero = zero
        self.q = []                 # [(도착 스텝, 송신 스텝, payload)]
        self.last = zero
        self.newest = -1
        self.n_sent = self.n_lost = self.n_stale = self.n_starved = self.n_late = 0
        # 굶은 **연속 구간**의 최대 길이. 평균 굶은 비율이 같아도 이 값이 크면 붙들고 있는 시간이
        # 길다는 뜻이라 성질이 다르다(exp 57 의 연집 손실이 겨냥하는 지점). 시작 구간(첫 수신 전)은
        # 세지 않는다 — 그건 공칭 지연이지 사건이 아니다.
        self.starve_run = self.max_starve_run = 0

    def send(self, k, payload):
        self.n_sent += 1
        if self.loss and self.rng.random() < self.loss:
            self.n_lost += 1
            return
        d = self.n0
        if self.nj:
            d += int(self.rng.integers(-self.nj, self.nj + 1))
        self.q.append((k + max(d, 1), k, payload))

    def recv(self, k, hold=True, n_buf=0):
        """(payload, 새 패킷인가) 반환. 굶으면 hold 면 마지막 값, 아니면 0.

        n_buf > 0 이면 **디지터(재생) 버퍼**를 둔다: 재생 시각을 송신+공칭지연+n_buf 로 못박고
        그때까지 온 것만 순서대로 내보낸다. 늦게 온 패킷은 버린다(n_late). 지터를 **추가 지연으로
        바꿔** 규칙적인 흐름을 사는 거래다.
        """
        best, keep = None, []
        for a, s, p in self.q:
            if n_buf:
                play = s + self.n0 + n_buf
                if a > play:                       # 재생 시각을 놓친 패킷
                    self.n_late += 1
                    continue
                if play > k:
                    keep.append((a, s, p))
                    continue
            elif a > k:
                keep.append((a, s, p))
                continue
            if s <= self.newest or (best is not None and s <= best[0]):
                self.n_stale += 1
                continue
            if best is not None:
                self.n_stale += 1
            best = (s, p)
        self.q = keep
        if best is None:
            self.n_starved += 1
            if self.newest >= 0:
                self.starve_run += 1
                self.max_starve_run = max(self.max_starve_run, self.starve_run)
            return (self.last if hold else self.zero), False
        self.starve_run = 0
        self.newest, self.last = best[0], best[1]
        return best[1], True


# --------------------------------------------------------------------------- #
# 시뮬레이션
# --------------------------------------------------------------------------- #
def run(mode="zoh", seed=0, jitter_ms=0.0, loss=0.0, delay_ms=DELAY_MS,
        b_wave=None, vf_stiffness=0.0, steps=STEPS, energy_mode="cumulative",
        lam_pos=None, buf_ms=0.0, rate_hz=None, chan=None, lam_gate=False,
        tissue_on=True, b_scale=1.0, estop=False, resume_ms=0.0, blind_mm=1.0,
        retract_mm=0.0, master_lock=False, breath_mm=0.0, breath_hz=0.25,
        tissue_obj=None, op_react_ms=0.0, op_lag_mm=3.0,
        op_force_N=0.0, op_learn=0.0, op_reverse_mm=0.0):
    """한 조건 시뮬레이션. exp 56·57·58·59 가 공유하는 1-DOF 원격조작 시뮬레이터다.

    mode:
      'zoh'  — 파동변수 + 굶으면 마지막 값 유지 (표준 구현)
      'zero' — 파동변수 + 굶으면 0
      'tdpa' — 파동변수 + 마지막 값 유지하되 **에너지 예산**으로 감쇠 (시간영역 수동성 제어)
      'bigb' — 'zoh' 인데 파동 임피던스를 키운 고정 보수화 기준선
      'pp'   — 직접 힘반사 (투명하지만 취약, 대조군)

    energy_mode: 'cumulative'(누적 에너지 전송) | 'increment'(증분 전송) — 손실 내성 대조용
    chan: 채널 생성자. 기본은 Channel(평균 0 균등 지터 + 독립 베르누이 손실). exp 57 이
          연집 손실·긴 지연 꼬리 채널을 여기에 끼워 넣어 **플랜트와 제어를 그대로 두고 채널만**
          바꾼다 — 이 실험이 exp 50 에 한 것과 같은 방식.
    lam_gate: 표류 보정 항도 같은 예산(β)으로 함께 죈다. exp 57 이 이걸 켜서 "예산은 파동만
          죄므로 도구를 못 세운다"를 확인했고, 결과가 **더 나빴다**(그 항이 브레이크였다).

    exp 58 이 쓰는 절제·보완 손잡이 — 끊긴 동안 도구를 멈추는 것이 **설계된 것인지 우연인지**를
    가리기 위한 것들이다:
      tissue_on=False — 조직 반력을 뺀다(자유공간 접근 구간). 정지 평형 하나가 사라진다.
      b_scale        — 팔의 국소 감쇠 배율. 1 미만이면 가볍고 잘 미끄러지는 축이 된다.
      estop          — **통신 상실 정지**를 켠다. 예산이 아무것도 허락하지 않는 상태(β=0)에서
                       **움직인 누적 거리**가 blind_mm 을 넘으면 팔이 자기 위치를 국소적으로
                       붙든다 — 채널을 거치지 않으므로 채널이 죽어도 동작한다(exp 50 의 "벽은
                       로컬 렌더링"과 같은 원칙을 실패 경로에 적용).
      blind_mm       — 그 누적 거리 한계 [mm]. **통신 파라미터가 아니라 사슬이 이미 선언해 둔
                       임상 여유**에서 온다(exp 45 shaft 2.17 mm, exp 48 통로 1.25 mm 계열).
                       β=0 자체를 트리거로 쓰면 지터가 방향별 80% 를 굶기는 채널에서 너무
                       자주 걸린다 — 처음에 그렇게 짰다가 정지가 98.5% 걸려 과제를 못 했다.
      resume_ms      — 정지 해제 시 원래 명령으로 되돌아가는 시간. 0 이면 즉시 복귀(돌진 위험).

    exp 59 가 쓰는 손잡이 — exp 58 이 남긴 두 자백("정지는 붙드는 것이지 후퇴가 아니다", "술자 쪽은
    그대로다")을 시험하기 위한 것들이다:
      breath_mm/hz   — **환자 움직임**. 조직 표면이 주기적으로 움직인다. 정지 중에는 도구가 고정된
                       장애물이 되므로 이 움직임이 곧 상대 침투가 된다 — "붙들고 있는 것이 정말
                       안전 상태인가"를 물을 수 있게 만드는 유일한 항이다(없으면 질문 자체가 없다).
      retract_mm     — 정지 시 붙드는 대신 이만큼 **후퇴**한다. 후퇴 자체가 '정보 없이 하는 움직임'
                       이라 공짜가 아니다 — 그 교환비가 exp 59 의 A 절이다.
      master_lock    — 정지 중 마스터를 **국소적으로** 제동한다(채널 불필요). 안 하면 술자는 계속
                       움직이고 그 어긋남이 복귀 돌진이 된다.
      tissue_obj     — 조직 모델 주입. 기본은 exp 47/50 의 절삭+마찰 모델인데, 그 모델은 관통 후
                       **탄성이 없어서** 도구를 붙들고 있는 동안 환자가 움직여도 힘이 쌓이지 않는다.
                       exp 59 가 stick-slip 파악 항을 넣은 모델을 여기로 끼운다.
      op_react_ms    — 술자의 반응. 0 이면 exp 50 이후 내내 쓴 **비적응** 술자(계획 궤적을 향한 고정
                       임피던스)다. >0 이면 "도구가 응답하지 않으면 손을 멈춘다"는 규칙이 붙는다 —
                       비적응 술자로는 마스터 제동을 평가할 수 없다(의도가 손 스프링에 저장된다).
    """
    rng = np.random.default_rng(20250805 + 1000 * seed)
    b = (B_WAVE_BIG if mode == "bigb" else (B_WAVE if b_wave is None else b_wave))
    hold = mode != "zero"
    is_wave = mode != "pp"
    lam = LAM_TASK if lam_pos is None else lam_pos
    zero = (0.0, 0.0, 0.0)
    n_buf = int(round(buf_ms * 1e-3 / DT))
    # 패킷 송신 주기. 표본 하나가 나르는 파동 에너지는 ½u²·T_s 이므로(T_s = 송신 간격) 장부도
    # 그 단위로 적어야 수신단이 붙들고 있는 동안 꺼내 쓰는 양과 맞는다.
    n_step = max(int(round(1.0 / (rate_hz * DT))), 1) if rate_hz else 1
    t_s = n_step * DT
    make = Channel if chan is None else chan
    ch_ms = make(rng, delay_ms, jitter_ms, loss, zero=zero)       # 마스터 → 팔
    ch_sm = make(rng, delay_ms, jitter_ms, loss, zero=zero)       # 팔 → 마스터
    tissue = tele.Tissue() if tissue_obj is None else tissue_obj
    sq = np.sqrt(2 * b)

    xm = xs = vm = vs = 0.0
    # 파동 에너지 장부. sent 는 **선로에 올린 전부**(잃은 것 포함 — 손실은 에너지를 없앨 뿐이다).
    e_u_sent = e_w_sent = 0.0
    e_u_ext = e_w_ext = 0.0          # 수신단이 실제로 꺼내 쓴 양
    bud_u = bud_w = 0.0              # 상대가 알려온 예산(누적 모드면 그 값 그대로)
    e_u_inc = e_w_inc = 0.0          # 증분 모드에서 이번 패킷에 실어 보낼 몫

    log = dict(t=[], xm=[], xs=[], fe=[], fm=[], e_ch=[], e_sys=[], beta=[],
               starved=[], held=[])
    e_sys_in = e_sys_out = 0.0       # exp 50 식 시스템 에너지 수지(비교용)
    e_ctrl = e_ctrl_max = 0.0        # 제어기 **전체**가 두 몸체에 해 준 일(exp 65)
    e_ctrl_nd = e_ctrl_nd_max = 0.0  # 같은 것에서 항상 소산인 국소 감쇠를 뺀 것
    n_att = 0
    beta_sum = 0.0
    diverged = False
    # 통신 상실 정지 상태(exp 58)
    stopped, x_hold, blend, n_estop = False, 0.0, 1.0, 0
    blind_acc = 0.0                  # 새 정보 없이 움직인 누적 거리 [m]
    blind_limit = blind_mm * 1e-3    # 선언된 임상 여유 [m] — 통신 파라미터가 아니다
    n_held = 0
    resume_vmax = 0.0
    resume_win = 0                   # 해제 직후 관찰 창 [스텝]

    punct_k = None
    surf = 0.0
    xm_hold = 0.0
    f_e_held_max = depth_held_max = mismatch_release = 0.0
    df_held_max = f_e_at_stop = 0.0
    f_e_held_lo, f_e_held_hi, f_e_held_swing = np.inf, 0.0, 0.0
    dose_held = secs_held = 0.0
    drag_held = drag_last = 0.0                   # 회복 불가 조직 변형(exp 63); 모델이 세면 쓴다
    op_seen, op_still, op_frozen = 0.0, 0, None   # 적응형 술자 상태(op_react_ms > 0)
    t_op, op_rate = 0.0, 1.0                      # 술자의 내부 시계와 그 진행 속도(exp 62)
    n_adverse = n_force_cue = 0
    f_ml_prev = 0.0                               # 손이 느끼는 힘에 잠금의 저항도 들어간다
    xs_rx = 0.0                                    # 술자가 화면으로 보는 팔 위치(지연된 값)
    for k in range(steps):
        t = k * DT
        # 환자 움직임: 조직 표면이 움직인다. 상대 침투가 xs − surf − X_SURFACE 가 되므로, 도구를
        # 붙들고 있으면 표면이 다가오는 만큼 그대로 더 박힌다.
        surf = breath_mm * 1e-3 * np.sin(2 * np.pi * breath_hz * t) if breath_mm else 0.0
        f_e = tissue.force(xs - surf) if tissue_on else 0.0
        if tissue.punctured and punct_k is None:
            punct_k = k
        do_send = (k % n_step == 0)

        if is_wave:
            # ---- 마스터: 반사파 수신 → 표시할 힘 → 전진파 송신 ----
            (w_in, bw_rx, xs_rx), fresh_w = ch_sm.recv(k, hold, n_buf)
            if energy_mode == "cumulative":
                bud_w = bw_rx
            elif fresh_w:
                bud_w += bw_rx
            beta_w = 1.0
            if mode == "tdpa":
                avail = bud_w - e_w_ext
                step_e = 0.5 * w_in * w_in * DT
                if step_e > avail + ATT_EPS:
                    beta_w = np.sqrt(max(avail, 0.0) / max(step_e, 1e-18))
                    w_in = w_in * beta_w
            e_w_ext += 0.5 * w_in * w_in * DT
            f_m_ch = -(b * vm - sq * w_in)          # 손에 표시(+x 방향 힘 부호)

            u_out = sq * vm - w_in
            if do_send:
                e_u_inc = 0.5 * u_out * u_out * t_s
                e_u_sent += e_u_inc
                ch_ms.send(k, (u_out,
                               e_u_sent if energy_mode == "cumulative" else e_u_inc, xm))

            # ---- 팔: 전진파 수신 → 속도 명령 → 반사파 송신 ----
            (u_in, bu_rx, xm_rx), fresh_u = ch_ms.recv(k, hold, n_buf)
            if energy_mode == "cumulative":
                bud_u = bu_rx
            elif fresh_u:
                bud_u += bu_rx
            beta_u = 1.0
            if mode == "tdpa":
                avail = bud_u - e_u_ext
                step_e = 0.5 * u_in * u_in * DT
                if step_e > avail + ATT_EPS:
                    beta_u = np.sqrt(max(avail, 0.0) / max(step_e, 1e-18))
                    u_in = u_in * beta_u
            e_u_ext += 0.5 * u_in * u_in * DT

            F_s = -f_e
            vs_cmd = (sq * u_in - F_s) / b
            if lam:
                # exp 50 이 표류를 잡으려고 얹은 위치 보정. **이 항은 파동 변환 밖**이라 위
                # 수동성 장부가 보증하지 않는다 — 이 실험에서 결국 물리는 자리가 여기다.
                # lam_gate=True 면 이 항도 같은 예산 신호(β)로 함께 죈다. exp 57 이 "예산은
                # 파동만 죄므로 도구를 세우지 못한다"를 확인하고 그 처방으로 쓰는 스위치다.
                vs_cmd += lam * (beta_u if lam_gate else 1.0) * (xm_rx - xs)
            w_out = (b * vs - F_s) / sq
            if do_send:
                e_w_inc = 0.5 * w_out * w_out * t_s
                e_w_sent += e_w_inc
                ch_sm.send(k, (w_out,
                               e_w_sent if energy_mode == "cumulative" else e_w_inc, xs))

            f_coup = D_S * vs_cmd
            f_loc = -D_S * vs * b_scale
            beta = min(beta_u, beta_w)
            if beta < 1.0:
                n_att += 1
                beta_sum += beta
        else:                                        # 'pp' 직접 힘반사
            if do_send:
                ch_ms.send(k, (xm, 0.0, 0.0))
                ch_sm.send(k, (xs, 0.0, 0.0))
            (xm_rx, _, _), fresh_u = ch_ms.recv(k, hold, n_buf)
            (xs_rx, _, _), fresh_w = ch_sm.recv(k, hold, n_buf)
            f_m_ch = -K_C * (xm - xs_rx)
            f_coup = K_S * (xm_rx - xs)
            f_loc = -D_S * vs * b_scale
            beta_u = beta = 1.0

        # ---- 통신 상실 정지 (exp 58) ----
        # **판정 기준을 네트워크가 아니라 오차 예산에서 가져온다.** β=0(예산이 아무것도 허락하지
        # 않음)은 "지금 새 정보 없이 재생 중"이라는 정확한 물리 신호이지만, 지터가 방향별로 80% 를
        # 굶기는 채널에서는 그 자체로는 너무 자주 성립한다(처음에 그걸 트리거로 썼다가 정지가
        # 98.5% 걸려 과제를 아예 못 했다). 그래서 **그 상태로 움직인 누적 거리**를 세고, 그 값이
        # 사슬이 이미 선언해 둔 여유(exp 45 의 shaft 여유 2.17 mm, exp 48 의 통로 1.25 mm 계열)를
        # 넘을 때 멈춘다. 임계값이 통신 파라미터가 아니라 **해부·계획에서 오는 숫자**가 된다.
        # 집행은 **국소적**이다: 팔이 자기 위치를 스프링-댐퍼로 붙든다. 채널을 거치지 않으므로
        # 채널이 죽어 있어도 동작하고, 고정점을 향한 소산 항이라 에너지를 만들지 않는다.
        if estop and is_wave:
            # **감시 대상은 위해의 정의와 같은 양이어야 한다.** 처음엔 β=0(예산 고갈) 구간의 이동을
            # 셌는데, 안전 지표는 "새 표본이 없는 동안 움직인 거리"(굶은 스텝)를 센다 — 전자는
            # 후자의 부분집합이라 감시가 위해를 덜 본다(맹행 4.14 → 3.23 mm 밖에 못 줄였다).
            # 그래서 굶은 스텝 기준으로 세고, 새 표본이 들어오면 카운터를 접는다.
            dry = (k > ch_ms.n0) and not (fresh_u and fresh_w)
            if not stopped:
                if dry:
                    blind_acc += abs(vs) * DT
                    if blind_acc > blind_limit:
                        # 붙들 목표. retract_mm 이 있으면 그만큼 뒤로 물러난 지점을 목표로 한다
                        # (조직 안에서만 후퇴하고 표면보다 얕게는 가지 않는다 — 도구를 완전히 빼는
                        #  것은 별개의 결정이다). retract_mm=0 이면 **정확히 지금 위치**를 붙든다:
                        # 자유공간에서도 표면으로 클램프하면 목표가 앞으로 밀린다(처음에 그랬다).
                        stopped, blend = True, 0.0
                        x_hold = xs
                        if retract_mm > 0.0:
                            x_hold = max(xs - retract_mm * 1e-3, X_SURFACE + surf)
                        xm_hold = xm
                        f_e_at_stop = abs(f_e)   # 정지 시점의 조직력(절삭 기저)
                        f_e_held_lo = f_e_held_hi = abs(f_e)   # 진폭은 **정지 1회 단위**로
                        n_estop += 1
                else:
                    blind_acc = 0.0
            if stopped:
                f_e_held_max = max(f_e_held_max, abs(f_e))
                # **증분**이 실제로 붙들기가 조직에 더 얹은 몫이다. 절삭 기저(2~3 N)는 어느 정책에서든
                # 있으므로 총합으로 보면 파악 항의 ≤0.8 N 이 잡음에 묻힌다 — 처음에 그랬다.
                df_held_max = max(df_held_max, abs(f_e) - f_e_at_stop)
                # 증분은 **정지가 걸린 위상**에 민감하다(호흡 마루에서 멈추면 작게 나온다). 정지 구간
                # 안에서의 **진폭**은 그 위상에 무관하므로 정책 비교에는 이쪽이 맞다(exp 60 에서
                # 증분만 보다가 환자 움직임이 클수록 부하가 작아지는 비단조를 만났다).
                # 단 **정지 1회 단위**로 재야 한다 — 전체 구간에 걸쳐 min/max 를 누적하면 서로 다른
                # 깊이의 정지가 섞이고, 관통 순간을 걸친 정지가 하나라도 있으면 스윙이 조직 파라미터와
                # 무관하게 F_PUNC 로 찍힌다(exp 61 에서 시드 절반이 정확히 4.00 N 으로 나와 잡았다).
                f_e_held_lo = min(f_e_held_lo, abs(f_e))
                f_e_held_hi = max(f_e_held_hi, abs(f_e))
                f_e_held_swing = max(f_e_held_swing, f_e_held_hi - f_e_held_lo)
                # 최댓값 계열 지표는 **지속시간을 못 본다.** 조직 손상이 누적이라면 이쪽이 맞는
                # 지표다(exp 60 이 "지표가 정보의 값을 정한다"로 남긴 항목, exp 61 에서 씀).
                dose_held += abs(f_e) * DT
                secs_held += DT
                # 조직 모델이 **회복 불가 변형**을 세고 있으면(exp 63) 정지 구간 몫만 따로 모은다.
                # 힘 계열 지표 셋(증분·진폭·누적)이 전부 못 보는 축이다 — 이완하면 힘은 사라져도
                # 끌려간 조직은 그대로 남는다.
                drag_held += max(0.0, getattr(tissue, "drag", 0.0) - drag_last)
                depth_held_max = max(depth_held_max, xs - surf - X_SURFACE)
                # 램프는 **정보가 실제로 오는 스텝에서만** 올라간다. 굶은 스텝에서는 리셋하지 않고
                # 그대로 멈춰 둔다 — 연속 fresh 를 요구하면 지터 채널에서 영구히 래치된다(처음에
                # 그렇게 짰다가 정지가 85% 걸려 도구가 12 mm 에서 멈췄다). 부수적으로 좋은 성질이
                # 하나 붙는다: **링크가 나쁘면 복귀도 느려진다.**
                if not dry:
                    blend += DT / max(resume_ms * 1e-3, DT)
                    if blend >= 1.0:
                        mismatch_release = max(mismatch_release, abs(xm - xs))
                        stopped, blend, blind_acc = False, 1.0, 0.0
                        resume_win = 100                          # 복귀 후 100 ms 를 관찰한다
                wgt = min(max(blend, 0.0), 1.0)
                f_coup = wgt * f_coup + (1.0 - wgt) * (K_HOLD * (x_hold - xs)
                                                       - D_HOLD * vs)
                n_held += 1
            elif resume_win > 0:
                # 램프 길이와 무관하게 **해제 직후 같은 창**에서 최대 속도를 본다. 램프 구간만
                # 보면 즉시 복귀(램프 0)가 가장 얌전해 보이는 착시가 생긴다 — 처음에 그랬다.
                resume_win -= 1
                resume_vmax = max(resume_vmax, abs(vs))
            # 정지 여부와 무관하게 매 스텝 갱신해야 위의 정지 구간 델타가 맞는다.
            drag_last = getattr(tissue, "drag", 0.0)

        # ---- 채널 에너지(파동 좌표, 주장이 걸린 블록) ----
        e_ch = (e_u_sent + e_w_sent) - (e_u_ext + e_w_ext) if is_wave else np.nan
        # ---- exp 50 식 시스템 수지(비교용): 국소 소산까지 섞인 지표 ----
        e_sys_in += max(-f_m_ch * vm, 0.0) * DT + max(-f_loc * vs, 0.0) * DT
        e_sys_out += max(f_e * vs, 0.0) * DT
        e_sys = e_sys_in - e_sys_out

        # ---- 가상 고정구(로컬 렌더링) ----
        f_vf = 0.0
        if vf_stiffness > 0.0 and xm > X_WALL:
            f_vf = -vf_stiffness * (xm - X_WALL) - 0.05 * np.sqrt(vf_stiffness) * vm

        # ---- 술자 ----
        # op_react_ms > 0 이면 "도구가 응답하지 않으면 손을 멈춘다". exp 50 은 시각 폐루프를 **이득
        # 있는 루프**로 걸었다가 사람 루프 자체가 발산해서 폐기했는데, 이건 이득이 아니라 **목표를
        # 그 자리에 얼려두는** 규칙이라 발산하지 않는다. 반응 지연은 사람의 반응시간이다.
        # 술자의 **내부 시계**로 목표를 읽는다. 얼어 있는 동안에는 시계도 안 간다 — 손을 멈춘
        # 술자는 과제를 진행하고 있지 않다. op_learn 이 이 시계의 진행 **속도**를 늦춘다.
        tgt = tele.operator_target(t_op)
        if op_react_ms > 0.0:
            # 판정 기준은 **술자가 실제로 보는/느끼는 양**이다.
            #  · 시각 단서: 화면의 도구가 손보다 얼마나 뒤처졌는가(exp 59). 처음엔 "도구가 안
            #    움직이면"으로 짰는데, 붙들려 있는 동안에도 팔이 기어가서 판정이 계속 리셋됐다.
            #  · 힘 단서(exp 62): 마스터에 표시되는 힘. exp 50 이 파동변수로 **전달**해 놓고도
            #    술자 모델이 그걸 **쓰지 않고** 있었다 — 못 느끼는 사람에게 힘을 보내고 있었다.
            cue = abs(xm - xs_rx) > op_lag_mm * 1e-3
            # 손이 느끼는 것은 채널이 보낸 힘 **더하기 마스터가 스스로 거는 힘**이다. 잠금을
            # 빼고 채널 힘만 느끼게 하면 "잠금이 단서를 가리는가"라는 질문이 공정하지 않다
            # (잠금의 저항 자체가 술자에게는 단서다). 한 스텝 전 값을 쓴다 — 1 kHz 에서 무해.
            if op_force_N > 0.0 and abs(f_m_ch + f_ml_prev) > op_force_N:
                cue = True
                n_force_cue += 1
            if cue:
                op_still += 1
            else:
                op_still, op_frozen = 0, None
            if op_still * DT * 1e3 > op_react_ms:
                if op_frozen is None:
                    op_frozen = tgt                       # 목표를 그 자리에 얼린다(손을 멈춘다)
                    n_adverse += 1
                    if op_learn > 0.0:
                        # **학습**: 겪을 때마다 더 조심스러워진다(고전적인 move-and-wait).
                        op_rate = max(op_rate * op_learn, OP_RATE_MIN)
                # **되돌림**: 얼기만 하는 게 아니라 손을 빼기도 한다.
                tgt = op_frozen - op_reverse_mm * 1e-3
        if op_frozen is None:
            t_op += op_rate * DT                          # 얼어 있으면 과제가 진행되지 않는다
            if op_learn > 0.0:
                op_rate = min(op_rate + OP_RATE_UP * DT, 1.0)   # 아무 일 없으면 천천히 회복
        f_h = K_OP * (tgt - xm) - D_OP * vm

        # ---- 정지 중 마스터 제동 (exp 59) ----
        # 팔만 멈추면 술자는 계속 움직이고 그 어긋남이 복귀 돌진이 된다. 제동도 **국소**로 한다 —
        # 마스터가 자기 위치를 붙드는 것이므로 채널이 죽어도 동작한다(exp 50 의 로컬 렌더링 원칙).
        f_ml = 0.0
        if master_lock and estop and stopped:
            f_ml = K_HOLD * (xm_hold - xm) - D_HOLD * vm
        f_ml_prev = f_ml

        # ---- 제어기 **전체**의 에너지 장부 (exp 65) ----
        # 위의 e_ch 는 **파동 블록**만 감싼다 — 56~64 가 아홉 실험 동안 "장부는 그 블록만 덮는다"고
        # 적어 왔지만, 정작 **제어기 전체가 수동인가**는 아무도 안 물었다. 여기서 센다:
        # 제어기가 두 기계 몸체에 **해 준 일**의 누적. 내부 전원이 없는(수동) 제어기라면 초기
        # 저장량을 넘어 공급할 수 없으므로 이 값이 위로 유계여야 한다. 자라면 **에너지를 만든 것**이다.
        p_ctrl = (f_m_ch + f_vf + f_ml) * vm + (f_coup + f_loc) * vs
        e_ctrl += p_ctrl * DT
        e_ctrl_max = max(e_ctrl_max, e_ctrl)
        # f_loc 은 항상 소산(-D_S·vs)이라 위 합계를 **관대하게** 만든다 — exp 50 의 시스템 장부가
        # 위반을 못 본 것과 같은 이유(R17). 그 항을 뺀 장부도 같이 둬서 둘을 비교한다.
        e_ctrl_nd += (p_ctrl - f_loc * vs) * DT
        e_ctrl_nd_max = max(e_ctrl_nd_max, e_ctrl_nd)

        am = (f_h + f_m_ch + f_vf + f_ml - B_M * vm) / M_M
        a_s = (f_coup + f_loc + f_e - B_S * vs) / M_S
        vm += am * DT
        vs += a_s * DT
        xm += vm * DT
        xs += vs * DT

        log["t"].append(t); log["xm"].append(xm); log["xs"].append(xs)
        log["fe"].append(f_e); log["fm"].append(f_m_ch)
        log["e_ch"].append(e_ch); log["e_sys"].append(e_sys); log["beta"].append(beta)
        # 이번 스텝에 **새 표본이 없었나**. 굶은 구간에 도구가 얼마나 움직였는지(= 모르는 채로 간
        # 거리)를 뒤에서 재려고 남긴다 — exp 57 이 연집 손실의 안전 비용을 여기서 뽑는다.
        log["starved"].append(0.0 if (fresh_u and fresh_w) else 1.0)
        log["held"].append(1.0 if (estop and stopped) else 0.0)

        if not (abs(xm) < 0.5 and abs(xs) < 0.5 and abs(vm) < 50 and abs(vs) < 50):
            diverged = True
            break

    arr = {kk: np.array(v) for kk, v in log.items()}
    n = len(arr["t"])
    starved = ch_ms.n_starved + ch_sm.n_starved
    res = dict(mode=mode, diverged=diverged, log=arr, jitter_ms=jitter_ms, loss=loss,
               chans=(ch_ms, ch_sm),        # 채널 통계(굶음·폐기·지각)를 바깥에서 볼 수 있게
               n_starved=starved, starve_frac=starved / max(2 * n, 1),
               n_lost=ch_ms.n_lost + ch_sm.n_lost,
               n_stale=ch_ms.n_stale + ch_sm.n_stale,
               n_late=ch_ms.n_late + ch_sm.n_late,
               n_estop=n_estop, held_frac=n_held / max(len(log["t"]), 1),
               resume_vmax_mms=resume_vmax * 1e3,
               f_e_held_max=f_e_held_max, df_held_max=df_held_max,
               f_e_held_swing=f_e_held_swing,
               n_adverse=n_adverse, n_force_cue=n_force_cue,
               op_rate_end=op_rate, t_op_end=t_op,
               f_e_held_dose=dose_held, secs_held=secs_held,
               e_ctrl_max=e_ctrl_max, e_ctrl_nd_max=e_ctrl_nd_max, e_ctrl_end=e_ctrl,
               drag_held_mm=drag_held * 1e3,
               drag_total_mm=getattr(tissue, "drag", 0.0) * 1e3,
               mismatch_release_mm=mismatch_release * 1e3,
               depth_held_max_mm=depth_held_max * 1e3,
               waste_frac=(ch_ms.n_stale + ch_sm.n_stale + ch_ms.n_late
                           + ch_sm.n_late) / max(ch_ms.n_sent + ch_sm.n_sent, 1),
               att_duty=n_att / max(n, 1),
               beta_mean=(beta_sum / n_att) if n_att else 1.0)
    if diverged or n < 10:
        res.update(stable=False, osc_mm=np.inf, e_min=-np.inf, e_sys_min=-np.inf,
                   force_err_N=np.inf, pos_err_mm=np.inf, final_depth_mm=np.nan,
                   tool_pen_mm=np.nan, master_pen_mm=np.nan, lag_mm=np.inf,
                   force_err_max_N=np.inf, force_err_punc_N=np.inf, punct_ms=np.nan,
                   e_drawdown=np.inf)
        return res
    res["osc_mm"] = float(np.std(arr["xs"][int(0.8 * n):]) * 1e3)
    res["stable"] = bool(res["osc_mm"] <= 0.5)
    res["e_min"] = float(np.nanmin(arr["e_ch"])) if is_wave else np.nan
    # **최대 낙폭**: 어느 구간에서든 채널이 순수하게 꺼내 쓴 양의 최대치. 누적 최솟값(e_min)은
    # 선로에 떠 있는 에너지 저수지에 가려지므로, 한 사건의 크기를 보려면 이쪽을 봐야 한다.
    res["e_drawdown"] = (float(np.max(np.maximum.accumulate(arr["e_ch"]) - arr["e_ch"]))
                         if is_wave else np.nan)
    res["e_sys_min"] = float(arr["e_sys"].min())
    res["force_err_N"] = float(np.sqrt(np.mean((arr["fm"] - arr["fe"]) ** 2)))
    # RMS 는 **사건을 가린다**. 손이 실제로 정보를 받는 순간은 관통 과도(수 ms)이고, 그 구간의
    # 충실도가 채널이 얼마나 빨라야 하는지를 정한다 — 전 구간 RMS 로는 안 보인다.
    dfm = np.abs(arr["fm"] - arr["fe"])
    res["force_err_max_N"] = float(dfm.max())
    if punct_k is not None and punct_k + 50 < n:
        res["force_err_punc_N"] = float(dfm[punct_k:punct_k + 50].max())
        res["punct_ms"] = float(punct_k * DT * 1e3)
    else:
        res["force_err_punc_N"] = np.nan
        res["punct_ms"] = np.nan
    res["pos_err_mm"] = float(np.sqrt(np.mean((arr["xm"] - arr["xs"]) ** 2)) * 1e3)
    res["final_depth_mm"] = float(arr["xs"][-1] * 1e3)
    res["master_pen_mm"] = float(max(arr["xm"].max() - X_WALL, 0.0) * 1e3)
    res["tool_pen_mm"] = float(max(arr["xs"].max() - X_WALL, 0.0) * 1e3)
    # 벽에 손이 닿아 있는 동안 도구가 어디 있었나 — 벽이 지키는 것과 실제로 지켜지는 것의 차이
    at_wall = arr["xm"] > X_WALL - 1e-4
    res["lag_mm"] = float(np.median(arr["xm"][at_wall] - arr["xs"][at_wall]) * 1e3) \
        if at_wall.sum() > 5 else np.nan
    return res


def sweep(mode, seeds=N_SEEDS, **kw):
    """지터·손실이 확률적이므로 시드 여러 개를 돌려 요약한다(안정 개수는 최악 기준)."""
    rs = [run(mode=mode, seed=s, **kw) for s in range(seeds)]
    ok = [r for r in rs if not r["diverged"]]

    def med(key, default=np.nan):
        return float(np.median([r[key] for r in ok])) if ok else default

    return dict(runs=rs, n_stable=sum(r["stable"] for r in rs), n=len(rs),
                osc_mm=med("osc_mm", np.inf),
                e_min=(float(np.min([r["e_min"] for r in ok]))
                       if ok and np.isfinite(ok[0]["e_min"]) else
                       (np.nan if ok else -np.inf)),
                e_sys_min=(min(r["e_sys_min"] for r in rs) if ok else -np.inf),
                force_err_N=med("force_err_N", np.inf),
                force_err_max_N=med("force_err_max_N", np.inf),
                force_err_punc_N=med("force_err_punc_N", np.inf),
                pos_err_mm=med("pos_err_mm", np.inf),
                final_depth_mm=med("final_depth_mm"),
                att_duty=float(np.mean([r["att_duty"] for r in rs])),
                starve_frac=float(np.mean([r["starve_frac"] for r in rs])),
                waste_frac=float(np.mean([r["waste_frac"] for r in rs])),
                tool_pen_mm=med("tool_pen_mm"), master_pen_mm=med("master_pen_mm"),
                lag_mm=med("lag_mm"), n_div=sum(r["diverged"] for r in rs))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    print("=== 56. 지터·패킷 손실 채널: 상수 지연 가정을 걷어내면 ===")
    print(f"exp 50 의 플랜트 그대로. 공칭 편도 지연 {DELAY_MS:.0f} ms — 상수 지연이면 파동변수가 "
          f"200 ms 까지 버티던 조건.")
    print("채널 에너지 = ½∫(보낸 파동² − 꺼낸 파동²). 상수 지연·무손실이면 '선로 위의 에너지'라 ≥0,"
          " 음수면 채널이 없던 에너지를 만든 것.")

    jitters = [0.0, 10.0, 40.0] if quick else [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    losses = [0.0, 0.10, 0.40] if quick else [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    rates = [1000, 100, 20] if quick else [1000, 500, 200, 100, 50, 20, 10]
    lams = [LAM_50, LAM_TASK] if quick else [3.0, 6.0, 12.0, 24.0, 48.0]

    # ---------------- A ----------------
    print("-" * 94)
    print(f"[A] exp 50 의 설정 그대로(위치보정 λ={LAM_50:.0f}/s) 지터와 손실을 넣는다")
    print(f"{'지터[±ms]':>9s} | {'E_min[mJ]':>10s} | {'진동[mm]':>8s} | {'안정':>6s} | "
          f"{'굶은 스텝':>8s} | {'버린 패킷':>8s} | {'도달깊이[mm]':>11s}")
    A = {}
    for j in jitters:
        s = sweep("zoh", seeds=seeds, jitter_ms=j, lam_pos=LAM_50)
        A[j] = s
        print(f"{j:9.0f} | {s['e_min']*1e3:10.4f} | {s['osc_mm']:8.3f} | "
              f"{s['n_stable']:3d}/{s['n']:<2d} | {s['starve_frac']*100:7.1f}% | "
              f"{s['waste_frac']*100:7.1f}% | {s['final_depth_mm']:11.1f}")
    print()
    print(f"{'손실률':>9s} | {'E_min[mJ]':>10s} | {'진동[mm]':>8s} | {'안정':>6s} | "
          f"{'굶은 스텝':>8s} | {'버린 패킷':>8s} | {'도달깊이[mm]':>11s}")
    A2 = {}
    for p in losses:
        s = sweep("zoh", seeds=seeds, loss=p, lam_pos=LAM_50)
        A2[p] = s
        print(f"{p*100:8.0f}% | {s['e_min']*1e3:10.4f} | {s['osc_mm']:8.3f} | "
              f"{s['n_stable']:3d}/{s['n']:<2d} | {s['starve_frac']*100:7.1f}% | "
              f"{s['waste_frac']*100:7.1f}% | {s['final_depth_mm']:11.1f}")
    print("  **아무 일도 일어나지 않았다.** 걷어낸 가정이 결론을 바꾸지 않았다.")
    print("  이유 셋은 채널 쪽에 있다:")
    print("   (1) 지연이 늘면 굶어서 같은 파동을 두 번 꺼내지만(생성), 줄면 낡은 패킷을 버린다"
          "(소멸) — 평균 0 인 지터에서 상쇄된다.")
    print("   (2) 손실은 애초에 채널 **안에서** 에너지를 없애는 사건이라 수동성을 깨지 않는다.")
    print("   (3) 1 kHz 는 이 과제의 정보량에 비해 터무니없이 과표본이다(아래 표).")

    # A-2. 패킷률
    print()
    print("  [패킷률] 지터·손실 없이 송신 주기만 늘린다 — 이 링크는 몇 Hz 가 필요한가")
    print(f"{'송신률[Hz]':>11s} | {'도달깊이[mm]':>11s} | {'위치오차[mm]':>12s} | "
          f"{'진동[mm]':>8s} | {'관통시 힘오차[N]':>15s}")
    R = {}
    for r in rates:
        s = sweep("zoh", seeds=1, rate_hz=r)
        R[r] = s
        fp = s["force_err_punc_N"]
        print(f"{r:11d} | {s['final_depth_mm']:11.2f} | {s['pos_err_mm']:12.2f} | "
              f"{s['osc_mm']:8.3f} | "
              f"{(f'{fp:.2f}' if np.isfinite(fp) else '—'):>15s}")
    print("   → 50 Hz 까지 사실상 차이가 없다. 1 kHz 는 **국소 제어 루프**가 요구하는 속도이지 "
          "채널이 날라야 할 정보량이 아니다.")
    print("   (20 Hz 행은 이산화 공진이다 — 단조 감쇠가 아니라 특정 주기에서 튄다.)")
    print("   관통 순간 손이 느끼는 힘은 어느 송신률에서도 3 N 가량 틀린다 — 그건 속도가 아니라 "
          "**지연**이 정하는 몫이다.")

    # ---------------- B ----------------
    print("-" * 94)
    print("  이유 (4) 가 진짜다: **이 설정은 과제를 완주하지 못한다.**")
    print(f"  조직 반력 아래 정상상태 오차가 |f_e|/(D_S·λ) 만큼 남아 도구가 "
          f"{A[0.0]['final_depth_mm']:.1f} mm 에 멈춘다(표적 {X_TARGET*1e3:.0f} mm). "
          "채널이 거의 여기되지 않는다.")
    print("  **실패할 수 없는 시험은 시험이 아니다.** 표류 보정 이득을 올려 완주시킨 뒤 다시 잰다.")
    print("-" * 94)
    print("[B] 과제를 완주시키면 지터가 문다")
    print(f"{'λ[1/s]':>7s} | {'무지터: 깊이 / 진동 / E_min':>34s} | "
          f"{'±20 ms: 깊이 / 진동 / E_min':>34s}")
    B = {}
    for lam in lams:
        cells = []
        for j in (0.0, 20.0):
            s = sweep("zoh", seeds=(1 if j == 0 else seeds), jitter_ms=j, lam_pos=lam)
            B[(lam, j)] = s
            cells.append(f"{s['final_depth_mm']:5.1f} mm / {s['osc_mm']:5.2f} mm / "
                         f"{s['e_min']*1e3:+8.4f} mJ")
        print(f"{lam:7.0f} | " + " | ".join(f"{c:>34s}" for c in cells))
    e_lo, e_hi = B[(lams[0], 20.0)]["e_min"], B[(LAM_TASK, 20.0)]["e_min"]
    print(f"  λ={lams[0]:.0f} 에서는 지터가 만든 에너지가 {e_lo*1e3:+.4f} mJ 로 무시할 크기인데, "
          f"완주하는 λ={LAM_TASK:.0f} 에서는 {e_hi*1e3:+.3f} mJ — "
          f"{abs(e_hi/e_lo):.0f} 배다.")
    print("  생성량은 신호 크기의 제곱을 따라간다. 결함은 내내 있었고 **시험이 그걸 여기하지 "
          "못했을 뿐**이다.")
    print("  게다가 구조적으로 예견된 자리다: 표류 보정 λ(x_m − x_s) 항은 **파동 변환 밖**이라")
    print("  수동성 증명이 애초에 덮은 적이 없다. **증명은 멀쩡한데, 일을 하는 부분을 안 덮고 있었다.**")
    print("  (λ=12·무지터 행의 큰 진동은 채널 불안정이 아니라 exp 47 의 관통 직후 돌진이다.)")

    # ---------------- C ----------------
    print("-" * 94)
    print("[C] 표준 처방 — 디지터(재생) 버퍼로 지터를 **추가 상수 지연**으로 바꾼다")
    print(f"    (exp 50 이 상수 지연 200 ms 까지 안정임을 보였으니 지연 예산은 남아 있다. "
          f"λ={LAM_TASK:.0f}, 지터 ±20 ms, 손실 5%)")
    print(f"{'버퍼[ms]':>8s} | {'실효 편도[ms]':>12s} | {'굶은 스텝':>8s} | {'늦어 버림':>9s} | "
          f"{'E_min[mJ]':>10s} | {'진동[mm]':>8s} | {'위치오차[mm]':>12s}")
    C = {}
    for bm in ([0.0, 20.0, 45.0] if quick else [0.0, 5.0, 10.0, 20.0, 30.0, 45.0]):
        s = sweep("zoh", seeds=seeds, jitter_ms=20.0, loss=0.05, buf_ms=bm)
        C[bm] = s
        late = float(np.mean([r["n_late"] for r in s["runs"]])) / (2 * STEPS) * 100
        print(f"{bm:8.0f} | {DELAY_MS+bm:12.0f} | {s['starve_frac']*100:7.1f}% | "
              f"{late:8.1f}% | {s['e_min']*1e3:10.4f} | {s['osc_mm']:8.3f} | "
              f"{s['pos_err_mm']:12.2f}")
    b0, bb = min(C), max(C)
    print(f"  버퍼는 **수동성을 되찾아 준다**: E_min {C[b0]['e_min']*1e3:+.3f} → "
          f"{C[bb]['e_min']*1e3:+.3f} mJ, 굶은 스텝 {C[b0]['starve_frac']*100:.0f}% → "
          f"{C[bb]['starve_frac']*100:.0f}%.")
    print(f"  그런데 성능은 **나빠진다**: 진동 {C[b0]['osc_mm']:.2f} → {C[bb]['osc_mm']:.2f} mm, "
          f"위치오차 {C[b0]['pos_err_mm']:.2f} → {C[bb]['pos_err_mm']:.2f} mm.")
    print("  이유가 B 와 같다 — 추가 지연이 때리는 곳이 바로 **보증 밖에 있는 그 위치 루프**다.")
    print("  **증명을 사고 성능을 팔았다.** 수동적인 것과 잘 도는 것은 같은 성질이 아니다.")

    # ---------------- D ----------------
    print("-" * 94)
    print(f"[D] 굶었을 때 무엇을 할 것인가 (λ={LAM_TASK:.0f}, 지터 ±20 ms, 손실 20%)")
    print(f"{'굶었을 때':>22s} | {'버퍼':>5s} | {'E_min[mJ]':>10s} | {'수동?':>5s} | "
          f"{'도달깊이[mm]':>11s} | {'진동[mm]':>8s} | {'위치오차[mm]':>12s} | {'감쇠율':>7s}")
    D = {}
    for lbl, m in (("마지막 값 유지(ZOH)", "zoh"), ("0 으로 채움", "zero"),
                   ("에너지 예산(TDPA)", "tdpa")):
        for bm in (0.0, 30.0):
            s = sweep(m, seeds=seeds, jitter_ms=20.0, loss=0.20, buf_ms=bm)
            D[(m, bm)] = s
            print(f"{lbl if bm == 0 else '':>22s} | {bm:5.0f} | {s['e_min']*1e3:10.4f} | "
                  f"{'예' if s['e_min'] >= -1e-9 else '아니오':>5s} | "
                  f"{s['final_depth_mm']:11.1f} | {s['osc_mm']:8.3f} | "
                  f"{s['pos_err_mm']:12.2f} | {s['att_duty']*100:6.1f}%")
    print("  0 채움은 에너지를 만들지 않는다 — 그리고 도구가 조직 표면도 못 넘는다. "
          "**안전한데 못 쓴다.**")
    print("  에너지 예산(TDPA)은 붙들되 **보내온 에너지가 허락하는 만큼만** 꺼낸다: 버퍼 없이도 "
          "수동성을 회복하고,")
    print(f"  진동은 셋 중 가장 낮으며({D[('tdpa', 0.0)]['osc_mm']:.2f} mm) 감쇠기는 "
          f"{D[('tdpa', 0.0)]['att_duty']*100:.0f}% 만 켜진다. 예산이 마르면 스스로 0 으로 "
          "수렴하므로 최악의 경우 0 채움으로 퇴화한다(안전한 쪽으로).")
    print("  **버퍼는 지연을 상시로 내고 예산은 사건이 있을 때만 낸다** — exp 54 의 절제 논리와 "
          "같은 모양.")

    # 관측기 위치
    r_bad = run("zoh", seed=0, jitter_ms=20.0, loss=0.20)
    print(f"  [관측기 위치] 같은 실행 — 파동 채널: {r_bad['e_min']*1e3:+.3f} mJ / "
          f"exp 50 식 시스템 수지: {r_bad['e_sys_min']*1e3:+.3f} mJ")
    print("    → 항상-소산 항이 섞인 지표로는 채널의 수동성을 판정할 수 없다. "
          "**관측기는 자기가 판정할 블록만 감싸야 한다.**")

    # ---------------- E ----------------
    print("-" * 94)
    print("[E] 손실 내성은 알고리즘이 아니라 **보내는 양의 성질**에서 나온다 (지터 ±20 ms)")
    print(f"{'손실률':>8s} | {'누적: 감쇠율 / 깊이 / 위치오차':>36s} | "
          f"{'증분: 감쇠율 / 깊이 / 위치오차':>36s}")
    M = {}
    for p in ([0.0, 0.20, 0.40] if quick else [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]):
        cells = []
        for em in ("cumulative", "increment"):
            s = sweep("tdpa", seeds=seeds, jitter_ms=20.0, loss=p, energy_mode=em)
            M[(em, p)] = s
            cells.append(f"{s['att_duty']*100:5.1f}% / {s['final_depth_mm']:5.1f} mm / "
                         f"{s['pos_err_mm']:5.2f} mm")
        print(f"{p*100:7.0f}% | " + " | ".join(f"{c:>36s}" for c in cells))
    print("  증분을 보내면 잃은 패킷의 예산이 **영영** 사라진다 → 예산이 실제보다 작아 보여 "
          "감쇠기가 상시 켜진다.")
    print("  누적값은 **단조**라서 다음 패킷 하나가 손실 이력 전체를 복구한다. "
          "같은 알고리즘, 다른 페이로드.")

    # ---------------- F ----------------
    print("-" * 94)
    print("[F] '그냥 보수적으로' 기준선 — 이득의 출처를 가른다 (지터 ±20 ms, 손실 20%)")
    print(f"{'구조':>26s} | {'E_min[mJ]':>10s} | {'진동[mm]':>8s} | {'도달깊이[mm]':>11s} | "
          f"{'힘오차[N]':>9s} | {'문제없을때 힘오차':>16s}")
    F = {}
    arms = [(f"파동 b={B_WAVE:.0f} (공칭)", dict(mode="zoh")),
            (f"파동 b={B_WAVE_BIG:.0f} (고정 보수화)", dict(mode="bigb")),
            ("P-P 직접 (대조)", dict(mode="pp")),
            ("디지터 버퍼 30 ms", dict(mode="zoh", buf_ms=30.0)),
            ("TDPA (에너지 예산)", dict(mode="tdpa"))]
    for lbl, kw in arms:
        s = sweep(seeds=seeds, jitter_ms=20.0, loss=0.20, **kw)
        s0 = sweep(seeds=1, **kw)
        F[lbl] = (s, s0)
        em = f"{s['e_min']*1e3:10.4f}" if np.isfinite(s["e_min"]) else f"{'—':>10s}"
        print(f"{lbl:>26s} | {em} | {s['osc_mm']:8.3f} | {s['final_depth_mm']:11.1f} | "
              f"{s['force_err_N']:9.2f} | {s0['force_err_N']:16.2f}")
    print("  고정 보수화(b 키우기)는 **문제가 없을 때도 항상** 투명성을 낸다. 그리고 지터가 만든 "
          "문제를 겨냥하지도 못한다 —")
    print("  exp 53 의 '대책은 자기가 겨냥한 오차원만 고친다'가 통신 채널에서 다시 나온다.")

    # ---------------- G ----------------
    print("-" * 94)
    print(f"[G] 가상 고정구({X_WALL*1e3:.0f} mm, 로컬 렌더링 K_vf=12000 N/m) — "
          "exp 50 은 침범을 **마스터**에서 쟀다")
    print(f"{'지터[±ms]':>9s} | {'마스터 침범[mm]':>14s} | {'도구 침범[mm]':>13s} | "
          f"{'손이 벽일 때 도구 뒤처짐[mm]':>24s} | {'+버퍼 30ms':>11s}")
    G = {}
    for j in ([0.0, 20.0, 40.0] if quick else [0.0, 5.0, 10.0, 20.0, 40.0]):
        s_z = sweep("zoh", seeds=seeds, jitter_ms=j, loss=0.05, vf_stiffness=12000.0)
        s_b = sweep("zoh", seeds=seeds, jitter_ms=j, loss=0.05, vf_stiffness=12000.0,
                    buf_ms=30.0)
        G[j] = (s_z, s_b)
        print(f"{j:9.0f} | {s_z['master_pen_mm']:14.2f} | {s_z['tool_pen_mm']:13.2f} | "
              f"{s_z['lag_mm']:24.2f} | {s_b['lag_mm']:11.2f}")
    jmax = max(G)
    print("  벽은 마스터에서 렌더링하니 **손은 잘 막힌다** — 지터와 거의 무관하다(exp 50 의 숫자).")
    print(f"  그런데 금지구역에 들어가는 것은 손이 아니라 **도구**이고, 손이 벽에 닿아 있는 동안 "
          f"도구는 {G[jmax][0]['lag_mm']:.1f} mm 뒤에 있다 — 마스터 침범 "
          f"{G[jmax][0]['master_pen_mm']:.2f} mm 로는 알 수 없는 값이다.")
    print("  exp 50 의 '침범 168배 감소'는 **술자의 손**에 대한 참말이다. 여기서는 도구가 뒤처져 "
          "과보호로 기울지만, 부호가 반대인 배치(도구가 앞서는 경우)면 그대로 사고다.")
    print("  → 안전 지표는 **위해가 발생하는 지점**에서 재야 한다(exp 41 의 FRE≠TRE 와 같은 병).")

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    for lbl, kw, c in [
            (f"exp 50 gain (lam={LAM_50:.0f}), jitter +/-20", dict(lam_pos=LAM_50,
                                                                   jitter_ms=20.0),
             "0.55"),
            ("task gain, constant delay", dict(), "seagreen"),
            ("task gain, jitter +/-20 + 20% loss", dict(jitter_ms=20.0, loss=0.20),
             "crimson")]:
        lg = run("zoh", seed=0, **kw)["log"]
        ax.plot(lg["t"], lg["e_ch"] * 1e3, color=c, lw=1.3, label=lbl)
    lg = run("tdpa", seed=0, jitter_ms=20.0, loss=0.20)["log"]
    ax.plot(lg["t"], lg["e_ch"] * 1e3, color="tab:blue", lw=1.4,
            label="same, with energy budget")
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("wave-channel stored energy [mJ]")
    ax.set_title("The violation shows up only once the task is exercised", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ls = sorted({l for l, _ in B})
    ax.plot(ls, [-B[(l, 20.0)]["e_min"] * 1e3 for l in ls], "-o", color="crimson",
            label="energy created by jitter [mJ]")
    ax2 = ax.twinx()
    ax2.plot(ls, [B[(l, 20.0)]["final_depth_mm"] for l in ls], "-^", color="tab:blue")
    ax2.axhline(X_TARGET * 1e3, color="0.4", ls="--", lw=1)
    ax2.set_ylabel("depth reached [mm]", color="tab:blue", fontsize=9)
    ax.set_xlabel("drift-correction gain lambda [1/s]")
    ax.set_ylabel("energy created [mJ]", color="crimson")
    ax.set_title("A test the system cannot fail is not a test", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")

    ax = axes[0, 2]
    rs_ = sorted(R)
    ax.semilogx(rs_, [R[r]["final_depth_mm"] for r in rs_], "-o", color="tab:blue",
                label="depth reached [mm]")
    ax.semilogx(rs_, [R[r]["force_err_punc_N"] for r in rs_], "-s", color="crimson",
                label="force error at puncture [N]")
    ax.semilogx(rs_, [R[r]["pos_err_mm"] for r in rs_], "-^", color="seagreen",
                label="master-tool error [mm]")
    ax.set_xlabel("packet rate [Hz]"); ax.set_ylabel("[mm] / [N]")
    ax.set_title("1 kHz is what the local loop needs, not the channel", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    bs = sorted(C)
    ax.plot([DELAY_MS + b for b in bs], [-C[b]["e_min"] * 1e3 for b in bs], "-o",
            color="crimson", label="energy created [mJ]")
    ax2 = ax.twinx()
    ax2.plot([DELAY_MS + b for b in bs], [C[b]["osc_mm"] for b in bs], "-^",
             color="tab:blue", label="settled oscillation [mm]")
    ax.set_xlabel("effective one-way delay [ms]  (de-jitter buffer added)")
    ax.set_ylabel("energy created [mJ]", color="crimson")
    ax2.set_ylabel("oscillation [mm]", color="tab:blue", fontsize=9)
    ax.set_title("The buffer buys the proof and sells the performance", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ps = sorted({p for _, p in M})
    for em, c, lbl in [("cumulative", "tab:blue", "send cumulative energy"),
                       ("increment", "crimson", "send increments")]:
        ax.plot([p * 100 for p in ps], [M[(em, p)]["att_duty"] * 100 for p in ps],
                "-o", color=c, label=lbl)
    ax.set_xlabel("packet loss [%]"); ax.set_ylabel("attenuator duty cycle [%]")
    ax.set_title("Loss tolerance comes from what you send", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 2]
    js = sorted(G)
    ax.plot(js, [G[j][0]["master_pen_mm"] for j in js], "-o", color="seagreen",
            label="master penetration (exp 50's number)")
    ax.plot(js, [G[j][0]["lag_mm"] for j in js], "-s", color="crimson",
            label="tool behind the master at the wall")
    ax.plot(js, [G[j][1]["lag_mm"] for j in js], "-^", color="tab:blue",
            label="same, with a 30 ms de-jitter buffer")
    ax.set_xlabel("jitter [+/- ms]"); ax.set_ylabel("distance [mm]")
    ax.set_title("The wall protects the hand; the patient meets the tool", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("56. A jittery, lossy channel - the proof held, and it was never "
                 "covering the part that does the work", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "56_jittery_channel.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/56_jittery_channel.png, assets/56_jittery_channel.png")

    return dict(A=A, A2=A2, R=R, B=B, C=C, D=D, M=M, F=F, G=G)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 1-DOF 삽입축, 술자는 선형 임피던스로 계획 궤적을 따른다(exp 50 과 동일). 사람이 지터를
#     느끼고 속도를 줄이는 적응은 없다 — 실제로는 그게 가장 큰 안정화 요인일 수 있다.
#   - 지터를 평균 0 인 균등분포로 뒀다. 실제 망 지연은 **한쪽으로 긴 꼬리**(late tail)라 생성과
#     소멸이 상쇄되지 않을 수 있다. A 절의 상쇄 논리는 그 대칭성에 기대고 있다.
#   - 손실도 독립 베르누이다. 실제 손실은 **연집(burst)** 이라 같은 손실률에서도 굶는 구간이 길다.
#     예산이 마르는 속도가 달라지므로 TDPA 의 감쇠 프로파일은 이 결과보다 거칠 것이다.
#   - 수신기를 '최신 것만 채택, 낡은 것은 폐기'로 고정했다. 이 선택이 A 절의 상쇄를 만든다 —
#     도착분을 모두 재생하는 수신기라면 지연 감소 구간에서 파동이 압축돼 결과가 달라진다.
#   - 채널 에너지는 이산 합이다. 수동성의 엄밀한 증명이 아니라 성립 여부의 수치 확인이다.
#   - 표류 보정 λ 는 파동 변환 밖의 항이라 이 장부가 덮지 않는다. 그 사실 자체가 B·C 절의 결론
#     이지만, 그렇다고 λ 를 포함한 전체 시스템의 수동성을 여기서 증명한 것은 아니다.
#   - '완주시키는 이득'을 λ=24 로 고른 것은 이 플랜트·이 조직 모델에서의 값이다. 임상 시스템은
#     보통 위치 보정 대신 별도의 위치 채널이나 하이브리드 구조를 쓴다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
