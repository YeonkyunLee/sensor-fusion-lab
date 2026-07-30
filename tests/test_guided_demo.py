"""브라우저 데모(guided_demo.html) 코어의 헤드리스 검증. node 없으면 skip.

데모도 "돌아가는 것처럼 보이는" 것으로 끝내지 않기 위해, 데모가 주장하는 순서
(매끄러운 영역 < 특징적 영역 < 흩어 찍기)를 node 로 숫자 검증한다.

실행: pytest -q  /  직접: node tests/guided_demo_check.js
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "guided_demo.html"
CHECK = Path(__file__).with_name("guided_demo_check.js")


def test_demo_file_is_wellformed_and_page_ready():
    """Pages 서빙 조건: front matter, 코어 블록, 캔버스, 조작 버튼이 존재."""
    html = DEMO.read_text(encoding="utf-8")
    assert html.startswith("---\nlayout: none\n---"), "Jekyll front matter 누락"
    assert re.search(r'<script id="core">[\s\S]{500,}?</script>', html), "코어 블록 누락"
    assert "<canvas" in html
    for btn in ("btnLand", "btnReg", "btnSmooth", "btnFeature", "btnSpread"):
        assert f'id="{btn}"' in html, f"{btn} 버튼 없음"
    # 외부 라이브러리 의존 없음(자기완결)
    assert "<script src=" not in html, "외부 스크립트를 불러오면 안 됨"


def test_demo_core_reproduces_the_where_to_probe_ordering():
    """node 로 코어를 돌려 데모의 핵심 주장이 성립하는지 확인."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 없음 — 브라우저 데모 코어 검증 생략")
    r = subprocess.run([node, str(CHECK)], capture_output=True, text=True,
                       encoding="utf-8", timeout=300)
    assert r.returncode == 0, f"헤드리스 검증 실패:\n{r.stdout}\n{r.stderr}"
    assert "모두 통과" in r.stdout
