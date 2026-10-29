from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from chatgpt_to_handoff import (
    HandoffError,
    list_conversations,
    main,
    write_handoff,
)

FIXTURES = Path(__file__).parent / "fixtures"

_AGENTS_INSTRUCTION = (
    "This file is a handoff from a ChatGPT conversation. Follow the "
    "requirements and decisions below. Treat open questions as unresolved."
)
_NONE_EXTRACTED = "_None extracted._"


def test_active_branch_excludes_sibling(tmp_path: Path) -> None:
    source = FIXTURES / "active_branch.json"
    out = tmp_path / "HANDOFF.md"
    write_handoff(source, output=out)
    text = out.read_text(encoding="utf-8")
    assert "ACTIVE_USER_TURN" in text
    assert "ACTIVE_ASSISTANT_TURN" in text
    assert "SIBLING_EDIT_MUST_NOT_APPEAR" not in text


def test_hidden_system_omitted(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    write_handoff(FIXTURES / "hidden_system.json", output=out)
    text = out.read_text(encoding="utf-8")
    assert "SECRET_SYSTEM_PROMPT" not in text
    assert "Visible user line" in text


def test_multimodal_image_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    write_handoff(FIXTURES / "multimodal.json", output=out)
    text = out.read_text(encoding="utf-8")
    assert "[image]" in text


def test_main_two_conversations_no_selector(tmp_path: Path, capsys) -> None:
    out = tmp_path / "HANDOFF.md"
    code = main([str(FIXTURES / "two_conversations.json"), "-o", str(out)])
    assert code == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "Alpha chat" in err
    assert "Beta chat" in err


def test_main_select_by_id(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    code = main(
        [
            str(FIXTURES / "two_conversations.json"),
            "--id",
            "bbbb0000-0000-4000-8000-000000000002",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    assert "beta" in out.read_text(encoding="utf-8")


def test_main_duplicate_title_returns_one(tmp_path: Path, capsys) -> None:
    out = tmp_path / "HANDOFF.md"
    code = main(
        [str(FIXTURES / "duplicate_titles.json"), "--title", "Same title", "-o", str(out)]
    )
    assert code == 1
    assert not out.exists()
    assert capsys.readouterr().err


def test_single_conversation_no_selector(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    assert main([str(FIXTURES / "single_conversation.json"), "-o", str(out)]) == 0
    assert out.exists()


def test_zip_with_conversations_json(tmp_path: Path) -> None:
    zpath = tmp_path / "export.zip"
    data = (FIXTURES / "single_conversation.json").read_bytes()
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("conversations.json", data)
    out = tmp_path / "HANDOFF.md"
    write_handoff(zpath, output=out)
    assert "solo" in out.read_text(encoding="utf-8")


def test_conversations_shard_filename(tmp_path: Path) -> None:
    shard = tmp_path / "conversations-001.json"
    shard.write_bytes((FIXTURES / "single_conversation.json").read_bytes())
    out = tmp_path / "HANDOFF.md"
    write_handoff(shard, output=out)
    assert out.exists()


def test_nested_zip(tmp_path: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr(
            "conversations.json",
            (FIXTURES / "single_conversation.json").read_bytes(),
        )
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("nested.zip", inner.getvalue())
    out = tmp_path / "HANDOFF.md"
    write_handoff(outer, output=out)
    assert "solo" in out.read_text(encoding="utf-8")


def test_agents_md_heading(tmp_path: Path) -> None:
    out = tmp_path / "AGENTS.md"
    write_handoff(FIXTURES / "single_conversation.json", agents_md=True, output=out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# AGENTS.md\n")
    assert _AGENTS_INSTRUCTION in text


def test_code_fence_not_open_question(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    write_handoff(FIXTURES / "code_and_question.json", output=out)
    text = out.read_text(encoding="utf-8")
    code_section = text.split("## Code", 1)[1].split("## Transcript", 1)[0]
    assert "```python" in code_section
    assert "print('hi')" in code_section
    questions = text.split("## Open questions", 1)[1].split("## Code", 1)[0]
    assert "What is the status?" in questions
    assert "print('hi')" not in questions


def test_visible_question_line(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    write_handoff(FIXTURES / "code_and_question.json", output=out)
    questions = (
        out.read_text(encoding="utf-8").split("## Open questions", 1)[1].split("## Code", 1)[0]
    )
    assert "- What is the status?" in questions


def test_deterministic_newlines(tmp_path: Path) -> None:
    source = FIXTURES / "single_conversation.json"
    out = tmp_path / "HANDOFF.md"
    write_handoff(source, output=out)
    first = out.read_bytes()
    write_handoff(source, output=out)
    second = out.read_bytes()
    assert first == second
    assert b"\r" not in first


def test_bad_json_no_output(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    assert main([str(FIXTURES / "bad.json"), "-o", str(out)]) == 1
    assert not out.exists()


def test_share_url_patched(tmp_path: Path) -> None:
    body = (FIXTURES / "share_response.json").read_bytes()

    class FakeResp:
        def getcode(self) -> int:
            return 200

        def read(self) -> bytes:
            return body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    out = tmp_path / "HANDOFF.md"
    url = "https://chatgpt.com/share/69b1c492-1540-8006-aa29-ee2e0a831385"
    with patch("chatgpt_to_handoff.urllib_request.urlopen", return_value=FakeResp()):
        write_handoff(url, output=out)
    assert "SHARE_FIXTURE_USER_VISIBLE" in out.read_text(encoding="utf-8")


def test_list_conversations(tmp_path: Path, capsys) -> None:
    handoff = tmp_path / "HANDOFF.md"
    rows = list_conversations(FIXTURES / "two_conversations.json")
    assert len(rows) == 2
    assert rows[0].title == "Alpha chat"
    assert rows[0].update_time == "2023-11-14T22:13:20Z"
    code = main([str(FIXTURES / "two_conversations.json"), "--list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Alpha chat\taaaa0000-0000-4000-8000-000000000001\t2023-11-14T22:13:20Z" in out
    assert not handoff.exists()


def test_list_with_id_usage_error() -> None:
    assert main([str(FIXTURES / "two_conversations.json"), "--list", "--id", "x"]) == 2


def test_decision_section(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    write_handoff(FIXTURES / "decision.json", output=out)
    decisions = (
        out.read_text(encoding="utf-8").split("## Decisions", 1)[1].split("## Requirements", 1)[0]
    )
    assert "we will" in decisions


def test_empty_sections_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    write_handoff(FIXTURES / "empty_sections.json", output=out)
    text = out.read_text(encoding="utf-8")
    assert text.count(_NONE_EXTRACTED) >= 3


def test_write_handoff_raises_on_ambiguous() -> None:
    with pytest.raises(HandoffError) as exc:
        write_handoff(FIXTURES / "two_conversations.json")
    assert exc.value.summaries


def test_existing_uuid_filename_is_not_a_share_label(tmp_path: Path) -> None:
    source = tmp_path / "69b1c492-1540-8006-aa29-ee2e0a831385"
    source.write_bytes((FIXTURES / "single_conversation.json").read_bytes())
    out = tmp_path / "HANDOFF.md"
    write_handoff(source, output=out)
    text = out.read_text(encoding="utf-8")
    assert "- Source: 69b1c492-1540-8006-aa29-ee2e0a831385" in text
    assert "https://chatgpt.com/share/" not in text


def test_failed_run_leaves_existing_handoff(tmp_path: Path) -> None:
    out = tmp_path / "HANDOFF.md"
    out.write_text("keep me\n", encoding="utf-8")
    code = main([str(FIXTURES / "bad.json"), "-o", str(out)])
    assert code == 1
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_readme_contains_heuristic_patterns() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "(?i)^(goal|context|objective)\\b" in readme
    assert (
        "(?i)\\b(decided|decision|we will|we'll|let's go with|chosen approach)\\b" in readme
    )
    assert "(?i)\\b(must|should|requirement)\\b" in readme
    assert "^(?:[-*+]|\\d+[.)])\\s+(.+)$" in readme
