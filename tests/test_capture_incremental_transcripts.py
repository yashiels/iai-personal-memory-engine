from __future__ import annotations

import json
import platform
import re
from pathlib import Path

import pytest
from test_capture_turn_deferred import _build_args
from test_turn_capture_offset_advance import _extract_py_script, _run_py_script

from iai_mcp.capture import reconcile_capture_delta
from iai_mcp.cli import cmd_capture_turn_deferred

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows", reason="POSIX hook carriers"
)

FIXTURES = Path(__file__).parent / "fixtures" / "capture_incremental"


def _events(home, session_id):
    path = home / ".iai-mcp" / ".deferred-captures" / f"{session_id}.live.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()[1:]]


def _offset(home, session_id):
    return int(
        (home / ".iai-mcp" / ".capture-state" / f"{session_id}.offset").read_text()
    )


def _capture(carrier, home, transcript, session_id, cap=200):
    if carrier == "cli":
        assert cmd_capture_turn_deferred(_build_args(session_id, transcript, cap)) == 0
    else:
        script = _extract_py_script()
        if carrier == "plugin":
            hook = (
                Path(__file__).parent.parent
                / "plugin"
                / "hooks"
                / "iai-mcp-turn-capture.sh"
            )
            script = re.search(
                r"PY_SCRIPT='(.*?)'\s*\n", hook.read_text(), re.DOTALL
            ).group(1)
        assert (
            _run_py_script(
                script.replace("MAX_TURNS = 100_000", f"MAX_TURNS = {cap}"),
                session_id,
                transcript,
                home,
            )[0]
            == 0
        )


@pytest.mark.parametrize("carrier", ["cli", "hook", "plugin"])
@pytest.mark.parametrize(
    "fixture,expected", [("claude-resumed.jsonl", 410), ("codex-acp.jsonl", 4)]
)
def test_incremental_fixture_capture_is_lossless_and_repeatable(
    tmp_path,
    monkeypatch,
    caplog,
    carrier,
    fixture,
    expected,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    lines = (FIXTURES / fixture).read_text().splitlines(keepends=True)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(lines[:3]))
    _capture(carrier, tmp_path, transcript, "fixture")
    first = _events(tmp_path, "fixture")
    _capture(carrier, tmp_path, transcript, "fixture")
    assert _events(tmp_path, "fixture") == first
    spool = tmp_path / ".iai-mcp" / ".deferred-captures" / "fixture.live.jsonl"
    if spool.exists():
        rotated = spool.with_name("fixture.rotated.jsonl")
        spool.rename(rotated)
        rotated.unlink()
    with transcript.open("a") as fh:
        fh.write("".join(lines[3:]))
    for _ in range(4):
        _capture(carrier, tmp_path, transcript, "fixture")
    assert _offset(tmp_path, "fixture") == len(lines)
    events = first + _events(tmp_path, "fixture")
    assert len(events) == expected
    identities = [(ev["role"], ev.get("source_uuid") or ev["ts"]) for ev in events]
    assert len(set(identities)) == expected
    _capture(carrier, tmp_path, transcript, "fixture")
    assert first + _events(tmp_path, "fixture") == events
    assert "shortfall" not in caplog.text


@pytest.mark.parametrize("carrier", ["cli", "hook", "plugin"])
@pytest.mark.parametrize("valid_json", [False, True])
def test_unfinished_tail_is_retried_once_completed(
    tmp_path, monkeypatch, carrier, valid_json
):
    monkeypatch.setenv("HOME", str(tmp_path))
    lines = (FIXTURES / "codex-acp.jsonl").read_text().splitlines(keepends=True)
    transcript = tmp_path / "transcript.jsonl"
    tail = lines[-1]
    split = len(tail) - 1 if valid_json else len(tail) // 2
    transcript.write_text("".join(lines[:-1]) + tail[:split])
    _capture(carrier, tmp_path, transcript, "tail")
    assert _offset(tmp_path, "tail") == len(lines) - (not valid_json)
    assert len(_events(tmp_path, "tail")) == 3 + valid_json
    with transcript.open("a") as fh:
        fh.write(tail[split:])
    _capture(carrier, tmp_path, transcript, "tail")
    assert _offset(tmp_path, "tail") == len(lines)
    assert len(_events(tmp_path, "tail")) == 4
    _capture(carrier, tmp_path, transcript, "tail")
    assert len(_events(tmp_path, "tail")) == 4


def test_delta_flags_a_dropped_turn_and_malformed_line():
    lines = (FIXTURES / "codex-acp.jsonl").read_text().splitlines(keepends=True)
    result = reconcile_capture_delta(lines, 3)
    assert result["expected_turns"] == 4
    assert result["missing_turns"] == 1
    assert result["status"] == "partial"
    result = reconcile_capture_delta(lines + ["{bad json\n"], 4)
    assert result["parse_failed"] == result["missing_turns"] == 1


def test_capped_delta_does_not_report_unconsumed_backlog(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text((FIXTURES / "claude-resumed.jsonl").read_text())
    _capture("cli", tmp_path, transcript, "capped", cap=3)
    assert len(_events(tmp_path, "capped")) == 3
    assert _offset(tmp_path, "capped") < len(transcript.read_text().splitlines())
    assert "shortfall" not in caplog.text


def test_delta_detects_parser_drop_in_cli(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp import capture

    original = capture._parse_transcript_obj
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text((FIXTURES / "codex-acp.jsonl").read_text())
    calls = 0

    def drop_once(obj):
        nonlocal calls
        calls += 1
        return None if calls == 3 else original(obj)

    monkeypatch.setattr(capture, "_parse_transcript_obj", drop_once)
    _capture("cli", tmp_path, transcript, "drop")
    assert "expected=4 captured=3 missing=1" in caplog.text


@pytest.mark.parametrize("carrier", ["cli", "hook", "plugin"])
def test_single_complete_tail_survives_later_newline(tmp_path, monkeypatch, carrier):
    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = tmp_path / "transcript.jsonl"
    line = json.dumps(
        {
            "type": "assistant",
            "uuid": "single-tail",
            "message": {"content": "A complete response without a final newline."},
        }
    )
    transcript.write_text(line)
    _capture(carrier, tmp_path, transcript, "single")
    events = _events(tmp_path, "single")
    assert len(events) == _offset(tmp_path, "single") == 1
    assert reconcile_capture_delta([line], 1)["status"] == "ok"
    _capture(carrier, tmp_path, transcript, "single")
    assert _events(tmp_path, "single") == events
    with transcript.open("a") as fh:
        fh.write("\n")
    _capture(carrier, tmp_path, transcript, "single")
    assert _events(tmp_path, "single") == events
    assert _offset(tmp_path, "single") == 1


@pytest.mark.parametrize("carrier", ["cli", "hook", "plugin"])
@pytest.mark.parametrize("tail", ["{", "[]", "null", '"text"', "42"])
def test_incomplete_or_non_object_tail_waits(tmp_path, monkeypatch, carrier, tail):
    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(tail)
    _capture(carrier, tmp_path, transcript, "unsupported")
    assert _offset(tmp_path, "unsupported") == 0
    assert _events(tmp_path, "unsupported") == []
    result = reconcile_capture_delta([tail], 0)
    assert result["status"] == "ok"
    assert result["expected_turns"] == result["parse_failed"] == 0


def _tool_line(format, name):
    if format == "codex":
        obj = {
            "type": "response_item",
            "payload": {"type": "function_call", "name": name},
        }
    else:
        obj = {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": name}]},
        }
    return json.dumps(obj) + "\n"


def _message_line(format, role, text, identity):
    if format == "codex":
        obj = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "id": identity,
                "content": [{"type": "output_text", "text": text}],
            },
        }
    else:
        obj = {"type": role, "uuid": identity, "message": {"content": text}}
    return json.dumps(obj) + "\n"


@pytest.mark.parametrize("carrier", ["cli", "hook", "plugin"])
@pytest.mark.parametrize("format", ["claude", "codex"])
@pytest.mark.parametrize("boundary", ["partial_tail", "max_emit"])
def test_tool_trailers_survive_incremental_boundaries(
    tmp_path, monkeypatch, caplog, carrier, format, boundary
):
    monkeypatch.setenv("HOME", str(tmp_path))
    lines = [
        _message_line(format, "user", "Please inspect the source files.", "user"),
        _tool_line(format, "Read"),
    ]
    if boundary == "max_emit":
        lines += [
            _message_line(format, "assistant", "ok", "stub"),
            _tool_line(format, "Search"),
        ]
    lines.append(
        _message_line(
            format, "assistant", "The source files have been inspected.", "response"
        )
    )
    transcript = tmp_path / "transcript.jsonl"
    full_text = "".join(lines)
    split = (
        len(full_text)
        if boundary == "max_emit"
        else len(full_text) - len(lines[-1]) // 2
    )
    transcript.write_text(full_text[:split])
    state_path = tmp_path / ".iai-mcp" / ".capture-state" / "tools.turnstate.json"

    def capture_delta():
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        before = _events(tmp_path, "tools")
        _capture(carrier, tmp_path, transcript, "tools", cap=2)
        current = json.loads(state_path.read_text())
        after = _events(tmp_path, "tools")
        consumed = transcript.read_text().splitlines(keepends=True)[
            previous.get("offset", 0) : current["offset"]
        ]
        result = reconcile_capture_delta(
            consumed, len(after) - len(before), pending_tools=previous.get("pending")
        )
        assert result["status"] == "ok"
        assert result["missing_turns"] == result["parse_failed"] == 0
        return current, after

    state, first = capture_delta()
    assert state["pending"] == ["Read"]
    assert state["offset"] == (3 if boundary == "max_emit" else 2)
    if boundary == "partial_tail":
        _, repeated = capture_delta()
        assert repeated == first
        with transcript.open("a") as fh:
            fh.write(full_text[split:])
    state, events = capture_delta()
    assert state["pending"] == []
    assert state["offset"] == len(lines)
    assert len(events) == (3 if boundary == "max_emit" else 2)
    assert len({event["source_uuid"] for event in events}) == len(events)
    trailer = "[tools: Read, Search]" if boundary == "max_emit" else "[tools: Read]"
    assert events[-1]["text"].endswith(trailer)
    assert events[-1]["text"].count(trailer) == 1
    _, repeated = capture_delta()
    assert repeated == events
    assert "shortfall" not in caplog.text
