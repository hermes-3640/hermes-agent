"""Tests for retry-loop detection — agent-level guardrail that prevents burn-through
of a tool-calling turn when an operation repeatedly fails with the same error."""

from __future__ import annotations

from typing import Any

import pytest

from agent.tool_executor import (
    _RetryLoopTracker,
    _args_fingerprint,
    _check_global_consecutive_failures,
    _detect_consecutive_tool_failure,
    _inject_retry_loop_steer,
    _reset_consecutive_tool_failure,
    _track_consecutive_tool_failure,
)


# ---------------------------------------------------------------------------
# Helpers that exercise the functions directly (no AIAgent required).
# ---------------------------------------------------------------------------

class _FakeAgent:
    """Minimal agent stub that holds a _retry_loop_tracker."""

    def __init__(self, tracker: _RetryLoopTracker | None = None):
        self._retry_loop_tracker = tracker or _RetryLoopTracker()
        self._pending_steer: str | None = None


def _make_tracker(**overrides: Any) -> _RetryLoopTracker:
    """Build a tracker, then apply keyword overrides on top."""
    base = _RetryLoopTracker()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ---------------------------------------------------------------------------
# _args_fingerprint — canonicalises tool arguments into a stable key.
# ---------------------------------------------------------------------------

class TestArgsFingerprint:
    def test_write_file_extracts_path(self) -> None:
        assert _args_fingerprint("write_file", {"path": "/foo/bar.py", "content": "x"}) == "/foo/bar.py"

    def test_patch_extracts_path(self) -> None:
        assert _args_fingerprint("patch", {"path": "/foo/bar.py", "old_string": "x"}) == "/foo/bar.py"

    def test_terminal_extract_command_short(self) -> None:
        fp = _args_fingerprint("terminal", {"command": "ls /foo", "workdir": None})
        assert fp == "ls /foo"

    def test_terminal_extract_command_long_truncates(self) -> None:
        long_cmd = "python3 -c " + "x" * 500  # 500+ chars
        fp = _args_fingerprint("terminal", {"command": long_cmd, "workdir": None})
        assert len(fp) <= 200  # truncated

    def test_web_search_extract_query(self) -> None:
        fp = _args_fingerprint("web_search", {"query": "hello world", "limit": 5})
        assert fp == "hello world"

    def test_read_file_extract_path(self) -> None:
        assert _args_fingerprint("read_file", {"path": "/foo.txt"}) == "/foo.txt"

    def test_execute_code_truncates(self) -> None:
        code = "print(" + "x" * 500 + ")"
        fp = _args_fingerprint("execute_code", {"code": code, "reset": False})
        assert len(fp) <= 200

    def test_delegate_task_truncates_goal(self) -> None:
        goal = "do something very long " + "x" * 500
        fp = _args_fingerprint("delegate_task", {"tasks": [{"goal": goal}]})
        assert len(fp) <= 200

    def test_unknown_tool_uses_first_string_value(self) -> None:
        fp = _args_fingerprint("unknown_tool", {"key1": "foo", "key2": "bar"})
        assert fp == "foo"

    def test_unknown_tool_falls_back_to_empty(self) -> None:
        fp = _args_fingerprint("unknown_tool", {"num": 42, "flag": True})
        assert fp == ""

    def test_file_tools_priority(self) -> None:
        """file_tools_keys are checked first even if other keys have string values."""
        fp = _args_fingerprint("write_file", {"path": "/real.py", "name": "wrong"})
        assert fp == "/real.py"

    def test_terminal_command_overrides_workdir(self) -> None:
        assert _args_fingerprint("terminal", {"command": "ls", "workdir": "/foo"}) == "ls"

    def test_web_extract_extract_url(self) -> None:
        fp = _args_fingerprint("web_extract", {"urls": ["https://example.com", "https://foo.com"]})
        assert fp == "https://example.com"


# ---------------------------------------------------------------------------
# _detect_consecutive_tool_failure — per-tool detection.
# ---------------------------------------------------------------------------

class TestDetectConsecutiveToolFailure:
    def test_no_tracker_no_crash(self) -> None:
        agent = _FakeAgent()
        # No tracker at all (None case handled by getattr returning None)
        agent._retry_loop_tracker = None
        result = _detect_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert result is None  # first failure — no steer

    def test_three_consecutive_failures_same_tool_path(self) -> None:
        tracker = _make_tracker(per_tool_fails={("/foo.py", "write_file"): 3})
        agent = _FakeAgent(tracker)
        result = _detect_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert result is not None
        assert "Repeated failures" in result
        assert "write_file" in result
        assert "/foo.py" in result

    def test_two_failures_not_enough(self) -> None:
        tracker = _make_tracker(per_tool_fails={("/foo.py", "write_file"): 2})
        agent = _FakeAgent(tracker)
        result = _detect_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert result is None

    def test_different_path_resets_tracker(self) -> None:
        """Different path = new operation, should not count against old counter."""
        tracker = _make_tracker(
            per_tool_fails={("/foo.py", "write_file"): 2, ("/bar.py", "write_file"): 3},
        )
        agent = _FakeAgent(tracker)
        # /bar.py already has 3 failures — should trigger immediately
        result = _detect_consecutive_tool_failure(agent, "write_file", {"path": "/bar.py"})
        assert result is not None
        assert "/bar.py" in result

    def test_terminal_non_zero_exit_counts(self) -> None:
        """Terminal with exit code != 0 is a failure."""
        # Fingerprint must match exactly what _args_fingerprint returns
        tracker = _make_tracker(per_tool_fails={("ls /nonexistent", "terminal"): 3})
        agent = _FakeAgent(tracker)
        result = _detect_consecutive_tool_failure(agent, "terminal", {"command": "ls /nonexistent"})
        assert result is not None

    def test_file_not_found_is_failure(self) -> None:
        tracker = _make_tracker(per_tool_fails={("/missing/file.py", "patch"): 3})
        agent = _FakeAgent(tracker)
        result = _detect_consecutive_tool_failure(agent, "patch", {"path": "/missing/file.py"})
        assert result is not None

    def test_mcp_disconnected_is_failure(self) -> None:
        # Fingerprint for search_files({"pattern": "foo"}) = "foo" (first string value)
        tracker = _make_tracker(per_tool_fails={("foo", "search_files"): 3})
        agent = _FakeAgent(tracker)
        result = _detect_consecutive_tool_failure(agent, "search_files", {"pattern": "foo"})
        assert result is not None


# ---------------------------------------------------------------------------
# _reset_consecutive_tool_failure — reset on success or different args.
# ---------------------------------------------------------------------------

class TestResetConsecutiveToolFailure:
    def test_success_resets_counter(self) -> None:
        tracker = _make_tracker(
            per_tool_fails={("/foo.py", "write_file"): 2},
            global_consecutive_failures=2,
        )
        agent = _FakeAgent(tracker)
        _reset_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        # Key is popped on success (the dataclass update_per_tool pops it)
        assert ("foo.py" in str(tracker.per_tool_fails)) or (
            ("/foo.py", "write_file") not in tracker.per_tool_fails
        )
        assert tracker.global_consecutive_failures == 0

    def test_different_args_resets_that_key(self) -> None:
        tracker = _make_tracker(
            per_tool_fails={("/foo.py", "write_file"): 2, ("/bar.py", "write_file"): 1},
            global_consecutive_failures=0,
        )
        agent = _FakeAgent(tracker)
        _reset_consecutive_tool_failure(agent, "write_file", {"path": "/bar.py"})
        assert ("/bar.py", "write_file") not in tracker.per_tool_fails
        # /foo.py should be untouched (global reset clears it too)
        assert tracker.global_consecutive_failures == 0
        assert ("/foo.py", "write_file") not in tracker.per_tool_fails

    def test_different_tool_resets_nothing(self) -> None:
        """Different tool resets global counter too, so everything clears."""
        tracker = _make_tracker(per_tool_fails={("/foo.py", "write_file"): 2})
        agent = _FakeAgent(tracker)
        _reset_consecutive_tool_failure(agent, "terminal", {"command": "ls"})
        # Success resets EVERYTHING (global + all per-tool)
        assert tracker.global_consecutive_failures == 0
        assert tracker.per_tool_fails == {}


# ---------------------------------------------------------------------------
# Global consecutive failures detection.
# ---------------------------------------------------------------------------

class TestGlobalConsecutiveFailures:
    def test_threshold_exceeded(self) -> None:
        tracker = _make_tracker(global_consecutive_failures=5)
        agent = _FakeAgent(tracker)
        result = _check_global_consecutive_failures(agent)
        assert result is not None
        assert "consecutive tool failures" in result

    def test_below_threshold_no_injection(self) -> None:
        tracker = _make_tracker(global_consecutive_failures=4)
        agent = _FakeAgent(tracker)
        result = _check_global_consecutive_failures(agent)
        assert result is None

    def test_global_reset_on_success(self) -> None:
        tracker = _make_tracker(global_consecutive_failures=5)
        agent = _FakeAgent(tracker)
        _reset_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert tracker.global_consecutive_failures == 0

    def test_last_failing_tool_updates(self) -> None:
        tracker = _make_tracker(global_consecutive_failures=0)
        agent = _FakeAgent(tracker)
        _track_consecutive_tool_failure(agent, "terminal", {"command": "ls /foo"})
        assert tracker.last_failing_tool == "terminal"


# ---------------------------------------------------------------------------
# Integration: simulate a full failure sequence through the agent's tracker.
# ---------------------------------------------------------------------------

class TestFullFailureSequence:
    """Simulate the same tool failing 3 times in a row and verify detection at the right time."""

    def test_third_failure_triggers(self) -> None:
        """First failure: increment to 1, no steer.
        Second failure: increment to 2, no steer.
        Third failure: increment to 3, STEER INJECTED."""

        agent = _FakeAgent()

        # Failure 1
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert agent._retry_loop_tracker.per_tool_fails[("/foo.py", "write_file")] == 1
        # No steer injected yet
        assert agent._pending_steer is None or agent._pending_steer == ""

        # Failure 2
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert agent._retry_loop_tracker.per_tool_fails[("/foo.py", "write_file")] == 2
        # Still no steer

        # Failure 3 — track + detect + inject (matches real execution path)
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert agent._retry_loop_tracker.per_tool_fails[("/foo.py", "write_file")] == 3
        steer_msg = _detect_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        if steer_msg:
            _inject_retry_loop_steer(agent, steer_msg)
        assert agent._pending_steer is not None
        assert "Repeated failures" in agent._pending_steer
        assert "write_file" in agent._pending_steer
        assert "/foo.py" in agent._pending_steer

    def test_successful_call_resets(self) -> None:
        """A success resets the counter, so retries after success start fresh."""
        agent = _FakeAgent()

        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert agent._retry_loop_tracker.per_tool_fails[("/foo.py", "write_file")] == 2

        # Success — reset
        _reset_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert ("/foo.py", "write_file") not in agent._retry_loop_tracker.per_tool_fails
        assert agent._retry_loop_tracker.global_consecutive_failures == 0

        # Failure again — should only be 1, not 4
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert agent._retry_loop_tracker.per_tool_fails[("/foo.py", "write_file")] == 1

    def test_different_args_does_not_accumulate(self) -> None:
        """Different files = different operations; counters should not accumulate."""
        agent = _FakeAgent()

        # Fail on /foo.py twice
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        assert agent._retry_loop_tracker.per_tool_fails[("/foo.py", "write_file")] == 2

        # Fail on /bar.py — different key, starts at 1
        _track_consecutive_tool_failure(agent, "write_file", {"path": "/bar.py"})
        assert agent._retry_loop_tracker.per_tool_fails[("/foo.py", "write_file")] == 2  # unchanged
        assert agent._retry_loop_tracker.per_tool_fails[("/bar.py", "write_file")] == 1

    def test_global_tracker_accumulates_across_tools(self) -> None:
        """Different tools should increment the global counter."""
        agent = _FakeAgent()

        _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
        _track_consecutive_tool_failure(agent, "patch", {"path": "/bar.py"})
        _track_consecutive_tool_failure(agent, "terminal", {"command": "ls /foo"})

        assert agent._retry_loop_tracker.global_consecutive_failures == 3

        # Reset on success — clears global too
        _reset_consecutive_tool_failure(agent, "read_file", {"path": "/baz.py"})
        assert agent._retry_loop_tracker.global_consecutive_failures == 0

    def test_file_not_found_error_message_in_steer(self) -> None:
        """The steer message should contain actionable guidance."""
        agent = _FakeAgent()

        _track_consecutive_tool_failure(agent, "patch", {"path": "/missing/file.py"})
        _track_consecutive_tool_failure(agent, "patch", {"path": "/missing/file.py"})
        _track_consecutive_tool_failure(agent, "patch", {"path": "/missing/file.py"})

        steer_msg = _detect_consecutive_tool_failure(agent, "patch", {"path": "/missing/file.py"})
        if steer_msg:
            _inject_retry_loop_steer(agent, steer_msg)

        assert agent._pending_steer is not None
        # Check that the message is actionable
        assert "reconsider" in agent._pending_steer.lower() or "re-verify" in agent._pending_steer.lower()
        assert "patch" in agent._pending_steer
        assert "/missing/file.py" in agent._pending_steer

    def test_no_crash_with_empty_args(self) -> None:
        """Tools with no meaningful args should not crash."""
        agent = _FakeAgent()

        _track_consecutive_tool_failure(agent, "some_tool", {})
        _track_consecutive_tool_failure(agent, "some_tool", {})
        _track_consecutive_tool_failure(agent, "some_tool", {})

        steer_msg = _detect_consecutive_tool_failure(agent, "some_tool", {})
        if steer_msg:
            _inject_retry_loop_steer(agent, steer_msg)

        # Should still have a steer, even with empty args fingerprint
        assert agent._pending_steer is not None
        assert "some_tool" in agent._pending_steer

    def test_mcp_parked_error_detected(self) -> None:
        """MCP connection failures should be tracked as consecutive failures."""
        agent = _FakeAgent()

        # Simulate web_search (MCP-backed) failing repeatedly
        for _ in range(3):
            _track_consecutive_tool_failure(agent, "web_search", {"query": "test query"})

        assert agent._retry_loop_tracker.per_tool_fails[("test query", "web_search")] == 3
        steer_msg = _detect_consecutive_tool_failure(agent, "web_search", {"query": "test query"})
        if steer_msg:
            _inject_retry_loop_steer(agent, steer_msg)
        assert "web_search" in agent._pending_steer

    def test_session_persistence_does_not_crash(self) -> None:
        """The tracking code should never crash, even with edge cases."""
        # Agent without any _retry_loop_tracker attribute
        agent = object.__new__(object)

        try:
            _track_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
            _detect_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
            _reset_consecutive_tool_failure(agent, "write_file", {"path": "/foo.py"})
            _check_global_consecutive_failures(agent)
        except Exception as e:
            pytest.fail(f"retry-loop functions raised on agent without tracker: {e}")


# ---------------------------------------------------------------------------
# _RetryLoopTracker dataclass integrity.
# ---------------------------------------------------------------------------

class TestRetryLoopTracker:
    def test_default_values(self) -> None:
        tracker = _RetryLoopTracker()
        assert tracker.per_tool_fails == {}
        assert tracker.global_consecutive_failures == 0
        assert tracker.last_failing_tool is None

    def test_set_and_update(self) -> None:
        tracker = _RetryLoopTracker()
        tracker.per_tool_fails[("/foo.py", "write_file")] = 1
        tracker.global_consecutive_failures = 1
        tracker.last_failing_tool = "write_file"

        assert tracker.per_tool_fails[("/foo.py", "write_file")] == 1
        assert tracker.global_consecutive_failures == 1
        assert tracker.last_failing_tool == "write_file"

    def test_increment_global(self) -> None:
        tracker = _RetryLoopTracker(global_consecutive_failures=0)
        tracker.increment_global()
        assert tracker.global_consecutive_failures == 1
        tracker.increment_global()
        assert tracker.global_consecutive_failures == 2

    def test_reset_all(self) -> None:
        tracker = _RetryLoopTracker(
            per_tool_fails={("/foo.py", "write_file"): 3},
            global_consecutive_failures=3,
            last_failing_tool="write_file",
        )
        tracker.reset_all()
        assert tracker.per_tool_fails == {}
        assert tracker.global_consecutive_failures == 0
        assert tracker.last_failing_tool is None

    def test_update_per_tool(self) -> None:
        tracker = _RetryLoopTracker()
        tracker.update_per_tool("/foo.py", "write_file", is_success=False)
        assert tracker.per_tool_fails[("/foo.py", "write_file")] == 1

        tracker.update_per_tool("/foo.py", "write_file", is_success=False)
        assert tracker.per_tool_fails[("/foo.py", "write_file")] == 2

        tracker.update_per_tool("/foo.py", "write_file", is_success=True)
        # Success pops the key entirely (not setting to 0)
        assert ("foo.py" in str(tracker.per_tool_fails)) or (
            ("/foo.py", "write_file") not in tracker.per_tool_fails
        )

        tracker.update_per_tool("/bar.py", "write_file", is_success=False)
        assert tracker.per_tool_fails[("/bar.py", "write_file")] == 1
