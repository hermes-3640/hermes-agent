"""Behavior tests for session lease deadlock detection (t_a1055472).

Covers:
- Stale lease detection: if a lease holder is idle beyond TTL, it's force-released
- Concurrent waiter cap: rejects new waiters when the queue is too long
- Idle timeout configurable per registry
- Stale grace period prevents false positives on slow tool calls
- Cleanup of stale leases on acquire (not just a background loop)
- DB-level stale lease sweep (sweep_stale_turn_leases)
- Logging of stale detection events
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.turn_lease import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_MAX_WAITERS,
    STALE_GRACE_SECS,
    SessionTurnLeaseRegistry,
    TurnLeaseStaleError,
    TurnLeaseTimeoutError,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Stale lease detection
# ---------------------------------------------------------------------------


def test_stale_lease_is_force_released_on_acquire():
    """When a lease is held but idle beyond TTL + grace, force-release lets the new caller in."""

    async def scenario():
        idle_timeout = 0.05  # 50ms for fast test
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)
        token = await registry.acquire(
            "sess-stale", owner_key="holder-a", generation=1, timeout=1
        )
        assert token is not None

        # Age the lease past TTL + grace.
        lease = registry._leases["sess-stale"]
        grace = STALE_GRACE_SECS  # default 60s, but idle_timeout is tiny so
        # we just need idle > idle_timeout + grace. Since grace=60s is fixed,
        # we age way past it.
        lease.last_used = time.time() - (idle_timeout + grace + 10.0)
        lease.acquired_at = lease.last_used

        # A new acquire should detect the stale lease and force-release it.
        new_token = await registry.acquire(
            "sess-stale", owner_key="holder-b", generation=2, timeout=1
        )
        assert new_token is not None
        assert new_token.owner_key == "holder-b"
        assert new_token.generation == 2
        # The old holder should be marked released.
        assert token.released is True
        # The old holder should be cleared.
        assert lease.holder is new_token

        registry.release(new_token)

    _run(scenario())


def test_stale_grace_prevents_false_positive_on_slow_tool():
    """A lease that is idle for less than TTL + grace is NOT force-released."""

    async def scenario():
        idle_timeout = 10.0  # 10s timeout
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)
        token = await registry.acquire(
            "sess-safe", owner_key="holder-a", generation=1, timeout=1
        )
        assert token is not None

        # Age the lease past TTL but WITHIN grace. Grace is 60s.
        lease = registry._leases["sess-safe"]
        lease.last_used = time.time() - (idle_timeout + 0.5)  # 10.5s old, grace=60s

        # The acquire cleanup should NOT force-release because we're within grace.
        # Since holder-a still holds it, the new caller will wait and timeout.
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(
                "sess-safe", owner_key="holder-b", generation=2, timeout=1
            )

        # The original holder is still valid.
        assert lease.holder is token

        registry.release(token)

    _run(scenario())


def test_stale_force_release_wakes_existing_waiters():
    """When a stale lease is force-released, existing waiters can re-acquire."""

    async def scenario():
        idle_timeout = 0.05  # 50ms for fast test
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)

        token_a = await registry.acquire(
            "sess-wake", owner_key="holder-a", generation=1, timeout=1
        )
        assert token_a is not None

        # Age the lease past TTL + grace.
        lease = registry._leases["sess-wake"]
        lease.last_used = time.time() - (idle_timeout + STALE_GRACE_SECS + 10.0)

        # Start a waiter — it will timeout waiting for holder-a,
        # but we'll force-release before that.
        waiter_task = asyncio.create_task(
            registry.acquire(
                "sess-wake", owner_key="holder-b", generation=2, timeout=5
            )
        )
        await asyncio.sleep(0.1)

        # Force-release via cleanup.
        registry._cleanup()

        # The waiter was woken by force-release. It will now acquire because
        # no one holds the lease anymore.
        waiter_token = await waiter_task
        assert waiter_token is not None
        assert waiter_token.owner_key == "holder-b"

        registry.release(waiter_token)
        # holder-a is already released; idempotent.
        registry.release(token_a)

    _run(scenario())


# ---------------------------------------------------------------------------
# Concurrent waiter cap
# ---------------------------------------------------------------------------


def test_waiter_cap_rejects_excessive_waiters():
    """When too many threads wait for one lease, new acquires fail immediately."""

    async def scenario():
        max_waiters = 2
        registry = SessionTurnLeaseRegistry(idle_timeout=DEFAULT_IDLE_TIMEOUT, max_waiters=max_waiters)

        # First holder holds the lease.
        holder = await registry.acquire(
            "sess-limited", owner_key="holder-a", generation=1, timeout=10
        )
        assert holder is not None

        # First waiter queues.
        waiter1_task = asyncio.create_task(
            registry.acquire("sess-limited", owner_key="waiter-a", generation=2, timeout=5)
        )
        await asyncio.sleep(0.05)
        assert not waiter1_task.done()

        # Second waiter queues.
        waiter2_task = asyncio.create_task(
            registry.acquire("sess-limited", owner_key="waiter-b", generation=3, timeout=5)
        )
        await asyncio.sleep(0.05)
        assert not waiter2_task.done()

        # Third waiter should be rejected (cap is 2).
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(
                "sess-limited", owner_key="waiter-c", generation=4, timeout=0.02
            )

        # Release holder to unblock the first waiter.
        registry.release(holder)

        # First waiter gets the lock.
        t1 = await waiter1_task
        assert t1 is not None

        # Release first waiter so second waiter can acquire.
        registry.release(t1)

        # Second waiter now acquires.
        t2 = await waiter2_task
        assert t2 is not None
        registry.release(t2)

    _run(scenario())


def test_waiter_cap_default_is_three():
    """Default max_waiters is 3."""
    assert DEFAULT_MAX_WAITERS == 3


# ---------------------------------------------------------------------------
# Idle timeout configuration
# ---------------------------------------------------------------------------


def test_idle_timeout_configurable():
    """Idle timeout is configurable per registry instance."""
    custom_timeout = 600.0  # 10 min
    registry = SessionTurnLeaseRegistry(idle_timeout=custom_timeout)
    assert registry._idle_timeout == custom_timeout

    # New leases inherit the timeout.
    token = asyncio.run(registry.acquire("test", owner_key="k", generation=1, timeout=1))
    assert registry._leases["test"].idle_timeout == custom_timeout
    registry.release(token)


# ---------------------------------------------------------------------------
# Cleanup during acquire
# ---------------------------------------------------------------------------


def test_cleanup_finds_and_forces_releases_stale_leases():
    """_cleanup() scans all leases and force-releases stale ones with holders."""

    async def scenario():
        idle_timeout = 0.05
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)

        # Create two leases, one stale and one fresh.
        token_a = await registry.acquire(
            "sess-clean-1", owner_key="holder-a", generation=1, timeout=1
        )
        token_b = await registry.acquire(
            "sess-clean-2", owner_key="holder-b", generation=1, timeout=1
        )

        # Age sess-clean-1 past TTL + grace.
        lease_a = registry._leases["sess-clean-1"]
        lease_a.last_used = time.time() - (idle_timeout + STALE_GRACE_SECS + 10.0)

        # sess-clean-2 is fresh.
        lease_b = registry._leases["sess-clean-2"]

        # Run cleanup.
        registry._cleanup()

        # sess-clean-1 should have been force-released.
        assert lease_a.detected_stale is True
        assert lease_a.holder is None
        assert token_a.released is True

        # sess-clean-2 should be untouched (still held).
        assert lease_b.holder is token_b
        assert lease_b.detected_stale is False

        registry.release(token_b)

    _run(scenario())


def test_cleanup_does_not_force_release_idle_leases():
    """_cleanup() does not force-release leases with no holder."""

    async def scenario():
        idle_timeout = 0.05
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)

        token = await registry.acquire(
            "sess-idle", owner_key="holder-a", generation=1, timeout=1
        )
        registry.release(token)  # release, making the lease idle

        # Age the idle lease past TTL + grace.
        lease = registry._leases["sess-idle"]
        lease.last_used = time.time() - (idle_timeout + STALE_GRACE_SECS + 10.0)

        # Cleanup should NOT force-release (no holder to release).
        registry._cleanup()

        # The lease should still exist but be idle (no holder).
        assert lease.holder is None
        assert lease.detected_stale is False  # no force-release happened

    _run(scenario())


def test_cleanup_on_acquire_before_contended_acquire():
    """Stale cleanup runs before the lock check on acquire."""

    async def scenario():
        idle_timeout = 0.05
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)

        token_a = await registry.acquire(
            "sess-preflight", owner_key="holder-a", generation=1, timeout=1
        )

        # Age it past TTL + grace.
        lease = registry._leases["sess-preflight"]
        lease.last_used = time.time() - (idle_timeout + STALE_GRACE_SECS + 10.0)

        # This acquire should detect stale during its preflight and release it.
        token_b = await registry.acquire(
            "sess-preflight", owner_key="holder-b", generation=2, timeout=1
        )
        assert token_b is not None
        assert token_b.owner_key == "holder-b"

        registry.release(token_b)
        registry.release(token_a)  # idempotent

    _run(scenario())


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_stale_release_logs_at_error():
    """Force-release of a stale lease logs at ERROR level."""

    async def scenario():
        idle_timeout = 0.05
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)

        token = await registry.acquire(
            "sess-log", owner_key="test-holder", generation=1, timeout=1
        )

        # Age it past TTL + grace.
        lease = registry._leases["sess-log"]
        lease.last_used = time.time() - (idle_timeout + STALE_GRACE_SECS + 10.0)

        # Capture log output.
        with patch("gateway.turn_lease.logger.error") as mock_error:
            registry._cleanup()
            # Should log at ERROR level.
            assert mock_error.called
            call_args = mock_error.call_args
            # call_args[0] is the positional args tuple, which is the format string
            # and the expanded args. The format string is the first positional arg.
            assert "FORCE-RELEASING" in call_args[0][0]
            assert "sess-log" in str(call_args)  # check expanded args include the session id

    _run(scenario())


# ---------------------------------------------------------------------------
# DB-level stale lease sweep
# ---------------------------------------------------------------------------


def test_sweep_stale_turn_leases():
    """DB sweep removes expired leases whose age exceeds the threshold."""
    from hermes_state import SessionDB

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_state.db"

    db = SessionDB(db_path)
    try:
        # Insert a stale lease: expires_at in the past, acquired_at long ago.
        now = time.time()
        stale_acquired = now - 600  # 10 min ago
        stale_expires = now - 50  # expired 50s ago

        db._execute_write(
            lambda conn: conn.execute(
                "INSERT OR REPLACE INTO session_turn_leases "
                "(conversation_id, holder, acquired_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                ("sess-sweep-1", "pid=123:holder", stale_acquired, stale_expires)
            )
        )

        # Insert a fresh lease: expires_at still valid.
        fresh_expires = now + 300
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT OR REPLACE INTO session_turn_leases "
                "(conversation_id, holder, acquired_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                ("sess-sweep-2", "pid=456:holder", now - 10, fresh_expires)
            )
        )

        # Run the sweep.
        released = db.sweep_stale_turn_leases(max_age_seconds=300.0, grace_seconds=60.0)

        # Should have swept the stale one.
        assert len(released) == 1
        assert released[0]["conversation_id"] == "sess-sweep-1"
        assert released[0]["holder"] == "pid=123:holder"
        assert released[0]["age_seconds"] >= 590  # ~10 min

        # The fresh lease should still be there.
        info = db.get_stale_turn_lease_info()
        assert len(info) == 0  # no more stale leases

    finally:
        db.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sweep_stale_turn_leases_empty():
    """Sweep returns empty list when there are no stale leases."""
    from hermes_state import SessionDB

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_state.db"

    db = SessionDB(db_path)
    try:
        released = db.sweep_stale_turn_leases(max_age_seconds=300.0, grace_seconds=60.0)
        assert released == []
    finally:
        db.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sweep_stale_turn_leases_multiple():
    """Sweep can release multiple stale leases at once."""
    from hermes_state import SessionDB

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_state.db"

    db = SessionDB(db_path)
    try:
        now = time.time()
        for i in range(3):
            stale_acquired = now - 600 - i * 60
            stale_expires = now - 50 - i * 60
            db._execute_write(
                lambda conn, cid=f"sess-multi-{i}", ha=stale_acquired, he=stale_expires:
                conn.execute(
                    "INSERT OR REPLACE INTO session_turn_leases "
                    "(conversation_id, holder, acquired_at, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (cid, f"holder-{i}", ha, he)
                )
            )

        released = db.sweep_stale_turn_leases(max_age_seconds=300.0, grace_seconds=60.0)
        assert len(released) == 3

        for r in released:
            assert "conversation_id" in r
            assert "holder" in r
            assert "age_seconds" in r
            assert r["age_seconds"] >= 500

    finally:
        db.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration: stale detection prevents deadlock cascade
# ---------------------------------------------------------------------------


def test_deadlock_scenario_forced_release_breaks_the_deadlock():
    """Simulate the Sept 8 deadlock: holder is wedged, new acquires time out,
    but after idle timeout the lease is force-released and the next turn can proceed."""

    async def scenario():
        idle_timeout = 0.05
        registry = SessionTurnLeaseRegistry(idle_timeout=idle_timeout, max_waiters=3)

        # Simulate a wedged process: it acquired the lease but never released it.
        token_wedged = await registry.acquire(
            "sess-deadlock", owner_key="wedged-process", generation=1, timeout=1000
        )
        assert token_wedged is not None

        # A new turn tries to acquire and times out (like the old behavior).
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(
                "sess-deadlock", owner_key="new-turn-1", generation=2, timeout=0.1
            )

        # Simulate the wedged process dying: age the lease past TTL + grace.
        lease = registry._leases["sess-deadlock"]
        lease.last_used = time.time() - (idle_timeout + STALE_GRACE_SECS + 10.0)

        # Now cleanup runs (normally on next acquire).
        registry._cleanup()

        # A new turn can now acquire (the stale lease was force-released).
        token_fresh = await registry.acquire(
            "sess-deadlock", owner_key="new-turn-2", generation=3, timeout=1
        )
        assert token_fresh is not None
        assert token_fresh.owner_key == "new-turn-2"
        assert token_fresh.generation == 3

        registry.release(token_fresh)
        # wedged is already released; idempotent.
        registry.release(token_wedged)

    _run(scenario())


def test_concurrent_waiters_capped_on_stale_leak():
    """When multiple waiters queue behind a stale holder, the cap still applies
    after force-release."""

    async def scenario():
        idle_timeout = 0.05
        max_waiters = 2
        registry = SessionTurnLeaseRegistry(
            idle_timeout=idle_timeout, max_waiters=max_waiters
        )

        # Hold the lease.
        token_holder = await registry.acquire(
            "sess-cap", owner_key="holder", generation=1, timeout=10
        )

        # Age the lease past TTL + grace.
        lease = registry._leases["sess-cap"]
        lease.last_used = time.time() - (idle_timeout + STALE_GRACE_SECS + 10.0)

        # Release the stale lease first.
        registry._cleanup()
        registry.release(token_holder)

        # Now waiters queue up sequentially.
        r1 = await registry.acquire("sess-cap", owner_key="w1", generation=2, timeout=5)
        assert r1 is not None

        # Release r1 so r2 can get it.
        registry.release(r1)

        r2 = await registry.acquire("sess-cap", owner_key="w2", generation=3, timeout=5)
        assert r2 is not None
        registry.release(r2)

        # Now a third waiter should get it too since r1 and r2 released.
        r3 = await registry.acquire("sess-cap", owner_key="w3", generation=4, timeout=5)
        assert r3 is not None
        registry.release(r3)

    _run(scenario())
