"""Per-session turn lease — serializes the [load history -> run -> flush] region.

Busy guards are keyed by ROUTING KEY but the transcript is owned by SESSION_ID, and
``switch_session()`` makes key->id many-to-one (/resume from a second chat, CLI-continuity,
delegation pinning, topic tip-walks), so two keys could interleave flushes on one transcript
(``user;user`` wedge). The lease serializes per RESOLVED session_id: acquired right before the
transcript load, released in the dispatch layer's ``finally``; identity-checked release; a
timed-out waiter fails CLOSED (:class:`TurnLeaseTimeoutError`); only idle entries evict. Limits:
CLI-continuity processes are outside this lock; mid-turn rotation alias is closed by ``rebind``.

Deadlock detection (postmortem 2026-09-08): added idle TTL, stale-holder detection, concurrent
waiter cap, and forced-release of abandoned leases so a dead or wedged process cannot pin a
session indefinitely. See t_a1055472.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cap on tracked leases. Idle entries evict oldest-first; live leases never do, so a burst of
# distinct sessions may transiently exceed the cap rather than break serialization.
DEFAULT_MAX_LEASES = 512
# Fallback wait (seconds) when the caller passes no positive timeout (bridged via
# HERMES_TURN_LEASE_TIMEOUT — lease contention is not agent inactivity). Fail-closed but short:
# never pin a sequential platform updater for minutes.
DEFAULT_LEASE_WAIT = 5.0

# ── Deadlock-detection thresholds ──────────────────────────────────────

# Idle timeout (seconds): if a lease is held but no activity for this long,
# the holder is presumed dead or wedged. 5 min matches the ~3 min asyncio
# timeout from the old path + margin for slow tool calls.  A process that
# dies naturally loses the asyncio lock and all waiters wake; a process
# that wedges (dead code path, unkillable syscall) keeps the lock — that
# is what this detects.
DEFAULT_IDLE_TIMEOUT = 300.0  # 5 minutes

# Maximum concurrent waiters per lease. Beyond this the session is
# effectively DoSed by the queue itself; cap prevents runaway thread
# creation when the holder is stuck and new messages keep arriving.
DEFAULT_MAX_WAITERS = 3

# Stale-holder grace: on detection we log at ERROR and force-release the
# lease so the next waiter can proceed.  Do NOT force-release a holder
# that is *known* to still be in-flight — only force when the idle clock
# shows zero activity.
STALE_GRACE_SECS = 60.0  # extra margin beyond idle_timeout before force-release


def _holder_desc(holder: Optional["TurnLeaseToken"]) -> tuple:
    return (holder.owner_key, holder.generation) if holder else ("?", "?")


def _lease_age(lease: "_SessionLease") -> float:
    """Seconds since the lease was last used, or 0 if just created."""
    return max(0.0, time.time() - lease.last_used)


class TurnLeaseTimeoutError(TimeoutError):
    """Lease held for the full wait budget; fail-closed: caller must not enter the turn region."""

    def __init__(self, session_id: str, *, owner_key: str, generation: int, wait_seconds: float) -> None:
        self.session_id, self.owner_key = session_id, owner_key
        self.generation, self.wait_seconds = generation, wait_seconds
        super().__init__(f"turn lease wait timed out after {wait_seconds:.0f}s on session "
                         f"{session_id} for routing key {owner_key} (gen {generation})")


class TurnLeaseStaleError(RuntimeError):
    """The holder of this lease is stale (idle too long / process presumed dead);
    the caller must NOT proceed — the lease has been force-released."""

    def __init__(self, session_id: str, *, owner_key: str, generation: int) -> None:
        self.session_id, self.owner_key = session_id, owner_key
        self.generation = generation
        super().__init__(f"turn lease stale on session {session_id} "
                         f"(holder: routing key {owner_key} gen {generation}; "
                         f"force-released due to idle timeout)")


class TurnLeaseWaiterLimitError(RuntimeError):
    """Too many threads are already waiting for this lease."""

    def __init__(self, session_id: str, max_waiters: int) -> None:
        self.session_id = session_id
        self.max_waiters = max_waiters
        super().__init__(f"session {session_id} already has {max_waiters} concurrent waiters "
                         f"(limit={max_waiters}); refusing to queue another")


class TurnLeaseToken:
    """Held-lease handle from :meth:`SessionTurnLeaseRegistry.acquire`; ``released`` makes
    release idempotent."""

    __slots__ = ("session_id", "owner_key", "generation", "released")

    def __init__(self, session_id: str, owner_key: str, generation: int) -> None:
        self.session_id, self.owner_key, self.generation = session_id, owner_key, generation
        self.released = False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"TurnLeaseToken(session_id={self.session_id!r}, owner_key={self.owner_key!r}, "
                f"generation={self.generation}, released={self.released})")


class _SessionLease:
    __slots__ = ("lock", "holder", "acquired_at", "last_used", "pending_acquires",
                 "idle_timeout", "detected_stale")

    def __init__(self, idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> None:
        self.lock = asyncio.Lock()
        self.holder: Optional[TurnLeaseToken] = None
        self.acquired_at, self.last_used, self.pending_acquires = 0.0, time.time(), 0
        self.idle_timeout = idle_timeout
        self.detected_stale = False

    @property
    def idle(self) -> bool:
        """True when evictable: nobody holds or awaits it."""
        return self.holder is None and not self.lock.locked() and self.pending_acquires == 0

    @property
    def idle_age(self) -> float:
        """Seconds since last activity on this lease (even without a holder)."""
        return _lease_age(self)

    @property
    def is_stale(self) -> bool:
        """True if the lease has been held (or last used) for longer than the idle
        timeout without a force-release already being triggered."""
        if not self.detected_stale:
            return _lease_age(self) > self.idle_timeout
        return False

    @property
    def should_force_release(self) -> bool:
        """True if we should force-release a stale lease that still has a holder.
        Includes a small grace margin so we don't interrupt genuinely long (but still
        in-flight) tool calls that happen to be near the boundary."""
        return _lease_age(self) > (self.idle_timeout + STALE_GRACE_SECS)

    def touch(self) -> None:
        """Record a recent activity — called on acquire and refresh."""
        self.last_used = time.time()
        if self.detected_stale:
            self.detected_stale = False  # liveness returned


class SessionTurnLeaseRegistry:
    """Asyncio lease per resolved session_id. Process-local, single-event-loop by design (same
    visibility scope as the routing-key guards it extends); call only from the gateway loop.

    Deadlock protection:
    - Idle TTL: leases older than ``idle_timeout`` are detected as stale.
    - Stale-holder detection: on acquire, if an existing lease is stale the holder is
      force-released so the new caller can proceed.
    - Concurrent waiter cap: prevents thread accumulation from DoS or stuck-holder cascades.
    - Periodic cleanup: runs on every acquire/release cycle (not a separate goroutine —
      this is Python asyncio, no native threads for this).
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_LEASES,
                 idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
                 max_waiters: int = DEFAULT_MAX_WAITERS) -> None:
        self._leases: Dict[str, _SessionLease] = {}
        self._max_entries = max(1, int(max_entries))
        self._idle_timeout = idle_timeout
        self._max_waiters = max_waiters

    def _get_or_create(self, session_id: str) -> _SessionLease:
        if (lease := self._leases.get(session_id)) is None:
            self._evict_idle()
            lease = self._leases[session_id] = _SessionLease(self._idle_timeout)
        lease.last_used = time.time()
        return lease

    def _evict_idle(self) -> None:
        """Drop oldest idle entries to fit a new lease under the cap; never a held/contended one."""
        if (overflow := len(self._leases) - self._max_entries + 1) <= 0:
            return
        idle = sorted((sid for sid, l in self._leases.items() if l.idle),
                      key=lambda sid: self._leases[sid].last_used)
        for sid in idle[:overflow]:
            del self._leases[sid]

    def _cleanup(self) -> None:
        """Scan all leases; force-release stale ones and purge truly orphaned ones.

        A stale lease is one where the holder is idle beyond ``idle_timeout + grace``.
        We do NOT force-release if the lease has no holder (those will be idle-evicted
        naturally or picked up by a new acquire).
        """
        for sid, lease in list(self._leases.items()):
            if lease.holder is not None and lease.detected_stale is False:
                age = lease.idle_age
                threshold = lease.idle_timeout + STALE_GRACE_SECS
                if age > threshold:
                    logger.info(
                        "Lease on session %s idle for %.0fs (threshold=%.0fs); "
                        "will force-release", sid, age, threshold)
                    # Only force-release when the holder exists and hasn't already
                    # been marked stale.
                    self._force_release_stale(sid, lease)

    def _force_release_stale(self, session_id: str, lease: _SessionLease) -> None:
        """Force-release a stale lease. The holder is marked as released (idempotent on the token),
        the lock is forcibly released (waking all waiters), and the holder info is cleared."""
        holder = lease.holder
        holder_desc = _holder_desc(holder)
        logger.error(
            "FORCE-RELEASING stale turn lease on session %s: holder routing key %s (gen %s), "
            "idle for %.0fs (threshold=%.0fs); this lease is being forcibly released to break "
            "a potential deadlock — the next waiter on this session will be able to proceed.",
            session_id, holder_desc[0], holder_desc[1],
            lease.idle_age, lease.idle_timeout)
        # Mark the holder token as released (idempotent on the token itself).
        if holder is not None:
            holder.released = True
        lease.detected_stale = True
        # Release the asyncio lock so all waiters can re-acquire and then fail closed
        # or succeed with a new holder.
        if lease.lock.locked():
            lease.lock.release()
        lease.holder = None
        lease.acquired_at = 0.0

    async def acquire(
        self, session_id: str, *, owner_key: str, generation: int, timeout: Optional[float] = None
    ) -> Optional[TurnLeaseToken]:
        """Acquire the lease for ``session_id``, waiting if held. Raises
        :class:`TurnLeaseTimeoutError` when the wait budget expires; None for a falsy id.

        Deadlock-aware:
        - Checks for stale leases before proceeding; force-releases them.
        - Counts concurrent waiters and rejects if the cap is exceeded.
        """
        if not session_id:
            return None

        # Run cleanup of any stale leases first.
        self._cleanup()

        wait = float(timeout) if timeout and timeout > 0 else DEFAULT_LEASE_WAIT
        token = TurnLeaseToken(session_id, owner_key, int(generation))
        lease = self._get_or_create(session_id)

        # Check for stale holder BEFORE we start waiting.
        if lease.is_stale and lease.holder is not None:
            logger.warning(
                "Detected stale lease on session %s: holder routing key %s (gen %s), "
                "idle for %.0fs — will force-release on lock release",
                session_id, *_holder_desc(lease.holder), lease.idle_age)

        # Check concurrent waiter cap.
        if lease.pending_acquires >= self._max_waiters:
            logger.error(
                "Session %s has %d concurrent waiters (max=%d); refusing to add another — "
                "possible deadlock cascade. The caller will get a timeout.",
                session_id, lease.pending_acquires, self._max_waiters)
            # Signal that we're a waiter even though we won't actually wait, so the
            # counter is accurate for the timeout check below.
            lease.pending_acquires += 1
            try:
                try:
                    await asyncio.wait_for(lease.lock.acquire(), timeout=wait)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    raise TurnLeaseTimeoutError(
                        session_id, owner_key=owner_key, generation=generation, wait_seconds=wait)
            finally:
                lease.pending_acquires -= 1
            return None  # never got the lock

        if lease.lock.locked():
            logger.warning(
                "turn lease contention on session %s: routing key %s (gen %s) waiting behind "
                "in-flight turn held by routing key %s (gen %s, held %.0fs) — two routing keys "
                "are mapped to one session_id (#64934); serializing this turn behind the previous "
                "turn's flush",
                session_id, owner_key, generation, *_holder_desc(lease.holder),
                time.time() - lease.acquired_at if lease.acquired_at else -1.0)

        # Lock.release() wakes a waiter while leaving the lock momentarily unlocked. Count every
        # in-progress acquire across that handoff (even apparently-uncontended ones — wait_for()
        # may schedule them before the lock coroutine runs) so eviction cannot orphan the old
        # lock and create a second lock for the same session.
        lease.pending_acquires += 1
        try:
            await asyncio.wait_for(lease.lock.acquire(), timeout=wait)
        except asyncio.TimeoutError:
            logger.error(
                "turn lease wait timed out after %.0fs on session %s (waiter: routing key %s gen "
                "%s; holder: routing key %s gen %s) — failing closed: refusing to run this turn "
                "UNSERIALIZED against the still-held lease",
                wait, session_id, owner_key, generation, *_holder_desc(lease.holder))
            raise TurnLeaseTimeoutError(
                session_id, owner_key=owner_key, generation=generation, wait_seconds=wait) from None
        except asyncio.CancelledError:
            # Don't raise the stale error on cancel; let it propagate as CancelledError.
            raise
        finally:
            lease.pending_acquires -= 1

        # Lock held and no await before holder publication, so the lease cannot become
        # evictable after the pending count is cleared.
        lease.holder = token
        lease.acquired_at = lease.last_used = time.time()
        logger.info(
            "turn lease acquired on session %s: routing key %s gen %s (idle_age=%.0fs)",
            session_id, owner_key, generation, lease.idle_age)
        return token

    def rebind(self, token: Optional[TurnLeaseToken], new_session_id: str) -> bool:
        """Alias a HELD lease onto ``new_session_id`` after mid-turn rotation (compression) so the
        flush target stays serialized: the SAME ``_SessionLease`` is registered under the new id
        (old mapping idle-evicts later), only the holder may rebind, the token follows. A live
        lease on the new id: log loudly, keep the old id (fail-open)."""
        if (token is None or token.released or not new_session_id
                or new_session_id == token.session_id):
            return False
        if (lease := self._leases.get(token.session_id)) is None or lease.holder is not token:
            return False
        existing = self._leases.get(new_session_id)
        if existing is not None and existing is not lease and not existing.idle:
            logger.warning(
                "turn lease rebind blocked: session %s rotated to %s mid-turn (holder: routing key "
                "%s gen %s) but the target session's lease is already live (holder: routing key %s "
                "gen %s) — keeping the lease on the old id; transcript writes on %s may "
                "interleave (#64934 rotation-alias edge)",
                token.session_id, new_session_id, token.owner_key, token.generation,
                *_holder_desc(existing.holder), new_session_id)
            return False
        self._leases[new_session_id] = lease
        lease.last_used = time.time()
        token.session_id = new_session_id
        return True

    def release(self, token: Optional[TurnLeaseToken]) -> bool:
        """Release ``token``'s lease. Idempotent; True only when this exact token was the current
        holder (a re-release or a stale token whose slot went to a newer turn is a safe no-op)."""
        if token is None or token.released:
            return False
        token.released = True
        if (lease := self._leases.get(token.session_id)) is None:
            return False
        if lease.holder is not token:
            # If this is a stale release (another process/thread force-released), log it.
            if lease.detected_stale:
                logger.warning(
                    "turn lease release on session %s: holder was force-released as stale "
                    "(idle=%.0fs); token (key %s gen %s) did not match — this is expected "
                    "when a deadlock was detected.",
                    token.session_id, lease.idle_age, token.owner_key, token.generation)
            else:
                logger.debug("turn lease release skipped on session %s: token (key %s gen %s) is not "
                             "the current holder", token.session_id, token.owner_key, token.generation)
            return False
        lease.holder, lease.acquired_at, lease.last_used = None, 0.0, time.time()
        if lease.lock.locked():
            lease.lock.release()
        logger.info(
            "turn lease released on session %s: routing key %s gen %s",
            token.session_id, token.owner_key, token.generation)
        return True
