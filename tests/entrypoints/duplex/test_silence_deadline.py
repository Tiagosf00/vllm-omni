# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic unit tests for deadline-aligned silence continuation.

The native-duplex scheduler in
``vllm_omni/entrypoints/duplex/session_runner.py`` aligns each silence
continuation to ``submission_time_N + chunk_period`` and sleeps only the
remaining budget. These tests cover the pure deadline arithmetic and the
per-session reset semantics shared by every native-duplex serving adapter.
"""

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.duplex.session_runner import (
    compute_silence_continuation_deadline,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import (
    MiniCPMO45ServingSessionState,
)
from vllm_omni.model_executor.models.personaplex.duplex.serving_adapter import (
    PersonaPlexServingSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("chunk_period_s", "now", "last_submit", "prior_deadline", "delay_s", "next_deadline"),
    [
        # First continuation anchors to the last real submission: unit N was
        # submitted at t=1.0, audio N produced at 1.4 -> 0.6 s of sleep left.
        (1.0, 1.4, 1.0, None, 0.6, 3.0),
        # Only the remaining budget is slept (audio produced at 1.7).
        (1.0, 1.7, 1.0, None, 0.3, 3.0),
        # Overdue deadline: no sleep; the chain still advances by the period.
        (1.0, 2.5, 1.0, None, 0.0, 3.0),
        # No submission yet (None): the first continuation anchors to now.
        (1.0, 0.4, None, None, 1.0, 2.4),
        # Deadlines advance from the prior deadline, not from now, so pipeline
        # processing time does not accumulate as timer drift.
        (1.0, 4.2, 3.0, 5.0, 0.8, 6.0),
        # Two consecutive units that overrun their deadline by 0.2 s keep the
        # 1 s cadence: no sleep, and the chain advances one period each.
        (1.0, 2.2, 1.0, None, 0.0, 3.0),
        (1.0, 3.2, 2.0, 3.0, 0.0, 4.0),
        # A short overshoot (within one period of the prior deadline) still
        # chases the stale deadline: immediate submit, chain advances.
        (1.0, 2.5, 1.0, 2.0, 0.0, 3.0),
        # A long stall (more than one period past the prior deadline) submits
        # one continuation immediately and restarts from that submission.
        (1.0, 5.0, 1.0, 2.0, 0.0, 6.0),
        # The same recovery applies to the first continuation after a stall:
        # last_submit at 1.0, now at 5.0 -> submit now, restart at 6.0.
        (1.0, 5.0, 1.0, None, 0.0, 6.0),
        # Zero chunk period (guarded upstream): total, no sleep.
        (0.0, 10.0, None, None, 0.0, 10.0),
    ],
)
def test_compute_silence_continuation_deadline(
    chunk_period_s: float,
    now: float,
    last_submit: float | None,
    prior_deadline: float | None,
    delay_s: float,
    next_deadline: float,
) -> None:
    delay, next_dl = compute_silence_continuation_deadline(
        chunk_period_s=chunk_period_s,
        now=now,
        last_submit=last_submit,
        prior_deadline=prior_deadline,
    )
    assert delay == pytest.approx(delay_s)
    assert next_dl == pytest.approx(next_deadline)


class TestSilenceDeadlineSessionState:
    @pytest.mark.parametrize(
        "state",
        [
            MiniCPMO45ServingSessionState(),
            PersonaPlexServingSessionState(),
        ],
    )
    def test_clear_continuation_resets_the_deadline_chain(self, state: object) -> None:
        # Both native-duplex session states feed the shared runner, so both
        # must start with an unset chain and drop it at turn boundaries.
        assert state.last_native_submit_monotonic is None
        assert state.silence_deadline_monotonic is None
        state.last_native_submit_monotonic = 5.0
        state.silence_deadline_monotonic = 6.0
        state.clear_continuation()
        assert state.last_native_submit_monotonic is None
        assert state.silence_deadline_monotonic is None


class FakeMonotonicClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _commit_native_submit(
    state: MiniCPMO45ServingSessionState,
    *,
    submit_time: float,
    silence_continuation: bool,
    silence_deadline: float | None,
) -> None:
    """Mirror the runner's ``_run`` timing-commit contract."""
    state.last_native_submit_monotonic = submit_time
    if silence_continuation:
        if silence_deadline is not None:
            state.silence_deadline_monotonic = silence_deadline
    else:
        state.silence_deadline_monotonic = None


class TestSilenceDeadlineScheduler:
    """End-to-end schedule decisions with a controlled clock.

    These drive the real deadline helper and the runner's commit contract:
    a silence append is submitted at ``last_submit + chunk_period`` (or
    immediately after a stall), the timing state is committed only when the
    append actually submits in the captured epoch, and a real input resets
    the chain.
    """

    def _schedule(
        self,
        clock: FakeMonotonicClock,
        state: MiniCPMO45ServingSessionState,
        *,
        chunk_period_s: float = 1.0,
        real_input_pending: bool = False,
        append_ok: bool = True,
        epoch_matches: bool = True,
    ) -> tuple[float, float]:
        """Run one schedule cycle; return (submitted, deadline_if_submitted)."""
        delay_s, next_deadline = compute_silence_continuation_deadline(
            chunk_period_s=chunk_period_s,
            now=clock(),
            last_submit=state.last_native_submit_monotonic,
            prior_deadline=state.silence_deadline_monotonic,
        )
        if delay_s > 0:
            clock.advance(delay_s)
        if real_input_pending:
            return 0.0, next_deadline
        # Mirror _run: only a real submit in the captured epoch commits state.
        if append_ok and epoch_matches:
            _commit_native_submit(
                state,
                submit_time=clock(),
                silence_continuation=True,
                silence_deadline=next_deadline,
            )
            return 1.0, next_deadline
        return 0.0, next_deadline

    def test_real_input_pending_prevents_silence_append(self) -> None:
        clock = FakeMonotonicClock(1.4)
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 1.0
        submitted, _ = self._schedule(clock, state, real_input_pending=True)
        assert submitted == 0.0
        assert state.silence_deadline_monotonic is None

    def test_failed_append_does_not_advance_the_chain(self) -> None:
        clock = FakeMonotonicClock(1.4)
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 1.0
        submitted, _ = self._schedule(clock, state, append_ok=False)
        assert submitted == 0.0
        assert state.last_native_submit_monotonic == 1.0
        assert state.silence_deadline_monotonic is None

    def test_epoch_stale_append_does_not_advance_the_chain(self) -> None:
        clock = FakeMonotonicClock(1.4)
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 1.0
        submitted, _ = self._schedule(clock, state, epoch_matches=False)
        assert submitted == 0.0
        assert state.last_native_submit_monotonic == 1.0
        assert state.silence_deadline_monotonic is None

    def test_successful_append_commits_deadline_from_anchor(self) -> None:
        clock = FakeMonotonicClock(1.4)
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 1.0
        submitted, next_deadline = self._schedule(clock, state)
        assert submitted == 1.0
        assert state.silence_deadline_monotonic == next_deadline
        # The chain advanced from the prior deadline (2.0), not from now (1.4).
        assert next_deadline == pytest.approx(3.0)

    def test_small_overshoot_preserves_the_schedule(self) -> None:
        # Unit N+1 due at 2.0; audio produced at 2.2 (0.2 s over) -> submit
        # immediately, next deadline still advances one period.
        clock = FakeMonotonicClock(2.2)
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 1.0
        submitted, next_deadline = self._schedule(clock, state)
        assert submitted == 1.0
        assert next_deadline == pytest.approx(3.0)

    def test_long_stall_submits_one_and_restarts(self) -> None:
        # More than one period late: submit one continuation immediately and
        # restart the schedule from that submission (next = now + period).
        clock = FakeMonotonicClock(5.0)
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 1.0
        state.silence_deadline_monotonic = 2.0
        submitted, next_deadline = self._schedule(clock, state)
        assert submitted == 1.0
        assert next_deadline == pytest.approx(6.0)

    def test_real_input_after_silence_reanchors(self) -> None:
        clock = FakeMonotonicClock(1.4)
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 1.0
        submitted, next_deadline = self._schedule(clock, state)
        assert submitted == 1.0
        assert state.silence_deadline_monotonic == next_deadline
        # A real (non-silence) input drops the deadline; the next silence
        # continuation anchors to that input's submission, not the old chain.
        state.silence_deadline_monotonic = None
        state.last_native_submit_monotonic = 2.0
        clock.advance(0.4)
        submitted2, next2 = self._schedule(clock, state)
        assert submitted2 == 1.0
        assert next2 == pytest.approx(4.0)
        assert state.last_native_submit_monotonic == pytest.approx(3.0)