# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Focused scheduler tests for deadline-aligned silence continuation.

These tests exercise the production ``_schedule_silence_continuation`` path
(and the append acceptance callback it installs) rather than copying the
scheduling arithmetic. The monotonic clock is a fake so the deadline math is
deterministic, and ``asyncio.sleep`` is simulated by advancing that clock.
A gated stage port lets a test control when an append actually completes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

import vllm_omni.engine.duplex.session.model_channel as model_channel_module
import vllm_omni.engine.duplex.session.runner as runner_module
from tests.engine.duplex.test_session_runner import (
    Harness,
    append_audio,
    close_harness,
    open_harness,
    tts_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK_PERIOD_S = 1.0


class FakeClock:
    """A monotonic clock whose value the test controls directly."""

    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value


class GatedPort:
    """Wrap the recording stage port so a test controls when submits finish.

    ``gate`` starts open; set it to ``asyncio.Event()`` (or clear it) to hold
    appends in flight, then set it to release them.
    """

    def __init__(self, port: Any) -> None:
        self._port = port
        self.gate: asyncio.Event = asyncio.Event()
        self.gate.set()

    async def submit(self, submission: Any) -> Any:
        await self.gate.wait()
        return await self._port.submit(submission)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._port, name)


async def _active_response_harness() -> Harness:
    """Open a harness and drive a TTS segment so a response is active."""
    h = await open_harness()
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
    assert h.session.active_response_id is not None
    return h


def _continuation_kwargs(h: Harness) -> dict[str, object]:
    """The scheduler arguments the model channel would pass for this session."""
    return {
        "request_id": h.session.active_request_id,
        "owner_id": f"response:{h.session.active_response_id}",
        "response_id": h.session.active_response_id,
        "response_owned": True,
        "expected_epoch": h.session.epoch,
        "expected_model_turn_id": h.session.turn_id,
    }


def _install_fake_clock(
    monkeypatch: pytest.MonkeyPatch,
    *,
    clock: FakeClock,
    real_sleep: Callable[..., Any],
) -> None:
    """Point both modules at the fake clock and simulate ``asyncio.sleep``.

    ``real_sleep`` is the unpatched ``asyncio.sleep`` captured before patching;
    the simulated sleep advances the fake clock by the requested delay and then
    yields control so the append task can run. ``stall`` (a one-element list)
    is added once to the next sleep so a test can model a wake that happens
    long after the planned deadline.
    """
    monkeypatch.setattr(runner_module.time, "monotonic", clock)
    monkeypatch.setattr(model_channel_module.time, "monotonic", clock)
    stall: list[float] = [0.0]

    async def fake_sleep(delay: float) -> None:
        clock.value += delay + stall[0]
        stall[0] = 0.0
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return stall


@pytest.mark.asyncio
async def test_normal_cadence_keeps_deadlines_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Processing time reduces the remaining sleep; consecutive deadlines align.

    Each accepted unit becomes the new anchor (its submission time), and the
    following continuation advances from the stored deadline rather than
    resetting to now.
    """
    clock = FakeClock(start=100.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    try:
        # Real input accepted at t=100.0; the model processes for 0.4 s.
        h.runner.model_state.last_native_submit_monotonic = 100.0
        clock.value = 100.4

        # First continuation: due at 101.0, so only 0.6 s of sleep remains.
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        await asyncio.sleep(0.05)
        # The accepted unit re-anchored the chain to its own submission time
        # (101.0) and stored the following deadline (102.0).
        assert h.runner.model_state.last_native_submit_monotonic == pytest.approx(101.0)
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(102.0)

        # Model produces the continuation's audio; the next unit is planned
        # from the stored deadline (no drift): the following deadline is 103.0.
        clock.value = 101.3
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        await asyncio.sleep(0.05)
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(103.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_in_flight_real_append_skips_outdated_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real append accepted after the silence was planned re-anchors the chain.

    The planned silence queues behind the real append; once the real append
    completes and re-anchors, ``before_append`` detects the changed anchor and
    the outdated silence is skipped (no submission, no callback overwrite).
    """
    clock = FakeClock(start=200.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    gated = GatedPort(h.port)
    monkeypatch.setattr(h, "port", gated)
    try:
        anchor_before = h.runner.model_state.last_native_submit_monotonic
        # Hold the real append in flight; it will re-anchor on acceptance.
        gated.gate = asyncio.Event()
        real_task = asyncio.create_task(
            h.runner._start_append(
                h.runner.model.silence_unit_payload(),
                final=False,
                silence_continuation=False,
            )
        )
        await asyncio.sleep(0.05)
        assert not real_task.done()

        # The silence is planned while the real append is queued ahead of it.
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True

        # Release the real append: it accepts and re-anchors the chain to its
        # own submission time (a fresh object, distinct from the anchor the
        # silence planned against).
        gated.gate.set()
        await real_task
        await asyncio.sleep(0.05)

        # The outdated silence was skipped: its callback never overwrote the
        # new anchor, and the deadline the real append cleared stays cleared.
        assert h.runner.model_state.last_native_submit_monotonic is not anchor_before
        assert h.runner.model_state.last_native_submit_monotonic is not None
        assert h.runner.model_state.silence_deadline_monotonic is None

        # A fresh continuation uses the new anchor.
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        await asyncio.sleep(0.05)
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(
            h.runner.model_state.last_native_submit_monotonic + CHUNK_PERIOD_S
        )
    finally:
        gated.gate.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_long_stall_saves_future_deadline_from_actual_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submission more than one period past the planned deadline restarts.

    The schedule plans the following deadline at ``planned + period``, but the
    actual submission happens much later. The accepted continuation must save
    ``submit_time + period`` so the next unit is not immediately due.
    """
    clock = FakeClock(start=300.0)
    real_sleep = asyncio.sleep
    stall = _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    try:
        # Plan a continuation whose deadline is 1.0 s in the future. The wake
        # is delayed 2 s past the planned submission (clock 300 -> 303 during
        # the scheduler's sleep), so the append accepts 2 s late.
        h.runner.model_state.last_native_submit_monotonic = 300.0
        h.runner.model_state.silence_deadline_monotonic = None
        stall[0] = 2.0
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        await asyncio.sleep(0.05)
        # 303.0 > 302.0 (planned deadline + period) -> save 303.0 + 1.0.
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(304.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_small_wakeup_delay_preserves_planned_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary wakeup jitter (within one period) keeps the planned cadence.

    A submission slightly after the planned deadline must not reset the chain:
    the stored deadline stays at ``planned + period`` so drift does not
    accumulate.
    """
    clock = FakeClock(start=400.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    try:
        h.runner.model_state.last_native_submit_monotonic = 400.0
        h.runner.model_state.silence_deadline_monotonic = None
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        # Submitted 0.2 s late (within one period): keep the planned deadline.
        clock.value = 401.2
        await asyncio.sleep(0.05)
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(402.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_exact_one_period_late_preserves_planned_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly one period late is not "more than one period late".

    The strict ``>`` comparison means a submission exactly at the planned
    following deadline keeps the planned cadence (no extra catch-up unit).
    """
    clock = FakeClock(start=500.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    try:
        h.runner.model_state.last_native_submit_monotonic = 500.0
        h.runner.model_state.silence_deadline_monotonic = None
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        # Submitted exactly at the planned following deadline (502.0).
        clock.value = 502.0
        await asyncio.sleep(0.05)
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(502.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_failed_append_leaves_timing_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed append never advances the timing chain."""
    clock = FakeClock(start=600.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    try:
        h.port.fail_submit = RuntimeError("boom")
        before_anchor = h.runner.model_state.last_native_submit_monotonic
        before_deadline = h.runner.model_state.silence_deadline_monotonic
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        # The append fails; the scheduler reports the task, but timing is untouched.
        assert scheduled is True
        await asyncio.sleep(0.05)
        assert h.runner.model_state.last_native_submit_monotonic is before_anchor
        assert h.runner.model_state.silence_deadline_monotonic is before_deadline
    finally:
        await close_harness(h)
