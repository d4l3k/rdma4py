"""Completion queues and completion channels."""

from __future__ import annotations

import pytest

import ibverbs as ib


def test_create_cq(ctx):
    cq = ctx.create_cq(16)
    assert cq.cqe >= 16
    cq.close()


def test_poll_empty_returns_empty_list(ctx):
    cq = ctx.create_cq(16)
    assert cq.poll(8) == []
    cq.close()


def test_poll_zero_raises(ctx):
    cq = ctx.create_cq(16)
    with pytest.raises(ValueError):
        cq.poll(0)
    cq.close()


def test_comp_channel_fd(ctx):
    ch = ctx.create_comp_channel()
    assert ch.fd >= 0
    cq = ctx.create_cq(16, channel=ch)
    assert cq.channel is ch
    cq.close()
    ch.close()


def test_req_notify_on_channel_cq(ctx):
    ch = ctx.create_comp_channel()
    cq = ctx.create_cq(16, channel=ch)
    cq.req_notify()  # should not raise
    cq.close()
    ch.close()


def test_cannot_ack_undelivered_event(ctx):
    cq = ctx.create_cq(16)
    with pytest.raises(ValueError, match="more CQ events"):
        cq.ack_events()
    cq.close()


def test_cq_context_manager(ctx):
    with ctx.create_cq(32) as cq:
        assert cq.cqe >= 32
