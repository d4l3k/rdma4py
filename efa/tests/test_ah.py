"""Address handles: creation from GID / bytes / AHAttr, efadv query."""

from __future__ import annotations

import efa
import pytest

pytestmark = pytest.mark.integration


def test_create_ah_from_gid(ctx, pd):
    gid = ctx.query_gid()
    ah = pd.create_ah(gid)
    assert ah.pd is pd
    ah.close()
    ah.close()  # idempotent


def test_create_ah_from_raw_bytes(ctx, pd):
    ah = pd.create_ah(ctx.query_gid().raw)
    ah.close()


def test_create_ah_from_attr(ctx, pd):
    attr = efa.AHAttr(ctx.query_gid().raw, sgid_index=0, port_num=1)
    ah = pd.create_ah(attr)
    ah.close()


def test_ahattr_rejects_bad_gid():
    with pytest.raises(ValueError):
        efa.AHAttr(b"too-short")


def test_ahn(ctx, pd):
    ah = pd.create_ah(ctx.query_gid())
    assert ah.ahn >= 0
    ah.close()
    with pytest.raises(efa.EfaError):
        _ = ah.ahn


def test_endpoint_info_peer_owns_ah(ctx, pd):
    cq = ctx.create_cq(16)
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq)).prepare(qkey=5)
    info = efa.local_endpoint_info(qp, qkey=5)
    assert info.qp_num == qp.qp_num
    assert info.gid == ctx.query_gid().raw
    with info.peer(pd) as peer:
        assert peer.qp_num == qp.qp_num
        assert peer.qkey == 5
        assert peer.ah is not None
    assert peer.ah is None
    qp.close()
    cq.close()
