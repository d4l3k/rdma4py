import ibverbs as ib
import pytest


def test_rdmacm_is_optional_but_discoverable():
    assert isinstance(ib._ibverbs._has_rdmacm(), bool)


@pytest.mark.parametrize(
    "host, port", [("", 4420), ("localhost", 0), ("localhost", 65536)]
)
def test_cmid_validates_address(host, port):
    with pytest.raises(ValueError):
        ib.CMID.resolve(host, port)


@pytest.mark.parametrize("source", ["", "bad\x00source", 42])
def test_cmid_validates_source_address(source):
    with pytest.raises(ValueError):
        ib.CMID.resolve("localhost", source=source)
