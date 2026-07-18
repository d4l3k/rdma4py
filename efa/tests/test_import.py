def test_import_and_version():
    import efa

    assert isinstance(efa.__version__, str)
    assert efa.__version__


def test_extension_is_linked():
    import efa

    # The compiled extension is importable and linked against
    # libibverbs + libefa.
    assert efa._efa._linked() is True
