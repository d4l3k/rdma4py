def test_import_and_version():
    import ibverbs

    assert isinstance(ibverbs.__version__, str)
    assert ibverbs.__version__


def test_extension_is_linked():
    import ibverbs

    # The compiled extension is importable and linked against libibverbs.
    assert ibverbs._ibverbs._linked() is True
