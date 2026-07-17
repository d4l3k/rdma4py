def test_import_and_version():
    from importlib.metadata import version

    import ibverbs

    assert ibverbs.__version__ == version("ibverbs")


def test_extension_is_linked():
    import ibverbs

    # The compiled extension is importable and linked against libibverbs.
    assert ibverbs._ibverbs._linked() is True
