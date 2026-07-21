# rdma4py Repository Guide

This repository contains three independently installable packages:

- `efa`: low-level AWS EFA/SRD bindings and EFA direct-verbs support.
- `ibverbs`: general low-level libibverbs and RDMA-CM bindings.
- `nvmeof`: a pure-Python NVMe/RDMA initiator layered on `ibverbs`.

Keep package boundaries explicit. General verbs behavior belongs in
`ibverbs`; EFA provider behavior belongs in `efa`; NVMe controller, namespace,
and command policy belongs in `nvmeof`. Consult `efa/CLAUDE.md` and
`ibverbs/CLAUDE.md` before changing those bindings.

## Development

The Cython packages compile against rdma-core headers but load runtime
libraries with `dlopen`. Install the RDMA development headers before building
from source. Rebuild a package after changing its `.pyx` or `.pxd` files.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "./efa[test]"
pip install -e "./ibverbs[test]"
pip install -e "./nvmeof[test]"
```

Run the package suites independently:

```bash
RDMA4PY_SKIP_HARDWARE_TESTS=1 .venv/bin/python -m pytest -rs -m "not gpu" efa
RDMA4PY_SKIP_HARDWARE_TESTS=1 .venv/bin/python -m pytest -rs -m "not gpu" ibverbs
.venv/bin/python -m pytest -rs -m "not gpu" nvmeof
```

Hardware, GPU, and real NVMe/RDMA target tests must skip clearly when their
requirements are absent. Never run destructive NVMe-oF tests unless the target
namespace is explicitly disposable and `NVME4PY_DESTRUCTIVE=1` is set.

## Binding Conventions

- Preserve the thin, explicit resource model: one Python object per native
  resource, idempotent `close()`, context-manager support, and parent lifetime
  retention.
- Keep the data path compiled and release the GIL only around native calls.
- Do not add torch or CUDA as runtime imports to the base `efa` or `ibverbs`
  modules. CUDA helpers remain optional and duck-typed.
- Use structured protocol packing and parsing for NVMe wire formats. Keep
  command and transport validation close to the layer that owns it.
- Public functions, methods, constructor options, feature flags, and enum
  members require useful docstrings. Document lifecycle, ordering, capability,
  and hardware constraints where they affect correctness.

## Documentation

Documentation sources live in `docs/`, with package guides in each package
README. Always use direct `pip install ...` syntax in user-facing
documentation.

The site uses `pytorch-sphinx-theme2`. Its pinned version requires the
compatible Sphinx and MyST versions in `docs/requirements.txt`. Build with
warnings treated as errors:

```bash
pip install -r docs/requirements.txt
.venv/bin/sphinx-build -W --keep-going -E -a -b html docs docs/_build/html
```

`docs/conf.py` copies Font Awesome webfonts from the theme's pinned PyData
dependency because the Theme 2 wheel omits files referenced by its CSS. Keep
that workaround until the upstream wheel includes them.

## Scope And Verification

Keep edits scoped to the owning package and avoid generated C, extension,
wheel, cache, or built documentation artifacts. Before committing, run the
affected tests, formatting/lint checks, `git diff --check`, and the Sphinx
warning-as-error build for documentation changes.
