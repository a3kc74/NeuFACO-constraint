# FACO C++ Backend

This directory is the canonical refactor entry point for the FACO native backend.

- `binding.cpp`
- `mfaco_train.cpp`
- `mfaco_train.h`
- `kd_tree.h`
- `setup.py`

Build from the repository root with:

```raw
uv run python cpp/faco/setup.py build_ext --inplace
```

The wrapper builds the compiled `faco_opt` extension next to the canonical source in `cpp/faco/src/`.

