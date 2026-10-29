Eudoxys county-run environment — reconstructed 2026-10-29

Restore with uv (tested with uv 0.13.6 on macOS arm64):

    uv sync --locked
    uv run --locked python verify_environment.py

Keep pyproject.toml, uv.lock and .python-version together. Python is pinned
to 3.12.6 in .python-version. uv.lock supports Python 3.12.

This reconstructs the installed review/.venv environment. All 88 observed
packages are pinned. The eight formerly editable local Git installations are
replaced with immutable, full commit references. Their working trees were clean
when captured. Registry artifacts have hashes in uv.lock. Additional conditional
lock entries (colorama and nvidia-nccl-cu13) apply on other platforms; they were
not installed in the original macOS environment. Only macOS arm64 was tested.

Important provenance boundary
-----------------------------
This lock was created after the experiments, not before them. The original run
manifests recorded selected source hashes and seven core dependency versions,
not every installed dependency or repository commit. installed-environment.json
captures the environment observed during this reconstruction. validation.json
records matching package versions, Git commit checks, source comparisons against
the saved difference-regularization sweep, and two prediction reproduction checks.
This does not establish historical identity for every previously unrecorded file
or dependency across all earlier runs.

Explicit reconstruction overrides
---------------------------------
1. loads declares plotly, but plotly was absent in the observed run environment.
   The loads dependency-metadata entry omits that requirement to preserve the
   observed package set. This omission already existed during the experiments;
   plotting functionality requiring Plotly is not available. uv pip check still
   reports the original installed loads metadata's missing-plotly requirement.
2. weather declares a Git pvlib dependency, while the observed environment used
   the registry release pvlib==0.16.1. override-dependencies preserves that release.
3. All eight project Git dependencies are overridden with the captured commits,
   including transitive references that otherwise follow unpinned branch heads.

The lock covers Python runtime packages, not the operating system, system OpenMP
runtime, compilers or external datasets. Git package builds may need build-time
dependencies. Neither the original run environment nor its repositories were
modified by this reconstruction; validation used a separate .venv.

Contents
--------
uv.lock                    Complete generated uv lockfile
pyproject.toml             Exact package pins, Git sources and explicit overrides
.python-version            Python 3.12.6
installed-environment.json Captured package inventory and Git revisions
validation.json            Clean-install and prediction comparison results
verify_environment.py      Portable package/commit check for macOS arm64
SHA256SUMS                 Checksums of delivered files
