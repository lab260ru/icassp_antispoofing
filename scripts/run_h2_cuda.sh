#!/usr/bin/env bash
# Launch an H2 ONNX task with the CUDA 13 wheel libraries discoverable.
#
# ONNX Runtime 1.28 needs libcublasLt.so.13, which is supplied by the local
# Python environment under site-packages rather than the system CUDA 12.6
# installation. Set the loader path before starting Python; changing it inside
# an already-running interpreter is not sufficient for reliable provider load.
set -euo pipefail

task_python="${PYTHON:-python3}"
task_site="$(${task_python} -c 'import site; print(site.getsitepackages()[0])')"
task_cuda_libs="${task_site}/nvidia/cu13/lib:${task_site}/nvidia/cudnn/lib:${task_site}/nvidia/cusparselt/lib"

export LD_LIBRARY_PATH="${task_cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${task_python}" "$@"
