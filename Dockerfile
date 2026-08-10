# Spooky FastAPI service — CPU-only image (no CUDA, no D-Wave cloud token
# required: DWave_solver.py uses dimod's SimulatedAnnealingSampler, a local
# classical solver). Suitable for HF Spaces' Docker SDK on the free CPU tier.
#
# GPU (lightning.gpu) and qiskit.remote solver profiles are still listed in
# fastapi_app/config/solvers.yaml but will fail at solve time in this image —
# use the qaoa_CPU_* profiles (lightning.qubit) instead.
FROM python:3.12-slim

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision=$GIT_SHA

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only what the package build and the API need (see .dockerignore)
COPY pyproject.toml README.md ./
COPY quantum/ ./quantum/
COPY fastapi_app/ ./fastapi_app/

# Core install (numpy/h5py/pennylane/pyyaml) + fastapi extra + the
# lightweight extras the API actually imports at startup: dimod (classical
# D-Wave solver, no dwave-system/cloud client needed), plotly
# (quantum/visualizer.py, imported unconditionally by api.py), and
# pyomo/highspy (quantum/builder/ILPBuilder.py, imported unconditionally by
# quantum/builder/__init__.py — the ILP solver).
RUN pip install -e ".[fastapi, dwave, visualizer, classical]"

# Regenerating maps
RUN python quantum/maps/generate_all_maps.py

# HF Spaces runs containers as a non-root user; match that here.
RUN useradd -u 1000 -m spooky && chown -R spooky:spooky /app
USER spooky

WORKDIR /app/fastapi_app

# HF Spaces' Docker SDK expects the app on port 7860 by default.
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
