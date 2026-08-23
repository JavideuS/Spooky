"""
Fetch IBM Quantum backend CLOPS (Circuit Layer Operations Per Second) via the
IBM Cloud REST API. Not exposed on the IBMBackend/BackendV2 Python object —
only reachable through GET /v1/backends (see the original probe this was
lifted from: clops_test.py at the repo root).

CLOPS is IBM's own empirically measured throughput benchmark and, unlike the
calibration-based model in qpu_time_estimate.estimate_qpu_time(), already
folds in the classical control-loop overhead between shots (parameter
binding, primitive dispatch/feedback) that a physics-only gate-duration model
can't see. See qpu_time_estimate.estimate_qpu_time_clops() for how it's
applied to a circuit.

Requires IBM_TOKEN (IAM API key) and IBM_CRN (Service CRN) in the
environment/.env.
"""

import os

_CLOPS_CACHE = None  # {backend_name: {"value": int, "type": str} | None}, fetched once per process


def get_backend_clops(backend_name, force_refresh=False):
    """
    Return {"value": int, "type": str} for backend_name, or None if IBM
    hasn't published a CLOPS measurement for it.

    Fetches ALL backends' CLOPS in one call and caches the result for the
    life of the process — call sites shouldn't hit this once per window.

    Raises:
        RuntimeError: if IBM_TOKEN/IBM_CRN aren't set, or the request fails.
    """
    global _CLOPS_CACHE
    if _CLOPS_CACHE is None or force_refresh:
        import requests
        from dotenv import load_dotenv
        from ibm_cloud_sdk_core import IAMTokenManager

        load_dotenv()
        api_key = os.getenv("IBM_TOKEN")
        crn = os.getenv("IBM_CRN")
        if not api_key or not crn:
            raise RuntimeError(
                "IBM_TOKEN and IBM_CRN must be set (.env or environment) to "
                "fetch CLOPS via the IBM Cloud REST API."
            )

        token_manager = IAMTokenManager(apikey=api_key)
        access_token = token_manager.get_token()

        response = requests.get(
            "https://quantum.cloud.ibm.com/api/v1/backends",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
                "Service-CRN": crn,
                "IBM-API-Version": "2026-04-15",
            },
            timeout=30,
        )
        response.raise_for_status()

        _CLOPS_CACHE = {
            device["name"]: device.get("clops")
            for device in response.json().get("devices", [])
        }

    return _CLOPS_CACHE.get(backend_name)
