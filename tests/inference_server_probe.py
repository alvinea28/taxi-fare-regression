"""Exercise the real Azure inference WSGI loader with a local MLflow fixture.

This is a test scorer, not a replacement for Azure's generated scoring script.
No sockets, credentials, production data, or Azure endpoints are used.
"""

from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import sys


_model = None


def init():
    # Both imports are required by the generated MLflow scorer's startup path.
    from azureml.ai.monitoring import Collector
    import mlflow.pyfunc

    assert callable(Collector)
    global _model
    _model = mlflow.pyfunc.load_model(os.environ["AZUREML_MODEL_DIR"])


def run(raw_data):
    import pandas as pd

    if _model is None:
        raise RuntimeError("The server did not initialize the model")
    frame = pd.DataFrame(**json.loads(raw_data)["input_data"])
    return _model.predict(frame).tolist()


def main():
    model_path, request_path = map(Path, sys.argv[1:])

    def deny_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the inference server probe")

    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    os.environ.update({
        "AML_APP_ROOT": str(request_path.parent),
        "AZUREML_ENTRY_SCRIPT": str(Path(__file__).resolve()),
        "AZUREML_MODEL_DIR": str(model_path),
        "AML_APP_INSIGHTS_ENABLED": "false",
        "AML_MODEL_DC_STORAGE_ENABLED": "false",
    })

    # This is the actual Gunicorn worker application entry used on Linux.
    from azureml_inference_server_http.server.entry import app
    from azureml.contrib.services.aml_request import AMLRequest as LegacyRequest
    from azureml_inference_server_http.api.aml_request import AMLRequest
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd

    assert LegacyRequest is AMLRequest, "The server's legacy namespace patch did not run"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    with app.test_client() as client:
        health = client.get("/")
        assert health.status_code == 200, health.get_data(as_text=True)
        response = client.post("/score", json=payload)
        assert response.status_code == 200, response.get_data(as_text=True)
        predictions = response.get_json()
    expected = mlflow.pyfunc.load_model(str(model_path)).predict(pd.DataFrame(**payload["input_data"]))
    assert len(predictions) == len(payload["input_data"]["data"])
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, expected)
    print("TAXI_INFERENCE_SMOKE=" + json.dumps({
        "platform": platform.system(),
        "python": platform.python_version(),
        "server": version("azureml-inference-server-http"),
        "contrib": version("azureml-contrib-services"),
        "health_status": health.status_code,
        "score_status": response.status_code,
        "predictions": len(predictions),
    }))


if __name__ == "__main__":
    main()
