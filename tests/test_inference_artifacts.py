"""Verify real MLflow inference artifacts without Azure access.

Run with Python 3.11, exported serving requirements, matplotlib, and PyYAML.
Uses synthetic data, local tracking, mocked registration, and temporary files.
"""

from contextlib import ExitStack
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlparse
from urllib.request import url2pathname


ROOT = Path(__file__).resolve().parents[1]
PROVIDER = "setuptools==80.10.2"
MONITORING = "azureml-ai-monitoring==1.0.0"
CONTRIB = "azureml-contrib-services==1.57.0"
SERVER = "azureml-inference-server-http==1.2.0"


def load_source(name):
    path = ROOT / "data-science/src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"smoke_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InferenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stack = ExitStack()
        cls.addClassCleanup(stack.close)
        cls.root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="taxi-inference-test-")))
        original_directory = Path.cwd()
        stack.callback(os.chdir, original_directory)
        os.chdir(cls.root)
        tracking = (cls.root / "tracking").as_uri()
        stack.enter_context(patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": tracking,
            "MLFLOW_REGISTRY_URI": tracking,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cls.root / "matplotlib"),
        }))
        os.environ.pop("MLFLOW_RUN_ID", None)
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

        def deny_network(*args, **kwargs):
            raise AssertionError("Network access is forbidden in inference artifact tests")

        stack.enter_context(patch.object(socket.socket, "connect", deny_network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny_network))
        stack.enter_context(patch.object(socket, "create_connection", deny_network))

        import mlflow
        import mlflow.pyfunc
        import numpy as np
        import pandas as pd
        import yaml
        from matplotlib import pyplot as plt

        cls.mlflow, cls.np, cls.yaml = mlflow, np, yaml
        mlflow.set_tracking_uri(tracking)
        mlflow.set_registry_uri(tracking)
        mlflow.set_experiment("inference-artifact-regression")
        stack.callback(mlflow.end_run)
        stack.callback(plt.close, "all")
        train, register = load_source("train"), load_source("register")
        features = train.NUMERIC_COLS + train.CAT_NOM_COLS + train.CAT_ORD_COLS
        data = pd.DataFrame({name: np.arange(32, dtype=float) for name in features})
        data["store_forward"], data["vendor"] = 0, 2
        data["cost"] = np.arange(32, dtype=float) + 3.0
        cls.request = data[features].head(2)
        (cls.root / "train").mkdir()
        data.to_parquet(cls.root / "train/train.parquet", index=False)
        cls.trained = cls.root / "trained"
        with mlflow.start_run(run_name="train-export"):
            train.main(SimpleNamespace(
                train_data=str(cls.root / "train"), model_output=str(cls.trained),
                regressor__n_estimators=2, regressor__bootstrap=True,
                regressor__max_depth=3, regressor__max_features="sqrt",
                regressor__min_samples_leaf=1, regressor__min_samples_split=2,
            ))

        evaluation, output = cls.root / "evaluation", cls.root / "output"
        evaluation.mkdir()
        output.mkdir()
        (evaluation / "deploy_flag").write_text("1", encoding="utf-8")
        with patch.object(mlflow, "register_model", return_value=SimpleNamespace(version="test")) as mocked:
            with mlflow.start_run(run_name="registration-export"):
                register.main(SimpleNamespace(
                    model_name="taxi-model", model_path=str(cls.trained),
                    evaluation_output=str(evaluation), model_info_output_path=str(output),
                ))
                artifact_uri = urlparse(mlflow.get_artifact_uri("taxi-model"))
                assert artifact_uri.scheme == "file" and not artifact_uri.netloc
                cls.registered = Path(url2pathname(artifact_uri.path))
            mocked.assert_called_once()
        assert json.loads((output / "model_info.json").read_text()) == {"id": "taxi-model:test"}

    def assert_runtime_dependencies(self, model_path):
        requirements = (model_path / "requirements.txt").read_text().splitlines()
        self.assertIn(PROVIDER, requirements)
        self.assertIn(MONITORING, requirements)
        self.assertIn(CONTRIB, requirements)
        self.assertIn(SERVER, requirements)
        conda = self.yaml.safe_load((model_path / "conda.yaml").read_text())
        pip_requirements = [
            requirement
            for dependency in conda["dependencies"]
            if isinstance(dependency, dict)
            for requirement in dependency.get("pip", [])
        ]
        self.assertIn(PROVIDER, pip_requirements)
        self.assertIn(MONITORING, pip_requirements)
        self.assertIn(CONTRIB, pip_requirements)
        self.assertIn(SERVER, pip_requirements)
        self.assertIn("mlflow==2.9.2", pip_requirements)

    def test_training_export_includes_runtime_provider(self):
        self.assert_runtime_dependencies(self.trained)

    def test_registration_export_includes_runtime_provider(self):
        self.assert_runtime_dependencies(self.registered)

    def test_both_models_load_and_predict(self):
        for model_path in (self.trained, self.registered):
            with self.subTest(model=model_path.name):
                model = self.mlflow.pyfunc.load_model(str(model_path))
                predictions = model.predict(self.request)
                self.assertEqual(len(predictions), 2)
                self.assertTrue(self.np.isfinite(predictions).all())

    def test_installed_provider_satisfies_legacy_import(self):
        import pkg_resources
        self.assertEqual(pkg_resources.get_distribution("setuptools").version, "80.10.2")
        self.assertEqual(self.mlflow.__version__, "2.9.2")

    def test_generated_scorer_monitoring_import(self):
        # This is the exact import that failed in Azure's generated score script.
        # Do not instantiate a collector or enable collection in this smoke test.
        from azureml.ai.monitoring import Collector
        self.assertTrue(callable(Collector))
        self.assertTrue(callable(Collector.collect))
        self.assertEqual(version("azureml-ai-monitoring"), "1.0.0")

    def test_real_inference_server_initializes_and_scores(self):
        # A clean child process prevents earlier Azure SDK mocks from hiding an
        # import failure. The probe exercises the SDK WSGI entry without sockets.
        request_path = self.root / "http-request.json"
        request_path.write_text(json.dumps({"input_data": self.request.to_dict("split")}), encoding="utf-8")
        env = {
            name: value for name, value in os.environ.items()
            if not name.upper().startswith(("AZURE", "AML_", "MLFLOW_", "APPINSIGHTS", "APPLICATIONINSIGHTS"))
        }
        env["MLFLOW_TRACKING_URI"] = (self.root / "server-tracking").as_uri()
        env["MLFLOW_REGISTRY_URI"] = env["MLFLOW_TRACKING_URI"]
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "tests/inference_server_probe.py"),
             str(self.registered), str(request_path)],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        marker = "TAXI_INFERENCE_SMOKE="
        summaries = [line.split(marker, 1)[1] for line in result.stdout.splitlines() if marker in line]
        self.assertEqual(len(summaries), 1, result.stdout + result.stderr)
        summary = json.loads(summaries[0])
        self.assertEqual(summary["server"], "1.2.0")
        self.assertEqual(summary["contrib"], "1.57.0")
        self.assertEqual(summary["health_status"], 200)
        self.assertEqual(summary["score_status"], 200)
        self.assertEqual(summary["predictions"], len(self.request))


if __name__ == "__main__":
    unittest.main(verbosity=2)
