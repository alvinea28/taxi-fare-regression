"""Offline regressions for missing run history and model-promotion safety.

Requires the project's ML libraries plus azure-core. Azure registry/run APIs are
mocked, while the fallback loads a real locally serialized model. No cloud calls.
"""

from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
MISSING_RUN = "595f6bf4-cd30-4f1b-abac-fcc83b170d40"


def load_source(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data-science/src" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stack = ExitStack()
        cls.addClassCleanup(stack.close)
        cls.root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="taxi-comparison-test-")))
        previous = Path.cwd()
        stack.callback(os.chdir, previous)
        os.chdir(cls.root)
        stack.enter_context(patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": (cls.root / "tracking").as_uri(),
            "MLFLOW_REGISTRY_URI": (cls.root / "tracking").as_uri(),
            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(cls.root / "matplotlib"),
        }))

        def no_network(*args, **kwargs):
            raise AssertionError("Network access is forbidden in comparison tests")

        for name in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, name, no_network))
        stack.enter_context(patch.object(socket, "create_connection", no_network))

        import mlflow
        import mlflow.sklearn
        import numpy as np
        import pandas as pd
        from azure.core.exceptions import ResourceNotFoundError
        from matplotlib import pyplot as plt
        from sklearn.linear_model import LinearRegression

        cls.mlflow, cls.np, cls.plt = mlflow, np, plt
        cls.missing_type = ResourceNotFoundError
        cls.evaluate, cls.register = load_source("evaluate"), load_source("register")
        cls.inputs = pd.DataFrame({"x": np.arange(6, dtype=float)})
        cls.truth = np.arange(6, dtype=float) * 2 + 3
        cls.saved = cls.root / "registered-copy"
        mlflow.sklearn.save_model(
            LinearRegression().fit(cls.inputs, cls.truth), str(cls.saved),
            pip_requirements=["mlflow==2.9.2", "scikit-learn==1.5.2", "setuptools==80.10.2"],
        )
        cls.real_load = staticmethod(mlflow.pyfunc.load_model)
        stack.callback(plt.close, "all")

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.output = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=self.root)))
        self.registry = Mock()
        self.registry.search_model_versions.return_value = [SimpleNamespace(version="1", run_id=MISSING_RUN)]
        stack.enter_context(patch.object(self.evaluate, "MlflowClient", return_value=self.registry))
        stack.enter_context(patch.object(self.mlflow, "log_metric"))
        stack.enter_context(patch.object(self.mlflow, "log_artifact"))
        self.addCleanup(self.plt.close, "all")
        self.workspace = object()
        self.run = Mock()
        self.run.get_context.return_value = SimpleNamespace(experiment=SimpleNamespace(workspace=self.workspace))
        self.model_api = Mock()
        self.downloads = []

        def download(target_dir, exist_ok=False):
            self.assertFalse(exist_ok)
            self.downloads.append(Path(target_dir))
            return str(shutil.copytree(self.saved, Path(target_dir) / "taxi-model"))

        self.model_api.return_value.download.side_effect = download
        azureml, core = ModuleType("azureml"), ModuleType("azureml.core")
        core.Model, core.Run = self.model_api, self.run
        azureml.core = core
        stack.enter_context(patch.dict(sys.modules, {"azureml": azureml, "azureml.core": core}))

    def missing_run(self, message=None):
        error = self.missing_type(message=message or f"(UserError) Run {MISSING_RUN} was not found")
        error.status_code = 404
        return error

    def promote(self, score=0.5):
        return self.evaluate.model_promotion(
            "taxi-model", str(self.output), self.inputs, self.truth, self.truth, score,
        )

    def missing_run_loader(self, model_uri):
        if model_uri == "models:/taxi-model/1":
            raise self.missing_run()
        self.assertTrue(Path(model_uri).is_dir(), "Keep artifacts available through prediction.")
        return self.real_load(model_uri)

    def test_missing_origin_uses_registered_copy_without_changing_decision(self):
        with patch.object(self.mlflow.pyfunc, "load_model", side_effect=self.missing_run_loader):
            predictions, flag = self.promote(score=0.5)
        self.assertEqual(flag, 0, "A worse candidate must still be rejected.")
        self.np.testing.assert_allclose(predictions["taxi-model:1"], self.truth)
        self.model_api.assert_called_once_with(self.workspace, name="taxi-model", version=1, expand=False)
        self.run.get_context.assert_called_once_with(allow_offline=False)
        self.assertTrue(self.downloads)
        self.assertTrue(all(not path.exists() for path in self.downloads))

    def test_equal_candidate_can_pass_after_registered_copy_recovery(self):
        with patch.object(self.mlflow.pyfunc, "load_model", side_effect=self.missing_run_loader):
            _, flag = self.promote(score=1.0)
        self.assertEqual(flag, 1)

    def test_normal_mlflow_load_does_not_use_azure_fallback(self):
        model = self.real_load(str(self.saved))
        with patch.object(self.mlflow.pyfunc, "load_model", return_value=model):
            _, flag = self.promote(score=0.5)
        self.assertEqual(flag, 0)
        self.model_api.assert_not_called()
        self.run.get_context.assert_not_called()

    def test_first_model_policy_is_unchanged(self):
        self.registry.search_model_versions.return_value = []
        with patch.object(self.mlflow.pyfunc, "load_model") as load:
            _, flag = self.promote(score=0.5)
        self.assertEqual(flag, 1)
        load.assert_not_called()
        self.model_api.assert_not_called()

    def test_unrelated_missing_resource_is_not_treated_as_missing_run(self):
        with patch.object(self.mlflow.pyfunc, "load_model", side_effect=self.missing_run("Model was not found")):
            with self.assertRaises(self.missing_type):
                self.promote()
        self.model_api.assert_not_called()
        self.assertFalse((self.output / "deploy_flag").exists())

    def test_non_404_status_is_not_recovered_even_with_matching_message(self):
        error = self.missing_run()
        error.status_code = 403
        with patch.object(self.mlflow.pyfunc, "load_model", side_effect=error):
            with self.assertRaises(self.missing_type):
                self.promote()
        self.model_api.assert_not_called()

    def test_corrupt_registered_copy_fails_and_temporary_files_are_removed(self):
        def load(model_uri):
            if model_uri == "models:/taxi-model/1":
                raise self.missing_run()
            raise ValueError("corrupt registered copy")

        with patch.object(self.mlflow.pyfunc, "load_model", side_effect=load):
            with self.assertRaisesRegex(ValueError, "corrupt registered copy"):
                self.promote()
        self.assertFalse((self.output / "deploy_flag").exists())
        self.assertTrue(self.downloads)
        self.assertTrue(all(not path.exists() for path in self.downloads))

    def test_authentication_network_and_deserialization_errors_propagate(self):
        from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
        for error in (ClientAuthenticationError("denied"), HttpResponseError("forbidden"), TimeoutError("network"), ValueError("invalid model")):
            with self.subTest(error=type(error).__name__):
                with patch.object(self.mlflow.pyfunc, "load_model", side_effect=error):
                    with self.assertRaises(type(error)):
                        self.promote()
        self.model_api.assert_not_called()
        self.assertFalse((self.output / "deploy_flag").exists())

    def test_unavailable_registered_copy_still_fails_closed(self):
        self.model_api.return_value.download.side_effect = PermissionError("registered asset unavailable")
        with patch.object(self.mlflow.pyfunc, "load_model", side_effect=self.missing_run_loader):
            with self.assertRaisesRegex(PermissionError, "registered asset unavailable"):
                self.promote()
        self.assertFalse((self.output / "deploy_flag").exists())

    def test_prediction_failure_is_not_retried_or_skipped(self):
        model = Mock()
        model.predict.side_effect = ValueError("feature mismatch")
        with patch.object(self.mlflow.pyfunc, "load_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "feature mismatch"):
                self.promote()
        self.model_api.assert_not_called()

    def test_registry_search_failure_is_not_an_empty_registry(self):
        self.registry.search_model_versions.side_effect = PermissionError("registry denied")
        with self.assertRaisesRegex(PermissionError, "registry denied"):
            self.promote()
        self.assertFalse((self.output / "deploy_flag").exists())

    def test_rejected_candidate_never_registers(self):
        (self.output / "deploy_flag").write_text("0", encoding="utf-8")
        with patch.object(self.mlflow.sklearn, "load_model") as load, patch.object(self.mlflow.sklearn, "log_model"), patch.object(self.mlflow, "active_run", return_value=SimpleNamespace(info=SimpleNamespace(run_id="local-test"))), patch.object(self.mlflow, "register_model", return_value=SimpleNamespace(version="2")) as register:
            self.register.main(SimpleNamespace(
                evaluation_output=str(self.output), model_path=str(self.saved),
                model_name="taxi-model", model_info_output_path=str(self.output),
            ))
        load.assert_not_called()
        register.assert_not_called()
        self.assertFalse((self.output / "model_info.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
