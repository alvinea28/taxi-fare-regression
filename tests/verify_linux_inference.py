"""Reproduce the old server failure, then run real artifact/WSGI checks on Linux.

Run with a Linux Python 3 standard library (including WSL Ubuntu). A checksummed
uv executable, managed Python 3.11, packages, and caches live only in a temporary
directory. Public package downloads need network access; the actual inference
tests block networking and never contact Azure. No system packages are changed.
"""

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.9.26"
SERVER = "azureml-inference-server-http==1.2.0"
CONTRIB = "azureml-contrib-services==1.57.0"


def export_requirements(name):
    tree = ast.parse((ROOT / "data-science/src" / name).read_text(encoding="utf-8"))
    values = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pip_reqs" for target in node.targets)
    ]
    assert len(values) == 1, f"Expected one exported requirements list in {name}"
    return values[0]


def download_uv(directory):
    with urllib.request.urlopen(f"https://pypi.org/pypi/uv/{UV_VERSION}/json", timeout=60) as response:
        metadata = json.load(response)
    wheels = [
        item for item in metadata["urls"]
        if item["packagetype"] == "bdist_wheel"
        and "manylinux_2_17_x86_64" in item["filename"]
        and not item["yanked"]
    ]
    assert len(wheels) == 1, "Expected the pinned Linux x86_64 uv wheel"
    wheel = wheels[0]
    with urllib.request.urlopen(wheel["url"], timeout=120) as response:
        contents = response.read()
    digest = hashlib.sha256(contents).hexdigest()
    assert digest == wheel["digests"]["sha256"], "uv download checksum mismatch"
    with zipfile.ZipFile(io.BytesIO(contents)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/") and Path(name).name == "uv"]
        assert len(names) == 1, "Expected exactly one uv executable"
        binary = directory / "uv"
        binary.write_bytes(archive.read(names[0]))
    binary.chmod(0o700)
    print(json.dumps({"uv": UV_VERSION, "wheel_sha256": digest}), flush=True)
    return binary


def main():
    assert platform.system() == "Linux" and platform.machine() == "x86_64", "Run in Linux x86_64, e.g. WSL Ubuntu"
    requirements = export_requirements("register.py")
    assert requirements == export_requirements("train.py"), "Training and registration runtime lists diverged"
    assert SERVER in requirements and CONTRIB in requirements, "Missing serving repair"
    with tempfile.TemporaryDirectory(prefix="taxi-linux-inference-") as temporary:
        directory = Path(temporary)
        uv = download_uv(directory)
        env = {
            name: value for name, value in os.environ.items()
            if not name.upper().startswith(("AZURE", "AML_", "MLFLOW_", "UV_", "APPINSIGHTS", "APPLICATIONINSIGHTS"))
            and name not in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME")
        }
        env.update({
            "UV_CACHE_DIR": str(directory / "cache"),
            "UV_PYTHON_INSTALL_DIR": str(directory / "python"),
            "UV_PYTHON_PREFERENCE": "only-managed",
            "UV_NO_PROGRESS": "1",
            "UV_LINK_MODE": "copy",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(directory / "matplotlib"),
        })

        def execute(arguments, *, capture=False):
            return subprocess.run(
                list(map(str, arguments)), cwd=ROOT, env=env,
                check=True, timeout=600, capture_output=capture, text=True,
            )

        venv = directory / "baseline-env"
        # Do not seed unrelated latest build tools: wheel 0.48 requires a
        # packaging version incompatible with this legacy MLflow runtime.
        execute([uv, "venv", "--python", "3.11", venv])
        python = venv / "bin/python"
        execute([python, "-c", "import platform; print('Linux test Python: ' + platform.python_version())"])
        baseline = [requirement for requirement in requirements if requirement not in (SERVER, CONTRIB)]
        execute([uv, "pip", "install", "--python", python, *baseline,
             "pip==26.2.1", "azureml-inference-server-http==1.0.0"])
        broken = subprocess.run(
            [str(python), "-B", "-c", "import azureml_inference_server_http.server"],
            cwd=directory, env=env, capture_output=True, text=True, timeout=30,
        )
        assert broken.returncode != 0, "The original startup failure was not reproduced"
        assert "ModuleNotFoundError: No module named 'azureml.contrib'" in broken.stderr, broken.stderr
        print("BASELINE_REPRODUCED=ModuleNotFoundError: No module named 'azureml.contrib'", flush=True)

        # Validate a fresh install, not a repaired environment retaining older
        # transitive packages (for example Flask 2.2 from the baseline server).
        venv = directory / "fixed-env"
        execute([uv, "venv", "--python", "3.11", venv])
        python = venv / "bin/python"
        execute([uv, "pip", "install", "--python", python, *requirements,
             "pip==26.2.1", "matplotlib==3.9.2", "PyYAML==6.0.3"])
        execute([uv, "pip", "check", "--python", python])
        execute([python, "-B", "-W", "ignore::DeprecationWarning", "-m", "unittest", "discover",
                 "-s", "tests", "-p", "test_inference_artifacts.py", "-v"])
        print("LINUX_INFERENCE_VERIFIED=artifact manifests, model loads, native WSGI initialization, health and scoring", flush=True)


if __name__ == "__main__":
    main()
