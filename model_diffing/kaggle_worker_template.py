"""Private Kaggle script staged by model_diffing.kaggle_bridge.

The bridge replaces the request marker before uploading this file. Do not run
the template directly and do not commit a rendered worker containing a private
target mapping.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REQUEST_B64 = "__MODEL_DIFFING_REQUEST_B64__"
RESPONSE_PATH = Path("/kaggle/working/model_diffing_response.json")


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _prepare_repository(spec: dict[str, object]) -> Path:
    worker = spec["worker"]
    if not isinstance(worker, dict):
        raise ValueError("worker config is invalid")
    repository = worker["repository"]
    if not isinstance(repository, dict):
        raise ValueError("worker.repository config is invalid")
    destination = Path(str(worker["repo_root"]))
    mode = repository.get("mode")
    if mode == "path":
        if not destination.is_dir():
            raise FileNotFoundError(f"Attached repository path does not exist: {destination}")
        return destination
    if mode != "git":
        raise ValueError("worker.repository.mode must be git or path")

    working_root = Path("/kaggle/working").resolve()
    resolved_destination = destination.resolve()
    if resolved_destination == working_root or working_root not in resolved_destination.parents:
        raise ValueError("git repository destination must be a child of /kaggle/working")
    if destination.exists():
        shutil.rmtree(destination)
    git_env = os.environ.copy()
    token = None
    try:
        secret_name = repository.get("github_token_secret")
        if secret_name:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret(str(secret_name))
            if token:
                basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
                git_env.update(
                    {
                        "GIT_CONFIG_COUNT": "1",
                        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
                    }
                )
        _run(["git", "clone", "--filter=blob:none", str(repository["url"]), str(destination)], env=git_env)
        reference = repository.get("ref")
        if reference:
            _run(["git", "checkout", "--detach", str(reference)], cwd=destination, env=git_env)
    finally:
        token = None
        git_env.clear()
    return destination


def main() -> None:
    request = json.loads(base64.b64decode(REQUEST_B64).decode("utf-8"))
    repo = _prepare_repository(request)
    worker = request["worker"]
    if worker.get("install_requirements", True):
        requirements = repo / str(worker.get("requirements_path", "requirements-kaggle.txt"))
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
        _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])

    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ["HF_HOME"] = str(worker.get("hf_home", "/kaggle/working/hf_cache"))

    from model_diffing.inference import SharedAdapterTargetPair
    from model_diffing.registry import load_target_pair
    from safety_training.config import load_yaml

    artifact_root = Path(str(worker["artifact_root"]))
    target_a, target_b = load_target_pair(artifact_root, request["target_a"], request["target_b"])
    base_config = load_yaml(repo / str(worker.get("base_config", "configs/base.yaml")))
    backend = SharedAdapterTargetPair(
        target_a,
        target_b,
        base_config,
        hf_home=str(worker.get("hf_home", "/kaggle/working/hf_cache")),
    )
    results = backend.send_messages(
        request["prompts"],
        int(request["samples_per_model"]),
        turn=int(request["turn"]),
        sampling=request["sampling"],
    )
    response = {
        "schema_version": "1.0",
        "status": "ok",
        "request_id": request["request_id"],
        "batch_id": request["batch_id"],
        "results": results,
    }
    RESPONSE_PATH.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
