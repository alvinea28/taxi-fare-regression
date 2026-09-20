# Taxi fare regression - repaired MLOps project template

Standalone Azure Machine Learning **classical / AML CLI v2 / Bicep** project,
copied from the working taxi project at commit
`7223bcab85b9896474258be04dce0e7b08bff6a5` on 2026-09-20.

This repository is the taxi-template source for
[alvinea28/own-ml-ops-v2](https://github.com/alvinea28/own-ml-ops-v2/tree/main).
The original working project was not changed. This is a clean snapshot, not a
mirror of its Git history or a copy of its deployed Azure resources.

## Included fixes

- Working data preparation, training, evaluation, registration, batch and online
  pipeline files, Bicep infrastructure, and sample data.
- Python 3.11 / MLflow 2.9.2 compatibility, including the exported serving pins
  for setuptools, Azure ML monitoring, contrib services, and the inference server.
- Azure CLI task-isolation defaults and explicit failure guards, including online
  deployment testing **before** assigning production traffic.
- Application Insights kept separate from optional ADX training-data collection.
- Registered-model comparison recovery when the originating run is missing,
  while preserving rejection decisions and failing closed on other errors.
- Failed online-deployment diagnostics and the existing regression tests.

Model code, serving dependencies, and sample data are preserved from the source
snapshot. The environment configurations are generalized and compute defaults now
use the [ESv3 Dedicated profile](docs/compute-quotas.md). Documentation, licensing,
ignore rules, and regression checks are maintained separately. Personal work
reports, old Git history, local environments, credentials, and trained models are excluded.

## Use with the accelerator

Follow the [personal accelerator setup guide](https://github.com/alvinea28/own-ml-ops-v2/blob/main/docs/TAXI-TEMPLATE.md).

1. Import this repository into your Azure DevOps project as
   **taxi-fare-regression-template**, with `main` as its default branch. This
   distinct name prevents confusion with a generated application repository.
2. Import the personal accelerator's **main** branch and import
   [Azure/mlops-templates](https://github.com/Azure/mlops-templates) as
   **mlops-templates**. The shared helper repository is still required.
3. Create a **new** target Azure Repos repository (an initial README is fine).
   Do not select your existing working taxi repository as the target.
4. Run the normal taxi initializer with the Azure DevOps project, empty destination,
  and imported source repository names. Classical ML, CLI v2, and Bicep are automatic.
5. Configure the generated project before running its deployment pipelines.

Repository visibility is managed separately on GitHub. Readers need access and
must authenticate private Azure Repos imports when required. Never put a PAT
or other credential in YAML. Importing is a one-time copy: later GitHub changes
do not automatically update an existing Azure Repos import or generated project.

## Configure before deployment

- Edit [config-infra-dev.yml](config-infra-dev.yml) and
  [config-infra-prod.yml](config-infra-prod.yml): choose a unique short `postfix`,
  region, and your own workload-federated Azure service connection names. `demo01`
  is an example, not a deploy-ready globally unique name.
- The `main` application branch selects production configuration; other branch
  names select development configuration. The accelerator itself uses `main`.
- Create/authorize the required Azure DevOps environment and service connections
  in your organization. Grant only the Azure permissions required by the pipelines.
- Check regional Azure ML compute quota, image access, and managed-online SKU
  support. The online setting is one `Standard_E2s_v3` instance, not a
  high-availability deployment or a guarantee of capacity in another subscription.
- **Important infrastructure prerequisite:** the copied compute-cluster Bicep
  module does not declare a managed identity or an ACR pull role assignment.
  With ACR admin access disabled, configure a supported compute identity with
  registry-scoped `AcrPull` before training. Working Azure-side identity/RBAC state
  is not transported by copying a repository. The infrastructure/training cluster
  uses `Standard_E4s_v3`, Dedicated, 0–4 nodes. Batch uses the same size at 0–5 nodes.
  These and online deployment share the ESv3 budget; see [the 40-core calculation](docs/compute-quotas.md).
- The shared helper baseline inspected for this snapshot is recorded in
  [template-manifest.json](template-manifest.json). Pipeline helpers still follow
  the imported **mlops-templates/main** branch, so review changes before updating it.

See [docs/online-deployment-rerun.md](docs/online-deployment-rerun.md) when changing
model serving dependencies. Copying these files does not deploy or repair Azure
resources, carry credentials, or register a model in a new workspace.

## Local regression checks

Use Python **3.11**, Node.js **18+**, Bash, and an installed Azure CLI. The test
suite uses isolated temporary CLI configuration; it needs no Azure login.

In a separate Python environment, install [requirements-test.txt](requirements-test.txt),
then run:

```text
python -m pip install -r requirements-test.txt
node --test tests/ml-pipeline-config.test.cjs
node --test tests/compute-profile.test.cjs
python -B -m unittest discover -s tests -p "test_*.py" -v
```

The original suites contain 34 Node/Bash/CLI cases and 18 Python cases, plus the
new compute-profile regressions. Python tests
use synthetic data, local model artifacts, mocked Azure APIs, and blocked network
access. They include actual Azure inference-server WSGI initialization and scoring;
they are not an Azure-hosted end-to-end deployment test.

On Linux x86-64, [tests/verify_linux_inference.py](tests/verify_linux_inference.py)
can also create isolated environments to reproduce the previous failure and
verify the fixed serving dependencies. Its bootstrap downloads require network
access, but it does not deploy Azure resources.

## Provenance and license

Derived from [Azure/mlops-v2](https://github.com/Azure/mlops-v2) and
[Azure/mlops-project-template](https://github.com/Azure/mlops-project-template),
with repairs from the working taxi project. Original Microsoft copyright notices
and the [MIT license](LICENSE) are retained. This is a personal derivative, not an
official Microsoft release. Taxi sample data is preserved from the accelerator.
