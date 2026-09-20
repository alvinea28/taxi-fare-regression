# Rerun training and managed online deployment

This reusable guide replaces the source project's environment-specific incident
notes. It contains no subscription, tenant, workspace, or endpoint credentials.

## Serving-dependency changes require a new registered model

The managed MLflow deployment builds its serving environment from the registered
model's artifacts. Changing the training environment alone, or retrying deployment
of an old model version, does not update that model's exported dependencies.

1. Publish the corrected project files to the generated project's intended branch.
2. Run the training pipeline with environment registration enabled. Keep data and
   compute creation enabled unless the matching resources already exist.
3. Confirm evaluation accepts the candidate and registration produces a **new**
   model version. A rejected candidate must not be forced through registration.
4. Inspect both the registered model's pip requirements and Conda environment for:
   - `setuptools==80.10.2`
   - `azureml-ai-monitoring==1.0.0`
   - `azureml-contrib-services==1.57.0`
   - `azureml-inference-server-http==1.2.0`
5. Run the online deployment pipeline from the same intended branch. The copied
   deployment resolves `azureml:taxi-model@latest`; confirm this is the new model.
6. Confirm deployment startup and the named-deployment smoke test succeed before
   the pipeline allocates 100% traffic.

## Configuration-only changes

A supported online VM-size change does not by itself require retraining when the
existing model already contains the corrected serving dependencies. Check the
target region's managed-online SKU support and available **Azure ML** family quota.
Do not infer online capacity from a training-compute quota or ordinary VM SKU list.

## When deployment fails

- Preserve the failed deployment long enough to inspect its logs.
- The pipeline's failure-only diagnostics collect bounded inference-server and
  storage-initializer logs without retrieving credentials or changing resources.
- Inspect the concrete Python exception or platform error; do not assume every
  liveness-probe failure is memory pressure or a quota problem.
- Local regression success does not prove hosted container startup, image pull
  permission, regional capacity, or successful Azure DevOps service authentication.
