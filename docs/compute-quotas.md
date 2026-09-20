# ESv3 compute profile and quota budget

The reusable taxi template uses **Azure ML ESv3 Dedicated** compute by default.
This matches the family in the supplied quota screenshot; that screenshot is not
an exact VM size, an independently verified live quota, or a capacity guarantee.

| Component | Size | vCPUs / RAM per node | Scaling | Quota budget per environment |
| --- | --- | --- | --- | --- |
| Bicep-created `cpu-cluster`, reused by training | `Standard_E4s_v3` | 4 / 32 GiB | Dedicated, 0–4 nodes | 16 cores |
| `batch-cluster` | `Standard_E4s_v3` | 4 / 32 GiB | Dedicated, 0–5 nodes | 20 cores |
| Managed online deployment | `Standard_E2s_v3` | 2 / 16 GiB | 1 instance | 4 cores including upgrade reserve |
| **Combined maximum** | | | | **40 cores** |

Bicep and training configure the **same** `cpu-cluster`, not two training clusters.
Batch's job-level `instance_count: 2` is separate from the five-node cluster ceiling.
The manual batch-cluster definition uses the same defaults as the pipeline.

The reported 96 available ESv3 cores would cover one 40-core environment, or two
such environments (80 cores), **only if** that quota is for the correct Azure ML
subscription and region and no other workloads consume it. Dev/prod and other
workspaces share the family quota when they use the same subscription and region.
Also check total regional Dedicated cores, workspace-level caps, current usage,
and overlapping online deployments. Existing quota does not guarantee VM stock.

Online quota uses `ceil(1.2 × instance_count) × vCPUs` for these SKUs. One E2s_v3
instance therefore needs four quota cores even though it normally runs two vCPUs.
A single instance is not a high-availability configuration.

## Where to change the settings

- [config-infra-dev.yml](../config-infra-dev.yml) and
  [config-infra-prod.yml](../config-infra-prod.yml): `aml_compute_sku`,
  `aml_compute_min_instances`, `aml_compute_max_instances`, and
  `aml_batch_max_instances`. Both Bicep and the training pipeline read these values.
- [Batch cluster YAML](../mlops/azureml/deploy/batch/batch-cluster.yml): standalone
  CLI alternative; keep its literals aligned if you change the defaults.
- [Online deployment YAML](../mlops/azureml/deploy/online/online-deployment.yml):
  online SKU and instance count. Pipeline variables are not automatically expanded
  inside Azure ML asset files.
- Regenerate [the compiled ARM template](../infrastructure/main.json) after Bicep edits.

Do not select `low_priority` merely because Dedicated quota is available. Azure ML
has a separate low-priority quota. Dedicated nodes can cost more than Spot nodes;
the clusters scale to zero when idle, while the online instance remains allocated.

## Existing resources and deployment prerequisites

Changing these files does **not** resize existing Azure ML clusters. The imported
shared `create-compute` helper skips a cluster that already exists, regardless of
its old SKU or tier. Before running, inspect the actual cluster size and tier.
If they differ, plan a replacement/new cluster and update all consumers together;
do not delete a running cluster or assume Bicep can change immutable properties.

The environment YAMLs currently select `eastus`; verify quota and supported SKUs
there or in the region you explicitly configure. Direct Bicep usage must also
pass the intended `location` instead of assuming its default region matches.
The local CLI could not independently query the requested subscription's quota
during this change; the 96-core value above comes from the user's screenshot.

This VM-family change does not create service connections, grant permissions, or
configure compute managed identity/ACR image-pull access. Match the configured
Azure DevOps service connection names to real connections in your project, and
ensure the compute can pull the training image before submitting jobs. No Azure
resources or training/deployment jobs are started by updating this template.

## References

- [ESv3 sizes and specifications](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/ev3-esv3-series)
- [Managed online supported SKUs](https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list)
- [Azure ML quota sharing and online reserve](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas)
