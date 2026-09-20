// Offline ESv3 configuration regressions: node --test tests/compute-profile.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const expected = { aml_compute_sku: 'Standard_E4s_v3', aml_compute_min_instances: '0', aml_compute_max_instances: '4', aml_batch_max_instances: '5' };
const values = environment => Object.fromEntries(Object.keys(expected).map(key => {
  const value = read(`config-infra-${environment}.yml`).match(new RegExp(`^  ${key}: ([^\\s#]+)`, 'm'))?.[1];
  assert.ok(value, `Missing ${environment} setting: ${key}`);
  return [key, value];
}));

for (const environment of ['dev', 'prod']) {
  test(`${environment}: cluster configuration has the shared ESv3 defaults`, () => {
    assert.deepEqual(values(environment), expected);
  });
}

test('Bicep and CLI use the same training SKU and scaling settings', () => {
  const main = read('infrastructure/main.bicep');
  assert.match(main, /param computeSku string = 'Standard_E4s_v3'/);
  assert.match(main, /param computeMinInstances int = 0/);
  assert.match(main, /param computeMaxInstances int = 4/);
  assert.match(main, /computeSku: computeSku/);
  assert.match(main, /minInstances: computeMinInstances/);
  assert.match(main, /maxInstances: computeMaxInstances/);
  const module = read('infrastructure/modules/aml_computecluster.bicep');
  assert.match(module, /vmSize: computeSku/);
  assert.match(module, /vmPriority: 'Dedicated'/);
  assert.match(module, /minNodeCount: minInstances/);
  assert.match(module, /maxNodeCount: maxInstances/);
  assert.match(module, /parent: workspace/);
});

test('both Bicep validation and deployment forward the same profile', () => {
  const pipeline = read('infrastructure/pipelines/bicep-ado-deploy-infra.yml');
  for (const binding of ['computeSku=$(aml_compute_sku)', 'computeMinInstances=$(aml_compute_min_instances)', 'computeMaxInstances=$(aml_compute_max_instances)']) {
    assert.equal(pipeline.split(binding).length - 1, 2, binding);
  }
});

for (const [file, maximum] of [
  ['deploy-model-training-pipeline.yml', 'aml_compute_max_instances'],
  ['deploy-batch-endpoint-pipeline.yml', 'aml_batch_max_instances'],
]) {
  test(`${file}: Dedicated tier and compile-time numeric settings`, () => {
    const pipeline = read('mlops/devops-pipelines/' + file);
    assert.ok(pipeline.includes('size: ${{ variables.aml_compute_sku }}'));
    assert.ok(pipeline.includes('min_instances: ${{ variables.aml_compute_min_instances }}'));
    assert.ok(pipeline.includes(`max_instances: \${{ variables.${maximum} }}`));
    assert.match(pipeline, /cluster_tier: dedicated/);
    assert.doesNotMatch(pipeline, /cluster_tier: low_priority/);
    assert.doesNotMatch(pipeline, /(?:min_instances|max_instances): \$\(/, 'Numeric template parameters must not receive runtime macro strings.');
  });
}

test('manual batch cluster matches the pipeline defaults', () => {
  const manual = read('mlops/azureml/deploy/batch/batch-cluster.yml');
  assert.match(manual, /^name: batch-cluster\s*$/m);
  assert.match(manual, /^size: Standard_E4s_v3\s*$/m);
  assert.match(manual, /^min_instances: 0\s*$/m);
  assert.match(manual, /^max_instances: 5\s*$/m);
  assert.match(manual, /^tier: dedicated\s*$/m);
});

test('cluster consumers still refer to the existing cluster names', () => {
  assert.match(read('mlops/azureml/train/pipeline.yml'), /default_compute: azureml:cpu-cluster/);
  assert.match(read('mlops/azureml/deploy/batch/batch-deployment.yml'), /^compute: azureml:batch-cluster\s*$/m);
  assert.match(read('infrastructure/modules/aml_computecluster.bicep'), /param computeClusterName string = 'cpu-cluster'/);
});

test('compiled ARM reflects the authoritative Bicep compute profile', () => {
  const arm = JSON.parse(read('infrastructure/main.json'));
  assert.equal(arm.parameters.computeSku.defaultValue, 'Standard_E4s_v3');
  assert.equal(arm.parameters.computeMinInstances.defaultValue, 0);
  assert.equal(arm.parameters.computeMaxInstances.defaultValue, 4);
  const module = arm.resources.find(resource => resource.type === 'Microsoft.Resources/deployments' && resource.name === 'mlwcc');
  assert.ok(module);
  assert.equal(module.properties.parameters.computeSku.value, "[parameters('computeSku')]");
  const compute = module.properties.template.resources.find(resource => resource.type === 'Microsoft.MachineLearningServices/workspaces/computes');
  assert.ok(compute);
  assert.equal(compute.properties.properties.vmSize, "[parameters('computeSku')]");
  assert.equal(compute.properties.properties.vmPriority, 'Dedicated');
  assert.equal(compute.properties.properties.scaleSettings.minNodeCount, "[parameters('minInstances')]");
  assert.equal(compute.properties.properties.scaleSettings.maxNodeCount, "[parameters('maxInstances')]");
});

test('combined defaults fit the supplied 96-core family budget with online reserve', () => {
  const deployment = read('mlops/azureml/deploy/online/online-deployment.yml');
  assert.match(deployment, /^instance_type: Standard_E2s_v3\s*$/m);
  const instances = Number(deployment.match(/^instance_count: (\d+)\s*$/m)?.[1]);
  assert.equal(instances, 1);
  const config = values('prod');
  const training = 4 * Number(config.aml_compute_max_instances);
  const batch = 4 * Number(config.aml_batch_max_instances);
  const online = 2 * Math.ceil(1.2 * instances);
  const perEnvironment = training + batch + online;
  // Screenshot-based budget fixture, not a live regional-capacity assertion.
  assert.deepEqual({ training, batch, online, perEnvironment }, { training: 16, batch: 20, online: 4, perEnvironment: 40 });
  assert.ok(perEnvironment <= 96);
  assert.ok(2 * perEnvironment <= 96, 'Dev and prod share quota when colocated.');
  assert.ok(2 * perEnvironment + 20 > 96, 'Additional workspaces cannot be assumed to fit.');
});
