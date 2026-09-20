// Offline regression tests: node --test tests/ml-pipeline-config.test.cjs
// Requires Azure CLI and Bash; no Azure login or npm packages are needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { after, test } = require('node:test');

const root = path.resolve(__dirname, '..');
const pipelineDir = 'mlops/devops-pipelines/';
const pipelineNames = [
  'deploy-model-training-pipeline.yml',
  'deploy-online-endpoint-pipeline.yml',
  'deploy-batch-endpoint-pipeline.yml',
];
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const guard = read(`${pipelineDir}templates/require-success.yml`);
const scriptMatch = guard.match(/^      script: \|\r?\n((?:        [^\r\n]*(?:\r?\n|$))+)/m);
assert.ok(scriptMatch, 'The success guard must have an inline Bash script.');
const guardScript = scriptMatch[1].replace(/^        /gm, '').replace(/\r\n/g, '\n');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'aml-pipeline-config-test-'));
after(() => fs.rmSync(temp, { recursive: true, force: true }));

const windows = process.platform === 'win32';
const programFiles = process.env.ProgramFiles ?? 'C:/Program Files';
const cli = windows
  ? path.join(programFiles, 'Microsoft SDKs/Azure/CLI2/python.exe')
  : 'az';
const cliPrefix = windows ? ['-IBm', 'azure.cli'] : [];
const bash = windows ? path.join(programFiles, 'Git/bin/bash.exe') : 'bash';

function isolatedEnvironment(configDirectory) {
  const env = { ...process.env };
  for (const key of Object.keys(env)) {
    if (/^AZURE_/i.test(key)) delete env[key];
  }
  delete env.BASH_ENV;
  delete env.ENV;
  return {
    ...env,
    AZURE_CONFIG_DIR: configDirectory,
    AZURE_CORE_COLLECT_TELEMETRY: 'no',
    AZURE_EXTENSION_USE_DYNAMIC_INSTALL: 'no',
  };
}

function az(args, env) {
  const result = spawnSync(cli, [...cliPrefix, ...args], {
    cwd: temp,
    env,
    encoding: 'utf8',
    timeout: 30000,
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, `Azure CLI failed: ${result.stderr}`);
  return result.stdout;
}

function environmentNames(environment) {
  // Read the existing flat naming variables without coercing a postfix such as 0001.
  const config = read(`config-infra-${environment}.yml`);
  const values = {};
  for (const key of ['namespace', 'postfix', 'environment', 'resource_group', 'aml_workspace']) {
    const match = config.match(new RegExp(`^  ${key}: ([^\\s#]+)`, 'm'));
    assert.ok(match, `Missing configuration value: ${key}`);
    values[key] = match[1];
  }
  const resolve = (value) => value.replace(/\$\(([^)]+)\)/g, (_, key) => {
    assert.ok(Object.hasOwn(values, key), `Unknown naming variable: ${key}`);
    return values[key];
  });
  return { group: resolve(values.resource_group), workspace: resolve(values.aml_workspace) };
}

test('training runtime keeps a compatible pkg_resources provider for legacy MLflow', () => {
  const conda = read('data-science/environment/train-conda.yml');
  assert.match(conda, /^  - python=3\.11\s*$/m);
  assert.match(conda, /^      - mlflow==2\.9\.2\s*$/m);
  assert.match(conda, /^      - setuptools==80\.10\.2\s*$/m);
  assert.doesNotMatch(conda, /^\s*- pkg[-_]resources\b/m);
});

for (const file of ['train.py', 'register.py']) {
  test(`inference runtime: ${file} exports generated-scorer dependencies`, () => {
    const source = read(`data-science/src/${file}`);
    const requirements = source.match(/pip_reqs\s*=\s*\[([\s\S]*?)\]/)?.[1];
    assert.ok(requirements, `${file} must declare its exported inference dependencies.`);
    assert.match(requirements, /"setuptools==80\.10\.2"/);
    assert.match(requirements, /"azureml-ai-monitoring==1\.0\.0"/);
    assert.match(requirements, /"azureml-contrib-services==1\.57\.0"/);
    assert.match(requirements, /"azureml-inference-server-http==1\.2\.0"/);
    assert.match(requirements, /"mlflow==2\.9\.2"/);
    assert.match(source, /pip_requirements\s*=\s*pip_reqs/);
    assert.doesNotMatch(requirements, /"pkg[-_]resources/);
  });
}

test('inference runtime: active batch environment retains the same compatibility pin', () => {
  const file = 'mlops/azureml/deploy/batch/batch-deployment.yml';
  const deployment = read(file);
  const conda = deployment.match(/^  conda_file:\s*(\S+)\s*$/m)?.[1];
  assert.equal(conda, 'conda.yml');
  const environment = read(path.join(path.dirname(file), conda));
  assert.match(environment, /^      - setuptools==80\.10\.2\s*$/m);
  assert.match(environment, /^      - mlflow==2\.9\.2\s*$/m);
});

test('registered training environment reads the corrected Conda definition', () => {
  const file = 'mlops/azureml/train/train-env.yml';
  const environment = read(file);
  const conda = environment.match(/^conda_file:\s*(\S+)\s*$/m);
  assert.ok(conda, 'The registered environment must reference a Conda file.');
  assert.equal(
    path.resolve(root, path.dirname(file), conda[1]),
    path.join(root, 'data-science/environment/train-conda.yml'),
  );
  assert.doesNotMatch(environment, /^version:\s*1\s*$/m);
});

test('training registers its environment by default and all steps use the latest version', () => {
  const source = read(`${pipelineDir}${pipelineNames[0]}`);
  assert.match(source, /name: skipEnvironmentRegistration\r?\n    displayName: Skip Environment Registration\r?\n    type: boolean\r?\n    default: false/);
  assert.ok(source.includes('environment_file: mlops/azureml/train/train-env.yml'));
  const job = read('mlops/azureml/train/pipeline.yml');
  assert.equal([...job.matchAll(/^    environment: azureml:taxi-train-env@latest\s*$/gm)].length, 4);
  assert.doesNotMatch(job, /environment: azureml:taxi-train-env:1\b/);
});

for (const environment of ['dev', 'prod']) {
  test(`${environment}: monitoring separation keeps Application Insights on and ADX collection opt-in`, () => {
    const config = read(`config-infra-${environment}.yml`);
    assert.match(config, /^  enable_monitoring: true\s*$/m);
    assert.match(config, /^  enable_training_data_monitoring: false\s*$/m);
  });
}

test('training monitoring separation binds the collector to its own setting', () => {
  const source = read(`${pipelineDir}${pipelineNames[0]}`);
  assert.match(source, /^              enable_monitoring: \$\(enable_training_data_monitoring\)\s*$/m);
  assert.doesNotMatch(source, /^\s*enable_monitoring: \$\(enable_monitoring\)\s*$/m);
  const job = read('mlops/azureml/train/pipeline.yml');
  assert.match(job, /^  enable_monitoring: 'false'\s*$/m);
  assert.ok(job.includes('enable_monitoring: ${{parent.inputs.enable_monitoring}}'));
});

test('infrastructure monitoring separation preserves the required Application Insights flag', () => {
  const source = read('infrastructure/pipelines/bicep-ado-deploy-infra.yml');
  assert.equal([...source.matchAll(/enableMonitoring=\$\(enable_monitoring\)/g)].length, 2);
  assert.doesNotMatch(source, /enable_training_data_monitoring/);
});

for (const file of pipelineNames) {
  test(`${file}: job-scoped defaults and credential isolation`, () => {
    const source = read(`${pipelineDir}${file}`);
    assert.match(source, /^        variables:\r?\n(?:          #[^\r\n]*\r?\n)*          AZURE_DEFAULTS_GROUP: \$\(resource_group\)\r?\n          AZURE_DEFAULTS_WORKSPACE: \$\(aml_workspace\)/m);
    assert.doesNotMatch(source, /useGlobalConfig:\s*true|AZURE_CONFIG_DIR\s*:/);
    assert.doesNotMatch(source, /- template:.*install-az-cli\.yml@/);
    assert.match(source, /- template:.*install-aml-cli\.yml@mlops-templates/);
  });
}

test('all required training steps are immediately guarded in the same skip branch', () => {
  const source = read(`${pipelineDir}${pipelineNames[0]}`);
  const references = [...source.matchAll(/^( +)- template: ([^\r\n]+)/gm)]
    .map((match) => ({ indent: match[1].length, template: match[2].trim() }));
  const required = ['register-environment.yml', 'create-compute.yml', 'register-data.yml', 'run-pipeline.yml'];
  for (const file of required) {
    const index = references.findIndex((entry) => entry.template.endsWith(`${file}@mlops-templates`));
    assert.ok(index >= 0, `Missing required training template: ${file}`);
    assert.deepEqual(references[index + 1], {
      indent: references[index].indent,
      template: 'templates/require-success.yml',
    });
  }
  assert.equal(references.filter((entry) => entry.template === 'templates/require-success.yml').length, 4);
  for (const name of ['skipEnvironmentRegistration', 'skipComputeCreation', 'skipDataRegistration']) {
    assert.ok(source.includes(`if ne(parameters.${name}, true)`), `Skip option changed: ${name}`);
  }
  assert.match(guard, /condition: succeeded\(\)/);
  assert.match(guard, /continueOnError: false/);
});

test('online deployment uses the ESv3 profile and includes upgrade quota reserve', () => {
  const deployment = read('mlops/azureml/deploy/online/online-deployment.yml');
  const sku = deployment.match(/^instance_type:\s*(\S+)\s*$/m)?.[1];
  const instances = Number(deployment.match(/^instance_count:\s*(\d+)\s*$/m)?.[1]);
  assert.equal(sku, 'Standard_E2s_v3');
  assert.equal(instances, 1, 'Additional replicas require a new live quota check.');
  // The user supplied an ESv3 Dedicated quota screenshot showing 96 available cores.
  // This is a budget fixture, not an independent live quota/capacity guarantee.
  const coresPerInstance = 2;
  const requiredCores = Math.ceil(1.2 * instances) * coresPerInstance;
  const observedUsage = 0;
  const observedLimit = 96;
  assert.ok(
    observedUsage + requiredCores <= observedLimit,
    `${sku} needs ${requiredCores} quota cores; the supplied ESv3 budget was ${observedLimit}.`,
  );
  assert.equal(requiredCores, 4);
  assert.ok(Math.ceil(1.2 * 49) * coresPerInstance > observedLimit,
    'Large replica counts must not be assumed to fit the reported family quota.');
  assert.match(deployment, /^model: azureml:taxi-model@latest\s*$/m);
  assert.doesNotMatch(deployment, /^code_configuration:|^environment:/m);
});

test('online quota repair guards endpoint creation, deployment, and smoke testing', () => {
  const source = read(`${pipelineDir}${pipelineNames[1]}`);
  const references = [...source.matchAll(/^( +)- template: ([^\r\n]+)/gm)]
    .map((match) => ({ indent: match[1].length, template: match[2].trim() }));
  for (const file of ['create-endpoint.yml', 'create-deployment.yml', 'test-deployment.yml']) {
    const index = references.findIndex((entry) => entry.template.endsWith(`${file}@mlops-templates`));
    assert.ok(index >= 0, `Missing online operation: ${file}`);
    assert.deepEqual(references[index + 1], {
      indent: references[index].indent,
      template: 'templates/require-success.yml',
    }, `${file} must fail closed before the next operation.`);
  }
  assert.equal(references.filter((entry) => entry.template === 'templates/require-success.yml').length, 3);
});

test('online quota repair tests the named deployment before allocating traffic', () => {
  const source = read(`${pipelineDir}${pipelineNames[1]}`);
  const creation = source.indexOf('create-deployment.yml@mlops-templates');
  const testing = source.indexOf('test-deployment.yml@mlops-templates');
  const traffic = source.indexOf('allocate-traffic.yml@mlops-templates');
  assert.ok(creation >= 0 && testing > creation && traffic > testing,
    'Required order: create deployment, test deployment, allocate traffic.');
  const testBlock = source.slice(testing, traffic);
  assert.match(testBlock, /deployment_name: taxi-online-dp/);
  assert.match(testBlock, /request_type: json/);
  assert.match(testBlock, /template: templates\/require-success\.yml/);
  assert.match(source.slice(traffic), /traffic_allocation: taxi-online-dp=100/);
});

test('online quota repair smoke request uses the trained named-column contract', () => {
  const source = read(`${pipelineDir}${pipelineNames[1]}`);
  const requestPath = source.match(/^\s+sample_request:\s*(\S+)\s*$/m)?.[1];
  assert.equal(requestPath, 'data/test-request.json');
  const input = JSON.parse(read(requestPath)).input_data;
  assert.ok(input && !Array.isArray(input), 'Use a named DataFrame request, not positional raw-CSV values.');
  const training = read('data-science/src/train.py');
  const features = ['NUMERIC_COLS', 'CAT_NOM_COLS', 'CAT_ORD_COLS'].flatMap((name) => {
    const match = training.match(new RegExp(`^${name} = \\[([\\s\\S]*?)\\]`, 'm'));
    assert.ok(match, `Missing training feature declaration: ${name}`);
    return [...match[1].matchAll(/"([^"\r\n]+)"/g)].map((column) => column[1]);
  });
  assert.equal(features.length, 20);
  assert.deepEqual(input.columns, features);
  assert.equal(input.data.length, 2);
  assert.equal(input.index.length, input.data.length);
  for (const row of input.data) {
    assert.equal(row.length, features.length);
    assert.ok(row.every(Number.isFinite));
    assert.ok([0, 1].includes(row[features.indexOf('store_forward')]));
    assert.ok([1, 2].includes(row[features.indexOf('vendor')]));
  }
});

function onlineDiagnostics() {
  const source = read(`${pipelineDir}${pipelineNames[1]}`);
  const marker = source.indexOf('displayName: Collect failed online deployment diagnostics');
  assert.ok(marker >= 0, 'Missing failed-deployment diagnostics task.');
  const start = source.lastIndexOf('          - task: AzureCLI@2', marker);
  const end = source.indexOf('\n          - ', marker);
  const block = source.slice(start, end < 0 ? undefined : end);
  const script = block.match(/^              inlineScript: \|\r?\n((?:                [^\r\n]*(?:\r?\n|$))+)/m);
  assert.ok(script, 'Diagnostics must use an inline Bash script.');
  return { source, marker, block, script: script[1].replace(/^                /gm, '').replace(/\r\n/g, '\n') };
}

test('online diagnostics run after the hard failure guard and before smoke testing', () => {
  const { source, marker, block } = onlineDiagnostics();
  assert.ok(source.indexOf('stepName: online deployment creation') < marker);
  assert.ok(marker < source.indexOf('test-deployment.yml@mlops-templates'));
  assert.match(block, /^            condition: failed\(\)\s*$/m);
  assert.match(block, /^            continueOnError: true\s*$/m);
  assert.match(block, /^            timeoutInMinutes: 5\s*$/m);
  assert.match(block, /azureSubscription: \$\(ado_service_connection_rg\)/);
  assert.match(block, /AML_RESOURCE_GROUP: \$\(resource_group\)/);
  assert.match(block, /AML_WORKSPACE: \$\(aml_workspace\)/);
  assert.match(block, /AML_ENDPOINT: \$\(endpoint_name\)/);
  assert.match(block, /AML_DEPLOYMENT: taxi-online-dp/);
});

test('online diagnostics use bounded read-only commands without debug or secret retrieval', () => {
  const { script } = onlineDiagnostics();
  assert.match(script, /az ml online-deployment show/);
  assert.match(script, /az ml online-deployment get-logs/);
  assert.match(script, /for container in inference-server storage-initializer/);
  assert.match(script, /--lines 1000/);
  assert.doesNotMatch(script, /--debug|set\s+-[^\r\n]*x|\bget-credentials\b|\b(?:create|update|delete|invoke|login)\b/);
});

test('online diagnostics Bash syntax is valid', () => {
  const result = spawnSync(bash, ['--noprofile', '--norc', '-n'], {
    input: onlineDiagnostics().script,
    env: isolatedEnvironment(path.join(temp, 'diagnostic-syntax')),
    encoding: 'utf8',
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
});

for (const failure of ['none', 'inference-server', 'all']) {
  test(`online diagnostics attempt both containers when failure is ${failure}`, () => {
    const directory = path.join(temp, `diagnostic-${failure}`);
    fs.mkdirSync(directory);
    const fakeAz = `
az() {
  printf '%s\\t' "$@" >> calls.txt
  printf '\\n' >> calls.txt
  local container=''
  local previous=''
  local action="$3"
  for argument in "$@"; do
    if [[ "$previous" == '--container' ]]; then container="$argument"; fi
    previous="$argument"
  done
  if [[ "$TEST_FAILURE" == 'all' || "$TEST_FAILURE" == "$container" ]]; then
    echo 'Fixture: container log unavailable' >&2
    return 1
  fi
  echo "Fixture result: $action $container"
}
`;
    const result = spawnSync(bash, ['--noprofile', '--norc', '-s'], {
      input: `${fakeAz}\n${onlineDiagnostics().script}\nprintf 'JOB_STATUS=%s\\n' "$AGENT_JOBSTATUS"\n`,
      cwd: directory,
      env: {
        ...isolatedEnvironment(path.join(directory, 'az')),
        AML_RESOURCE_GROUP: 'rg-fixture', AML_WORKSPACE: 'workspace-fixture',
        AML_ENDPOINT: 'endpoint-fixture', AML_DEPLOYMENT: 'taxi-online-dp',
        AGENT_JOBSTATUS: 'Failed', TEST_FAILURE: failure,
      },
      encoding: 'utf8', timeout: 10000,
    });
    assert.ifError(result.error);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /JOB_STATUS=Failed/);
    const calls = fs.readFileSync(path.join(directory, 'calls.txt'), 'utf8').trim().split(/\r?\n/).map(line => line.trim().split('\t'));
    assert.equal(calls.length, 3, 'Metadata plus both containers must always be attempted.');
    assert.equal(calls[0][2], 'show');
    for (const args of calls) {
      for (const [flag, value] of [['--resource-group', 'rg-fixture'], ['--workspace-name', 'workspace-fixture'], ['--endpoint-name', 'endpoint-fixture'], ['--name', 'taxi-online-dp']]) {
        assert.equal(args[args.indexOf(flag) + 1], value);
      }
    }
    assert.equal(calls[1][calls[1].indexOf('--container') + 1], 'inference-server');
    assert.equal(calls[2][calls[2].indexOf('--container') + 1], 'storage-initializer');
    if (failure !== 'none') assert.match(result.stdout, /##vso\[task\.logissue type=warning\]/);
  });
}

test('guard Bash syntax is valid', () => {
  const result = spawnSync(bash, ['--noprofile', '--norc', '-n'], {
    input: guardScript,
    env: isolatedEnvironment(path.join(temp, 'syntax')),
    encoding: 'utf8',
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
});

for (const status of ['Succeeded', 'SucceededWithIssues', 'Failed', 'Canceled', 'Skipped', 'Unexpected', '']) {
  test(`guard handles job status: ${status || '(unset)'}`, () => {
    const env = isolatedEnvironment(path.join(temp, 'guard'));
    delete env.AGENT_JOBSTATUS;
    if (status) env.AGENT_JOBSTATUS = status;
    const result = spawnSync(bash, ['--noprofile', '--norc', '-s'], {
      input: guardScript,
      env,
      encoding: 'utf8',
      timeout: 10000,
    });
    assert.ifError(result.error);
    assert.equal(result.status, status === 'Succeeded' ? 0 : 1, result.stderr);
    if (status !== 'Succeeded') assert.match(result.stdout, /##vso\[task\.logissue type=error\]/);
  });
}

for (const environment of ['dev', 'prod']) {
  test(`${environment}: real CLI resolves defaults in two fresh task config directories`, () => {
    const names = environmentNames(environment);
    for (const task of ['register-environment', 'submit-job']) {
      const directory = path.join(temp, environment, task);
      assert.equal(fs.existsSync(directory), false, 'Each task must start with fresh configuration.');
      const env = {
        ...isolatedEnvironment(directory),
        AZURE_DEFAULTS_GROUP: names.group,
        AZURE_DEFAULTS_WORKSPACE: names.workspace,
      };
      const defaults = JSON.parse(az(['config', 'get', 'defaults', '--output', 'json', '--only-show-errors'], env));
      for (const key of ['group', 'workspace']) {
        const entry = defaults.find((item) => item.name === key);
        assert.ok(entry, `Missing CLI default: ${key}`);
        assert.equal(entry.value, names[key]);
        assert.equal(entry.source, `AZURE_DEFAULTS_${key.toUpperCase()}`);
      }
    }
  });
}