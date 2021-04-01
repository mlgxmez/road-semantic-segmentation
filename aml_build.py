# This script instantiates a model un uploads the model to Blob Storage
# A cheap machine with CPU
from azureml.core import Environment, Workspace, ScriptRunConfig, Experiment
from utils.agent import createAmlCompute

EXPERIMENT_NAME = 'road-segmentation-build'
ENV_NAME = 'AzureML-TensorFlow-2.3-CPU'
CLUSTER_NAME = 'CPU-cluster'
VM_SIZE = 'Standard_D1_v2'

ws = Workspace.from_config()

# Create an experiment
experiment = Experiment(ws, EXPERIMENT_NAME)

# Create an environment
tf_env = Environment.get(ws, ENV_NAME)

# Create compute target
compute_target = createAmlCompute(ws, CLUSTER_NAME, VM_SIZE)

# Create run configuration params
script_run_params = dict(
    source_directory='.',
    script='model.py',
    arguments=['--path_model', 'models/new'],
    compute_target=compute_target,
    environment=tf_env)

src = ScriptRunConfig(**script_run_params)

run = experiment.submit(src)
run.wait_for_completion()
