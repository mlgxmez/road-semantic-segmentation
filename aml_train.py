import argparse
import json

from azureml.core import Environment, Workspace, ScriptRunConfig, Experiment
from utils.agent import createAmlCompute

parser = argparse.ArgumentParser()
parser.add_argument('--path_data')
parser.add_argument('--cloud', action='store_true')

args = parser.parse_args()

EXPERIMENT_NAME = 'road-segmentation-train'
ENV_NAME = 'AzureML-TensorFlow-2.3-GPU'
CLUSTER_NAME = 'GPU-cluster'
VM_SIZE = 'Standard_NC6'
SEED_NUMBER = 5

# Create workspace from config file
ws = Workspace.from_config()

# Create experiment to submit training
experiment_name = 'road-segmentation-train'
experiment = Experiment(ws, EXPERIMENT_NAME)

script_run_params = dict(source_directory='.',
                         script='train.py')

if args.cloud:
    # Training in the cloud
    tf_env = Environment.get(ws, ENV_NAME)
else:
    # Training locally
    import os
    CONDA_ENV_NAME = os.environ['CONDA_DEFAULT_ENV']
    ENV_NAME = 'base'

    tf_env = Environment.from_existing_conda_environment(
        ENV_NAME,
        CONDA_ENV_NAME
        )
    tf_env.python.user_managed_dependencies = True

script_run_params['environment'] = tf_env

if args.cloud:
    compute_target = createAmlCompute(ws, CLUSTER_NAME, VM_SIZE)
    script_run_params['compute_target'] = compute_target

with open('settings/model/train_config.json') as f:
    train_config = json.load(f)

# Training in the cloud requires to include the training set as Dataset
if args.cloud:
    from utils.storage import DataManager
    dm = DataManager(ws)

    # Obtain training set
    images_dataset = dm.filterDataset('training', 'images/**/*.png')
    labels_dataset = dm.filterDataset('training', 'labels/**/*.png')
    _, training_images = dm.splitDataset(images_dataset, 0.2, seed=SEED_NUMBER)
    _, training_labels = dm.splitDataset(labels_dataset, 0.2, seed=SEED_NUMBER)
    train_config.update(
        {'training_images': training_images.as_mount(),
         'training_labels': training_labels.as_mount()
         })

# Adapt all arguments to be parsed inside train.py
args = []
for k, v in train_config.items():
    args += ['--'+k, v]

# Add arguments for ScriptRunConfig
script_run_params['arguments'] = args

src = ScriptRunConfig(**script_run_params)
run = experiment.submit(src)
run.wait_for_completion()
