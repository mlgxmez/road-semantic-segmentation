from azureml.core import Environment, Workspace, Experiment, RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from utils.agent import createAmlCompute
from utils.storage import DataManager, validateDataset

BASE_IMAGE = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04:20210301.v1'
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

# Create the environment
tf_env = Environment(ENV_NAME)
tf_env.docker.enabled = True
tf_env.docker.base_image = BASE_IMAGE

# Define additional packages to be installed
conda_dep = CondaDependencies()
conda_dep.add_pip_package('tensorflow-gpu==2.3.0')
conda_dep.add_pip_package('pillow')

# Add packages to the environment
tf_env.python.conda_dependencies = conda_dep

# Create the configuration of an experiment
aml_run_config = RunConfiguration()
aml_run_config.environment = tf_env
# The name of the custome environment must not start by 'AzureML'
# https://github.com/MicrosoftDocs/azure-docs/issues/65770#issuecomment-724536550
aml_run_config.environment.name = 'road-segmentation-GPU'

# Create the compute target
compute_target = createAmlCompute(ws, CLUSTER_NAME, VM_SIZE)

dm = DataManager(ws)

# Obtain training set
images_dataset = dm.filterDataset('training', 'images/**/*.png')
labels_dataset = dm.filterDataset('training', 'labels/**/*_road_*.png')
scoring_images, training_images = dm.splitDataset(
    images_dataset,
    0.2,
    seed=SEED_NUMBER)
scoring_labels, training_labels = dm.splitDataset(
    labels_dataset,
    0.2,
    seed=SEED_NUMBER)

validateDataset(training_images, training_labels)
validateDataset(scoring_images, scoring_labels)

input_train_step = (training_images
                    .as_named_input("training_images")
                    .as_mount("data/images"))

output_train_step = (training_labels
                     .as_named_input("training_labels")
                     .as_mount("data/labels"))

input_score_step = (scoring_images
                    .as_named_input("scoring_images")
                    .as_mount("data/images"))

output_score_step = (scoring_labels
                     .as_named_input("scoring_labels")
                     .as_mount("data/labels"))

path_model = PipelineData(
    "path_model",
    datastore=dm.datastore,
    is_directory=True)

args_train = ["--trained_model", path_model,
              "--input_train_step", input_train_step,
              "--output_train_step", output_train_step,
              "--augmentation_file", "settings/augmentation_config.json"]

args_score = ["--trained_model", path_model,
              "--input_score_step", input_score_step,
              "--output_score_step", output_score_step]

# Define the steps of the pipeline
train_step = PythonScriptStep(
    name="train",
    script_name="train.py",
    compute_target=compute_target,
    arguments=args_train,
    outputs=[path_model],
    runconfig=aml_run_config,
    source_directory='./train'
)

score_step = PythonScriptStep(
    name="score",
    script_name="score.py",
    compute_target=compute_target,
    arguments=args_score,
    inputs=[path_model, input_score_step, output_score_step],
    runconfig=aml_run_config,
    source_directory='./score'
)

steps = [train_step, score_step]

# Build the pipeline
pipeline = Pipeline(workspace=ws, steps=steps)

# Submit the experiment
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion()
