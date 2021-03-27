from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


def createAmlCompute(workspace, compute_name):
    try:
        compute_target = ComputeTarget(workspace, compute_name)
        print('Found existing compute target')
    except ComputeTargetException:
        print('Creating a new compute target')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_NC6',
            max_nodes=4)
        compute_target = AmlCompute.create(
            workspace,
            compute_name,
            compute_config)
        compute_target.wait_for_completion(show_output=True)
    return compute_target
