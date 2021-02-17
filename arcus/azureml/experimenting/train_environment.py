from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
import os
import numpy as np

def get_training_environment(ws: Workspace, name: str, pip_file: str, use_gpu: bool = False, include_prerelease: bool = False, environment_type: str = None):
    '''
    Creates a training environment, based on the required pip packages, the need for GPU and a given environment type
    Args:
        ws (Workspace): the AzureML workspace that will be used to register the environment
        name (str): the name for the environment that will be registered
        use_gpu (bool): indicating if a GPU is required or not
        include_prerelease (bool): indicates if the pip packages can be installed in prerelease mode
        environment_type (str): either the name of an existing environment that will be taken as base, or one of these values (tensorflow, sklearn, pytorch).  
    Returns:
        a registered environment , ready to use
    '''
    from azureml.train.estimator import Estimator
    from azureml.core import Environment, ScriptRunConfig
    from azureml.core.runconfig import RunConfiguration
    from azureml.core.runconfig import CondaDependencies

    print('Getting environment for type', environment_type)
    base_environment = environment_type if environment_type else 'AzureML-Minimal'
    if(environment_type == 'tensorflow'):
        # Using Tensorflow Estimator
        base_environment = 'AzureML-TensorFlow-2.3-GPU' if use_gpu else 'AzureML-TensorFlow-2.3-CPU'
    elif(environment_type == 'sklearn'):
        base_environment = 'AzureML-Scikit-learn-0.20.3'
    elif(environment_type == 'pytorch'):
        base_environment = 'AzureML-PyTorch-1.6-GPU' if use_gpu else 'AzureML-PyTorch-1.6-GPU'

    pip_packages=__get_package_list_from_requirements(pip_file)

    if base_environment is not None:
        print('Taking', base_environment, 'as base environment')
        training_env = Environment.get(ws, base_environment)
        training_env.name = name
        for pippkg in pip_packages:
            training_env.python.conda_dependencies.add_pip_package(pippkg)

    else:
        print('Creating new environment')
        training_env = Environment(name = name)
        training_env.python.conda_dependencies = CondaDependencies.create(pip_packages = pip_packages)

    if(include_prerelease):
        training_env.python.conda_dependencies.set_pip_option("--pre")

    training_env.docker.enabled = True
    _ = training_env.register(workspace=ws)
    return training_env

def __get_package_list_from_requirements(pip_file: str):
    pip_list = list()
    if(os.path.exists(pip_file)):
        with open(pip_file, 'r') as _file:
            for req in _file:
                req = req.strip()
                pip_list.append(req)
    return np.array(pip_list)