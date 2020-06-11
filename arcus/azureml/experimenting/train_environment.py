from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
import os
import numpy as np

def get_training_environment(ws: Workspace, name: str, pip_file: str, include_prerelease: bool = False):
    from azureml.train.estimator import Estimator
    from azureml.core import Environment, ScriptRunConfig
    from azureml.core.runconfig import RunConfiguration
    from azureml.core.runconfig import DataReferenceConfiguration
    from azureml.core.runconfig import CondaDependencies

    dependencies = CondaDependencies.create(pip_packages=__get_pip_list(pip_file))
    if(include_prerelease):
        dependencies.set_pip_option("--pre")

    training_env = Environment(name = name)
    training_env.python.conda_dependencies = dependencies
    training_env.docker.enabled = True
    _ = training_env.register(workspace=ws)
    return training_env

def __get_pip_list(pip_file: str):
    pip_list = list()
    if(os.path.exists(pip_file)):
        with open(pip_file, 'r') as _file:
            for req in _file:
                req = req.strip()
                pip_list.append(req)
    return np.array(pip_list)