import os
import arcus.azureml.environment.local_environment as loc
import arcus.azureml.environment.aml_environment as aml
import arcus.azureml.environment.environment_factory as fac

def is_interactive():
    # If the environment variable System_DefinitionId is not available, we run locally
    return 'SYSTEM_DEFINITIONID' not in os.environ

def test_local_creation():
    localenv = fac.WorkEnvironmentFactory.Create(False)
    assert type(localenv) == loc.LocalEnvironment
    assert localenv.isvalid()


def test_aml_creation():
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')
    
    amlenv = fac.WorkEnvironmentFactory.Create(connected = True, config_file='.azureml/config.json')
    assert type(amlenv) == aml.AzureMLEnvironment
    assert amlenv.isvalid()