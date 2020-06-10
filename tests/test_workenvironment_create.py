import os
import arcus.azureml.environment.aml_environment as aml
import arcus.azureml.environment.errors as errors
import arcus.azureml.environment.environment_factory as fac
import pytest

def is_interactive():
    # If the environment variable System_DefinitionId is not available, we run locally
    return 'SYSTEM_DEFINITIONID' not in os.environ

def test_local_creation():
    with pytest.raises(errors.EnvironmentException) as excinfo:
        localenv = fac.WorkEnvironmentFactory.Create(False)
    assert 'The creation of an environment is only supported in connected mode' in str(excinfo.value)


def test_aml_factory_creation():
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')
    
    amlenv = fac.WorkEnvironmentFactory.Create(connected = True, config_file='.azureml/config.json')
    assert type(amlenv) == aml.AzureMLEnvironment
    assert amlenv.isvalid()

def test_aml_creation():
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')
    
    amlenv = aml.AzureMLEnvironment.Create(config_file='.azureml/config.json')
    assert amlenv.isvalid()

def test_aml_default_creation():
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')
    
    amlenv = aml.AzureMLEnvironment.Create()
    assert amlenv.isvalid()

def test_aml_constructor():
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')
    
    amlenv = aml.AzureMLEnvironment(config_file='.azureml/config.json')
    assert amlenv.isvalid()
