import arcus.azureml.environment.workenvironment as wenv

def is_interactive():
    # If the environment variable System_DefinitionId is not available, we run locally
    return 'SYSTEM_DEFINITIONID' not in os.environ



def test_local_creation():
    loc = wenv.WorkEnvironment.Create(False)
    assert type(loc) == wenv.LocalEnvironment
    assert loc.isvalid()

def test_aml_creation():
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')
    
    aml = wenv.WorkEnvironment.Create(connected = True, config_file='.azureml/config.json')
    assert type(aml) == wenv.AzureMLEnvironment
    assert aml.isvalid()

    