import arcus.azureml.environment.workenvironment as wenv

def test_local_creation():
    loc = wenv.WorkEnvironment.Create(False)
    assert type(loc) == wenv.LocalEnvironment
    assert loc.isvalid()

def test_aml_creation():
    loc = wenv.WorkEnvironment.Create(connected = True, config_file='.azureml/config.json')
    assert type(loc) == wenv.AzureMLEnvironment
    assert loc.isvalid()

    