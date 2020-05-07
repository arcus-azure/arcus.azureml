from arcus.azureml.workenvironment import WorkEnvironment

def test_load_default_config():
    work_env = WorkEnvironment()
    assert work_env != None

def test_disconnected_config():
    work_env = WorkEnvironment(connected = False)
    assert work_env.is_connected == False
