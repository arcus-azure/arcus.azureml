import pytest
import os


def is_interactive():
    # If the environment variable System_DefinitionId is not available, we run locally
    return 'SYSTEM_DEFINITIONID' not in os.environ

def test_download_to_azure():
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')
    
    import arcus.azureml.environment.aml_environment as aml
    import arcus.azureml.datacollection.kagglecollection as kgl 

    amlenv = aml.AzureMLEnvironment.Create(config_file='.azureml/config.json')
    assert amlenv.isvalid()

    collector = kgl.KaggleDataCollector()
    collector.copy_to_azureml(amlenv, 'new-york-state/nys-farm-product-dealer-licenses-currently-issued', local_path='kaggle_data', force_download=True)
    assert(1==1)