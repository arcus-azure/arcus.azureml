from arcus.azureml.datacollection.kagglecollection import copy_to_azureml

def test_download_to_azure():
    copy_to_azureml('new-york-state/nys-farm-product-dealer-licenses-currently-issued',  local_path='kaggle_data')
    assert(1==1)