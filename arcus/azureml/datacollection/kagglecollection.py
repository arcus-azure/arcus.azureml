'''
This module allows to download kaggle datasets to disk
'''
import os
import arcus.azureml.environment.aml_environment as aml

def copy_to_azureml(environment: aml.AzureMLEnvironment, dataset_name:str, user_name:str = None, user_key:str = None, local_path:str = None, force_download: bool = False):
    '''Downloads a kaggle dataset and stores it into an AzureML dataset
    Args:
        dataset_name (str): The name of the kaggle dataset
    '''  
    local_path = os.path.join(local_path, dataset_name.replace('/', '_').replace(' ', '-'))

    if force_download or not os.path.exists(local_path):
        if user_name:
            os.environ['KAGGLE_USERNAME'] = user_name
        if user_key:
            os.environ['KAGGLE_KEY'] = user_key
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=local_path, unzip=True)

        environment.upload_dataset(dataset_name, local_path, overwrite=force_download, tags={'source': 'kaggle', 'url': f'https://www.kaggle.com/{dataset_name}'})
        print('ok')