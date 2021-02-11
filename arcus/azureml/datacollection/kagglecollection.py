import os
import arcus.azureml.environment.aml_environment as aml
import logging
from logging import Logger

class KaggleDataCollector():
    __logger: Logger = None

    def __init__(self):
        self.__logger = logging.getLogger()

    def copy_to_azureml(self, environment: aml.AzureMLEnvironment, dataset_name:str, user_name:str = None, user_key:str = None, use_key_vault:bool = True, local_path:str = None, force_download: bool = False):
        '''Downloads a kaggle dataset and stores it into an AzureML dataset
        Args:
            environment (aml.AzureMLEnvironment): The environment in which the dataset should be created
            dataset_name (str): The name of the kaggle dataset
            user_name (str): The kaggle user name (or the secret name in the KeyVault to it).  
            user_key (str): The kaggle secret key (or the secret name in the KeyVault to it).  
            use_key_vault (bool): Recommended, will retrieve the kaggle credentials from Key Vault
            local_path (str): The local folder in which to persist the downloaded Kaggle data
            force_download (bool): Will redownload and overwrite existing files
        '''  
        local_path = os.path.join(local_path, dataset_name.replace('/', '_').replace(' ', '-'))

        if force_download or not os.path.exists(local_path):
            if use_key_vault:
                self.__logger.info('Using KeyVault for kaggle authentication')
                # When no user_name / user_key is given, we'll take the default kaggle authentication names
                if not user_name: user_name = 'KAGGLE-USERNAME'
                if not user_key: user_key = 'KAGGLE-KEY'
                # When using key vault, we will replace the user_name & user_key values with the values from the secrets in key vault
                user_name = environment.get_secret(user_name)
                user_key = environment.get_secret(user_key)
                self.__logger.info(f'Authentication to kaggle with user {user_name}')
            
            # Kaggle authentication happens through environment variables (or a json file)
            if user_name:
                os.environ['KAGGLE_USERNAME'] = user_name
            if user_key:
                os.environ['KAGGLE_KEY'] = user_key

            import kaggle
            kaggle.api.authenticate()
            self.__logger.info('Successfully authenticated to kaggle.com')

            kaggle.api.dataset_download_files(dataset_name, path=local_path, unzip=True)
            self.__logger.info('Dataset successfully downloaded locally')

            environment.upload_dataset(dataset_name, local_path, overwrite=force_download, tags={'source': 'kaggle', 'url': f'https://www.kaggle.com/{dataset_name}'})
            self.__logger.info('Dataset successfully uploaded to AzureML Dataset')
