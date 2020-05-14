from abc import ABCMeta, abstractmethod
import pandas as pd 
import numpy as np
from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath
import os
import glob
from azureml.data.dataset_error_handling import DatasetValidationError, DatasetExecutionError
from azureml.data.dataset_type_definitions import PromoteHeadersBehavior
import arcus.azureml.environment.environment as env
from arcus.azureml.experimenting import trainer
from arcus.azureml.experimenting import aml_trainer

class AzureMLEnvironment(env.WorkEnvironment):
    is_connected: bool = False
    __config_file: str = '.azureml/config.json'
    __workspace: Workspace = None
    __datastore_path: str = 'data'

    def __init__(self, config_file: str = None, datastore_path: str = None, subscription_id: str = None, 
                resource_group: str = None, workspace_name: str = None, write_config: bool = False):
        '''
        This allows a user to specify to work connected or disconnected.  
        When connecting, the implementation will check if there's a local config file for AzureML 
        and connect to the Azure ML workspace in that case, or take the given Workspace
        Args:
            config_file (str): The name of the config file (defaulting to .azureml/config.json) that contains the Workspace parameters
            datastore_path (str): the name of a DataStore in AzureML that contains Datasets
            aml_workspace_name (azureml.core.Workspace): An already built Azure ML Workspace object that can be used to work connected
        '''
        self.__datastore_path = datastore_path
        # User wants to connect
        # Setting the config file to the passed argument
        if config_file:
            # Since a file name is passed, we should check if it exists
            self.__config_file = config_file

        # If workspace parameters are passed, the workspace will be taken and the config file will be ignored
        if(subscription_id and resource_group and workspace_name):
            self.__workspace = Workspace(subscription_id, resource_group, workspace_name)
            if write_config:
                self.__workspace.write_config(self.__config_file)
        
        else:
            # A config file is passed, so we'll validate the existance and connect
            if not os.path.exists(self.__config_file):
                raise FileNotFoundError('The config file ' + self.__config_file + ' does not exist.  Please verify and try again')
            # There is a config file, so we'll connect
            self.__connect_from_config_file(self.__config_file)

        self.is_connected = True

    def load_tabular_dataset(self, dataset_name: str) -> pd.DataFrame:
        '''
        Loads a tabular dataset by a given name. the implementation will load the Dataset by name from the AzureML Workspace
        Args:
            dataset_name (str): The name of the dataset to load
        Returns:
            pd.DataFrame: The dataset, loaded as a DataFrame
        '''
        # Connecting data set
        _dataset = Dataset.get_by_name(self.__workspace, name=dataset_name)
        return _dataset.to_pandas_dataframe()

    def load_tabular_partition(self, partition_name: str, datastore_name: str = None, columns: np.array = None, first_row_header: bool = False) -> pd.DataFrame:
        '''
        Loads a partition from a tabular dataset. the implementation will connect to the DataStore and get all delimited files matching the partition_name
        Args:
            partition_name (str): The name of the partition as a wildcard filter.  Example: B* will take all files starting with B, ending with csv
            columns: (np.array): The column names to assign to the dataframe
            datastore_path (str): The name of a DataStore in AzureML that contains Datasets
        Returns:
            pd.DataFrame: The dataset, loaded as a DataFrame
        '''
        if not datastore_name:
            # No datastore name is given, so we'll take the default one
            datastore_name = self.__datastore_path

        # Connecting data store
        datastore = Datastore(self.__workspace, name=datastore_name)
        try:
            _header = PromoteHeadersBehavior.ALL_FILES_HAVE_SAME_HEADERS if first_row_header else False
            _aml_dataset = Dataset.Tabular.from_delimited_files(header=_header,
                path=DataPath(datastore, '/' + partition_name + '.csv')) #, set_column_types=columns
            _df = _aml_dataset.to_pandas_dataframe()
        except DatasetValidationError as dsvalex:
            if 'provided path is not valid' in str(dsvalex):
                return None
            else:
                raise
        if columns != None:
            _df.columns = columns
        return _df

    def start_experiment(self, name: str) -> trainer.Trainer:
        '''
        Creates a new experiment (or connects to an existing one), using the give name

        Args:
            name (str): the name of the experiment which whill be used in AzureML
        Returns:
            Trainer: a Trainer object that can be used to perform trainings and add logging in AzureML

        '''
        return aml_trainer.AzureMLTrainer(name, self.__workspace)

    def isvalid(self) -> bool:
        return True

    def get_azureml_workspace(self):
        return self.__workspace

    def __connect_from_config_file(self, file_name:str):
        self.__workspace = Workspace.from_config(_file_name=file_name)
        print('Connected to AzureML workspace')
        print('>> Name:', self.__workspace._workspace_name)
        print('>> Subscription:', self.__workspace.subscription_id)
        print('>> Resource group:', self.__workspace.resource_group)

