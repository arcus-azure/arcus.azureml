from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath

import os
import pandas as pd
import glob
import numpy as np

class WorkEnvironment:
    is_connected: bool = False
    __config_file: str = '.azureml/config.json'
    __workspace: Workspace = None
    __datastore_path: str = 'data'

    def __init__(self, connected: bool = False, config_file: str = None, datastore_path: str = None, aml_workspace: Workspace = None):
        '''
        This allows a user to specify to work connected or disconnected.  
        When connecting, the implementation will check if there's a local config file for AzureML 
        and connect to the Azure ML workspace in that case, or take the given Workspace
        Args:
            connected (bool): Specifies if the WorkEnvironment should be running local or connected
            config_file (str): The name of the config file (defaulting to .azureml/config.json) that contains the Workspace parameters
            datastore_path (str): 
                When working locally: the location of a folder name where datasets will be loading from
                When working connected: the name of a DataStore in AzureML that contains Datasets
            aml_workspace_name (azureml.core.Workspace): An already built Azure ML Workspace object that can be used to work connected
        '''
        self.__datastore_path = datastore_path
        if connected == False:
            self.is_connected = False
        else:
            # User wants to connect
            # Setting the config file to the passed argument
            if config_file:
                # Since a file name is passed, we should check if it exists
                self.__config_file = config_file

            # If workspace is passed, the workspace will be taken and the config file will be ignored
            if(aml_workspace!=None):
                self.__workspace = aml_workspace
            else:
                # A config file is passed, so we'll validate the existance and connect
                if not os.path.exists(self.__config_file):
                    raise FileNotFoundError('The config file ' + self.__config_file + ' does not exist.  Please verify and try again')
                # There is a config file, so we'll connect
                self.__connect_from_config_file(self.__config_file)
            self.is_connected = True

    @classmethod
    def Create(cls, subscription_id: str, resource_group: str, workspace_name: str, write_config: bool = False,
                config_file: str = None, datastore_path: str = None):
        '''
        Connects to an existing AzureML workspace and can persist the configuration locally
        Args:
            subscription_id (str): The subscription id where the AzureML service resides
            resource_group (str): The resource group that contains the AzureML workspace
            workspace_name (str): Name of the AzureML workspace
            write_config (bool): If True, the WorkSpace configuration will be persisted in the given (or default) config file
            config_file (str): The name of the config file (defaulting to .azureml/config.json) that contains the Workspace parameters
            datastore_path (str): 
                When working locally: the location of a folder name where datasets will be loading from
                When working connected: the name of a DataStore in AzureML that contains Datasets
        Returns: 
            WorkEnvironment: an instance of WorkEnvironment allowing the user to work connected.
        '''    
        cls.__datastore_path = datastore_path

        _workspace = Workspace(subscription_id, resource_group, workspace_name)
        if config_file:
            # Writing the config_file
            cls.__config_file = config_file

        if write_config:
            _workspace.write_config(cls.__config_file)
        
        return WorkEnvironment(connected=True, aml_workspace=_workspace)

    def load_tabular_dataset(self, dataset_name: str) -> pd.DataFrame:
        '''
        Loads a tabular dataset by a given name.
            When working locally: the implementation will look for a file in the datastore_path with name {dataset_name}.csv
            When working connected: the implementation will load the Dataset by name from the AzureML Workspace
        Args:
            dataset_name (str): The name of the dataset to load
        Returns:
            pd.DataFrame: The dataset, loaded as a DataFrame
        '''
        if self.is_connected:
            # Connecting data set
            _dataset = Dataset.get_by_name(self.__workspace, name=dataset_name)
            return _dataset.to_pandas_dataframe()
        else:
            _file_name = os.path.join(self.__datastore_path, dataset_name + '.csv')
            return pd.read_csv(_file_name)

    def load_tabular_partition(self, partition_name: str, datastore_name: str = None, columns: np.array = None) -> pd.DataFrame:
        '''
        Loads a partition from a tabular dataset.
            When working locally: the implementation will append all files in the datastore_path with name {partition_name}.csv
            When working connected: the implementation will connect to the DataStore and get all delimited files matching the partition_name
        Args:
            partition_name (str): The name of the partition as a wildcard filter.  Example: B* will take all files starting with B, ending with csv
            columns: (np.array): The column names to assign to the dataframe
            datastore_path (str): 
                When working locally: the location of a folder name where datasets will be loading from
                When working connected: the name of a DataStore in AzureML that contains Datasets
        Returns:
            pd.DataFrame: The dataset, loaded as a DataFrame
        '''
        if not datastore_name:
            # No datastore name is given, so we'll take the default one
            datastore_name = self.__datastore_path

        if self.is_connected:
            # Connecting data store
            datastore = Datastore(self.__workspace, name=datastore_name)
            _aml_dataset = Dataset.Tabular.from_delimited_files(header=False,
                path=DataPath(datastore, '/' + partition_name + '.csv')) #, set_column_types=columns
            _df = _aml_dataset.to_pandas_dataframe()
            if columns != None:
                _df.columns = columns
            return _df

        else:
            # Reading data from sub files in a folder
            _folder_path = os.path.join(self.__datastore_path, datastore_name)
            _partition_files = glob.glob(_folder_path + '/' + partition_name + '*.csv')
            _dfs = []
            for filename in _partition_files:
                df = pd.read_csv(filename, index_col=None, header=0)
                _dfs.append(df)

            _df = pd.concat(_dfs, axis=0, ignore_index=True)
            if columns != None:
                _df.columns = columns
            return _df

    def __isvalid(self) -> bool:
        return True

    def __connect_from_config_file(self, file_name:str):
        self.__workspace = Workspace.from_config(_file_name=file_name)
        print('Connected to AzureML workspace')
        print('>> Name:', self.__workspace._workspace_name)
        print('>> Subscription:', self.__workspace.subscription_id)
        print('>> Resource group:', self.__workspace.resource_group)

    