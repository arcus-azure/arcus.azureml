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

from arcus.azureml.experimenting.trainer import Trainer
from arcus.azureml.experimenting.local_trainer import LocalMLTrainer

class LocalEnvironment(env.WorkEnvironment):
    is_connected: bool = False
    __datastore_path: str = 'data'

    def __init__(self, datastore_path: str = None):
        '''
        This allows a user to specify to work disconnected on the local environment.
        Args:
            datastore_path (str): the location of a folder name where datasets will be loading from
        '''
        self.__datastore_path = datastore_path
        self.is_connected = False

    def load_tabular_dataset(self, dataset_name: str) -> pd.DataFrame:
        '''
        Loads a tabular dataset by a given name.
            the implementation will look for a file in the datastore_path with name {dataset_name}.csv
        Args:
            dataset_name (str): The name of the dataset to load
        Returns:
            pd.DataFrame: The dataset, loaded as a DataFrame
        '''
        _file_name = os.path.join(self.__datastore_path, dataset_name + '.csv')
        return pd.read_csv(_file_name)

    def load_tabular_partition(self, partition_name: str, datastore_name: str = None, columns: np.array = None, first_row_header: bool = False) -> pd.DataFrame:
        '''
        Loads a partition from a tabular dataset.
            The implementation will append all files in the datastore_path with name {partition_name}.csv
        Args:
            partition_name (str): The name of the partition as a wildcard filter.  Example: B* will take all files starting with B, ending with csv
            columns: (np.array): The column names to assign to the dataframe
            datastore_path (str): the location of a folder name where datasets will be loading from
        Returns:
            pd.DataFrame: The dataset, loaded as a DataFrame
        '''

        if not datastore_name:
            # No datastore name is given, so we'll take the default one
            datastore_name = self.__datastore_path

        # Reading data from sub files in a folder
        _folder_path = datastore_name
        _partition_files = glob.glob(_folder_path + '/' + partition_name + '.csv')
        _record_found = False
        _result = None
        for filename in _partition_files:
            _header = 0 if first_row_header else None
            df = pd.read_csv(filename, index_col=None, header=_header)
            if not _record_found:
                _result = df
                _record_found = True
            else:
                _result = _result.append(df)

        if not _record_found:
            return None
        if columns:
            _result.columns = columns
        return _result

    def start_experiment(self, name: str) -> Trainer:
        '''
        Creates a new local experiment (or connects to an existing one), using the give name

        Args:
            name (str): the name of the experiment which whill be used as reference
        Returns:
            Trainer: a Trainer object that can be used to perform trainings and add logging

        '''
        return LocalMLTrainer(name, self)

    def isvalid(self) -> bool:
        return True

