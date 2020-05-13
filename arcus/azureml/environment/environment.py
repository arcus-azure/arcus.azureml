from abc import ABCMeta, abstractmethod
import pandas as pd 
import numpy as np
from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath
from azureml.data.dataset_error_handling import DatasetValidationError, DatasetExecutionError
from azureml.data.dataset_type_definitions import PromoteHeadersBehavior
import os
import glob
from arcus.azureml.experimenting import trainer

class WorkEnvironment:
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_tabular_dataset(self, dataset_name: str) -> pd.DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def load_tabular_partition(self, partition_name: str, datastore_name: str = None, columns: np.array = None, first_row_header: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def isvalid(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def start_experiment(self, name: str) -> trainer.Trainer:
        raise NotImplementedError
