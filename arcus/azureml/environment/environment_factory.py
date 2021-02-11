from abc import ABCMeta, abstractmethod
import pandas as pd 
import numpy as np
from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath
import os
import glob
from azureml.data.dataset_error_handling import DatasetValidationError, DatasetExecutionError
from azureml.data.dataset_type_definitions import PromoteHeadersBehavior
import arcus.azureml.environment.aml_environment as aml 
import arcus.azureml.environment.environment as env
import arcus.azureml.environment.errors as errors

class WorkEnvironmentFactory:
    __metaclass__ = ABCMeta

    @classmethod
    def Create(cls, connected: bool = True, subscription_id: str = None, resource_group: str = None, workspace_name: str = None, 
                write_config: bool = False, config_file: str = None, datastore_path: str = None):
        '''
        Creates a WorkEnvironment and returns the correct implementation, based on the configuration
        Args:
            connected (bool): Deprecated, should be True
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
        if(connected):
            return aml.AzureMLEnvironment(config_file = config_file, datastore_path = datastore_path, 
                                        subscription_id=subscription_id, resource_group=resource_group, 
                                        workspace_name= workspace_name, write_config = write_config)
        else:
            raise errors.EnvironmentException('The creation of an environment is only supported in connected mode')

    @classmethod
    def CreateFromContext(cls, connected: bool = True, datastore_path: str = None):
        '''
        Creates a WorkEnvironment and returns the correct implementation, based on the configuration
        Args:
            connected (bool): Deprecated, should be True
            datastore_path (str): 
                When working locally: the location of a folder name where datasets will be loading from
                When working connected: the name of a DataStore in AzureML that contains Datasets
        Returns: 
            WorkEnvironment: an instance of WorkEnvironment allowing the user to work connected.
        '''    
        if(connected):
            return aml.AzureMLEnvironment(datastore_path = datastore_path, from_context=True)
        else:
            raise errors.EnvironmentException('The creation of an environment is only supported in connected mode')

