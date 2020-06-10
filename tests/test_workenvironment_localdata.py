from arcus.azureml.environment.aml_environment import AzureMLEnvironment 
from collections import Counter
import pandas as pd 
import numpy as np
from sklearn import linear_model
import pytest
from datetime import datetime

import tensorflow as tf
from arcus.azureml.experimenting.tuning import LocalArcusGridSearchCV
from sklearn.model_selection import train_test_split

def test_dataset():
    work_env = AzureMLEnvironment(connect_workspace=False, datastore_path='tests/resources/datasets')
    _df = work_env.load_tabular_dataset('student-admission', cloud_storage=False)
    _df = _df.tail(20)
    assert _df.shape == (20,3) # 3 columns expected

def test_partitions():
    work_env = AzureMLEnvironment(connect_workspace=False, datastore_path='tests/resources/datastore')
    partition_df = work_env.load_tabular_partition('stock_AT*', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'], cloud_storage=False)
    assert partition_df.shape == (30, 9)

    assert len(Counter(partition_df['Isin'])) == 3

def test_partitions_header():
    work_env = AzureMLEnvironment(connect_workspace=False, datastore_path='tests/resources/datastore')
    partition_df = work_env.load_tabular_partition('stock_header_AT*', first_row_header=True, cloud_storage=False)
    assert 'Isin' in partition_df.columns
    assert partition_df.shape == (24, 9)
    assert len(Counter(partition_df['Isin'])) == 3


def test_partitions_noheader_and_columns():
    work_env = AzureMLEnvironment(connect_workspace=False, datastore_path='tests/resources/datastore')
    partition_df = work_env.load_tabular_partition('stock_header_AT*', first_row_header=False, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'], cloud_storage=False)
    assert 'C' in partition_df.columns
    assert partition_df.shape == (27, 9)

def test_partitions_header_and_columns():
    work_env = AzureMLEnvironment(connect_workspace=False, datastore_path='tests/resources/datastore')
    partition_df = work_env.load_tabular_partition('stock_header_AT*', first_row_header=True, columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'], cloud_storage=False)
    assert 'Isin' in partition_df.columns
    assert partition_df.shape == (24, 9)
    assert len(Counter(partition_df['Isin'])) == 3

def test_partitions_notexisting():
    work_env = AzureMLEnvironment(connect_workspace=False, datastore_path='tests/resources/datastore')
    partition_df = work_env.load_tabular_partition('stock_BE*', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'], cloud_storage=False)
    assert partition_df == None
