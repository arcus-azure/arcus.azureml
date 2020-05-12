import os
from arcus.azureml.environment.environment_factory import WorkEnvironmentFactory as fac
from collections import Counter

def is_interactive():
    # If the environment variable System_DefinitionId is not available, we run locally
    return 'SYSTEM_DEFINITIONID' not in os.environ

def setup_function(func):
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')




datastore_name = 'smart_devops'
partitioned_datastore_name = 'aiincubators_covid'

def test_dataset():
    work_env = fac.Create(connected=True)
    _df = work_env.load_tabular_dataset('smart-devops-changesets')
    _df = _df.tail(20)
    assert _df.shape == (20,16) # 16 columns expected

def test_partitions():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition(partition_name= 'test-partitioning/stock_AT*', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
    assert partition_df is not None
    assert partition_df.shape == (30, 9)

    assert len(Counter(partition_df['Isin'])) == 3

def test_partitions_header():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_header_AT*', first_row_header=True)
    assert 'Isin' in partition_df.columns
    assert partition_df.shape == (24, 9)
    assert len(Counter(partition_df['Isin'])) == 3


def test_partitions_noheader_and_columns():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_header_AT*', first_row_header=False, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
    assert partition_df is not None

    assert 'C' in partition_df.columns
    assert partition_df.shape == (27, 9)

def test_partitions_header_and_columns():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_header_AT*', first_row_header=True, columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
    assert partition_df is not None

    assert 'Isin' in partition_df.columns
    assert partition_df.shape == (24, 9)
    assert len(Counter(partition_df['Isin'])) == 3

def test_partitions_notexisting():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_BE*', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
    assert partition_df is None