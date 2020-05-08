# from arcus.azureml.workenvironment import WorkEnvironment
# from collections import Counter


# def test_load_default_config():
#     work_env = WorkEnvironment()
#     assert work_env != None

# def test_disconnected_config():
#     work_env = WorkEnvironment(connected = False)
#     assert work_env.is_connected == False

# def test_local_partitions():
#     work_env = WorkEnvironment(connected = False, datastore_path='tests/resources/datastore')
#     partition_df = work_env.load_tabular_partition('stock_AT', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
#     assert partition_df.shape == (30, 9)

#     assert len(Counter(partition_df['Isin'])) == 3

# def test_local_partitions_header():
#     work_env = WorkEnvironment(connected = False, datastore_path='tests/resources/datastore')
#     partition_df = work_env.load_tabular_partition('stock_header_AT', first_row_header=True)
#     assert 'Isin' in partition_df.columns
#     assert partition_df.shape == (24, 9)
#     assert len(Counter(partition_df['Isin'])) == 3


# def test_local_partitions_noheader_and_columns():
#     work_env = WorkEnvironment(connected = False, datastore_path='tests/resources/datastore')
#     partition_df = work_env.load_tabular_partition('stock_header_AT', first_row_header=False, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
#     assert 'C' in partition_df.columns
#     assert partition_df.shape == (27, 9)

# def test_local_partitions_header_and_columns():
#     work_env = WorkEnvironment(connected = False, datastore_path='tests/resources/datastore')
#     partition_df = work_env.load_tabular_partition('stock_header_AT', first_row_header=True, columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
#     assert 'Isin' in partition_df.columns
#     assert partition_df.shape == (24, 9)
#     assert len(Counter(partition_df['Isin'])) == 3

# def test_local_partitions_notexisting():
#     work_env = WorkEnvironment(connected = False, datastore_path='tests/resources/datastore')
#     partition_df = work_env.load_tabular_partition('stock_BE', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
#     assert partition_df == None