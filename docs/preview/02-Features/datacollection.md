---
title: "Data collection"
layout: default
---

# Data collection

The data collection modules allow users to download data from different sources into an AzureML workspace (DataStore & Dataset)

## Kaggle datasets

It is possible to download a Kaggle dataset, by executing the following code:

```python
# Creating the AzureML environment
aml_environment = AzureMLEnvironment.CreateFromContext()
# Create the KaggleDataCollector class
collector = KaggleDataCollector()
# Copy the dataset to Azure ML
collector.copy_to_azureml(aml_environment, 'dataset-name', local_path='kaggle_data')
```

Multiple arguments can be passed to the `copy_to_azureml` method:

- `environment` (aml.AzureMLEnvironment): The environment in which the dataset should be created
- `dataset_name` (str): The name of the kaggle dataset
- `user_name` (str): The kaggle user name (or the secret name in the KeyVault to it).  
- `user_key` (str): The kaggle secret key (or the secret name in the KeyVault to it).  
- `use_key_vault` (bool): Recommended, will retrieve the kaggle credentials from Key Vault
- `local_path` (str): The local folder in which to persist the downloaded Kaggle data
- `force_download` (bool): Will redownload and overwrite existing files

It is highly recommended to provide the Kaggle credentials in the linked Azure KeyVault as you are not having those credentials persisted into log files, etc.  If you don't provide the user_name and user_key attributes, the credentials will be retrieved from the `KAGGLE-USERNAME` and the `KAGGLE-KEY` secrets.

## Sample notebooks

[This notebook](https://github.com/arcus-azure/arcus.azureml/tree/master/samples/datacollection.ipynb) can be used to quickly execute data collection scripts.
