
# General references
import argparse
import os

# Add arcus references
from arcus.azureml.datacollection.kagglecollection import KaggleDataCollector
from arcus.azureml.environment.aml_environment import AzureMLEnvironment

##########################################
### Parse arguments and prepare environment
##########################################

parser = argparse.ArgumentParser()

# If you want to parse arguments that get passed through the estimator, this can be done here
parser.add_argument('--kaggle_user', type=str, dest='user', default=None, help='Kaggle user name')
parser.add_argument('--kaggle_key', type=str, dest='key', default=None, help='Kaggle user secret')
parser.add_argument('--kaggle_dataset', type=str, dest='dataset', default=None, help='Kaggle data set name')
parser.add_argument('--use_keyvault', type=bool, dest='usekeyvault', default=True, help='Indicate to use Key Vault')

args, unknown = parser.parse_known_args()
kaggle_user = args.user
kaggle_secret = args.key
kaggle_dataset = args.dataset
use_key_vault = args.usekeyvault

# Load the environment from the Run context, so you can access any dataset
aml_environment = AzureMLEnvironment.CreateFromContext()
collector = KaggleDataCollector()
collector.copy_to_azureml(aml_environment, kaggle_dataset, local_path='kaggle_data', user_name = kaggle_user, user_key = kaggle_secret, use_key_vault=use_key_vault, force_download=True)

print('Training finished')
