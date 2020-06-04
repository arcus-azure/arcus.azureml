# Force latest version of arcus packages
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "--upgrade", "--pre", package])

install('arcus-ml')
install('arcus-azureml')

# General references
import argparse
import os
import numpy as np
import pandas as pd
import joblib

# Add arcus references
from arcus.ml import dataframes as adf
from arcus.ml.timeseries import timeops
from arcus.ml.images import *
from arcus.ml.evaluation import classification as clev
from arcus.azureml.environment.environment_factory import WorkEnvironmentFactory as fac
from arcus.azureml.experimenting.trainer import Trainer

# Add AzureML references
from azureml.core import Workspace, Dataset, Datastore, Experiment, Run

# This section enables to use the module code referenced in the repo
import os
import os.path
import sys
import time
from tqdm.notebook import tqdm

from datetime import date

import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

# Tensorflow / Keras references.  Feel free to remove when not used
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy, cosine_similarity
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

##########################################
### Parse arguments
##########################################

parser = argparse.ArgumentParser()

# If you want to parse arguments that get passed through the estimator, this can be done here
# parser.add_argument('--dataset_name', type=str, dest='dataset_name', help='Data set name')
args, unknown = parser.parse_known_args()
# dataset_name = args.dataset_name

# Load the environment from the Run context, so you can access any dataset
aml_environment = fac.CreateFromContext()
ws = Run.get_context().experiment.workspace

##########################################
### Access datasets
##########################################

# Closings time frame
# time_df = aml_environment.load_tabular_dataset('time-dataset')

# mount file data set
#file_dataset = ws.datasets['file-dataset']
#mounted_files = file_dataset.mount()
#mounted_files.start()


##########################################
### Generic functions
##########################################


##########################################
### Perform training
##########################################


##########################################
### Save model
##########################################

print('Training finished')