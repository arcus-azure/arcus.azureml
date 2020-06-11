# Force latest version of arcus packages
import subprocess
import sys

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
from arcus.azureml.environment.aml_environment import AzureMLEnvironment
from arcus.azureml.experimenting.aml_trainer import AzureMLTrainer

# Add AzureML references
from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
from azureml.core import VERSION

# This section enables to use the module code referenced in the repo
import os
import os.path
import sys
import time
from datetime import date

import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
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
### Parse arguments and prepare environment
##########################################

parser = argparse.ArgumentParser()

# If you want to parse arguments that get passed through the estimator, this can be done here
parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='Epoch count')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32, help='Batch size')
parser.add_argument('--es_patience', type=int, dest='es_patience', default=-1, help='Early stopping patience. If less than zero, no Early stopping')
parser.add_argument('--train_test_split_ratio', type=float, dest='train_test_split_ratio', default=0.3, help='Train test split ratio')

args, unknown = parser.parse_known_args()
epoch_count = args.epochs
batch_size = args.batch_size
es_patience = args.es_patience
train_test_split_ratio = args.train_test_split_ratio

# Load the environment from the Run context, so you can access any dataset
aml_environment = AzureMLEnvironment.CreateFromContext()
trainer = AzureMLTrainer.CreateFromContext()

if not os.path.exists('outputs'):
    os.makedirs('outputs')

##########################################
### Access datasets
##########################################

# Access tabular dataset (which is not passed as input)
# df = aml_environment.load_tabular_dataset('mydataset')


# access file data set just in the current sub directory
# every dataset that was passed as input to the start_training 
# will be available on the local image in a directory named like the dataset
# print(os.listdir('./datasetname/'))


##########################################
### Generic functions
##########################################
def perform_training(model, x, y, epoch_count, batch_size, val_split = 0.2, es_patience=-1):
    cbs = list()
    cbs.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=es_patience))

    if es_patience >= 0:
        cbs.append(ModelCheckpoint('outputs/best_model.h5', monitor='val_loss', save_best_only=True, mode='min'))
    
    model.fit(x, y,
                    epochs=epoch_count,
                    batch_size=batch_size,
                    validation_split = val_split,
                    callbacks = cbs)

    return model

def build_model(input_shape, output_shape):
    model = Sequential()
    # Build model architecture here
    # You can take input parameters from the arg parser to specify hyper parameters
    return model

##########################################
### Perform training
##########################################

# Load data 
X = None # replace with actual input vals
y = None # replace with actual output features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio, random_state=0)

# Build model 
model = build_model()
fitted_model = perform_training(model, X_train, y_train, epoch_count=epoch_count, batch_size=batch_size, es_patience=es_patience)

# Custom metrics tracking
# trainer._log_metrics('dice_coef_loss', list(fitted_model.history.history['dice_coef_loss'])[-1], description='')


##########################################
### Save model
##########################################
fitted_model.save('outputs/model')

print('Training finished')