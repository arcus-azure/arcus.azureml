---
title: "Interactive experimentations"
layout: default
---

# Interactive experimentations

With `arcus-azureml`, we offer an out of the box library that implements the best practices to experiment interactively with the machine learning models you want to apply to a problem.  You will typically do this from your preferred notebook experience (such as Visual Studio Code or Jupyter notebooks).  

With every attempt in an experiment, the following actions can happen without having to write the extra code:

1. Start a new experiment
1. Launch a new run in the experiment, indicating the specific parameters or description
1. Evaluate the tested model (classification or regression, for example) and have the results tracked to your run
1. Persist the actual script/code to the Azure ML backend
1. Save the trained model to the Azure ML backend

## Common imports

The following imports are advised for most cases in interactive experimentation mode.

```python
from arcus.azureml.environment.aml_environment import AzureMLEnvironment
```

## Step by step guide

### Connecting to your AzureML workspace

The first thing that is needed is to connect to the AzureML workspace.  As arcus-azureml is abstracting the azureml SDK, it is quite easy to get this done.  
The following possibilities exist and they are presented in order of recommendation:

__Using a local configuration file__

The `Create` method connects to the AzureML workspace, based on the settings in the given configuration file (defaults to the configuration location below).  If there is no cached authentication available, the browser will open and ask for authentication against the workspace.

```python
azure_config_file = '.azureml/config.json'
work_env = AzureMLEnvironment.Create(config_file=azure_config_file)
```

The layout of this configuration file is straight forward: 

```json
{
    "subscription_id": "",
    "resource_group": "",
    "workspace_name": ""
}
```
__By passing the configuration parameters in the constructor__

The following code will accept the subscription id, resource group and workspace name and will connect to the Workspace.  At the same time, the `write_config` parameter specifies if that information has to be stored in the config file, so that you can leverage it later.

```python
work_env = AzureMLEnvironment.Create(subscription_id='', resource_group='', workspace_name='', write_config=True)
```

__By getting a reference to the AzureML context in which the code is running__

The following code will just connect to the Workspace (without prompting for authentication) in which the code is running at that moment.  This is ideal for code that runs in Estimators or training scripts.

```python
work_env = AzureMLEnvironment.CreateFromContext()
```

### Accessing data in your workspace

There are two possibilities to access tabular data in our library.  

__Accessing a defined Tabular dataset by name__

The following code loads the dataset from the Workspace and returns it as a `pd.DataFrame`.

```python
df = work_env.load_tabular_dataset('datasetname')
```

__Accessing a partitioned Tabular datastore__

The following code loads all files from the configured datastore that match the given partition pattern and concatenates these into one `pd.DataFrame`.

```python
partition_df = work_env.load_tabular_partition('folder/AT*', columns=['Close', 'High', 'Isin'])
```

### Tracking an experiment

Now that the data is retrieved, we can start tracking an experiment.  In an experiment, there will be multiple runs.  All of this is implemented in the `AzureMLTrainer` class.  With this class, new runs can be started, results can be tracked and models can be saved in the AzureML workspace

Starting an experiment is as simple as running the following code 

```python
trainer = work_env.start_experiment('experiment_name')
```

__Loading a trainer, based on the context in which the code is running__

The following code will create a Trainer that is configured to the active Run in which the code is executed.  This is ideal for code that runs in Estimators or training scripts.

```python
trainer = AzureMLTrainer.CreateFromContext()
```

### Execute a training run and track to AzureML

The following code shows some typical extracts where a user is triggering a new run.  (by default the existing active run will be completed, before the new run gets started).  Every run can get a description.  It is also possible to indicate if the local folder (containing the script , etc) should be copied into the AzureML workspace.  (caution: for larger files, this will delay the execution) and it is possible to pass some metrics, such as hyper parameters that will be used in the run.  These values will be visible in the table of the run.

After the model has been trained, it can be evaluated (here we are showing a classifier that gets evaluated).  This evaluation will happen with the test data and the results (confusion matrix, classification report, the actual model and charts) will be saved and tracked in the cloud.

```python
trainer.new_run('description', copy_folder=True, metrics={'alpha': 0.01, 'kernel_size': 5})
# Perform training before evaluating the model
trainer.evaluate_classifier(logreg, X_test, y_test, show_roc = True, upload_model = True)   
```

## Sample notebooks

[This notebook](https://github.com/arcus-azure/arcus.azureml/tree/master/samples/interactive_experiments.ipynb) can be used to play around with Azure ML experiments.
