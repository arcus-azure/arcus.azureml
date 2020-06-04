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
from arcus.azureml.environment.environment_factory import WorkEnvironmentFactory as factory
```

## Step by step guide


## Sample notebooks

[This notebook](../../samples/aml_environment.ipynb) can be used to play around with Azure ML experiments.

[&larr; back](/)