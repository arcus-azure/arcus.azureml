# Arcus - Azure Azure Machine Learning
[![Build Status](https://dev.azure.com/codit/Arcus/_apis/build/status/Commit%20builds/CI%20-%20Arcus.ML?repoName=arcus-azure%2Farcus.ml&branchName=master)](https://dev.azure.com/codit/Arcus/_build/latest?definitionId=836&repoName=arcus-azure%2Farcus.ml&branchName=master)
[![PyPI version](https://badge.fury.io/py/arcus-azureml.svg)](https://badge.fury.io/py/arcus-azureml)

Azure Machine Learning development in a breeze.

![Arcus](https://raw.githubusercontent.com/arcus-azure/arcus/master/media/arcus.png)

# Positioning

With Arcus we are offering an open source library that streamlines Azure ML development, but lets ML engineers focus on the actual job at hand, without loosing time in tinkering with the AzureML SDK and all overhead that comes with it.

We offer the following concepts:

- Connectivity to an Azure ML workspace
- Start experiments on your local development environment (from within Jupyter notebooks or in plain .py scripts)
- Automatically track every run in an experiment on the Azure ML workspace (even if you execute everything locally).  This way you get a perfect overview of all training attempts, the actual parameters, the results and the persisted models
- Provide tracking & tracing of Grid Searches for Hyper parameter tuning
- Enable trainings (through standard scripts and dependency files) to be executed locally and in the cloud, using the exact same logic and code.

# Documentation
All documentation can be found on [here](https://azure-ml.arcus-azure.net/).

# Installation

The Arcus packages are available through PyPi and can be installed through pip, by executing the following command:

```shell
PM > pip3 install arcus-azureml
```

Upgrading to the latest version can be done by executing the following pip command:

```shell
PM > pip3 install --upgrade arcus-azureml 
```

Every CI build pushes a dev package to the PyPi feed.  And when committed, an alpha release is been published on the same feed.  These packages can be installed or upgrade, by leveraging the `--pre` argument for `pip`.

```shell
PM > pip3 install --upgrade --pre arcus-azureml
```

# Local development
    
It can be quite common that you are using the arcus-ml or arcus-azureml packages on other projects and you need some changes or additional functionality being added to the package.  Obviously, it's possible to follow the entire release pipeline (make a PR, get it approved and merged, wait for package to appear on the PyPi feed and upgrade the package).  This workflow is too tedious and will not increase your productivity.

The approach to go, is to leverage the following command, which will add a symbolic link to your development directory from the python packages.  That way, you always refer to the latest code that is on your development environment.  It is advised, though not required to leverage virtual environment (like with conda) for this.

```shell
pip install -e /path-to-arcus
```

# Customers
Are you an Arcus user? Let us know and [get listed](https://bit.ly/become-a-listed-arcus-user)!

# License Information
This is licensed under The MIT License (MIT). Which means that you can use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the web application. But you always need to state that Codit is the original author of this web application.
