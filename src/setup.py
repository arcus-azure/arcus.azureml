import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arcus-azureml", # Replace with your own username
    version="0.0.1",
    author="Sam Vanhoutte",
    author_email="sam.vanhoutte@codit.eu",
    description="A Python library to improve MLOps methodology on Azure Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arcus-azure/arcus.azureml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)