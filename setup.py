import setuptools
import arcus.azureml
import sys

from setuptools.command.test import test as TestCommand
from setuptools import find_namespace_packages

with open("package-description.md", "r") as fh:
    long_description = fh.read()
    
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

packages_to_import = setuptools.find_packages(exclude=['tests', 'docs', 'build'])

print('Package to import:')
print(packages_to_import)
print('=============')

setuptools.setup(
    name="arcus-azureml", # Replace with your own username
    version=arcus.azureml.__version__,
    author="Arcus",
    author_email="arcus-automation@codit.eu",
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    description="A Python library to improve MLOps methodology on Azure Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arcus-azure/arcus.azureml",
    packages=find_namespace_packages(include=['arcus.*'], exclude=['tests', 'docs', 'build', 'samples']),
    package_dir={'arcus.azureml': 'arcus/azureml'},
    include_package_data=True,
    namespace_packages=['arcus'],
    install_requires=['pandas', 'numpy', 'arcus-ml>=1.0.11', 'azureml-core', 'joblib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    extras_require={
        'testing': ['pytest'],
    }
)
