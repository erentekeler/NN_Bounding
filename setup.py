from setuptools import setup, find_packages

setup(
    name="nn_bounding",
    version="0.1.0",
    install_requires = ["h5py==3.16.0",
                    "matplotlib==3.10.8",
                    "numpy==2.4.4",
                    "openpyxl==3.1.5",
                    "pandas==3.0.2",
                    "seaborn==0.13.2",
                    "torch==2.8.0",
                    "gurobipy==13.0.0",
                    "onnx==1.17.0",
                    "scipy==1.15.2"],
    packages=find_packages())