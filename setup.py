from setuptools import setup, find_packages

setup(
    name="offline_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "d3rlpy>=2.0.0",
        "wandb>=0.15.0",
        "mlflow>=2.7.0",
    ],
)
