"""
Setup script for the sleep_dataset_generator package.

Allows `pip install -e .` so all notebooks can import from src/
without path manipulation.

Authors: Rushav Dash, Lisa Li — Team 7, TECHIN 513, UW
"""

from setuptools import setup, find_packages

setup(
    name="sleep_dataset_generator",
    version="1.0.0",
    description="Synthetic Sleep Environment Dataset Generator — TECHIN 513 Team 7",
    author="Rushav Dash, Lisa Li",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "statsmodels>=0.14.0",
        "kagglehub>=0.2.0",
    ],
)
