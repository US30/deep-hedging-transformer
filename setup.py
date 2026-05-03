from setuptools import setup, find_packages

setup(
    name="deephedge",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26",
        "scipy>=1.12",
        "pandas>=2.0",
        "yfinance>=0.2.36",
        "matplotlib>=3.8",
    ],
)
