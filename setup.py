from setuptools import setup, find_packages

setup(
    name="ml-multiclass-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "tensorflow>=2.5.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning application for multi-class classification",
)