from setuptools import setup, find_packages

setup(
    name="stocks_forecasting",
    version="0.4.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
        "plotly",
        "statsmodels",
        "xgboost",
        "tbats",
        "joblib",
    ],
    python_requires=">=3.8",
    author="Arthur Ferreira, Augusto Camargo",  
    author_email="200056981@aluno.unb.br, augusto.camargo@aluno.unb.br",
    description="A collection of stock prediction models including LSTM, SARIMA, TBATS and more",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EPS-ALM/stocks-forecasting",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 