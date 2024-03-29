# Machine learning methods on stock data

<div align="center">
  <a href="https://github.com/HoangNguyenM/" target="_blank">Hoang&nbsp;Nguyen</a>
</div>
</p>

--------------------

This repo contains my implementation on some potential deep learning models on stock trading, including black-box Bayesian optimization with different strategies, CNN, DQN Reinforcement Learning, and a transformer based model called MANet that I am developing. You can perform downloading data with yahoofinance or AlphaVantage by running main_download.py, try Bayesian optimization by running main_bayes.py or try DL models with main.py. This project mainly serves as my practice of ML models, hence the code will change regularly, and the documentation unfortunately may not be very detailed. However, if you find my code useful and would like to use it, please kindly consider crediting the source.

## Requirements

This project is built in Python 3.11.4 using CUDA 12.2 and CUDNN 8.9.4. Please use the following command to install the requirements:
```shell script
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu118
``` 
