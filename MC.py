#date 23/06/2025
import yfinance as yf, numpy as np, matplotlib.pyplot as plt , datetime as dt , random, string, io
import matplotlib 
from io import BytesIO
import json
from flask import Flask, request, jsonify, Response, render_template
from PIL import Image
import os, uuid, base64
from flask_cors import CORS
import pandas as pd 
import pandas_datareader as pdr
from flask import Flask

#importing the data from yahoo,  and bilding a function that returns the meanReturns and a covarianve matrix
monte_carlo_stockList = [ 'TSLA', 'AAPL', 'AMZN' , 'MSFT']   #definig stocklist 
stocks = monte_carlo_stockList
def get_data(stocks, start, end):
    stocksData = yf.download(stocks, start, end)
    stocksData =stocksData['Close']            #only taking "close" prices since where are only intrested in daily disturbences in the stock price
    returns = stocksData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix       # using the returns, we calculate th emeans return and the covariance matrix
    
endDate = dt.datetime.now()
startDate= endDate - dt.timedelta(days=300)
meanReturns,covMatrix = get_data(stocks, startDate, endDate)

#assigning random wieghts to the portfolio
weights = np.random.random(len(meanReturns))
weights /= (np.sum(weights))

#setting up the Monte Carlo simulation
#number of simulations 
mc_sims =100 
T = 100 # timeframe in days 

#generating 100 radndom future price paths over 100 trading days
meanM =np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
#creating a constant matrix of mean returns for the simulation
portfolio_sims= np.full(shape=(T, mc_sims),fill_value=0.0 )
initialPortfolio =10000

#runing the monte carlo simulation
for m in range(0, mc_sims): # Mc loops
    Z= np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)  # will workout what the triangle is for a cholesky decompisition 
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio
plt.ylabel('portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a portfolio')
plt.plot(portfolio_sims)

#saving the plot to a buffer, 
buf = io.BytesIO()
plt.savefig(buf, format='png')
plt.close()
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')

app= Flask(__name__)
with open('my_graph.json', 'w') as f:
    json.dump({'image': img_base64},f)
    
@app.route('/')
def send_graph():
    with open('my_graph.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__Backtest__':
    app.run(debug=True)
CORS(app)
#take the ploted graph store it an a server using flask, fetch the graph from the flask server using react

@app.after_request
def add_csp(response):
    response.headers['Content-Security-Policy'] = "script-src 'self' 'unsafe-eval';"
    return response

def id ():
    chars = string.ascii_letters
    chars += string.digits
    chars += string.punctuation
    password= ""
    for i in range(10):
        nextchar= random.choice(chars)
        password += nextchar  
    next
    return password

