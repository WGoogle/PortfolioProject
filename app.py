from flask import Flask, jsonify 
app = Flask(__name__)

@app.route("/")
def index(): #return a basic message when link clicked
    return "This is your Portfolio Analyzer, look around."


import yfinance as y
import pandas as pd
import numpy as np
"""
when user requests/clicks url matching stock and ticker symbol, I need to create a function that returns stock info.
Now, I have created a Flask API, that if you add the "stock/ticker symbol" to the end of the link that was created from above programming
will give you a ton of info. on the stock
"""
@app.route("/stock/<ticker>")
def data_from_stock(ticker):
    try:
        stock = y.Ticker(ticker)
        history_stock = stock.history(period = "1mo")
        if history_stock.empty:
            return jsonify({"Error": f"There is no exisiting data for the ticker {ticker}!"})
        return history_stock.to_json()
    except Exception as error:
        return jsonify({"error": "uh oh", "message": str(error)})

@app.route("/stock/<ticker>/history/<period>")
def periods_stock(ticker, period):
    try:
        desired_periods = ["1d", "1wk", "1mo", "1y", "5y", "max"]
        if period not in desired_periods:
            return jsonify({"Error": "That is not a valid period, the valid periods are 1d, 1wk, 1mo, 1y, 5y, max."})
        stock = y.Ticker(ticker)
        history_stock = stock.history(period = period)
        return history_stock.to_json()
    except Exception as error:
        return jsonify({"error": "uh oh", "message": str(error)})

@app.route("/stock/<ticker>/price")
def curr_stock_price(ticker):
    try:
        stock = y.Ticker(ticker)
        history_stock = stock.history(period = "1d").tail(1)["Close"].iloc[0] #get the most recent price with pandas documentation
        return jsonify({"ticker": ticker, "live price": history_stock})
    except Exception as error:
        return jsonify({"error": "uh oh", "message": str(error)})
    
#now that I have created the html side of input form, I need to use flask for backend
from flask import render_template, request, redirect, url_for

@app.route("/input")
def input():
    return render_template("inputform.html")

import sqlite3
@app.route("/addstock", methods = ["POST"])
def add():
    ticker = request.form.get("ticker")
    quantity = request.form.get("quantity")
    price = request.form.get("price")
    purchase_date = request.form.get("purchase_date")


    #now need to put information into db
    conn = sqlite3.connect("portfolio.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO portfolio (ticker, quantity, price, purchase_date) VALUES (?, ?, ?, ?)",
                   (ticker, quantity, price, purchase_date))
    conn.commit()
    conn.close()
    
    return redirect(url_for("portfolio_information"))
from datetime import datetime
@app.route("/portfolio")
def portfolio_information():
    
    conn = sqlite3.connect("portfolio.db")
    cur = conn.cursor()

    cur.execute("""SELECT ticker, quantity, price, purchase_date FROM portfolio""")
    stocks = cur.fetchall()

    conn.close()

    # Need to get a visaulized table of the stocks the user has 
    portfolio_information = []
    for stock in stocks:
        ticker, quantity, price, purchase_date = stock
        if purchase_date:
            purchase_date = datetime.strptime(purchase_date, "%Y-%m-%d")
            current_date = datetime.now()
            years_kept = (current_date - purchase_date).days / 365.25 
        # Get the current stock price using the yfinance lib
            real_stock = y.Ticker(ticker)
            current_price = real_stock.history(period="1d")["Close"]
            total_value = quantity * current_price
            plus_or_minus = (current_price - price) * quantity
            total_return_val = total_return(price, current_price)
            annualized_return_val = annualized_return(price, current_price, years_kept)
            volatility_val = volatility(ticker)
            sharpe_val = sharpe_ratio(ticker)
            sortino_val = sortino_ratio(ticker)

            portfolio_information.append({
                "ticker": ticker,
                "quantity": quantity,
                "purchase_price": price,
                "current_price": current_price,
                "purchase_date": purchase_date,
                "total_value": total_value,
                "plus_or_minus": plus_or_minus,
                "years_kept": years_kept,
                "total_return": total_return_val,
                "annualized_return": annualized_return_val,
                "volatility": volatility_val,
                "sharpe_ratio": sharpe_val,
                "sortino_ratio": sortino_val,
            })
    return render_template("portfolio_info.html", portfolio = portfolio_information)

#now I want to get more "advanced stats" rather than just taking stock info.
def total_return(price, current_price):
    return (current_price - price) / price * 100
def annualized_return(price, current_price, years_kept):
    total_ret = total_return(price, current_price)
    return (1 + total_ret) ** (1/ years_kept) - 1

import numpy as np
def volatility(ticker):
    stock = y.Ticker(ticker)
    history = stock.history(period="1y")
    history["Daily Return"] = history["Close"].pct_change()
    daily_returns = history["Daily Return"].dropna()
    
    sd = np.std(daily_returns) # get the standard dev. for formula
    annualized_volatility = sd * np.sqrt(252)  # 252 days for trading per year
    
    return annualized_volatility

def sharpe_ratio(ticker, risk_free_rate = 0.0377): #free risk rate for US Treasury 10 year
    stock = y.Ticker(ticker)
    history = stock.history(period="1y")
    history["Daily Return"] = history["Close"].pct_change()
    daily_returns = history["Daily Return"].dropna()
    
    avg_return = np.mean(daily_returns)
    sd = np.std(daily_returns)
    
    return (avg_return - risk_free_rate / 252) / sd

def sortino_ratio(ticker, target_return = 0.0377):
    stock = y.Ticker(ticker)
    history = stock.history(period="1y")
    history["Daily Return"] = history["Close"].pct_change()
    daily_returns = history["Daily Return"].dropna()
    
    down_returns = daily_returns[daily_returns < target_return]
    down_deviation = np.sqrt(np.mean(down_returns**2))
    
    avg_return = np.mean(daily_returns)
    
    return (avg_return - target_return) / down_deviation


# now I want to have some fun with my previous functions, so likely will try to visualize these risks. 
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import io
import base64

@app.route("/sharpe")
def sharpe_plot():
    conn = sqlite3.connect("portfolio.db")
    cur = conn.cursor()
    cur.execute("""SELECT ticker FROM portfolio""")
    stocks = cur.fetchall()
    conn.close()

    tickers = []
    sharpe_ratios = []

    for stock in stocks:
        ticker = stock[0]
        sharpe_val = sharpe_ratio(ticker)
        tickers.append(ticker)
        sharpe_ratios.append(sharpe_val)
    
    total_window, axis = plt.subplots()
    axis.bar(tickers, sharpe_ratios, color = "green")
    axis.set_title("The Sharpe Ratio of each Stock")
    axis.set_xlabel("Stock")
    axis.set_ylabel("Sharpe Ratio")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(total_window)

    return f'<img src="data:image/png;base64,{plot_url}" />'
    
@app.route("/sortino")
def sortino_plot():
    conn = sqlite3.connect("portfolio.db")
    cur = conn.cursor()
    cur.execute("""SELECT ticker FROM portfolio""")
    stocks = cur.fetchall()
    conn.close()

    tickers = []
    sortino_ratios = []

    for stock in stocks:
        ticker = stock[0]
        sortino_val = sortino_ratio(ticker)
        tickers.append(ticker)
        sortino_ratios.append(sortino_val)
    
    total_window, axis = plt.subplots()
    axis.bar(tickers, sortino_ratios, color = "green")
    axis.set_title("The Sortino of each Stock")
    axis.set_xlabel("Stock")
    axis.set_ylabel("Sortino Ratio")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(total_window)

    return f'<img src="data:image/png;base64,{plot_url}" />'
    

def monte_carlo_simulation(tickers, weights):
    historical_data = {}
    for t in tickers:
        stock = y.Ticker(t)
        hist = stock.history(period="5y")["Close"].pct_change().dropna()
        historical_data[t] = hist

    total_returns_table = pd.DataFrame(historical_data)
    total_returns_table = total_returns_table.dropna()



    simulations = []
    for i in range(10000):
        simulated_return = np.zeros(252)
        for t, w in zip(tickers, weights):
            sampled_data = np.random.choice(total_returns_table[t], 252)
            sampled_data = np.nan_to_num(sampled_data)
            simulated_return += sampled_data * w
        simulations.append(np.sum(simulated_return))
    
    results_table = pd.DataFrame(simulations, columns=["Return"])


    return results_table

def data_portfolio():
    conn = sqlite3.connect("portfolio.db")
    cur = conn.cursor()
    cur.execute("""SELECT ticker, quantity, price FROM portfolio""")
    stocks = cur.fetchall()
    conn.close()

    tickers = []
    quantities = []
    total_value = 0

    for stock in stocks:
        ticker, quantity, price = stock
        if quantity and price:
            tickers.append(ticker)
            quantity = float(quantity)
            price = float(price)
            total_value += quantity * price
            quantities.append(quantity * price)

    weights = [quantity / total_value for quantity in quantities]
    return tickers, weights

@app.route("/montecarlo")

def monte_carlo():
    tickers, weights = data_portfolio()

    resulting_table = monte_carlo_simulation(tickers, weights)

    figure, axis = plt.subplots()
    axis.hist(resulting_table["Return"], bins = 50, color = "green", alpha = 0.7)
    axis.set_title("Monte Carlo Simulation of Portfolio")
    axis.set_xlabel("Simulated Return")
    axis.set_ylabel("Times")

    image = io.BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)
    plot_url = base64.b64encode(image.getvalue()).decode()

    results_summarized = resulting_table.describe().to_html()
    return render_template("monte_carlo.html", results = results_summarized, plot_url=f"data:image/png;base64,{plot_url}")




if __name__ == "__main__":
    app.run(debug=True)
