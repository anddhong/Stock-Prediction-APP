from flask import Flask,render_template, abort, request
import iex_collect, threeOptions
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import json


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('profile.html')

@app.route('/recommend/')
@app.route('/recommend/<string:ticker>')
def recommend(ticker='aapl'):
	model=threeOptions.execution(ticker)
	if type(model)==str: works=False
	else: works=True
	return render_template('recommend.html',ticker=ticker,model=model,works=works)

@app.route('/minuteData/')
@app.route('/minuteData/<string:ticker>/')
def minute(ticker='aapl'):
	stock=iex_collect.stock_info(ticker)
	stock_info=stock.create_ranged_dataset('1d').tail(20)
	return render_template('minuteData.html', stock_info=stock_info.to_html(classes="table table-striped"),ticker=ticker)

#-----------------------------------#

if __name__=='__main__':
	app.run(debug=True)


#-----------

