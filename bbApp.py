from flask import Flask,render_template, abort, request
import iex_collect, threeOptions
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import json


app = Flask(__name__)

app.config.update(
    SQLALCHEMY_DATABASE_URI = "postgresql://sfdbo:sfdbo0@10.1.10.39:5432/ara",
    SQLALCHEMY_TRACK_MODIFICATIONS=False)
db = SQLAlchemy(app)

'''  
@app.route('/', methods=['POST','GET'])
def dbinfo():
    errMsg=''
    try:
        name=request.args['category']
    except:
        errMsg='ticker not assign, use AAPL as default'
        name='indicator'
    df=pd.read_sql("SELECT * from custom_cmt_temp",db.engine)
    cat=pd.read_sql("SELECT DISTINCT category FROM custom_cmt_temp",db.engine)
    cmt=pd.read_sql("SELECT DISTINCT cmt_type FROM custom_cmt_temp",db.engine)
    sub=pd.read_sql("SELECT DISTINCT sub_type FROM custom_cmt_temp",db.engine)



    ds=df.to_json(orient='records')
    #return(ds)
    d=json.loads(ds)[0]
    d.update(error_message=errMsg,len=len)

    return(render_template("bbHtmlTemplate.html",**d,cat,cmd,sub))
'''


@app.route('/')
def index():
	return render_template('bbHtmlTemplate.html')

@app.route('/mp4')
def mp4():
	return render_template('ticker_mp4.html')

@app.route('/mp4', methods=['POST'])
def mp4_post():
	ticker = request.form['text']
	return render_template('ticker_mp4.html',video=str('/static/'+ticker+'.mp4'))
	#r'C:\Users\andy\PycharmProjects\summer'+ '\\'+

#-----------------------------------#

if __name__=='__main__':
	app.run(debug=True)
