from flask import Flask, request, render_template, session
import pandas as pd
import pickle


app = Flask(__name__)
app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'


champions = pd.read_json("champions.json")
countvectorizer = pickle.load(open( "countvectorizer.p", "rb" ))
logisticregressor = pickle.load(open("logistic_regression.p", "rb"))
scores = pickle.load(open("scores.p", "rb"))




@app.route('/')
def home():

	return render_template('index.html', champions = champions)
	
@app.route('/predict',methods=['POST'])
def predict():
	comp = [str(champions.key[x]) for x in request.form.values()]
	session["comp"] = [*request.form.values()]
	
	
	blue = ",".join(comp[:5])
	red = ",".join(comp[5:])
	
	
	
	result = logisticregressor.predict_proba((countvectorizer.transform([blue]) - countvectorizer.transform([red])).toarray())[0]
	
	return render_template('result.html',champions=champions,result=result,comp = session["comp"])
	
@app.route('/scoring',methods=['POST','GET'])
def scoring():

	with pd.option_context('display.max_colwidth', -1): 
		#render dataframe as html
		html = scores.to_html(escape=False,index=False,justify="center")

	return render_template('scores.html',table=html)
	
if __name__ == "__main__":
    app.run(debug=True)