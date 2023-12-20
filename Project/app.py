#flask code
from flask import Flask,render_template,url_for,request
import pandas as pd 
import joblib

model  = joblib.load('xgboost_model.lb')
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == "POST":
        # get the user input
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])


        test_point = pd.DataFrame({'age':[age],'sex':[sex],'bmi':[bmi],'children':[children],'smoker':[smoker],'region':[region]})
        pred = model.predict(test_point)

        return render_template('result.html',prediction=round(pred[0],2))



if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8080",debug=True)
