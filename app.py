import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app=Flask(__name__)

reg_model=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    transformed_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=reg_model.predict(transformed_data)
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)