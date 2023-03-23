import  numpy as np
import pickle
from flask import Flask, render_template, request
model=pickle.load(open('model.pkl','rb'))
label=pickle.load(open('label.pkl','rb'))
app=Flask(__name__)
@app.route('/')  
def home():
    return render_template('home.html')   
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    age=request.form['age']
    workclass=request.form['workclass']
    education=request.form['education']
  
    occupation=request.form['occupation']
    sex=request.form['sex']
   
    hoursperweek=request.form['hoursperweek']
   
    
    values=[age,workclass,education,occupation,
            sex,hoursperweek]
    train=[]
    for i in values:
        train.append(i)
    train = label.fit_transform(train)
    
    result=np.array(train).reshape(1,-1)
    prediction=model.predict(result)
    if(prediction[0]==0):
        output="less than or equal to 50k USD"
    else:
        output="greater than 50k USD"
    
    #output=prediction.item()
    
    return render_template('index1.html',prediction_text=format(output))


if __name__ == "__main__":                            
    app.run()
    
