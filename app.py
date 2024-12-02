import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask , request , render_template

model = pickle.load(open("credit_card_model.pkl" , "rb"))
ss = pickle.load(open("Scaling_data.pkl" , "rb"))

app = Flask(__name__)

@app.route("/")
def home() :
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def predict():
    a = [i for i in request.form.values()]
    s = ss.transform(np.array(a[10 : 15]).reshape(1,-1))
    rem = np.array(a[:10]).reshape(1,-1)
    res = np.concatenate( [rem , s] , axis = 1)
    result = model.predict(res)[0]
    if result == 0:
        return render_template('index.html', prediction_text='Bad Customer')
    else:
        return render_template('index.html', prediction_text='Good Customer')


if __name__ == '__main__':
    app.run(debug=True)