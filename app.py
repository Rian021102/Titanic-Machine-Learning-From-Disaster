import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model_classifier.pkl','rb'))

@app.route('/',methods=["GET","POST"])
def home():
    return render_template('Index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if prediction==0:
            return render_template('Index.html',
                                  prediction_text='Survived'.format(prediction),
                                  )
    else:
            return render_template('Index.html',
                                  prediction_text='Not Survived'.format(prediction),
                                  )
if __name__ == "__main__":
    app.run(debug=True)