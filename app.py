import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# read our pickle file and label our logisticmodel as model
model = pickle.load(open('model_classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    
    if prediction==0:
        return render_template('Index.html',
                               prediction_text='Survived'.format(prediction[0]),
                               )
    else:
        return render_template('Index.html',
                               prediction_text='Not Survived'.format(prediction[0]),
                              )



if __name__ == "__main__":
    app.run(debug=True)