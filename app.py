from flask import Flask, request, url_for, redirect, render_template, jsonify
import pickle
import numpy as np
from tensorboard.compat.tensorflow_stub.tensor_shape import vector

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    output = ''
    return render_template('index.html')
vect = np.vectorize(np.float_)
ab = []
@app.route('/predict', methods=['POST','GET'])
def predict():
    for x in request.form.values():
        ab.append(x)

    print(ab)
    float_features = [float(x) for x in request.form.values()]
    print(float_features)
    final_features = [np.array(float_features)]
    prediction = model.predict_proba(final_features)
    print("Prediction",prediction)
    pred_in_float = prediction
    print(pred_in_float)
    output = '{0:.{1}f}'.format(pred_in_float[0][1],2)
    #return render_template('index.html', output="pred_in_float")


    if output > '0.5':
             return render_template('index.html',
                             output='Patient needs to be admitted in in-care ward!\nProbability of in-care treatment is {}'.format(output))
    else:
           return render_template('index.html',
                             output='Patient can go home.\n Probability of out-care treatment is is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
