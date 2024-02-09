import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    reason_code =''

    if output == 0:
        output_text = 'Good'
        decision_text = 'Approved'
        reason_code = 'Customer approved'

    else:
        output_text = 'Bad'
        decision_text = 'Rejected'

        if final_features[0][0] == 33:
            reason_code = "Maximum Delinquency is 60 days (CUstomer shows highly delinquent behavior)"

        elif final_features[0][0] == 6:
            reason_code = "rc2"

        else:
            reason_code = "rc3"
        

    #prediction_string = str(output)+"Credit Prediction is:"+output_text+"\n"+"Credit Decision is:"+decision_text+"Reason Code is:"+reason_code
    
    return render_template('index.html', prediction_text = "Credit Prediction for this customer is:"+output_text+ ",   Credit Decision for this customer is:"+decision_text+",    Reason Code for this decision is:"+reason_code)



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
