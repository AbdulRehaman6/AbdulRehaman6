import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
df = pd.read_csv('liver.csv')
#df = pd.read_csv('cleaned_liver_disease_dataset.csv')
X = df.drop('Result', axis=1)
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model
# model = joblib.load('best_model .pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    input_array = [input_features]
    prediction = model.predict(input_array)
    output = "Liver disease is present." if prediction[0] == 1 else "Liver disease is not present."
    
    return render_template('index.html', prediction_text='Predicted Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
