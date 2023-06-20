from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

app = Flask(__name__)

# Prints the name of iris species from the predicted number
def decode(num):
    for i in num:
        if i==0:
            return "setosa"
        elif i==1:
            return "versicolor"
            
        else:
            return "virginica"

# Define the class labels specific to your Iris flower dataset
target_names = ['Setosa', 'Versicolor', 'Virginica']

iris = load_iris()
test_ids = []

for i in range(0, 20):
    test_ids.append(i)
for i in range(50, 70):
    test_ids.append(i)
for i in range(100, 120):
    test_ids.append(i)

# Training data
train_data = np.delete(iris.data, test_ids, axis=0)
train_target = np.delete(iris.target, test_ids)

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    sepal_length = float(request.json['sepal_length'])
    sepal_width = float(request.json['sepal_width'])
    petal_length = float(request.json['petal_length'])
    petal_width = float(request.json['petal_width'])

    # Reshape the input data to be a 2D array
    data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    # Make predictions using the decision tree classifier
    prediction = clf.predict(data)

    # Map the predicted class index to the class label
    predicted_class = target_names[prediction[0]]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
