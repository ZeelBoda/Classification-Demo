import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Flask related imports
from flask import Flask, request, render_template

# Streamlit related imports
import streamlit as st

# Load the model (ensure the model file is correct and exists)
model_path = r'C:\Users\ZEEL\Desktop\Classification\Fruits_Vegetables\Fruits_Vegetables\Image_classify.keras'
model = load_model(model_path)

data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
            'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
            'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
            'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
            'turnip', 'watermelon']

img_height = 300
img_width = 300

# Flask App setup
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['fruit']
        filename = f.filename

        target = os.path.join(APP_ROOT, 'images/')
        if not os.path.isdir(target):
            os.mkdir(target)
        
        des = os.path.join(target, filename)
        f.save(des)

        if not os.path.exists(des):
            return "File not found", 404

        try:
            test_image = tf.keras.utils.load_img(des, target_size=(img_height, img_width))
            test_image = tf.keras.utils.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0

            prediction = model.predict(test_image)
            predicted_class = data_cat[np.argmax(prediction[0])]
           
            confidence = round(np.max(prediction[0]) * 100)
            print("prediction is " , predicted_class + " with confidence " + str(round(np.max(prediction[0]) * 100) ) + "%")
            print(np)

            return render_template("prediction.html", confidence="Chances -> " + str(confidence) + "%", prediction="Prediction -> " + str(predicted_class))
        except Exception as e:
            return f"Error processing file: {e}", 500
    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.debug = True
    app.run()

# Streamlit setup
st.header('Image Classification Model')

# Streamlit image input
image = st.text_input('Enter Image name', r'C:\Users\ZEEL\Desktop\Classification\Fruits_Vegetables\Fruits_Vegetables\corn.jpg')
image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image_load)
img_bat = np.expand_dims(img_arr, 0)

# Prediction
predict = model.predict(img_bat)
score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write(f'Vegetable/Fruit in image is {data_cat[np.argmax(score)]}')
st.write(f'With accuracy of {np.max(score) * 100:.2f}%')
