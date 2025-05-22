import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('mnist_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert to numpy array
    image = image.reshape((1, 28, 28, 1))  # Reshape for the model
    image = image.astype('float32') / 255  # Normalize
    return image

# Create the Streamlit app
st.title('MNIST Digit Classifier')
st.write('Upload an image of a handwritten digit (0-9)')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.write(f'Predicted Class: {predicted_class}')
