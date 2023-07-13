from flask import Flask, render_template, request
from tensorflow import keras
import numpy as np
from PIL import Image

app = Flask(__name__)
model = keras.models.load_model('D:\Project\Model\project4_model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded file from the POST request
        file = request.files['file']
        # Read the image file using PIL
        image = Image.open(file)
        # Preprocess the image
        image = preprocess_image(image)
        # Make a prediction using the loaded model
        prediction = model.predict(image)
        # Get the predicted class label
        predicted_class = np.argmax(prediction)
        # Get the class name based on the predicted class
        class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet'] 
        predicted_class_name = class_names[predicted_class]
        # Render the HTML template with the prediction result
        return render_template('result.html', predicted_class=predicted_class_name)
    # Render the initial HTML template for image upload
    return render_template('index.html')

def preprocess_image(image):
    # Resize the image to match the input dimensions expected by the model
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Normalize the pixel values
    image_array = image_array / 255.0
    # Expand dimensions to match the input shape expected by the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if __name__ == '__main__':
    app.run()




