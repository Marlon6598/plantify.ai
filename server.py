from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model and define class_names
model = tf.keras.models.load_model('D:\Project 4\model.h5')
class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']  # List of class names corresponding to your model's output classes

# Define a dictionary mapping class names to descriptions
class_descriptions = {
    'Black-grass': 'Black-grass (Alopecurus myosuroides) is an annual grass weed commonly found in arable fields, known for its competitive nature and resistance to herbicides.',
    'Charlock': 'Charlock (Rhamphospermum arvense) is an annual weed with bright yellow flowers, commonly found in agricultural fields and disturbed areas.',
    'Cleavers': 'Cleavers (Galium aparine) is a herbaceous plant with clinging stems and small hooked hairs, commonly found in moist, shady areas.',
    'Common Chickweed' : 'Common Chickweed (Stellaria media) is a low-growing annual plant with small white flowers, often considered a common weed in gardens and lawns.',
    'Common wheat' : 'Fat Hen (Chenopodium album) is an annual weed with broad, triangular leaves, commonly found in agricultural fields and disturbed areas, known for its edible and nutritious young leaves.',
    'Fat Hen': 'Fat Hen (Chenopodium album) is an annual weed with broad, triangular leaves, commonly found in agricultural fields and disturbed areas, known for its edible and nutritious young leaves.',
    'Loose Silky-bent' : 'Loose Silky-bent (Apera spica-venti) is an annual grass weed with loose, drooping seed heads and silky, slender leaves, often found in agricultural fields and disturbed areas.',
    'Maize' : 'Maize, also known as corn (Zea mays), is a cereal crop widely cultivated for its edible kernels, used in various food and industrial applications.',
    'Scentless Mayweed' : 'Scentless Mayweed (Tripleurospermum inodorum) is a common annual plant with daisy-like flowers and finely divided leaves, found in various habitats, known for its lack of noticeable fragrance.',
    'Shepherds Purse' : 'Shepherds Purse (Capsella bursa-pastoris) is a small annual weed with heart-shaped seed pods, commonly found in disturbed areas, known for its distinctive triangular leaves and medicinal properties.',
    'Small-flowered Cranesbill' : 'Small-flowered Cranesbill (Geranium pusillum) is a perennial plant with small, delicate flowers, typically found in meadows and open woodlands, known for its ornamental value and attractive foliage.',
    'Sugar beet' : 'Sugar beet (Beta vulgaris) is a cultivated crop known for its large, white, and fleshy root that contains high sugar content, commonly used for sugar production and animal feed.'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image = request.files['image']

    # Save the image to a temporary location
    image_path = 'temp.jpg'
    image.save(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    class_index = np.argmax(predictions[0])
    class_name = class_names[class_index]
    confidence = predictions[0][class_index]

    # Get the description for the predicted class
    description = class_descriptions.get(class_name, 'Description not available.')

    # Prepare the response as JSON
    response = {
        'class_name': class_name,
        'confidence': float(confidence),
        'description': description
    }

    return jsonify(response)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Adjust the size based on your model's input shape
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == '__main__':
    app.run()
