from flask import Flask, jsonify, request
import tensorflow as tf
from PIL import Image
import os
import boto3
import json
from dotenv import load_dotenv

app = Flask(__name__)

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '.env.local'))

S3_BUCKET = os.getenv('S3_BUCKET')
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.getenv('SECRET_KEY_ID')

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
)

# TF model
model = tf.keras.models.load_model('image_recognition_model_cifar100.h5')

# Model's labels
labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]


def process_image(filename):
    try:
        local_filename = filename
        s3_client.download_file(S3_BUCKET, filename, local_filename)
        print('File downloaded locally.')

        img = Image.open(local_filename)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = img.resize((32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_index = tf.argmax(prediction[0]).numpy()
        predicted_label = labels[predicted_index]

        result = {
            'prediction': prediction.tolist(),
            'predicted_label': predicted_label
        }

        result_filename = f'{filename}.json'
        s3_client.put_object(Bucket=S3_BUCKET, Key=result_filename, Body=json.dumps(result))
        print(f"Prediction stored in S3: {result_filename}")

    except Exception as e:
        print(f"Error processing image: {e}")
        error_result = {'error': 'Error processing image'}
        result_filename = f'{filename}.json'
        s3_client.put_object(Bucket=S3_BUCKET, Key=result_filename, Body=json.dumps(error_result))

    finally:
        try:
            os.remove(local_filename)
            print(f"Deleted local file: {local_filename}")
        except Exception as e:
            print(f"Error deleting file: {e}")

        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=filename)
            print(f"Deleted file from S3: {filename}")
        except Exception as e:
            print(f"Error deleting file from S3: {e}")


@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    data = request.get_json()
    filename = data['filename']
    process_image(filename)
    return jsonify({'message': 'Processing started'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)