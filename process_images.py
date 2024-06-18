from flask import Flask, jsonify, request
from flask_cors import CORS
import boto3
import tensorflow as tf
import os
import json
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

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

model = tf.keras.models.load_model('image_recognition_model.h5')

@app.route('/api/process_image')
def process_image(filename):
    s3_client.download_file(S3_BUCKET, filename, '/tmp/' + filename)
    
    img = tf.keras.preprocessing.image.load_img('/tmp/' + filename, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    
    results = {'filename': filename, 'predicted_class': int(predicted_class)}

    results_key = f'results/{filename}.json'
    s3_client.put_object(Bucket=S3_BUCKET, Key=results_key, Body=json.dumps(results))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)