import tensorflow as tf
import numpy as np
import requests
import json
import keras
import base64
from PIL import Image
import io
from tensorflow.python.ops.numpy_ops import np_config

char_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'a', 'b', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'p', 'q', 's', 't', 'u', 'v', 'x', 'y', 'z']
batch_size = 20
img_width = 280
img_height = 70
max_length = 6

char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=list(char_), num_oov_indices=1,
)
num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), invert=True
)
char_ = list(set(char_))
encoded_text = char_to_num(char_)
encoded_text = num_to_char(encoded_text)
def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Padding độ dài nhãn để đảm bảo chúng có cùng độ dài
    label = tf.pad(label, [[0, max_length - tf.shape(label)[0]]])
    return {"image": img, "label": label}


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def preprocess_base64_image(base64_data):
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))
    img = image.convert("L")
    img = img.resize((280, 70))
    img = np.array(img)
    img = img.astype(np.float32)/255.0
    img = tf.convert_to_tensor(img)
    img = tf.transpose(img, perm=[1, 0])
    img = tf.expand_dims(img, axis=-1)

    return img
def make_prediction(base64_string):
    url = 'http://localhost:8601/v1/models/solveCaptchas:predict'
    img = preprocess_base64_image(base64_string)
    np_config.enable_numpy_behavior()
    zeros_array = np.zeros((1, 6))
    instances = [
        {
            "image": img.tolist(),
            "label": zeros_array.tolist()
        }
    ]
    data = json.dumps({"signature_name": "serving_default", "instances": instances})
    headers = {'Content-Type': 'application/json'}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions
def UseModel(base64_string):
    arrayPrediction = np.array(make_prediction(base64_string))
    decoded_prediction = decode_batch_predictions(arrayPrediction)[0]
    return decoded_prediction
