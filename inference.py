import argparse
import json
import time
from typing import Optional

import cv2
import numpy as np
import pybase64
from flask import Flask, jsonify, request, abort, Response

from models.model_context import ModelsContext, model_type_to_model_type_name
from structs import Size2D

app = Flask(__name__)
models_context = ModelsContext()
current_model_hash: Optional[str] = None
labels_to_names = {0: 'Pedestrian'}


@app.route('/')
def index():
    return jsonify({'status': "server is running"}), 200


@app.route('/create_model/', methods=['POST'])
def create_model():
    global current_model_hash

    required_keys = 'model_type', 'model_params'

    if not all([k in request.args for k in required_keys]):
        abort(400)

    model_type_name = request.args['model_type'].lower()
    if model_type_name not in model_type_to_model_type_name:
        abort(Response(f'No such detector type: {model_type_name}. '
                       f'Get one from this variants: {model_type_to_model_type_name.keys()}'))

    current_model_hash = models_context.create_model(model_type_name,
                                                     json.loads(request.args['model_params']))
    return current_model_hash, 200


@app.route('/image/', methods=['POST'])
def predict_image():
    if not request.json or not 'data' in request.json:
        abort(400)

    if current_model_hash is None:
        abort(Response('No models created'))

    caption = run_detection_image(request.json['data'])
    return caption, 200


def run_detection_image(image_data):
    imgdata = pybase64.b64decode(image_data)
    file_bytes = np.asarray(bytearray(imgdata), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_size = Size2D(width=image.shape[1], height=image.shape[0])
    model = models_context.model(current_model_hash)

    print("start predict...")
    start_time = time.time()
    bboxes = model.predict(image)
    print("done in {} s".format(time.time() - start_time))

    objects = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(str, bbox.xyxy(image_size))
        objects.append({
            'name': labels_to_names[bbox.category_id],
            'score': str(bbox.confidence),
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    return json.dumps(dict(objects=objects))


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    app.run(debug=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
