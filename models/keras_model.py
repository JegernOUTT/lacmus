from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import tensorflow as tf
import keras.models as KM
import numpy as np

from models.keras_retinanet import custom_objects
from structs import Bbox, Size2D, bbox_from_xyxy
from models.model import Model
from utils.gpu import setup_gpu_if_exists


__all__ = ['KerasRetinaNet']


class KerasModel(Model):
    def __init__(self, config: Dict[str, Any]):
        assert 'filename' in config, f'Set filename path for {str(self)} model'
        super().__init__(config)
        self._filename = Path(config['filename'])
        assert self._filename.exists(), f'Model path invalid: {self._filename}'
        self._model = None
        self._session = None
        self._graph = None

    def init(self):
        self._session = tf.InteractiveSession()
        self._graph = tf.get_default_graph()
        setup_gpu_if_exists(self._config['gpu'] if 'gpu' in self._config else 0)
        self._model = KM.load_model(str(self._filename), custom_objects=self._get_custom_objects())

    def predict(self, image: np.ndarray) -> List[Bbox]:
        assert self._model is not None, f'Init model first'
        assert self._session is not None, f'Init model first'
        assert self._graph is not None, f'Init model first'

        with self._session.as_default():
            with self._graph.as_default():
                preprocessed_image = self._preprocess_image(image)
                output = self._model.predict_on_batch(preprocessed_image)
                return self._postprocess_bboxes(output)

    @abstractmethod
    def _get_custom_objects(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess_bboxes(self, model_output: Any) -> List[Bbox]:
        pass


class KerasRetinaNet(KerasModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert 'backbone_type' in config, 'Backbone type required in model config'
        self._backbone_type: str = config['backbone_type']
        assert 'width' in config and 'height' in config, 'Width and height required in model config'
        self._model_image_size = Size2D(width=config['width'], height=config['height'])

        self._current_scale = None
        self._current_image_size = None

    def _get_custom_objects(self) -> Dict[str, Any]:
        objects = custom_objects
        if self._backbone_type.lower().find('resnet') != -1:
            import keras_resnet
            objects.update(keras_resnet.custom_objects)
        return objects

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        self._current_image_size = Size2D(width=image.shape[1], height=image.shape[0])
        image = self._normalize_image(image)
        image, scale = self._resize_image(image)
        self._current_scale = scale
        return image[np.newaxis, ...]

    def _postprocess_bboxes(self, model_output: Any) -> List[Bbox]:
        boxes, scores, labels = model_output
        boxes /= self._current_scale

        bboxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            bbox = bbox_from_xyxy(box, image_size=self._current_image_size,
                                  confidence=score, category_id=label)
            bboxes.append(bbox)
        return bboxes

    def __repr__(self):
        return f'KerasRetinaNet: [{self._width}x{self._height}]'

    def _normalize_image(self, image, mode='caffe'):
        """ Preprocess an image by subtracting the ImageNet mean.

        Args
            x: np.array of shape (None, None, 3) or (3, None, None).
            mode: One of "caffe" or "tf".
                - caffe: will zero-center each color channel with
                    respect to the ImageNet dataset, without scaling.
                - tf: will scale pixels between -1 and 1, sample-wise.

        Returns
            The input with the ImageNet mean subtracted.
        """
        # mostly identical to
        # "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
        # except for converting RGB -> BGR since we assume BGR already
        x = image.astype(np.float32)

        if mode == 'tf':
            x /= 127.5
            x -= 1.
        elif mode == 'caffe':
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
        return x

    def _compute_resize_scale(self, image_size: Size2D) -> float:
        """ Compute an image scale such that the image size is constrained to min_side and max_side.

        Returns
            A resizing scale.
        """
        smallest_side = min(image_size.height, image_size.width)
        min_side, max_side = min(self._width, self._height), max(self._width, self._height)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(image_size.height, image_size.width)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        return scale

    def _resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """ Resize an image such that the size is constrained to min_side and max_side.

        Args
            image: Image to resize

        Returns
            A resized image.
        """
        # compute scale to resize the image
        scale = self._compute_resize_scale(Size2D(width=image.shape[1], height=image.shape[0]))

        # resize the image with the computed scale
        img = cv2.resize(image, None, fx=scale, fy=scale)

        return img, scale
