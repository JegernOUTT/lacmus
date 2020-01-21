from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as rt

from models.model import Model
from structs import Bbox, Size2D


class OnnxModel(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert 'filename' in config, f'Set filename path for {str(self)} model'
        self._filename = Path(config['filename'])
        self._onnx_session: Optional[rt.InferenceSession] = None
        self._input_name: Optional[str] = None

    def init(self):
        so = rt.SessionOptions()
        so.intra_op_num_threads = 8
        so.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._onnx_session = rt.InferenceSession(str(self._filename), sess_options=so)
        inputs = self._onnx_session.get_inputs()
        assert len(inputs) == 1, "Onnx model must contain 1 image input"
        self._input_name = inputs[0].name

    def predict(self, image: np.ndarray) -> List[Bbox]:
        assert self._onnx_session is not None, f'Init session first'
        assert self._input_name is not None, f'Init session first'

        preprocessed_image = self._preprocess_image(image)
        output = self._onnx_session.run(None, {self._input_name: preprocessed_image})
        return self._postprocess_bboxes(output)

    @abstractmethod
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess_bboxes(self, model_output: Any) -> List[Bbox]:
        pass


class TTFNetOnnxModel(OnnxModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert 'width' in config and 'height' in config, 'Width and height required in model config'
        self._width = config['width']
        self._height = config['height']

        self._current_scale = None
        self._current_image_size = None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image, scale = self._resize_image(image)
        self._current_scale = scale
        return np.transpose((image / 255.).astype(np.float32), (2, 0, 1))[np.newaxis, ...]

    def _postprocess_bboxes(self, model_output: Any) -> List[Bbox]:
        return []

    def __repr__(self):
        return f'TTFNetOnnxModel: [{self._width}x{self._height}]'

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
