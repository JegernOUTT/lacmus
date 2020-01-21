from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.model import Model

import numpy as np
import onnxruntime as rt

from structs import Bbox


class OnnxModel(ABC, Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert 'filename' in config, f'Set filename path for {str(self)} model'
        super().__init__(config)
        self._filename = Path(config['filename'])
        self._onnx_session: Optional[rt.InferenceSession] = None
        self._input_name: Optional[str] = None

    def init(self):
        self._onnx_session = rt.InferenceSession(str(self._filename))
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
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        return np.transpose((image / 255.).astype(np.float32), (2, 0, 1))[np.newaxis, ...]

    def _postprocess_bboxes(self, model_output: Any) -> List[Bbox]:
        pass
