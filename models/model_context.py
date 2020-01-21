import logging
import uuid
from typing import Optional, Dict, Any

from models.keras_model import KerasRetinaNet
from models.model import Model
from models.onnx_model import TTFNetOnnxModel

__all__ = ['model_type_to_model_type_name', 'ModelsContext']

model_type_to_model_type_name = {
    'kerasretinanet': KerasRetinaNet,
    'ttfnetonnx': TTFNetOnnxModel
}


class ModelsContext:
    def __init__(self):
        self._models: Dict[str, Model] = dict()

    def model(self, uuid_str: str) -> Optional[Model]:
        if uuid_str not in self._models:
            return None
        return self._models[uuid_str]

    def create_model(self, model: str, new_model_config: Dict[str, Any]) -> Optional[str]:
        model_type = model_type_to_model_type_name[model]

        try:
            model = model_type(new_model_config)
            model.init()
            uuid_str = str(uuid.uuid4())
            self._models[uuid_str] = model
            return uuid_str
        except Exception as e:
            logging.error(f'Error occured due model creation: {e}')
            return None

    def del_model(self, uuid_str: str):
        if uuid_str not in self._models:
            return
        del self._models[uuid_str]
