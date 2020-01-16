import numpy as np

from abc import ABC, abstractmethod

from typing import List, Any, Dict

from structs import Bbox


class Model(ABC):
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Bbox]:
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return self.__repr__()
