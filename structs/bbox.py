from dataclasses import dataclass
from typing import List, Tuple, Union, Any

import numpy as np

BboxIterableType = Union[Tuple[float, float, float, float], List[float], np.ndarray]

__all__ = ['Bbox', 'iou', 'bbox_from_xyxy', 'bbox_from_xywh', 'bbox_from_center_xywh']


def _make_list(some):
    if isinstance(some, list):
        return some
    else:
        return [some]


@dataclass
class Bbox:
    xyxy_data: np.ndarray
    confidence: float
    category_id: int
    meta: Any = None

    def clip_coords(self):
        self.xyxy_data = np.clip(self.xyxy_data, 0., 1.)

    def width(self, image_size: Union['Size2D', None] = None) -> Union[int, float]:
        return float(self.xyxy_data[2] - self.xyxy_data[0]) if image_size is None else \
            int((self.xyxy_data[2] - self.xyxy_data[0]) * image_size.width)

    def height(self, image_size: Union['Size2D', None] = None) -> Union[int, float]:
        return float(self.xyxy_data[3] - self.xyxy_data[1]) if image_size is None else \
            int((self.xyxy_data[3] - self.xyxy_data[1]) * image_size.height)

    def area(self, image_size: Union['Size2D', None] = None) -> float:
        return float(self.width(image_size) * self.height(image_size))

    def _normalize_if_necessary(self, image_size: Union['Size2D', None] = None):
        if image_size:
            return self.xyxy_data * [image_size.width, image_size.height,
                                     image_size.width, image_size.height]
        else:
            return self.xyxy_data.copy()

    def xyxy(self, image_size: Union['Size2D', None] = None) -> np.ndarray:
        return self._normalize_if_necessary(image_size=image_size)

    def xywh(self, image_size: Union['Size2D', None] = None) -> np.ndarray:
        xyxy = self._normalize_if_necessary(image_size=image_size)
        return np.array([*xyxy[:2], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]], dtype=np.float32)

    def center_xywh(self, image_size: Union['Size2D', None] = None) -> np.ndarray:
        xyxy = self._normalize_if_necessary(image_size=image_size)
        return np.array([(xyxy[2] + xyxy[0]) / 2, (xyxy[3] + xyxy[1]) / 2, xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                        dtype=np.float32)


def iou(bbox_1: Union[Bbox, List[Bbox]], bbox_2: Union[Bbox, List[Bbox]]) -> Union[np.ndarray, float]:
    bbox_1_list, bbox_2_list = _make_list(bbox_1), _make_list(bbox_2)

    bbox_xyxy_data_1 = np.concatenate([bbox.xyxy_data for bbox in bbox_1_list], axis=0)
    bbox_xyxy_data_2 = np.concatenate([bbox.xyxy_data for bbox in bbox_2_list], axis=0)

    area = (bbox_xyxy_data_1[:, 2] - bbox_xyxy_data_1[:, 0]) * (bbox_xyxy_data_1[:, 3] - bbox_xyxy_data_1[:, 1])

    iw = np.minimum(np.expand_dims(bbox_xyxy_data_2[:, 2], axis=1), bbox_xyxy_data_1[:, 2]) \
         - np.maximum(np.expand_dims(bbox_xyxy_data_2[:, 0], 1), bbox_xyxy_data_1[:, 0])
    ih = np.minimum(np.expand_dims(bbox_xyxy_data_2[:, 3], axis=1), bbox_xyxy_data_1[:, 3]) \
         - np.maximum(np.expand_dims(bbox_xyxy_data_2[:, 1], 1), bbox_xyxy_data_1[:, 1])

    intersection_area = np.maximum(iw, 0) * np.maximum(ih, 0)

    union_area = np.expand_dims((bbox_xyxy_data_2[:, 2] - bbox_xyxy_data_2[:, 0]) \
                                * (bbox_xyxy_data_2[:, 3] - bbox_xyxy_data_2[:, 1]), axis=1) + area - iw * ih
    union_area = np.maximum(union_area, np.finfo(float).eps)

    iou = intersection_area / union_area

    if isinstance(bbox_1, Bbox) and isinstance(bbox_2, Bbox):
        return float(iou)
    return iou


def _normalize_by_image_size(xyxy, image_size):
    if image_size is not None:
        xyxy = xyxy[0] / image_size.width, xyxy[1] / image_size.height, \
               xyxy[2] / image_size.width, xyxy[3] / image_size.height
    return xyxy


def bbox_from_xyxy(xyxy: BboxIterableType,
                   image_size: Union['Size2D', None] = None,
                   *args, **kwargs):
    xyxy = _normalize_by_image_size(xyxy=xyxy, image_size=image_size)
    return Bbox(xyxy_data=np.array(xyxy, dtype=np.float32), *args, **kwargs)


def bbox_from_xywh(xywh: BboxIterableType,
                   image_size: Union['Size2D', None] = None,
                   *args, **kwargs):
    xyxy = xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]
    xyxy = _normalize_by_image_size(xyxy=xyxy, image_size=image_size)
    return Bbox(xyxy_data=np.array(xyxy, dtype=np.float32), *args, **kwargs)


def bbox_from_center_xywh(center_xywh: BboxIterableType,
                          image_size: Union['Size2D', None] = None,
                          *args, **kwargs):
    x1, y1 = center_xywh[0] - (center_xywh[2] / 2), center_xywh[1] - (center_xywh[3] / 2)
    x2, y2 = center_xywh[0] + (center_xywh[2] / 2), center_xywh[1] + (center_xywh[3] / 2)
    xyxy = _normalize_by_image_size(xyxy=[x1, y1, x2, y2], image_size=image_size)
    return Bbox(xyxy_data=np.array(xyxy, dtype=np.float32), *args, **kwargs)
