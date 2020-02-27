from typing import Tuple, Callable, Dict

import cv2
import numpy as np

from structs import Size2D


def compute_resize_scale(current_size: Size2D, new_size: Size2D) -> float:
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Returns
        A resizing scale.
    """
    smallest_side = min(current_size.width, current_size.height)
    min_side, max_side = min(new_size.width, new_size.height), max(new_size.width, new_size.height)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(current_size.height, current_size.width)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    return scale


def resize_image_with_padding(image: np.ndarray,
                              new_size: Size2D,
                              interpolation: int = cv2.INTER_AREA) -> Tuple[np.ndarray, Dict]:
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        image: Image to resize

    Returns
        A resized image.
    """
    scale = compute_resize_scale(current_size=Size2D(width=image.shape[1], height=image.shape[0]),
                                 new_size=new_size)

    return (cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation),
            dict(scale=scale))


def norm(image: np.ndarray, mean: float, std: float):
    image = image.astype(np.float32)
    image -= mean
    image /= std
    return image


def preprocess_image(image: np.ndarray,
                     image_norm_func: Callable[[np.ndarray], np.ndarray],
                     image_resize_func: Callable[[np.ndarray], Tuple[np.ndarray, Dict]]) \
        -> Tuple[np.ndarray, Dict]:
    image = image_norm_func(image)
    image, resize_info = image_resize_func(image)
    return image, resize_info
