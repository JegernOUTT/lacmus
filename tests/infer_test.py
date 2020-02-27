from time import time

import numpy as np
from tqdm import tqdm

from models.model_context import ModelsContext


def infer_time_test(iter_count=6):
    assert iter_count > 5, "Set iter more than 5 cause of warmup stage"
    model_type_2_model_config = {
        'kerasretinanet': {"filename": "./resnet50_liza_alert_v1_interface.h5",
                           "backbone_type": "resnet",
                           "width": 1366,
                           "height": 800},
        'ttfnetonnx': {"filename": "./ttfnet_v1.onnx",
                       "width": 1440,
                       "height": 800}
    }

    models_context = ModelsContext()

    model_ids = []
    for model_type, model_config in model_type_2_model_config.items():
        model_id = models_context.create_model(model_type, new_model_config=model_config)
        model_ids.append(model_id)

    for model_id, model_config in zip(model_ids, model_type_2_model_config.values()):
        dummy_img = np.zeros((model_config['height'], model_config['width'], 3), dtype=np.uint8)
        model = models_context.model(model_id)

        times = []
        for _ in tqdm(range(iter_count), desc=f'{model} infer iters: '):
            time_from = time()
            model.predict(dummy_img)
            time_to = time()
            times.append(time_to - time_from)
        print(f'Model {model} infer time: {np.mean(times[:5])}')


if __name__ == '__main__':
    infer_time_test()
