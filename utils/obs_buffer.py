import os
import time
import torch
from collections import deque
from torchvision.transforms import ToPILImage
from collections import defaultdict

class Observation:
    def __init__(self, multi_view_images: dict):
        self.multi_view_images = multi_view_images

class ObservationBuffer:
    def __init__(self, max_recent_observations: int):
        self.max_recent_observations = max_recent_observations
        self.recent_observations = deque(maxlen = self.max_recent_observations)

    def add_observation(self, multi_view_images: dict):
        ob = Observation(multi_view_images = multi_view_images)
        self.recent_observations.append(ob)
    
    def reset(self):
        self.recent_observations = deque(maxlen = self.max_recent_observations)

    def aggregate_observations(self, observations: list[Observation], camera_keys: dict):
        agg_observations = defaultdict(list)

        for ob in observations:
            for camera_key in camera_keys:
                agg_observations[camera_key].append(ob.multi_view_images[camera_key])

        for key in agg_observations.keys():
            agg_observations[key] = torch.stack(agg_observations[key], dim=0)

        return dict(agg_observations)

    def get_inference_time_observations(self, camera_keys: dict, visualize: bool = False):
        inference_time_observations = []
        
        for recent_ob in list(self.recent_observations):
            inference_time_observations.append(recent_ob)

        if len(inference_time_observations) < self.max_recent_observations:
            # left pad with the earliest observations
            padding_observations = [inference_time_observations[0]] * (self.max_recent_observations - len(inference_time_observations))
            inference_time_observations = padding_observations + inference_time_observations

        agg_observations = self.aggregate_observations(inference_time_observations, camera_keys)

        if visualize:
            timestamp = str(int(time.time()))
            debug_root = os.environ.get("ZR0_OBSERVATION_DEBUG_ROOT", "temp")
            for camera_key in camera_keys:
                os.makedirs(os.path.join(debug_root, timestamp, camera_key), exist_ok=True)
                for img_idx, torch_image in enumerate(agg_observations[camera_key]):
                    to_pil = ToPILImage()
                    pil_image = to_pil(torch_image)
                    pil_image.save(os.path.join(debug_root, timestamp, camera_key, f"{img_idx}.jpg"))

        return agg_observations
