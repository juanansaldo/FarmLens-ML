from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from functools import partial
import os

pipeline = InferencePipeline.init(
    api_key=os.getenv('ROBOFLOW_API_KEY'),
    model_id=os.getenv('ROBOFLOW_MODEL_ID'),
    video_reference=f"https://{os.getenv('HOLOLENS_IP_ADDRESS')}/api/holographic/stream/live.mp4",
    on_prediction=render_boxes
)

pipeline.start()
pipeline.join()
