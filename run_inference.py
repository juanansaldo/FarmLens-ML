from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
project = rf.workspace(os.getenv('ROBOFLOW_WORKSPACE')).project(os.getenv('ROBOFLOW_MODEL_ID'))
model = project.version(18).model

# visualize your prediction
model.predict("images/test_img4.jpg", confidence=90, overlap=30).save("prediction.jpg")