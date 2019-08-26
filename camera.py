import os
import cv2
from base_camera import BaseCamera

import os

from scipy.stats import mode
from predict import predict_pose,image_processing,get_model

# hand_cascade = cv2.CascadeClassifier('../cascade4.xml')

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        predictions = []

        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            ret, frame = camera.read()

            pred, value = predict_pose(frame)
            predictions.append(pred)

            if len(predictions) >=50:
                prediction_last_100, length = mode(predictions[-50:])
                prediction_last_100 = prediction_last_100[0]
                length = length[0]

                threshold = length/50

                if threshold >= 0.8:
                    #Keep showing the pervious predictions
                    live_prediction = prediction_last_100
                    cv2.putText(frame, live_prediction ,(4,420),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)



            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
