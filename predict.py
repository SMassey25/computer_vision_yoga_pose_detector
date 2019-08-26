from skimage.transform import resize
from tensorflow.keras.models import load_model
from skimage.color import rgb2gray
import numpy as np

def get_model():
    global model
    model = load_model('model/best_model/model.h5')
get_model()

def image_processing(image):
    # img = imread(image)
    img = rgb2gray(resize(image,(200,200),anti_aliasing=True))
    processed_image = np.array(img).astype('float32')
    return processed_image


def predict_pose(image):
    # np.set_printoptions(suppress=True)
    #Calling the image processing function
    processed_image = image_processing(image)

    prediction_raw = model.predict(processed_image.reshape(1,200,200,1))
    prediction = np.argmax(prediction_raw)
    value = np.max(prediction_raw)
    value = round(value*100,2)
    classes = ['Downward Dog', 'Tree', 'Triangle', 'Upward Dog', 'Warrior 1', 'Warrior 2']
    yoga_pose = classes[prediction]

    return yoga_pose, value
