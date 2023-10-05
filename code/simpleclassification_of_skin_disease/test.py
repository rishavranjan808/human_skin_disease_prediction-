import cv2
import keras

import numpy as np
CATEGORIES = ['acne', 'rosacea']

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr

model = keras.models.load_model('C:\\Users\\risha\\Desktop\\test\\test\\modeldvc.h5')
prediction = model.predict([image('C:\\Users\\risha\\Desktop\\test\\test\\acne\\acne-cystic-29.jpg')])
print(CATEGORIES[prediction.argmax()])