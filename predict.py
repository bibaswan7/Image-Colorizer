from engine import build_modelv1
from utils import loadImagesToArray, RGB2GRAY, RGB2ab, Lab2RGB, custom_img_to_array
import numpy as np

HEIGHT = 256
WIDTH = 256
ks = (3, 3)
actt = "sigmoid"
learning_rate = 0.001



def pred(image):
    mymodv1 = build_modelv1(act=actt, learning_rate=0.001)
    mymodv1.load_weights('./v1_coast_100epoch.h5')
    color_me = custom_img_to_array(image)
    gray = RGB2GRAY(color_me, True)
    gray2 = RGB2GRAY(color_me)
    gray = gray[np.newaxis,:,:, :]
    output = mymodv1.predict(gray)
    pred = Lab2RGB(gray[0], output[0])
    return pred.reshape(color_me.shape)


