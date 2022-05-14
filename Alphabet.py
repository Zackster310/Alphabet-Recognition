from tkinter import E, Frame
import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, time, ssl


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

x = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")['labels']

x=np.array(x)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n = len(classes)

x_train, x_test, y_train, y_test = tts(x , y, random_state = 50, train_size = 7500, test_size = 2500)
x_train = x_train/255.0
x_test = x_test/255.0

model = LogisticRegression(solver = "saga", multi_class = "multinomial")
model = model.fit(x_train, y_train)

yPred = model.predict(x_test)

accuracyScore = accuracy_score(y_test, yPred)
print(accuracyScore)

capture = cv2.VideoCapture(0)

while(capture.isOpened()):
    try:
        ret, Frame = capture.read()
        grey = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        
        height, width = grey.shape
        topLeft = (int(width/2 - 60), int(height/2 - 44))
        botRight = (int(width/2 + 60), int(height/2 + 44))
        cv2.rectangle(grey, topLeft, botRight, (0,255,0), 2)

        roi = grey[topLeft[0]:botRight[0], topLeft[1]:botRight[1]]

        impil = Image.fromarray(roi)
        img_bw = impil.convert('L')
        img_resized = img_bw.resize((22, 30), Image.ANTIALIAS)
        img_inverted = PIL.ImageOps.invert(img_resized)

        min_pixel = np.percentile(img_inverted, 20)
        img_scaled = np.clip(img_inverted - min_pixel, 0 ,255)
        max_pixel = np.max(img_inverted)
        img_scaled = np.asarray(img_scaled)/max_pixel

        testSample = np.array(img_scaled).reshape(1, 784)
        testPred = model.predict(testSample)


        print("Predicted Class is: ", testPred)
        cv2.imshow('Frame', grey)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    except Exception as E:
        pass

capture.release()
cv2.destroyAllWindows()