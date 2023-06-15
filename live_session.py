import cv2.data
import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv
import numpy as np

import config
from model import Deep_Emotion
import utils
import time
import scipy.stats as st

def main():

    """
        Loading the Model
    """
    model = Deep_Emotion()
    optmizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    checkpoint = torch.load(config.CHECKPOINT, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optmizer.load_state_dict(checkpoint["optimizer"])
    model.to(config.DEVICE)


    """
        Live session
    """
    #Using Web Cam
    cap = cv.VideoCapture(0)

    #Evaluaion
    confident = [0, 0, 0, 0, 0, 0, 0, 0]
    percentageEase = 0

    while cap.isOpened():

        success, frame = cap.read()
        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # # transform
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # to gray
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)  # detect face

        count = 0
        # finding the coordinate of a face
        for idx, (x, y, w, h) in enumerate(faces):
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv.rectangle(frame, (x, y), (x + w, y + h), (154,205,50), 10)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print('Face not detected')
            else:
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey: ey + eh, ex:ex + ew]

            graytemp = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
            final_image = cv2.resize(graytemp, (config.IMAGE_SIZE, config.IMAGE_SIZE))  # resize
            final_image = np.expand_dims(final_image, axis=0)  # change dim to 3 (expect 4 )   for conv
            final_image = np.expand_dims(final_image, axis=0)  # change dim to 4 (expect 4)    for conv
            final_image = final_image / 255.  # normalize

            dataa = torch.from_numpy(final_image)
            dataa = dataa.type(torch.FloatTensor)
            dataa = dataa.to(config.DEVICE)

            outputs = model(dataa)
            pred = nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(pred)

            #finding confident leve
            confident.append(prediction)
            confident.pop(0)
            result = utils.most_frequent(confident)
            percent = utils.how_frequent(result, confident) #how confident
            percent = utils.mask_percent(percentageEase, percent)

            if ((result) == 0 ):
                status = "ANGRY"
                utils.showText(frame, status, percent, x, y, w, h)

            elif ((result)== 1):
                status = "Disgust"
                utils.showText(frame, status, percent, x, y, w, h)

            elif ((result)== 2):
                status = "Fear"
                utils.showText(frame, status, percent, x, y, w, h)

            elif ((result)== 3):
                status = "Happy"
                utils.showText(frame, status, percent, x, y, w, h)

            elif ((result)== 4):
                status = "Sad"
                utils.showText(frame, status, percent, x, y, w, h)

            elif ((result)== 5):
                status = "Surprise"
                utils.showText(frame, status, percent, x, y, w, h)

            elif ((result)== 6):
                status = "Neutral"
                utils.showText(frame, status, percent, x, y, w, h)

            if success:
                cv.imshow("real smile revealer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    #
    cap.relase()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()