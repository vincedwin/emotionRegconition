import cv2 as cv

def main():
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # # transform
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # to gray
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)  # detect face
        if success:
            cv.imshow("real smile revealer", faces)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == "__main__":
    main()