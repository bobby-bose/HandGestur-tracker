from flask import Flask, render_template, Response
import cv2
import HandTrackingModule as htm
import numpy as np
import os

app = Flask(__name__)

camera = cv2.VideoCapture(0)

detector = htm.HandDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

folderPath = "images"
imageList = os.listdir(folderPath)
overlayList = []
for imPath in imageList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            img = detector.findHands(frame)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList)!= 0:
                fingers = []
                for id in range(0, 5):
                    if id!= 0:
                        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:
                        if lmList[tipIds[id]][1] < lmList[tipIds[id] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                totalFingers = fingers.count(1)

                # Add gesture recognition logic here
                if fingers == [0, 0, 0, 0, 0]:
                    overlay = overlayList[0]
                elif fingers == [0, 1, 0, 0, 0]:
                    overlay = overlayList[1]
                elif fingers == [0, 1, 1, 0, 0]:
                    overlay = overlayList[2]
                elif fingers == [0, 1, 1, 1, 0]:
                    overlay = overlayList[3]
                elif fingers == [0, 1, 1, 1, 1]:
                    overlay = overlayList[4]
                elif fingers == [1, 1, 1, 1, 1]:
                    overlay = overlayList[5]
                else:
                    overlay = np.zeros((100, 100, 3), np.uint8)

                h, w, c = overlay.shape
                frame[0:h, 0:w] = overlay

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)