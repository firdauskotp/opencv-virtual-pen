import cv2 as cv
#for mediapipe use python 64 bit.
import mediapipe as mp #use pip install mediapipe==0.8.3.1 . latest version has an error
import time

class hand_detection():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence



        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)

    def find_hands(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        #result.multi_hand_landmarks = detect the coordinates of hands if available, else shows none

        if self.result.multi_hand_landmarks:
            for each_hand in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, each_hand, self.mpHands.HAND_CONNECTIONS)
        return img

            # earlier code for drawing the hand nodes. not needed if using in class

            # for id, lm in enumerate(each_hand.landmark):
            #     print(id, lm)  # debug id and
            #     h, w, c = img.shape
            #     cx, cy = int(lm.x * w), int(lm.y * h)
            #     print(cx, cy)  # print all landmarks in pixel
            #     if id == 0:
            #         cv.circle(img, (cx, cy), 15, (0, 0, 255), cv.FILLED)
            #
            # self.mpDraw.draw_landmarks(img, each_hand, self.mpHands.HAND_CONNECTIONS)

    def find_position(self, img, hand_number=0, draw=True):

        landmark_list = []

        if self.result.multi_hand_landmarks:
            each_hand = self.result.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(each_hand.landmark):
                print(id, lm)  # debug id and
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(cx, cy)  # print all landmarks in pixel
                landmark_list.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)

        return landmark_list



def main():
    capt = cv.VideoCapture(0)

    # For me, 0 is webcam
    # 1 is obs camera

    # FPS
    previous_time = 0
    current_time = 0

    #for class
    detector = hand_detection()

    while True:
        success, img = capt.read()
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        current_time = time.time()
        # fps formula
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 50, 50), 5)

        cv.imshow("Tracker", img)
        cv.waitKey(1)


#if you are running this code, do this


if __name__ == "__main__":
    main()