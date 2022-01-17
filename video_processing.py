import cv2
from main import preprocessImage, getSlidingWindowAreas, getMaskAfterClassification
from multiprocessing import Pool


def processVideo(filePath: str, classifier):
    cap = cv2.VideoCapture(filePath)
    while not cap.isOpened():
        cap = cv2.VideoCapture(filePath)
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            if pos_frame % 10 == 0:
                frame = preprocessImage(frame)
                # shape 13,13
                slidingWindows = getSlidingWindowAreas(frame, 40, 40, 20)
                with Pool(6) as p:
                    mask = getMaskAfterClassification(frame, slidingWindows, classifier, p)
                '''imshow(frame)
                show()
                cv2.imshow("test", mask)
                cv2.waitKey()
                cv2.destroyAllWindows()'''
                print("Processed frame " + str(pos_frame))
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break