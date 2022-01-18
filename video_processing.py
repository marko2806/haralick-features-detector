import cv2


def processVideo(filePath: str, processFrame, skip_frames=1, verbose=False):
    cap = cv2.VideoCapture(filePath)
    while not cap.isOpened():
        cap = cv2.VideoCapture(filePath)
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print("Processing video:" + filePath)
    print("Number of frames:" + str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            if pos_frame % skip_frames == 0:
                if processFrame is not None:
                    processFrame(frame, filePath, pos_frame)
                if verbose:
                    print("Processed frame " + str(int(pos_frame)))
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready:" + str(pos_frame))
        if cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
