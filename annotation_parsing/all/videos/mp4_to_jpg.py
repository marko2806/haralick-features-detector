import cv2
import os
def save_image_from_video(path):
  vidcap = cv2.VideoCapture(path)
  success,image = vidcap.read()
  count = 0
  name = path.split("\\")[-1][0:-4]
  while success:
    cv2.imwrite(str(name) + "frame%d.jpg" % count, image)  # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)

    count += 1
video_dir = 'B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\annotation_parsing\\all\\videos'
video_list = ['B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\annotation_parsing\\all\\videos\\'+f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
print(video_list)
for video in video_list:
  save_image_from_video(video)