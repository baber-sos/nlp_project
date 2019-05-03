import numpy as np
import cv2
import os

# vid_dir = './example_set/';
# vid_dir = '../YouTubeClips/';
vid_dir = '/common/users/bk456/YouTubeClips/';
vid_file = open('val_input');
dest_dir = './val_images/train/';
image_size = (224, 224);

if __name__ == '__main__':
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir);
    for file in vid_file:
    # for file in os.listdir(vid_dir):
        # print(file);
        file = file.strip();
        vid_file = vid_dir + file;
        # print(vid_file);
        vidcap = cv2.VideoCapture(vid_file);
        success, image = vidcap.read();
        print(vid_file)
        print(success);
        count = 1;
        video_path = dest_dir + file + '/';
        if not os.path.exists(video_path):
            os.mkdir(video_path);
        while success:
            image = cv2.resize(image, image_size);
            cv2.imwrite(video_path + "%d.jpg" % count, image);
            success, image = vidcap.read();
            count += 1;