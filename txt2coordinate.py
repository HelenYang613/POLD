import numpy as np
import os
import cv2


def SearchFiles(directory, fileType):      
    fileList=[]    
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType):
                fileList.append(os.path.join(root,fileName))
    return fileList

def find_new_file(dir):
    file_lists = SearchFiles(dir,'.txt')
    # file_lists = os.listdir(dir)
    file_lists.sort(key = lambda fn: os.path.getmtime(fn)
                     if not os.path.isdir(fn) else 0)
    file = file_lists[-1]
    return file

def t2c(dir):
    fileName = find_new_file(dir)
    fileLabel = open(fileName, "r+")
    imagePath = "/home/yangpeng/Desktop/parking_2D/image/1.jpg"
    img = cv2.imread(imagePath)
    h_img, w_img = img.shape[:2]

    with open(fileLabel.name, 'r+', encoding='utf-8') as f:
        bbox = np.zeros([len(f.readlines()),5])

    with open(fileLabel.name, 'r+', encoding='utf-8') as f:
        row = 0
        for line in f.readlines():
            info = line[:-1].split(',')
            kind = info[0].split(" ")[0]
            if (int(kind) == 2) | (int(kind) == 5) | (int(kind) == 7):
                bbox[row,0] = int(kind)   # object type
                cenX = float(info[0].split(" ")[1])
                cenY = float(info[0].split(" ")[2])
                wide = float(info[0].split(" ")[3])
                high = float(info[0].split(" ")[4].split()[0])
                # anti-normalization
                x_t = cenX * w_img
                y_t = cenY * h_img
                w_t = wide * w_img
                h_t = high * h_img
                # save into bbox
                bbox[row,1] = x_t - w_t / 2    # x_lefttop
                bbox[row,2] = y_t - h_t / 2    # y_leftbottom
                bbox[row,3] = x_t + w_t / 2    # x_righttop
                bbox[row,4] = y_t + h_t / 2    # y_rightbottom
                row = row + 1
            else:
                continue
    bbox = bbox[[not np.all(bbox[i] == 0) for i in range(bbox.shape[0])], :]
    return bbox
