import numpy as np
from os import listdir

import cv2
path_to_img = "images\mirflickr"
path_to_l = "Code\l\gray_scale.npy"
path_to_ab = "Code\ab"

img_list = listdir(path_to_img)
bw_list = [0]*len(img_list)
rgb_list = [0]*len(img_list)

for i in range(len(img_list)):
    #img_list[i] = cv2.resize(cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2GRAY), (224,224))
    rgb_list[i] = cv2.imread("images\mirflickr\\"+img_list[i])
    rgb_list[i] = cv2.resize(rgb_list[i], (224, 224))
    bw_list[i] = cv2.cvtColor(rgb_list[i], cv2.COLOR_BGR2GRAY)
    
bw_array_train = np.asarray(bw_list[:20000])
bw_array_test = np.asarray(bw_list[20000:22500])
bw_array_valid = np.asarray(bw_list[22500:])

np.save("X_test.npy", bw_array_test)
np.save("X_train.npy", bw_array_train)
np.save("X_valid.npy", bw_array_valid)

rgb_array_train = np.asarray(rgb_list[:20000])
rgb_array_test = np.asarray(rgb_list[20000:22500])
rgb_array_valid = np.asarray(rgb_list[22500:])

np.save("y_test.npy", rgb_array_test)
np.save("y_train.npy", rgb_array_train)
np.save("y_valid.npy", rgb_array_valid)


ab1 = np.load(path_to_ab+"\\ab1.npy", 'r')
ab2 = np.load(path_to_ab+"\\ab2.npy", 'r')
ab3 = np.load(path_to_ab+"\\ab3.npy", 'r')
lab_y_train = ab1+ab2
lab_y_test = ab3[:2500]
lab_y_valid = ab3[2500:]

gray_lab = np.load(path_to_l, 'r')
lab_X_train = gray_lab[:20000]
lab_X_test = gray_lab[20000:22500]
lab_X_valid = gray_lab[22500:]

np.save("lab_X_train.npy", lab_X_train)
np.save("lab_X_test.npy", lab_X_test)
np.save("lab_X_valid.npy", lab_X_valid)

np.save("lab_y_train.npy", lab_y_train)
np.save("lab_y_test.npy", lab_y_test)
np.save("lab_y_valid.npy", lab_y_valid)