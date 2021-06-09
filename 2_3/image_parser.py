import cv2
import numpy as np
import os

from settings import FACE_COUNT
from settings import IMAGE_COUNT_PER_FACE
from settings import TRAIN_COUNT_PER_FACE


def get_image_path(face_num, image_num_of_this_face):
    serial_num = 10*face_num+image_num_of_this_face+1
    num_str = str(serial_num)
    if serial_num < 10:
        num_str = '00'+num_str
    elif serial_num < 100:
        num_str = '0'+num_str
    return './ORL/orl'+num_str+'.bmp'


def get_image_shape():
    image_path_0_0 = get_image_path(0, 0)
    im_0_0 = cv2.imread(image_path_0_0, 0)
    return im_0_0.shape[0], im_0_0.shape[1]


def read_an_image(face_num, image_num_of_this_face):
    image_path = get_image_path(face_num, image_num_of_this_face)
    im = cv2.imread(image_path, 0)
    resizedIm = np.resize(im, (1, im.shape[0]*im.shape[1]))
    return resizedIm[0]


def read_all_images():
    training_images, test_images = [], []
    for i in range(FACE_COUNT):
        for j in range(IMAGE_COUNT_PER_FACE):
            im = read_an_image(i, j)
            if j < TRAIN_COUNT_PER_FACE:
                assert len(training_images) == 0 or len(
                    training_images) > 0 and len(im) == len(training_images[0])
                training_images.append(im)
            else:
                assert len(test_images) == 0 or len(
                    test_images) > 0 and len(im) == len(test_images[0])
                test_images.append(im)
    return np.array(training_images).T, np.array(test_images).T


def write_an_image(im, image_dirname, image_num):
    shape_0, shape_1 = get_image_shape()
    assert len(im) == shape_0*shape_1
    resizedIm = np.resize(im, (shape_0, shape_1))
    output_dir_flag = os.path.exists('./output/'+image_dirname)
    if not output_dir_flag:
        os.makedirs('./output/'+image_dirname)
    cv2.imwrite('./output/'+image_dirname+'/'+str(image_num)+'.bmp', resizedIm)
