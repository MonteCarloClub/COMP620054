'''
Reference
Dabhade S A ,  Bewoor M S ,  Dabhade S A , et al. Face Recognition using Principle Component Analysis[J]. Core.kmi.open.ac.uk, 2013, ETCSIT(100):45 - 50.

> Perhaps, the simplest classification scheme is a nearest neighbor classifier in
> the image space. Under this scheme, an image in the test set is recognized
> (classified) by assigning to it the label of the closest point in the learning
> set, where distances are measured in the image space.
'''
import numpy as np

from settings import FACE_COUNT
from settings import IMAGE_COUNT_PER_FACE
from settings import TRAIN_COUNT_PER_FACE
from image_parser import read_an_image
from math_utils import get_dist


def get_center(face_num):
    center = read_an_image(face_num, 0)
    center_len = len(center)
    for i in range(1, TRAIN_COUNT_PER_FACE):
        im_i = read_an_image(face_num, i)
        assert center_len == len(im_i)
        for j in range(center_len):
            center[j] = (center[j]*i+im_i[j])/(i+1)
    return center


def get_centers():
    centers = [get_center(0)]
    for i in range(1, FACE_COUNT):
        centers = np.append(centers, [get_center(i)], axis=0)
    return centers


def recognize_face(face_num, image_num_of_this_face, centers):
    im = read_an_image(face_num, image_num_of_this_face)
    face_recognized = face_num
    key_dist = get_dist(im, centers[face_num])
    for i in range(FACE_COUNT):
        if i == face_num:
            continue
        dist = get_dist(im, centers[i])
        if dist < key_dist:
            face_recognized = i
    return face_recognized


if __name__ == '__main__':
    print('###### Minimum Distance Face Recognition ######')
    print('The face recognition results of each person are listed in the following format:')
    print('Face Number | Image Number | Face Recognized | Right or Wrong (T/F)\n...\nError Rate\n')
    centers = get_centers()
    error_count = 0
    for i in range(FACE_COUNT):
        error_count_i = 0
        for j in range(TRAIN_COUNT_PER_FACE, IMAGE_COUNT_PER_FACE):
            face_recognized = recognize_face(i, j, centers)
            t_or_f = 'T'
            if face_recognized != i:
                t_or_f = 'F'
                error_count_i += 1
            print(f'{i+1}\t{j+1}\t{face_recognized+1}\t{t_or_f}')
        error_count += error_count_i
        print(f'{error_count_i} / {IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE} = {error_count_i/(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)}\n')
    print('Total Error Rate:')
    print(f'{error_count} / {(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT} = {error_count/((IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT)}')
