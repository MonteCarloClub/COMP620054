import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys

from settings import FACE_COUNT
from settings import IMAGE_COUNT_PER_FACE
from settings import TRAIN_COUNT_PER_FACE
from image_parser import read_all_images
from pca_recognizer import get_omegas_of_training_set

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please specify a series of `M_` via command line parameters.')
        print(f'{FACE_COUNT-1} <= `M_` < {TRAIN_COUNT_PER_FACE*FACE_COUNT}')
        print('E.g. python pca_recognizer.py 39 160 (...)')
        sys.exit()

    print('###### Efficient Fisher’s Linear Discriminant Face Recognition ######')
    print('The face recognition results of each person are listed in the following format:')
    print('Face Number | Image Number | Face Recognized | Right or Wrong (T/F)\n...\nError Rate')
    training_images, test_images = read_all_images()
    for i in range(1, len(sys.argv)):
        # print(f'(N - c) (= {TRAIN_COUNT_PER_FACE*FACE_COUNT} - {FACE_COUNT} = {TRAIN_COUNT_PER_FACE*FACE_COUNT-FACE_COUNT}) eigenimages are selected.')
        m, E, omegas_of_training_set = get_omegas_of_training_set(
            training_images, int(sys.argv[i]))
        # print(
        #     f'`n_components` is set to (c - 1) (= {FACE_COUNT} - 1 = {FACE_COUNT-1}).')
        lda_obj = LinearDiscriminantAnalysis(n_components=FACE_COUNT-1)
        label = []
        for j in range(FACE_COUNT):
            for k in range(TRAIN_COUNT_PER_FACE):
                label.append(j)
        lda_obj.fit(omegas_of_training_set.T, label)
        error_count = 0
        for j in range(FACE_COUNT):
            error_count_j = 0
            for k in range(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE):
                face = np.array(
                    [test_images[:, (IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*j+k]]).T-m
                omega_of_face = np.dot(E.T, face)
                face_recognized = lda_obj.predict(omega_of_face.T)
                tOrF = 'T'
                if face_recognized != j:
                    tOrF = 'F'
                    error_count_j += 1
                print(f'{j+1}\t{k+1}\t{face_recognized+1}\t{tOrF}')
            error_count += error_count_j
            print(f'{error_count_j} / {IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE} = {error_count_j/(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)}\n')
        print('Total Error Rate:')
        print(f'{error_count} / {(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT} = {error_count/((IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT)}')
        print('################################################################################')
