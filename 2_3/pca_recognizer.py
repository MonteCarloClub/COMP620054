'''
Reference
[1] Dabhade S A ,  Bewoor M S ,  Dabhade S A , et al. Face Recognition using Principle Component Analysis[J]. Core.kmi.open.ac.uk, 2013, ETCSIT(100):45 - 50.

The variable names used in this source code are consistent with this paper.
'''
import numpy as np
import sys

from settings import EIGENIMAGES_CONTRAST
from settings import FACE_COUNT
from settings import IMAGE_COUNT_PER_FACE
from settings import TRAIN_COUNT_PER_FACE
from image_parser import read_all_images
from image_parser import write_an_image
from math_utils import get_dist


def get_omegas_of_training_set(X, M_):
    '''
    m represents the mean image.
    W represents the mean centered images.
    '''
    m = np.array([np.mean(X, axis=1)]).T
    W = X-m
    '''
    C = np.dot(W,W.T), which The size of C is N × N which could be enormous.
    The eigenvalues and eigenvectors of C are obtained by eigen decomposition of C_ (C').
    The first (M − 1) (λi, ei) of C are given by (μi, Wdi) of C_, respectively.
    Wdi needs to be normalized in order to be equal to ei.
    The dimension of C_ is much smaller than that of C, reducing operations greatly.
    '''
    C_ = np.dot(W.T, W)
    eigenvalues_C, eigenvectors_C_ = np.linalg.eig(C_)
    eigenvectors_C = np.dot(W, eigenvectors_C_)
    for i in range(len(eigenvectors_C[0])):
        eigenvectors_C[:, i] /= np.linalg.norm(eigenvectors_C[:, i])
    '''
    {Wdi} corresponding to nonzero {μi} of C_ produce an orthonormal basis for the subspace,
    within which most image data can be represented with a small amount of error.
    {Wdi} are sorted from high to low according to their corresponding {μi}.
    The 90% of the total variance is contained in the first 5% - 10% of the dimensions.
    Only the M_ eigenvectors with the largest eigenvalues are examined.
    '''
    assert M_ < len(eigenvalues_C)
    sorted_eigenvalues_rank = np.argsort(eigenvalues_C)
    '''
    The vectors ei are also images, so called, eigenimages, or eigenfaces.
    '''
    E = []
    max_rank = len(eigenvalues_C)-1
    for i in range(M_):
        rank = sorted_eigenvalues_rank[max_rank-i]
        E.append(eigenvectors_C[:, rank])
    E = np.array(E).T
    '''
    A facial image can be projected onto M_ dimensions by computing omega.
    '''
    omegas_of_training_set = np.dot(E.T, W)
    return m, E, omegas_of_training_set


def recognize_face(m, E, omegas_of_training_set, face):
    W = face-m
    omega_of_face = np.dot(E.T, W)
    assert len(omegas_of_training_set[0]) == TRAIN_COUNT_PER_FACE*FACE_COUNT
    min_dist = get_dist(
        np.array([omegas_of_training_set[:, 0]]).T, omega_of_face)
    face_recognized = 0
    for i in range(FACE_COUNT):
        for j in range(TRAIN_COUNT_PER_FACE):
            if i == 0 and j == 0:
                continue
            dist = get_dist(
                np.array([omegas_of_training_set[:, TRAIN_COUNT_PER_FACE*i+j]]).T, omega_of_face)
            if dist < min_dist:
                min_dist = dist
                face_recognized = i
    return face_recognized


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please specify a series of `M_` via command line parameters.')
        print(f'0 < `M_` < {TRAIN_COUNT_PER_FACE*FACE_COUNT}')
        print('E.g. python pca_recognizer.py 199 (...)')
        sys.exit()

    print('###### Principal Component Analysis (PCA) Face Recognition ######')
    print('The face recognition results of each person are listed in the following format:')
    print('Face Number | Image Number | Face Recognized | Right or Wrong (T/F)\n...\nError Rate')
    training_images, test_images = read_all_images()
    for i in range(1, len(sys.argv)):
        M_ = int(sys.argv[i])
        print('################################################################################')
        print(f'The dimension of omega is set to {M_}.')
        m, E, omegas_of_training_set = get_omegas_of_training_set(
            training_images, M_)
        for j in range(len(E[0])):
            write_an_image(E[:, j]*EIGENIMAGES_CONTRAST, 'eigenimages', j+1)
        print('The eigenimages of training set are as follows:')
        print(E)
        print(f'(Its shape: {len(E)} * {len(E[0])})')
        print('You can view them at file://./output/eigenimages/\n')
        error_count = 0
        for j in range(FACE_COUNT):
            error_count_j = 0
            for k in range(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE):
                face = np.array(
                    [test_images[:, (IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*j+k]]).T
                face_recognized = recognize_face(
                    m, E, omegas_of_training_set, face)
                t_or_f = 'T'
                if face_recognized != j:
                    t_or_f = 'F'
                    error_count_j += 1
                print(f'{j+1}\t{k+1}\t{face_recognized+1}\t{t_or_f}')
            error_count += error_count_j
            print(f'{error_count_j} / {IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE} = {error_count_j/(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)}\n')
        print('Total Error Rate:')
        print(f'{error_count} / {(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT} = {error_count/((IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT)}')
        print('################################################################################')
