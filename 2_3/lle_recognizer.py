'''
Reference
https://www.arocmag.com/article/1001-3695(2007)10-0176-02.html

The variable names used in this source code are consistent with this paper.
'''
import numpy as np
import sys

from settings import FACE_COUNT
from settings import IMAGE_COUNT_PER_FACE
from settings import TRAIN_COUNT_PER_FACE
from image_parser import read_all_images
from math_utils import get_dist


def get_knn(images, face, k, if_face_in_images):
    dists = []
    for i in range(len(images[0])):
        dists.append(get_dist(np.array([images[:, i]]).T, face))
    knn_ranks = np.argsort(dists)
    if if_face_in_images:
        assert k+1 < len(knn_ranks)
        knn_ranks = knn_ranks[1:k+1]
    else:
        assert k < len(knn_ranks)
        knn_ranks = knn_ranks[:k]
    knn_images = np.array(images[:, knn_ranks])
    return knn_ranks, knn_images


def get_W_i(knn_images, face):
    k = len(knn_images[0])
    Z_i = np.dot((face-knn_images).T, face-knn_images)
    W_i = np.dot(np.linalg.inv(Z_i), np.array([np.ones(k)]).T)
    return W_i/W_i.sum()  # [k, 1]


def get_W(training_images, k):
    N = len(training_images[0])
    W = np.zeros([N, N])
    for i in range(N):
        X_i = np.array([training_images[:, i]]).T
        knn_ranks, knn_images = get_knn(training_images, X_i, k, True)
        W_i = get_W_i(knn_images, X_i)
        for j in range(k):
            # W_i refers to the i-th column of W
            W[knn_ranks[j]][i] = W_i[j][0]
    return W  # [N, N]


def get_Y(training_images, W, d):
    N = len(training_images[0])
    I = np.eye(N)
    M = np.dot(I-W, (I-W).T)
    eigenvalues_M, eigenvectors_M = np.linalg.eig(M)
    sorted_eigenvalues_M_ranks = np.argsort(eigenvalues_M)
    assert d+1 < len(sorted_eigenvalues_M_ranks)
    return eigenvectors_M[:, sorted_eigenvalues_M_ranks[1:(d+1)]].T  # [d, N]


def recognize_face(training_images, k, W, Y, face):
    assert len(W) == len(W[0]) and len(W) == len(Y[0])
    knn_ranks, knn_images = get_knn(training_images, face, k, False)
    W_face = get_W_i(knn_images, face)  # [w_(N+1, j)].T
    Y_face = W_face[0]*np.array([Y[:, knn_ranks[0]]]).T
    for i in range(1, k):
        Y_face += W_face[i]*np.array([Y[:, knn_ranks[i]]]).T
    min_dist = get_dist(Y_face, np.array([Y[:, 0]]).T)
    face_recognized = 0
    for i in range(FACE_COUNT):
        for j in range(TRAIN_COUNT_PER_FACE):
            if i == 0 and j == 0:
                continue
            dist = get_dist(Y_face, np.array(
                [Y[:, TRAIN_COUNT_PER_FACE*i+j]]).T)
            if dist < min_dist:
                min_dist = dist
                face_recognized = i
    return face_recognized


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please specify `k` and `d` via command line parameters.')
        print(f'0 < `k` < {TRAIN_COUNT_PER_FACE*FACE_COUNT}')
        print(f'0 < `d` < {TRAIN_COUNT_PER_FACE*FACE_COUNT-1}')
        print('E.g. python pca_recognizer.py 2 2 (...)')
        sys.exit()

    if len(sys.argv) > 3:
        print(
            f'Redundant command line parameters will be ignored: {sys.argv[3]} ...')

    print('###### Locally Linear Embedding (LLE) Face Recognition ######')
    print('The face recognition results of each person are listed in the following format:')
    print('Face Number | Image Number | Face Recognized | Right or Wrong (T/F)\n...\nError Rate')
    training_images, test_images = read_all_images()
    k, d = int(sys.argv[1]), int(sys.argv[2])
    W = get_W(training_images, k)
    Y = get_Y(training_images, W, d)
    error_count = 0
    for i in range(FACE_COUNT):
        error_count_i = 0
        for j in range(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE):
            face = np.array(
                [test_images[:, (IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*i+j]]).T
            face_recognized = recognize_face(training_images, k, W, Y, face)
            t_or_f = 'T'
            if face_recognized != i:
                t_or_f = 'F'
                error_count_i += 1
            print(f'{i+1}\t{j+1}\t{face_recognized+1}\t{t_or_f}')
        error_count += error_count_i
        print(f'{error_count_i} / {IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE} = {error_count_i/(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)}\n')
    print('Total Error Rate:')
    print(f'{error_count} / {(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT} = {error_count/((IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*FACE_COUNT)}')
