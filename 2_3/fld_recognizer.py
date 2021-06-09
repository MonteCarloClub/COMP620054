'''
Reference
[1] P. N. Belhumeur, J. P. Hespanha and D. J. Kriegman, "Eigenfaces vs. Fisherfaces: recognition using class specific linear projection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 19, no. 7, pp. 711-720, July 1997, doi: 10.1109/34.598228.
> The variable names used in this source code are consistent with this paper.
[2] Bhattacharyya, Suman & Rahul, Kumar. (2013). Face recognition by linear discriminant analysis. International Journal of Communication Network Security. 2. 31-35. 
'''
import numpy as np
import sys

from settings import FACE_COUNT
from settings import IMAGE_COUNT_PER_FACE
from settings import TRAIN_COUNT_PER_FACE
from image_parser import read_all_images
from math_utils import get_dist


def get_W(training_set, m):
    assert len(training_set[0]) == TRAIN_COUNT_PER_FACE*FACE_COUNT
    n = len(training_set)
    '''
    The between-class scatter matrix is defined as S_B.
    The total scatter matrix is defined as S_T.
    The within-class scatter matrix is defined as S_W.

    '''
    S_B, S_T, S_W = np.zeros([n, n]), np.zeros([n, n]), np.zeros([n, n])
    '''
    mu is the mean image of all images in the training set.
    mu_i is is the mean image of the images of the i-th face (X_i).
    '''
    mu = np.array([np.mean(training_set, axis=1)]).T
    for i in range(FACE_COUNT):
        X_i = training_set[:, TRAIN_COUNT_PER_FACE *
                           i:TRAIN_COUNT_PER_FACE*(i+1)]
        mu_i = np.array([np.mean(X_i, axis=1)]).T
        S_B += np.dot(mu_i-mu, (mu_i-mu).T)*TRAIN_COUNT_PER_FACE
        for j in range(TRAIN_COUNT_PER_FACE):
            x_i_j = np.array([X_i[:, j]]).T
            S_T += np.dot(x_i_j-mu, (x_i_j-mu).T)
            S_W += np.dot(x_i_j-mu_i, (x_i_j-mu_i).T)
    '''
    S_W is always singular, for the rank of S_W is <= (N - c).
    N = TRAIN_COUNT_PER_FACE*FACE_COUNT (= 200), c = FACE_COUNT (=40); m <= (c-1) (=39).
    Fisherfaces is proposed to avoid this problem by projecting the images to a lower dimensional space.
    1. Use PCA to reduce the dimension of the feature space to (N - c);
    2. Applying the standard FLD to reduce the dimension to (c - 1).
    W_opt = W_pca * W_fld. W_opt is the most optimized W, that is, the return value.
    W_pca = argmax(X) det(W.T * S_T * W),
    performed over n * (N - c) matrices with orthonormal columns.
    W_fld = argmax(X) det(W.T * W_pca.T * S_B * W_pca * W) / det(W.T * W_pca.T * S_W * W_pca * W),
    performed over (N - c) * m matrices with orthonormal columns.
    
    W_pca is the set of eigenvectors of S_T corresponding to the (N - c) largest eigenvalues.
    S_T is a symmetric matrix, the purpose of calling `eigh` is to output eigenvectors without `+0.j`.
    '''
    eigenvalues_S_T, eigenvectors_S_T = np.linalg.eigh(S_T)
    sorted_eigenvalues_S_T_rank = np.argsort(eigenvalues_S_T)
    W_pca = []
    max_rank = len(eigenvalues_S_T)-1
    for i in range(TRAIN_COUNT_PER_FACE*FACE_COUNT-FACE_COUNT):
        rank = sorted_eigenvalues_S_T_rank[max_rank-i]
        if eigenvalues_S_T[i] == 0.0:
            break
        W_pca.append(eigenvectors_S_T[:, rank])
    W_pca = np.array(W_pca).T
    '''
    W_fld is the set of eigenvectors of inv(W_pca.T * S_W * W_pca) * W_pca.T * S_B * W_pca
    corresponding to the m largest eigenvalues.
    '''
    mat_fld_numerator = np.dot(np.dot(W_pca.T, S_B), W_pca)
    mat_fld_denominator = np.dot(np.dot(W_pca.T, S_W), W_pca)
    mat_fld = np.dot(np.linalg.inv(mat_fld_denominator), mat_fld_numerator)
    eigenvalues_fld, eigenvectors_fld = np.linalg.eig(mat_fld)
    sorted_eigenvalues_fld_rank = np.argsort(eigenvalues_fld)
    W_fld = []
    max_rank = len(eigenvalues_fld)-1
    assert m <= max_rank
    for i in range(m):
        rank = sorted_eigenvalues_fld_rank[max_rank-i]
        if eigenvalues_fld[i] == 0.0:
            break
        W_fld.append(eigenvectors_fld[:, rank])
    W_fld = np.array(W_fld).T
    '''
    Confirm that all elements of W_fld do not contain the imaginary part.
    Output W_fld without `+0.j`.
    '''
    if np.all(np.imag(W_fld) == 0.0):
        W_fld = np.real(W_fld)
    return np.dot(W_pca, W_fld)


def get_training_image_projections(W, training_images):
    assert len(W) == len(training_images)
    return np.dot(W.T, training_images)


def recognize_face(W, training_image_projections, face):
    assert len(training_image_projections[0]) == TRAIN_COUNT_PER_FACE * \
        FACE_COUNT and len(W) == len(face) and len(face[0]) == 1
    face_projection = np.dot(W.T, face)
    min_dist = get_dist(
        np.array([training_image_projections[:, 0]]).T, face_projection)
    face_recognized = 0
    for i in range(FACE_COUNT):
        for j in range(TRAIN_COUNT_PER_FACE):
            if i == 0 and j == 0:
                continue
            dist = get_dist(np.array(
                [training_image_projections[:, TRAIN_COUNT_PER_FACE*i+j]]).T, face_projection)
            if dist < min_dist:
                min_dist = dist
                face_recognized = i
    return face_recognized


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please specify a series of `m` via command line parameters.')
        print(f'0 < `m` < {FACE_COUNT}')
        print('E.g. python pca_recognizer.py 39 (...)')
        sys.exit()

    print('###### Fisher’s Linear Discriminant (FLD) Face Recognition ######')
    print('The face recognition results of each person are listed in the following format:')
    print('Face Number | Image Number | Face Recognized | Right or Wrong (T/F)\n...\nError Rate')
    training_images, test_images = read_all_images()
    for i in range(1, len(sys.argv)):
        m = int(sys.argv[i])
        print('################################################################################')
        print(f'The dimension of projections is set to {m}.')
        W = get_W(training_images, m)
        print('The projection matrix is as follows:')
        print(W)
        print(f'(Its shape: {len(W)} * {len(W[0])})')
        training_image_projections = get_training_image_projections(
            W, training_images)
        error_count = 0
        for j in range(FACE_COUNT):
            error_count_j = 0
            for k in range(IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE):
                face = np.array(
                    [test_images[:, (IMAGE_COUNT_PER_FACE-TRAIN_COUNT_PER_FACE)*j+k]]).T
                face_recognized = recognize_face(
                    W, training_image_projections, face)
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
