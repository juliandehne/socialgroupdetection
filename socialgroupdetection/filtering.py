import numpy as np
from sklearn.svm import OneClassSVM, SVC

from .embeddings import convert_terms_to_embeddings


def classify_with_svm(new_embeddings, white_list_centroids, oc_svm=None):
    # Train or use the provided OneClassSVM model
    if oc_svm is None:
        oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.2)
        oc_svm.fit(white_list_centroids)

    # Predict the classifications using SVM
    classifications_svm = oc_svm.predict(new_embeddings)

    return classifications_svm


def classify_with_linear(new_embeddings, white_list_centroids):
    # Compute the geometric center of the whitelist centroids
    geometric_center = np.mean(white_list_centroids, axis=0)

    # Compute the average distance from the geometric center to the white_list_centroids
    distances = np.linalg.norm(white_list_centroids - geometric_center, axis=1)
    average_distance = np.mean(distances)

    # Compute the distances for the new embeddings
    new_distances = np.linalg.norm(new_embeddings - geometric_center, axis=1)

    # Classify based on the average distance
    classifications_linear = np.where(new_distances > average_distance, 0, 1)

    return classifications_linear


def classify_with_two_class_svm(new_embeddings, white_list_centroids, black_list_centroids, tc_svm=None):
    if tc_svm is None:
        # Combine white list and black list embeddings to form the training data
        X_train = np.vstack([white_list_centroids, black_list_centroids])

        # Create labels: 1 for white list (positive class), -1 for black list (negative class)
        y_train = np.hstack([np.ones(len(white_list_centroids)), -1 * np.ones(len(black_list_centroids))])

        # Train a two-class SVM
        two_class_svm = SVC(kernel='linear')  # You can choose other kernels like 'rbf' if needed
        two_class_svm.fit(X_train, y_train)

        # Predict using the two-class SVM
        classifications_two_class_svm = two_class_svm.predict(new_embeddings)
    else:
        classifications_two_class_svm = tc_svm.predict(new_embeddings)

    return classifications_two_class_svm

