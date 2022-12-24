import os
import numpy as np
from skimage import io
from pickle import dump
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu, find_contours
from skimage.morphology import disk, erosion

CHARS_PER_IMG = 80
training = ['a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w']
Ytrue = np.asarray([training[i] for i in range(len(training)) for j in range(CHARS_PER_IMG)])

def extract_character_features_v4(img_path, out_features, thresh=240, min_aspect_ratio=0.5, chars_per_image=CHARS_PER_IMG, plot=False, retry=False):
    # Read the image
    img = io.imread(img_path)
    # Convert to binary image
    img_binary = (img < thresh).astype(np.double)
    img_binary = erosion(img_binary, disk(1))
    # Extract the connected components
    img_labels = label(img_binary, background=0)
    regions = regionprops(img_labels)
    # Extract the features
    features = []
    bboxes = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        roi = img_binary[minr:maxr, minc:maxc]
        # Omit regions that have small height or width
        if roi.shape[0] < 10 or roi.shape[1] < 10: continue
        # Omit regions that have large height or width
        if roi.shape[0] > 100 or roi.shape[1] > 100: continue
        # Omit regions that has small area
        if roi.size < 150: continue
        # Omit regions that has small aspect ratio
        if retry and roi.shape[1] / roi.shape[0] < min_aspect_ratio: continue
        bboxes.append((minr, minc, maxr, maxc))
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        # Find contours of the region
        contours = find_contours(img_binary[minr:maxr, minc:maxc], 0.5)
        # Find the longest contour
        longest_contour = max(contours, key=len)
        # Find the centroid of the longest contour
        centroid = np.mean(longest_contour, axis=0)
        # Find the angle of the longest contour
        angle = np.arctan2(centroid[1] - cr, centroid[0] - cc)
        # Find the length of the longest contour
        length = len(longest_contour)
        # Find the number of holes in the region
        holes = len(find_contours(1 - roi, 0.5))
        hu = np.append(hu, [centroid[0], centroid[1], angle, length, holes])
        features.append(hu)
    if retry: return features, bboxes, img_labels
    if len(features) != chars_per_image:
        features, bboxes, img_labels = extract_character_features_v4(img_path, out_features, min_aspect_ratio=min_aspect_ratio, thresh=250, chars_per_image=chars_per_image, plot=plot, retry=True)
    features = np.asarray(features)
    out_features.append(features)
    if plot:
        print(f"Number of detected characters: {features.shape[0]}")
        plt.figure(figsize=(10, 10))
        io.imshow(img_binary)
        ax = plt.gca()
        for (minr, minc, maxr, maxc) in bboxes:
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        ax.set_title('Bounding Boxes')
        io.show()
    return features, bboxes, img_labels

def train(img_path):
    feature_list = []
    for img in training:
        _, _, _ = extract_character_features_v4(f'{img_path}/{img}.bmp', feature_list)
    feature_list = np.asarray(feature_list).reshape(-1, 12)
    means, stds = np.mean(feature_list, axis=0), np.std(feature_list, axis=0)
    feature_list = (feature_list - means) / stds
    # Use a SVM classifier to classify the characters
    clf = svm.SVC(kernel='rbf', gamma='scale', C=200)
    clf.fit(feature_list, Ytrue)
    pred = clf.predict(feature_list)
    print("Training finished!")
    print("-" * 50)
    print("Class Recognition Rates")
    print("-" * 50)
    for i in range(len(training)):
        p = pred[i * CHARS_PER_IMG:(i + 1) * CHARS_PER_IMG]
        rate = np.mean(p == np.asarray(training[i]))
        print(f"Image {training[i]}: {rate * 100:.2f}%")
    confM = confusion_matrix(Ytrue, pred)
    plt.figure(figsize=(5, 5))
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(0, len(training), 1), training, rotation=90)
    plt.yticks(np.arange(0, len(training), 1), training)
    io.imshow(confM)
    io.show()
    # Save the classifier
    with open('classifier.pkl', 'wb') as f:
        dump(clf, f)
    # Save the means and stds
    with open('means_stds.pkl', 'wb') as f:
        dump((means, stds), f)

if __name__ == '__main__':
    train('./images')