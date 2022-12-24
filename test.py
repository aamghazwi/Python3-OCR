import numpy as np
from skimage import io
from pickle import load
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from train import extract_character_features_v4, training

def get_recognition_rate(pred, bboxes, classes, locations):
    correct = 0
    for i in range(len(pred)):
        for j in range(len(classes)):
            # Check whether the center is inside the bounding box
            if pred[i] == classes[j] and locations[j][0] >= bboxes[i][1] and locations[j][0] <= bboxes[i][3] and locations[j][1] >= bboxes[i][0] and locations[j][1] <= bboxes[i][2]:
                if pred[i] == classes[j]:
                    correct += 1
                break
    return correct / len(classes)

def test(img_path):
    features, bboxes, img_labels = extract_character_features_v4(f'{img_path}', [], min_aspect_ratio=0.4)
    with open('means_stds.pkl', 'rb') as f:
        means, stds = load(f)
    with open('classifier.pkl', 'rb') as f:
        clf = load(f)
    with open('test_gt_py3.pkl', 'rb') as f:
        test_gt = load(f)
    classes = test_gt[b'classes']
    locations = test_gt[b'locations']
    features = (features - means) / stds
    feature_list = np.asarray([features]).reshape(-1, 12)
    pred = clf.predict(features)
    print('-' * 50)
    print(f"Recognition Rate: {get_recognition_rate(pred, bboxes, classes, locations)}")
    print('-' * 50)
    for i in range(10):
        p = pred[7 * i:7 * (i + 1)]
        print(f"Predicted: {p}, Accuracy: {p[p == np.asarray(training[i])].shape[0] / 7 * 100:.2f}%")
    plt.figure(figsize=(5, 8))
    ax = plt.gca()
    for i, (minr, minc, maxr, maxc) in enumerate(bboxes):
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        ax.text(maxc, (minr + maxr) / 2, pred[i], color='red', fontsize=15)
    plt.imshow(img_labels)
    cb = plt.colorbar()
    cb.remove()
    ax.set_title('Bounding Boxes and Recognition Results')
    ax.set_axis_off()
    io.show()

if __name__ == '__main__':
    test('./images/test.bmp')