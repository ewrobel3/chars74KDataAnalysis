import numpy as np
from skimage import io, color, filters, feature, transform
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def imread_convert(f):
        return io.imread(f).astype(np.uint8)

def load_data():
        # loads 99 images per character, for all 62 characters
        ic = io.ImageCollection("./Fnt/Sample0*/img*-000*.png", conserve_memory=True, load_func=imread_convert)
        data = io.concatenate_images(ic)
        labelNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        labels = np.empty([99*62])
        for label in labelNames:
            np.append(labels, np.full(99, label))
        shuffled_idx = np.random.permutation(data.shape[0])
        cutoff = int(data.shape[0]*0.8)
        train_X = data[shuffled_idx[:cutoff]]
        test_X = data[shuffled_idx[cutoff:]]
        train_Y = labels[shuffled_idx[:cutoff]]
        test_Y = labels[shuffled_idx[cutoff:]]
        return train_X, train_Y, test_X, test_Y

def preprocess(imgs):
        # resize, grayscale, blur, and extract features here
        l = []
        for img in imgs:
            img = color.rgb2gray(img)
            #img_blur = filters.gaussian(img_gray, sigma=0.4)
            img = transform.resize(img, (32, 32), anti_aliasing=True) #anti_aliasing automatically blurs before resize 
            #f = feature.hog(img_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(2, 2), feature_vector=True, block_norm='L2-Hys')
            l.append(img)
        return np.array(l)

def main():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)

    plt.figure(figsize=(16,4))
    for index, (image, labels) in enumerate(zip(train_X, train_Y)):
        plt.subplot(1, 6, index + 1)
        plt.imshow(np.reshape(image, (32,32)), cmap=plt.cm.gray)
        plt.title('Training: %i\n' % labels, fontsize = 20)
    logisticRegr = LogisticRegression(solver = 'lbfgs')
    logisticRegr.fit(train_X, train_Y)
    predictions = logisticRegr.predict(test_X)

    score = logisticRegr.score(predictions, test_Y)
    print(score)
    cm = metrics.confusion_matrix(test_Y, predictions)
    print(cm)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15);

if __name__ == "__main__":
    main()