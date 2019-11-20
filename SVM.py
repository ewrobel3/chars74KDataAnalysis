import numpy as np
import matplotlib.pyplot as plt
import string
import os.path
from skimage import io, color, filters, feature, transform, exposure, util
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def imread_convert(f):
        return io.imread(f).astype(np.uint8)

def load_data(data_amnt):
        ic = io.ImageCollection("./Fnt/Sample0*/img*-000*.png", conserve_memory=True, load_func=imread_convert)
        data = io.concatenate_images(ic)
        data = data[0:62*data_amnt]
        labelNames = string.digits + string.ascii_uppercase + string.ascii_lowercase
        labels = np.empty(data_amnt*62, str)
        for l in range(len(labelNames)):
            labels[l*data_amnt : l*data_amnt+data_amnt] = np.full(data_amnt, labelNames[l])
        shuffled_idx = np.random.permutation(data.shape[0])
        cutoff = int(data.shape[0]*0.8)
        train_X = data[shuffled_idx[:cutoff]]
        test_X = data[shuffled_idx[cutoff:]]
        train_Y = labels[shuffled_idx[:cutoff]]
        test_Y = labels[shuffled_idx[cutoff:]]
        return train_X, train_Y, test_X, test_Y

def preprocess(imgs):
        l = []
        for img in imgs:

            # 1. RGB to Gray
            img = color.rgb2gray(img)
            # plt.figure()
            # plt.imshow(img, cmap='gray')

            # 2. Size normalization
            img = transform.resize(img, (64, 64), anti_aliasing=True, mode='reflect')
            # plt.figure()
            # plt.imshow(img, cmap='gray')

            # 3. Binarization
            thresh = filters.threshold_otsu(img)
            img = img > thresh
            img = util.invert(img)
            # plt.figure()
            # plt.imshow(img, cmap='gray')

            # 4. Skeletonization: thins lines in image to produce a skeleton of each character input
            img = skeletonize(img).astype(float)
            # plt.figure()
            # plt.imshow(img, cmap='gray')

            # 5. Feature Extraction: Histogram of Oriented Gradients
            f = feature.hog(img, orientations=10, pixels_per_cell=(8, 8), feature_vector=True, cells_per_block=(2, 2), block_norm='L2-Hys')
            # img = exposure.rescale_intensity(img, in_range=(0, 10))
            # plt.figure()
            # plt.imshow(img, cmap='gray')
            l.append(f)
            # plt.show()
        return np.array(l)

def optimize_svm(train_X, train_Y, test_X, test_Y, n_pca=50):
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)

    # Grid Search CV
    # svm = SVC(gamma='auto')
    # parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100, 1000]}
    # clf = GridSearchCV(svm, parameters, cv=5)
    # model = clf.fit(train_X, train_Y)
    # print(model)

    # Regular
    svm = SVC(gamma='scale', C=1.0, kernel='linear')
    # svm.fit(train_X, train_Y)

    # train_prediction = clf.predict(train_X)
    # test_prediction = clf.predict(test_X)


    PCA
    pca = PCA(n_components=n_pca)
    pca_model = pca.fit(train_X)
    pca_train = pca_model.transform(train_X)
    pca_test = pca_model.transform(test_X)
    svm.fit(pca_train, train_Y)
    train_prediction = svm.predict(pca_train)
    test_prediction = svm.predict(pca_test)

    train_accuracy = accuracy_score(train_Y, train_prediction)
    test_accuracy = accuracy_score(test_Y, test_prediction)
    return train_accuracy, test_accuracy

def main():
    train_X, train_Y, test_X, test_Y = load_data(200)
    # trn_acc, tst_acc = optimize_svm(train_X, train_Y, test_X, test_Y)
    train =[]
    test = []
        
    for n in range(5):
        trn_acc, tst_acc = optimize_svm(train_X, train_Y, test_X, test_Y, 50+n*10)
        train.append(trn_acc)
        test.append(tst_acc)
    
    print("Best number of components for PCA with SVM classifier:")
    plt.title("SVM Accuracy", fontsize=16, fontweight='bold')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Accuracy")
    xaxis = np.linspace(50, 100, 5)
    plt.xticks(np.arange(50, 101, 10))
    plt.plot(xaxis, train, c='b', label='train')
    plt.plot(xaxis, test, c='r', label='test')
    plt.show()

if __name__ == "__main__":
    main()
