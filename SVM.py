import numpy as np
import matplotlib.pyplot as plt
import string
import os.path
from skimage import io, color, filters, feature, transform
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
            img = color.rgb2gray(img)
            img = transform.resize(img, (32, 32), anti_aliasing=True, mode='reflect') 
            # img = filters.gaussian(img, 0.4)
            f = feature.hog(img, orientations=10, pixels_per_cell=(4, 4), cells_per_block=(4, 4), block_norm='L2-Hys')
            img = np.array(f).flatten()
            l.append(img)
        return np.array(l)

def optimize_svm(train_X, train_Y, test_X, test_Y, n_pca=50):
    # svm = SVC(gamma='auto')
    # parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}
    # clf = GridSearchCV(svm, parameters, cv=5)
    # model = clf.fit(train_X, train_Y)
    # print(model)
    # train_prediction = clf.predict(train_X)
    # test_prediction = clf.predict(test_X)


    pca = PCA(n_components=n_pca)
    pca_model = pca.fit(train_X)
    pca_train = pca_model.transform(train_X)
    pca_test = pca_model.transform(test_X)
    
    svm = SVC(gamma='scale', C=1.0, kernel='rbf')
    svm.fit(pca_train, train_Y)
    train_prediction = svm.predict(pca_train)
    test_prediction = svm.predict(pca_test)

    train_accuracy = accuracy_score(train_Y, train_prediction)
    test_accuracy = accuracy_score(test_Y, test_prediction)
    return train_accuracy, test_accuracy

def main():
    train_X, train_Y, test_X, test_Y = load_data(200)
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    trn_acc, tst_acc = optimize_svm(train_X, train_Y, test_X, test_Y)
        
    print("Train and test accuracy with hog", trn_acc, tst_acc)
    # print("Best number of components for PCA with SVM classifier:", max_n)
    # plt.title("SVM Accuracy", fontsize=16, fontweight='bold')
    # plt.xlabel("Number of PCA Components")
    # plt.ylabel("Accuracy")
    # xaxis = np.linspace(50, 65, 16)
    # plt.xticks(np.arange(50, 65))
    # train_plt = plt.plot(xaxis, train_accuracy, c='b')
    # test_plt = plt.plot(xaxis, test_accuracy, c='r')
    # plt.legend([train_plt, test_plt], ['Train', 'Test'])
    # plt.show()

if __name__ == "__main__":
    main()
