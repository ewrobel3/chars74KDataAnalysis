import numpy as np
import matplotlib.pyplot as plt
import os.path
import string
from skimage import io, color, filters, feature, transform
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def imread_convert(f):
        return io.imread(f).astype(np.uint8)

def load_data(data_amnt):
        if (data_amnt > 1016):
            print("data amount invalid")
            return -1
        ic = io.ImageCollection("./Fnt/Sample0*/img*-00*.png", conserve_memory=True, load_func=imread_convert)
        data = io.concatenate_images(ic)
        data = data[0:62*data_amnt]
        labelNames = string.digits + string.ascii_uppercase + string.ascii_lowercase
        labels = np.empty(data_amnt*62, str)
        for l in range(len(labelNames)):
            labels[l*data_amnt : l*data_amnt + data_amnt] = np.full(data_amnt, labelNames[l])
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
            img = transform.resize(img, (32, 32), anti_aliasing=True, mode='reflect') #anti_aliasing automatically blurs before resize 
            # fd, img = feature.hog(img, orientations=10, pixels_per_cell=(5, 5), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
            img = np.array(img).flatten()
            l.append(img)
        return np.array(l)

def knn_error(train_X, train_Y, test_X, test_Y, n_pca=49, n_neighbors=4):
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)

    # pca = PCA(n_components=n_pca)
    # pca_model = pca.fit(train_X)
    # pca_train = pca_model.transform(train_X)
    # pca_test = pca_model.transform(test_X)

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # knn_model.fit(pca_train, train_Y)
    knn_model.fit(train_X, train_Y)
    # train_prediction = np.array(knn_model.predict(pca_train))
    train_prediction = np.array(knn_model.predict(train_X))
    # test_prediction = np.array(knn_model.predict(pca_test))
    test_prediction = np.array(knn_model.predict(test_X))
    train_accuracy = accuracy_score(train_Y, train_prediction)
    test_accuracy = accuracy_score(test_Y, test_prediction)
    return train_accuracy, test_accuracy


def main():
    train_X, train_Y, test_X, test_Y = load_data(500)
    train_accuracy = []
    test_accuracy = []
    max_k = 10
    for k in range(max_k):
        tr_acc, tst_acc = knn_error(train_X, train_Y, test_X, test_Y, n_neighbors=k+1)
        print(tr_acc, tst_acc)
        train_accuracy.append(tr_acc)
        test_accuracy.append(tst_acc)
    plt.title("KNN Accuracy", fontsize=16, fontweight='bold')
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    xaxis = np.linspace(1, max_k, max_k)
    plt.xticks(np.arange(0, max_k+1))
    plt.plot(xaxis, train_accuracy, c='b')
    plt.plot(xaxis, test_accuracy, c='r')
    plt.show()
    

if __name__ == "__main__":
    main()