# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from skimage import io, color, transform
import matplotlib as ml
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn import metrics
from matplotlib.legend_handler import HandlerLine2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def imread_convert(f):
        return io.imread(f).astype(np.uint8)

def load_data():
        # loads 99 images per character, for all 62 characters
        ic = io.ImageCollection("./Fnt/Sample0*/img*-000*.png", conserve_memory=True, load_func=imread_convert)
        data = io.concatenate_images(ic)
        labelNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
        labels = np.empty([0], dtype=str)
        for label in labelNames:
            labels = np.append(labels, np.full([99], label, dtype=str))
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
            img = img.flatten() #line added my mary to make code work in sklearn tree class
            #f = feature.hog(img_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(2, 2), feature_vector=True, block_norm='L2-Hys')
            l.append(img)
        return np.array(l)
#this method does not error but it gives really really inaccurate resuts when used
def pca(train_X):
    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    pca = PCA(n_components = 100)
    pca.fit(train_X)
    X_pca = pca.transform(train_X)
    print("origional shape: ", train_X.shape)
    print("new shape: ", X_pca.shape)
    return X_pca

def svm(train_X, train_Y, test_X, test_Y):
    classification = SVC(kernel = 'linear')
    classification = classification.fit(train_X, train_Y)
    svmPrediction = classification.predict(test_X)
    return svmPrediction
    

def randomForrest(train_X, train_Y, test_X, test_Y):
    n_estim = [1,2,4,8,16,32,64,100,200]
    train_results = []
    test_results = []
    for estimator in n_estim:
        forrest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                         max_depth=None, max_features='auto', max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2,
                                         min_weight_fraction_leaf=0.0, n_estimators=estimator, n_jobs=1,
                                         oob_score=False, random_state=None, verbose=0,
                                         warm_start=False)
        forrest = forrest.fit(train_X, train_Y)
        train_results.append(metrics.accuracy_score(train_Y,forrest.predict(train_X)))
        forrestPrediction = forrest.predict(test_X)
        test_results.append(metrics.accuracy_score(test_Y, forrestPrediction))
        
    line1, = plt.plot(n_estim,test_results,'r',label = "Test Max Estimators")
    line2, = plt.plot(n_estim, train_results, 'b', label = "Train Max Estimators")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("number of estimators")
    plt.show()
    
    return forrestPrediction

def decisionTree(train_X, train_Y, test_X):
    tree= DecisionTreeClassifier(class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None, splitter ='best')
    tree = tree.fit(train_X, train_Y)
    treePrediction = tree.predict(test_X)
    #plot_tree(tree, filled='true')
    #plt.show()
    
    return treePrediction

def logistic(train_X, train_Y, test_X, test_Y):
    mullr = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter = 1000)
    mullr = mullr.fit(train_X, train_Y)
    logPrediction = mullr.predict(test_X)
    return logPrediction

def knn(train_X, train_Y, test_X, test_Y):
    #n_estim = [1,2,4,8,16,32,64,100,200]
    #train_results = []
    #test_results = []
    #for estimator in n_estim:
    classifier = KNeighborsClassifier(n_neighbors = 5)
    classifier.fit(train_X, train_Y)
        #train_results.append(metrics.accuracy_score(train_Y,classifier.predict(train_X)))
    knnPrediction = classifier.predict(test_X)
        #test_results.append(metrics.accuracy_score(test_Y, knnPrediction))
        
    #line1, = plt.plot(n_estim,test_results,'r',label = "Test Max Estimators")
    #line2, = plt.plot(n_estim, train_results, 'b', label = "Train Max Estimators")
    #plt.legend()
    #plt.ylabel("Accuracy")
    #plt.xlabel("number of estimators")
    #plt.show()
        
    return knnPrediction
    
def main():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    train_Xpca =pca(train_X)
    test_Xpca = pca(test_X)
    treePrediction = decisionTree(train_Xpca, train_Y, test_Xpca)
    print("Decision Tree Accuracy:", metrics.accuracy_score(test_Y, treePrediction))

    
    #svmPrediction = svm(train_X, train_Y, test_X, test_Y)
    #print("SVM Accuracy", metrics.accuracy_score(test_Y, svmPrediction))
    #print(classification_report(test_Y, svmPrediction))
    
    #knnPrediction = knn(train_X, train_Y, test_X, test_Y)
    #print(classification_report(test_Y, knnPrediction))
    #print("KNN Accurary", metrics.accuracy_score(test_Y, knnPrediction))
    
    #logPrediction = logistic(train_X, train_Y, test_X, test_Y)
    #print(classification_report(test_Y,logPrediction))
    #forrestPrediction = randomForrest(train_X, train_Y, test_X, test_Y)
    #treePrediction = decisionTree(train_X, train_Y, test_X)
    #print("Random Forrest Accuracy:", metrics.accuracy_score(test_Y, forrestPrediction))
    #print("Decision Tree Accuracy:", metrics.accuracy_score(test_Y, treePrediction))
    #print("Random Forrest Classification Report")
    #print(classification_report(test_Y,forrestPrediction))


if __name__ == "__main__":
    main()