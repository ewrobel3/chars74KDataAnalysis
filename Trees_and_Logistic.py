# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from skimage import io, color, filters, feature, transform
import matplotlib as ml
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


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

def decisionTree(train_X, train_Y, test_X,):
    tree = RandomForestClassifier()
    tree = tree.fit(train_X, train_Y)
    prediction = tree.predict(test_X)
    #plot_tree(tree, filled='true')
    #plt.show()
    return prediction

def logistic(train_X, train_Y, test_X, test_Y):
    mullr = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter = 1000)
    mullr = mullr.fit(train_X, train_Y)
    prediction = mullr.predict(test_X)
    return prediction
    
def main():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    prediction = logistic(train_X, train_Y, test_X, test_Y)
    print(classification_report(test_Y,prediction))
    #prediction = decisionTree(train_X, train_Y, test_X)
    #print(confusion_matrix(test_Y,prediction))
    #print(classification_report(test_Y,prediction))


if __name__ == "__main__":
    main()