# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from skimage import io, color, filters, feature, transform
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift

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
            #f = feature.hog(img_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(2, 2), feature_vector=True, block_norm='L2-Hys')
            img = img.flatten()
            l.append(img)
        return np.array(l)

def kMeans():
    #load and process data
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    #runs KMeans
    kmeans = KMeans(n_clusters=62)
    kmnfit = kmeans.fit_predict(train_X)
    kmnpredict = kmeans.predict(test_X)
    
    fig, ax = plt.subplots(7, 10, figsize=(20,20))
    centers = kmeans.cluster_centers_.reshape(62, 32, 32)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
        
    clusterdict = {}
    #create dictionary with cluster numbers as keys and lists of indices in test_X as values
    for i in range(len(kmnfit)):
        if kmnfit[i] in clusterdict:
            clusterdict[kmnfit[i]].append(i)
        else:
            clusterdict[kmnfit[i]] = [i]
            #now clusterdict has an entry for each cluster that contains a list of the indices in train_X that make up that cluster
    #Make values of clusterdict be dictionaries of format {character: number of occurrences}
    for cluster, indices in clusterdict.items():
        clusterdict[cluster] = {}
        for i in indices:
            if (train_Y[i]) in clusterdict[cluster]:
                clusterdict[cluster][train_Y[i]] += 1
            else:
                clusterdict[cluster][train_Y[i]] = 1
    labelNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    clusterTable = pd.DataFrame.from_dict(clusterdict, orient='index', columns=labelNames)
    clusterTable.fillna(0)
    clusterTable.to_excel("ClusterTable1.xlsx")
    plt.savefig("Clusters.png")
    print(clusterTable)
    
        
        
def spectralClustering():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    
    spectral = SpectralClustering(n_clusters=62, affinity='nearest_neighbors', 
                                  assign_labels='kmeans')
    spectralfit = spectral.fit_predict(train_X)
    print(spectral.affinity_matrix_)
    
def meanShift():
    #load and process data
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    #runs MeanShift
    mnshift = MeanShift()
    mnsfit = mnshift.fit_predict(train_X)
    mnspredict = mnshift.predict(test_X)
    
    fig, ax = plt.subplots(7, 10, figsize=(20,20))
    centers = mnshift.cluster_centers_.reshape(62, 32, 32)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
        
def main():
    kMeans()
        
if __name__== "__main__":
  main()

