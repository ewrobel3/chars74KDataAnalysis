import numpy as np
from skimage import io, color, filters, feature, transform
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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

def main():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    
    # Prints the first 6 images in the training data and their corresponding labels
    plt.figure(figsize=(20,4))
    for index, (image, labels) in enumerate(zip(train_X[0:5], train_Y[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (32,32)), cmap=plt.cm.gray)
        plt.title('Training: ' + labels + '\n', fontsize = 20)
    
    #Runs logistic regression on the training data
    logisticRegr = LogisticRegression(solver = 'lbfgs')
    logisticRegr.fit(train_X, train_Y)
    predictions = logisticRegr.predict(test_X)
    score = logisticRegr.score(test_X, test_Y)
    #Prints the percentage correctly classified from the test data
    print(score)
    
    #Produces a colorplot confusion matrix for the test data
    cm = metrics.confusion_matrix(test_Y, predictions)
    plt.figure(figsize=(40,40))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion Matrix', size = 15)
    plt.colorbar()
    tick_marks = np.arange(62)
    plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], rotation=45, size = 10)
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual Label', size = 15)
    plt.xlabel('Predicted Label', size = 15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')
    
    #Collects the misclassified characters into an array
    #There is something wrong with this part
    index = 0
    misclassifiedIndexes = []
    for label, predict in zip(test_Y, predictions):
        if label != predict:
            misclassifiedIndexes.append(index)
            index +=1
    
    #Prints the misclassififed characters
    #There is something wrong with this part
    plt.figure(figsize=(20, 4))
    for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
        plt.subplot(1, 5, plotIndex + 1)
        plt.imshow(np.reshape(test_X[badIndex], (32,32)), cmap=plt.cm.gray)
        plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], test_Y[badIndex]), fontsize = 15)
        


if __name__ == "__main__":
    main()