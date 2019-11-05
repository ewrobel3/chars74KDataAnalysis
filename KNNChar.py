import numpy as np
from skimage import io, color, filters, feature, transform

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

class KNNCharClassifier:
    def __init__(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y

    def query_single_pt(self,query_X,k):
        dists = np.zeros((self.train_X.shape[0],1))
        for idx, pt in enumerate(self.train_X) :
            dist = np.linalg.norm(pt - query_X)
            dists[idx] = dist
        order = np.argpartition(dists, k, axis=0)
        tally = {}
        max_tally = 0
        label = None
        for i in range(k):
            n_label = self.train_Y[order[i]][0]
            if n_label in tally:
                new_tally = tally[n_label] + 1
            else:
                new_tally = 1
            tally[n_label] = new_tally
            if (new_tally > max_tally):
                max_tally = new_tally 
                label = n_label
        return label
    
    def query(self,data_X,k):
        return np.array([self.query_single_pt(x_pt,k) for x_pt in data_X]).reshape(-1,1)
    
    def test_loss(self,max_k,test_X,test_Y):
        loss = np.zeros(max_k)
        for k in range(1,max_k+1):
            loss[k-1] = (test_Y != self.query(test_X,k)).sum()/float(test_X.shape[0])
        return loss
    

def main():
    train_X, train_Y, test_X, test_Y = load_data()
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    img_clf = KNNCharClassifier(train_X, train_Y)
    loss = img_clf.test_loss(5, test_X, test_Y)
    print(loss)

if __name__ == "__main__":
    main()