import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets

def build_image(digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    

def load(image):
    digits = datasets.load_digits()

    if image: build_image(digits)
    
    return digits
        
def separate(digits):
    X, y = digits.data, digits.target

    # partition data with train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,stratify=y,random_state=42)

    print("train size:",len(X_train),"\ntest size:",len(X_test))
    
    return X_train, X_test, y_train, y_test
    
def learn(X_train, y_train):
    knn3= KNeighborsClassifier(n_neighbors=3)
    knn30= KNeighborsClassifier(n_neighbors=30)
    gauss=GaussianNB()

    knn3.fit(X_train, y_train)
    knn30.fit(X_train, y_train)
    gauss.fit(X_train, y_train)
    
    return knn3, knn30, gauss
    
def test(X_test, y_test, knn3, knn30, gauss):
    y_pred3 = knn3.predict(X_test)
    y_pred30 = knn30.predict(X_test)
    y_predg = gauss.predict(X_test)
    return [
        round(metrics.accuracy_score(y_test, y_pred3),2), 
        round(metrics.accuracy_score(y_test, y_pred30),2), 
        round(metrics.accuracy_score(y_test, y_predg),2)
    ]
    
def do(image):
    digits = load(image)
    
    X_train, X_test, y_train, y_test = separate(digits)
    
    knn3, knn30, gauss = learn(X_train, y_train)
    
    return test(X_test, y_test, knn3, knn30, gauss)