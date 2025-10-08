from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def train(X_train, X_test, y_train, y_test):
    predictor = LogisticRegression(max_iter=2000) 
    predictor.fit(X_train, y_train)

    y_pred = predictor.predict(X_test)
    print("accuracy on testing set:",  round(metrics.accuracy_score(y_test, y_pred),2))
