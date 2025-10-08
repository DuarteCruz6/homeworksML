from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def train(X_train, X_test, y_train, y_test, neurons, architecture):

    predictor = MLPClassifier(hidden_layer_sizes=neurons,random_state=42,activation = architecture,solver='sgd',max_iter=2000)
    predictor.fit(X_train, y_train)

    y_pred = predictor.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("accuracy on testing set:",  round(acc,4))
    
    return acc, predictor.loss_curve_

