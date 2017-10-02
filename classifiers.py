import numpy as np

def precision(Y, Ypred):
    return np.sum((Y==Ypred).astype(int)) / float(len(Y))

def GradientBoosting_Classifier(self,max_depth,min_samples_leaf, n_estimators,learning_rate):
    from sklearn.ensemble import GradientBoostingClassifier
    cls =GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,learning_rate=learning_rate,random_state=42)
    return cls

def RandomForest_Classifier(self,n_estimators):
    from sklearn.ensemble import RandomForestClassifier
    cls = RandomForestClassifier(n_estimators = n_estimators)
    return cls

def SVC_Classifier():
    from sklearn.svm import SVC
    cls =  SVC(kernel='rbf', C=1.0, gamma='auto')
    return cls

def MLP_Classifier():
    from sklearn.neural_network import MLPClassifier
    cls =MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200)
    return cls

def fit(self, X, y):
    self.cls.fit(X, y)

def predict(self, X):
    return self.cls.predict(X)

def score(self, X, Y):
    return self.cls.score(X,Y)

def SVM_GridSearch(clf, C_range, gamma_range, crossVal, data, labels):
    from sklearn.model_selection import GridSearchCV
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(clf, param_grid=param_grid, cv=crossVal)
    grid.fit(data, labels)
return grid.best_params_, grid.best_score_