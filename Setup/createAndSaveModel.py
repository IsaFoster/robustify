from _readData import getXandYShortFromFile, getXandYFromFile
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'************* Load Data *******************'
X_train, y_train, _, _ = getXandYFromFile()
X_train_short, y_train_short, _, _ = getXandYShortFromFile()
'*******************************************'

'********** Make and save models ***********'
modelName = "SVC_full_set"
model = SVC(kernel='linear')
model.fit(X_train, y_train.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "SVC_reduced_set"
model = SVC(kernel='linear')
model.fit(X_train_short, y_train_short.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "RF_full_set"
model = RandomForestClassifier()
model.fit(X_train, y_train.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "RF_reduced_set"
model = RandomForestClassifier()
model.fit(X_train_short, y_train_short.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "MLP_full_set"
model = MLPClassifier()
model.fit(X_train, y_train.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "MLP_reduced_set"
model = MLPClassifier()
model.fit(X_train_short, y_train_short.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "LDA_full_set"
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

modelName = "LDA_reduced_set"
model = LinearDiscriminantAnalysis()
model.fit(X_train_short, y_train_short.values.ravel())
pickle.dump(model, open('../Models/' + modelName, 'wb'))

'*******************************************'