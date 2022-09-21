from testFunctions import addNoise, evaluate
from sklearn import svm, tree, datasets
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'***************** Set seed *****************'
np.random.seed(1234)
'********************************************'


'***************** Load Data ****************'
iris = datasets.load_iris()
data=pd.DataFrame(iris.data)
data['Species']=iris.target
data.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'Species']
print(data.head())
X = data.drop(['Species'], axis=1)
y = data['Species']
'********************************************'


'**************** Split Data ****************'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
'********************************************'


'**************** Train Models **************'
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
'********************************************'


'*************** Plot Accuracy **************'
mdl1_scores, jitters = evaluate(logreg, X_train, y_train, X_test, y_test)
mdl2_scores, jitters = evaluate(knn, X_train, y_train, X_train, y_train)

plt.figure()
lw = 2
plt.plot(jitters, mdl1_scores, color='darkorange',
         lw=lw, label='Logistic Regression')
plt.plot(jitters, mdl2_scores, color='blue',
         lw=lw, label='KNN')
plt.xlabel('Amount of Noise')
plt.ylabel('Accuracy')
plt.title('Accuracy for increasing noise')
plt.legend(loc="lower right")
plt.show()
'********************************************'