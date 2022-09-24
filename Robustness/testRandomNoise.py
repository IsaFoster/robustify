from RandomNoise import addNoise, evaluate
from readData import getData
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


'***************** Load and Split Data ****************'
df_train, df_val, df_test = getData()
y_train = df_train[['data_type']]
X_train = df_train.drop(['data_type'], axis=1)
y_val = df_val[['data_type']]
X_val =  df_val.drop(['data_type'], axis=1)
y_test = df_test[['data_type']]
X_test = df_test.drop(['data_type'], axis=1)
'********************************************'

'**************** Train Models **************'
m2 = RandomForestClassifier(random_state=42)
m2.fit(X_train, y_train)

preds = m2.predict(X_val)
print("Accuracy:", accuracy_score(y_val, preds))
'********************************************'

'*************** Plot Accuracy **************'
conf = confusion_matrix(y_val, preds)
conf_matrix = ConfusionMatrixDisplay(conf)
conf_matrix.plot()
plt.show()

accuracy_scores = evaluate(m2, X_train, y_train, X_test, y_test)

print('base accuracy:', accuracy_score[0])
print('pertubateed up accuracy:', accuracy_score[1])
print('pertubated down accuracy:', accuracy_score[2])

'''plt.figure()
lw = 2
plt.plot(jitters, mdl1_scores, color='darkorange',
         lw=lw, label='random forrest')
plt.xlabel('Amount of Noise')
plt.ylabel('Accuracy')
plt.title('Accuracy for increasing noise')
plt.legend(loc="lower right")
plt.show()'''
'********************************************'