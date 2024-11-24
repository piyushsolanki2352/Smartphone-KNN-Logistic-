# Smartphone-(KNN-Logistic)-

aimport pandas as pd
import numpy as np
new = pd.read_excel("/content/Book21.xlsx")
print(new)
x = new[['weight(x2)','height(y2)']].values
y = new['class'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train, y_train)

new = np.array([[57,170]])
y_pred = knn.predict(new)
print(y_pred)


(KNN, Logistic)
import pandas as pd
import numpy as np
new = pd.read_excel("/content/Book22.xlsx")
print(new)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
new['size']= le.fit_transform(new['size'])
# new['type']= le.fit_transform(new['type'])
x = new[['weight','color intensity','size']].values
y = new['type'].values
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

new = np.array([[140,7,1]])
y_pred = knn.predict(new)
print(y_pred)

(KNN, Logistic)

import pandas as pd
new = pd.read_csv("/content/Iris.csv")
print(new)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
new['Species'] = le.fit_transform(new['Species'])
x = new[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','SepalWidthCm']].values
y = new['Species'].values
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
