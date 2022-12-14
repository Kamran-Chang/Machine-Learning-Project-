import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from scipy.stats import sem
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import classification_report, matthews_corrcoef

warnings.filterwarnings('ignore')
import random
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("training_set.csv")
df.info()

df.head(-1)

df.SEX = df.SEX.replace(to_replace=['F', 'M'], value=[0, 1])

df.head(-1)

"""# **KNN**"""

# lets check accuracy of our model first

model = KNeighborsClassifier(5)


def evaluate_model(X, y, repeats):
    cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def callModel(X, y):
    repeats = range(1, 4)
    results = []
    for r in repeats:
        scores = evaluate_model(X, y, r)
        results.append(scores)
    plt.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    plt.title('Boxplot for each RepeatedKFold')
    plt.xlabel('Repeats')
    plt.ylabel('accuracy')
    plt.show()


x = df.drop(['SOURCE'], axis=1).values
y = df[['SOURCE']].values
callModel(x, y)

# check for null values -- there is none
df.isnull().sum()

# NO DUPLICATES
df.duplicated()

"""# **Check for Outliers**"""

plt.boxplot(df.HAEMATOCRIT)

plt.boxplot(df.ERYTHROCYTE)

plt.boxplot(df.HAEMOGLOBINS)

plt.boxplot(df.LEUCOCYTE)

plt.boxplot(df.THROMBOCYTE)

plt.boxplot(df.MCH)

plt.boxplot(df.MCHC)

plt.boxplot(df.MCV)

cols = df.select_dtypes(['int64', 'float64']).columns
for column in cols:
    q1 = df[column].quantile(0.25)  # First Quartile
    q3 = df[column].quantile(0.75)  # Third Quartile
    IQR = q3 - q1  # Inter Quartile Range

    ll = q1 - 1.5 * IQR  # Lower Limit
    ul = q3 + 1.5 * IQR  # Upper Limit

    outliers = df[(df[column] < ll) | (df[column] > ul)]
    print('Number of outliers in "' + column + '" : ' + str(len(outliers)))

"""# **Train Test Sample**"""

Y = df["SOURCE"].values
X = df.drop(["SOURCE"], axis=1)

# train test samples
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# X=df.drop(["SOURCE"],axis=1)
# Y=df['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

"""# **Removing Outliers**"""

cols = df.select_dtypes(['int64', 'float64']).columns
for column in cols:
    q1 = df[column].quantile(0.25)  # First Quartile
    q3 = df[column].quantile(0.75)  # Third Quartile
    IQR = q3 - q1  # Inter Quartile Range

    ll = q1 - 1.5 * IQR  # Lower Limit
    ul = q3 + 1.5 * IQR  # Upper Limit

    outliers = df[(df[column] < ll) | (df[column] > ul)]
    print('Number of outliers in "' + column + '" : ' + str(len(outliers)))


def outlier_capping(x):
    x = x.clip(upper=x.quantile(0.95))
    x = x.clip(lower=x.quantile(0.05))
    return x


df = df.apply(outlier_capping)

cols = df.select_dtypes(['int64', 'float64']).columns
for column in cols:
    q1 = df[column].quantile(0.25)  # First Quartile
    q3 = df[column].quantile(0.75)  # Third Quartile
    IQR = q3 - q1  # Inter Quartile Range

    ll = q1 - 1.5 * IQR  # Lower Limit
    ul = q3 + 1.5 * IQR  # Upper Limit

    outliers = df[(df[column] < ll) | (df[column] > ul)]
    print('Number of outliers in "' + column + '" : ' + str(len(outliers)))

df.drop('MCH', inplace=True, axis=1)

df.head()

"""# **Box PLots**"""

df.head()


"""# **Co-related Features**"""

features_mean = list(df.columns[1:11])

df.head()

# check highly co-related fetaures
plt.figure(figsize=(10, 10))
sns.heatmap(df[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()

cor_matrix = df.corr().abs()
print(cor_matrix)

df.head()

# printing upper triangle
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
print(upper_tri)

df.head()

"""# **Removing Highly Co-related Features**"""

# Removing highly correlated features having correlation > 0.90
cor_matrix = df.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
data = df.drop(to_drop, axis=1)
print(data)

df.head()

"""# **Check For Class Balance**"""

# check for class imbalance
sns.countplot(Y, label='SOURCE')

oversample = SMOTE()
X, y = oversample.fit_resample(X, y.ravel())

# check for class imbalance
sns.countplot(y, label='SOURCE')

df['AGE'] = df['AGE'].replace([df['AGE'] > 85], random.randint(1, 85))

plt.figure(figsize=(20, 10))
for i, col in enumerate(df.columns[:-1]):
    k = i + 1
    plt.subplot(2, 6, int(k))
    sns.distplot(x=df[col],
                 color='darkblue', kde=False).set(title=f"{col}");
    plt.xlabel("")
    plt.ylabel("")
    plt.grid()
plt.show()

data.AGE.max()

data.columns

"""# **Scatter Plots**"""

count_ter = sns.scatterplot(x="AGE", y="HAEMATOCRIT", data=df)
count_ter.set_title("Scatter Plot with Respect to Tests")
count_ter.set_xlabel("AGE");
count_ter.set_ylabel("HAEMATOCRIT");

count_ter = sns.scatterplot(x="AGE", y="HAEMOGLOBINS", data=df)
count_ter.set_title("Scatter Plot with Respect to Tests")
count_ter.set_xlabel("AGE");
count_ter.set_ylabel("HAEMOGLOBINS");

count_ter = sns.scatterplot(x="AGE", y="ERYTHROCYTE", data=df)
count_ter.set_title("Scatter Plot with Respect to Tests")
count_ter.set_xlabel("AGE");
count_ter.set_ylabel("ERYTHROCYTE");

count_ter = sns.scatterplot(x="AGE", y="THROMBOCYTE", data=df)
count_ter.set_title("Scatter Plot with Respect to Tests")
count_ter.set_xlabel("AGE");
count_ter.set_ylabel("THROMBOCYTE");

count_ter = sns.scatterplot(x="AGE", y="MCHC", data=df)
count_ter.set_title("Scatter Plot with Respect to Tests")
count_ter.set_xlabel("AGE");
count_ter.set_ylabel("MCHC");

count_ter = sns.scatterplot(x="AGE", y="MCV", data=df)
count_ter.set_title("Scatter Plot with Respect to Tests")
count_ter.set_xlabel("AGE");
count_ter.set_ylabel("MCV");

callModel(df.drop(['SOURCE'], axis=1), df[['SOURCE']].values)

X = data.drop(["SOURCE"], axis=1)
Y = data['SOURCE'].values
X.head()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
callModel(X, y)

"""# **KNN**"""

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# =data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=13)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=17)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=19)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

# X=data.drop(["SOURCE"],axis=1)
# Y=data['SOURCE']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
classifier = KNeighborsClassifier(n_neighbors=21)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('accuracy score is: ' + str(accuracy_score(y_test, y_pred)))

"""# **Logistic Regression**"""

# y = data["SOURCE"].values
# = data.drop(["SOURCE"],axis=1)

y = data["SOURCE"].values
x = data.drop(["SOURCE"],axis=1)
x.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn import linear_model
logreg = linear_model.LogisticRegression()
logreg.fit(X_train_std, y_train)

y_pred = logreg.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

pickle.dump(logreg, open('model.pkl', 'wb'))

pickled_model = pickle.load(open('model.pkl', 'rb'))

"""# **Binary Classification With Neural Network**"""

x = df.drop(["SOURCE"], axis=1)
y = df["SOURCE"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=9))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)

scores = model.evaluate(np.array(x), np.array(y))
print("Loss:", scores[0])
print("Accuracy", scores[1] * 100)


"""# **AUC Curve**"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

sns.set()

acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()

"""# **Confusion Matrix**"""

from sklearn.metrics import confusion_matrix

y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)

labels = ['Care', 'Not Care']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# confusion matrix
print('Confusion matrix : \n', mat)

# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(y_test, y_predicted, labels=[1, 0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy
matrix_detail = classification_report(y_test, y_predicted, labels=[1, 0])
print('Classification report : \n', matrix_detail)