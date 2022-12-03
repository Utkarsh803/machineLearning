import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords
import emoji
from googletrans import Translator
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve


stemmer = SnowballStemmer('english')
translator = Translator()
stop = stopwords.words('english')

df = pd.concat(
    map(pd.read_csv, ['dataset.csv']), ignore_index=True)

print(df.shape)
#df = pd.read_csv("imdbReviews.csv")
df=df.dropna().reset_index(drop=True)



def stemStr(text):
    review = []
    reviewToken = [] 
    tokens = word_tokenize(text)
    for token in tokens:
        reviewToken = token
        reviewToken = stemmer.stem(token)    
        review.append(reviewToken) 
    result = " ".join(review) 
    return result


for index, row in df.iterrows():
    if type(row["Review"]) == str:
        tweet = emoji.demojize(row["Review"])
        tweet = tweet.replace(":"," ")
        tweet = ' '.join(tweet.split())
        tweet=stemStr(tweet)
        df.at[index,'Review'] = re.sub(r'\w*\d\w*', '', tweet).strip()

X1 = df.iloc[:, 1]
X2 = df.iloc[:, 2]  

vectorizer=CountVectorizer(ngram_range=(1,2), min_df = 2, stop_words='english', max_features=500)
x_train_bow=vectorizer.fit_transform(X2.values.astype('U'))
df_bow_sklearn = pd.DataFrame(x_train_bow.toarray(),columns=vectorizer.get_feature_names())
print(df_bow_sklearn.head)



k_values = []
accuracy = []
errorList = []

#crossvalidation scores
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    crossValScore = cross_val_score(knn, df_bow_sklearn, X1 , scoring="f1", cv=5)
    accuracy.append(np.array(crossValScore).mean())
    k_values.append(i)
    errorList.append(np.array(crossValScore).std())
    print("k: ", i)
    print("accuracy: ", crossValScore)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(
    "f1 score vs K of KNN model")
ax.errorbar(k_values, accuracy, label='Mean score', yerr=errorList)
ax.set_xlabel("K")
ax.set_ylabel("f1 score")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(df_bow_sklearn, X1, test_size=0.33)

k = [14]
##confusion matrix and ROC
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    print("KNN")
    print('K:', i)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("AUC", auc)
    print("CM", cm)


knn = DummyClassifier(strategy='uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[::, 1]
fpr2, tpr2, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)
print("KNN")
print("DummyClassifier - uniform")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("AUC", auc)
print("CM", cm)

knn = DummyClassifier(strategy='most_frequent')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[::, 1]
fpr3, tpr3, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)
print("KNN")
print("DummyClassifier - most_frequent")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("AUC", auc)
print("CM", cm)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("ROC curves for different models")
ax.plot(fpr, tpr, color='green')
ax.plot(fpr2, tpr2, color='blue', linewidth=3)
ax.plot(fpr3, tpr3, color='red')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
plt.legend(["k=14", "uniform", "most_frequent"])
plt.show()




df["Review1"] = df["Review"].str.lower()
df["new_column"] = df['Review1'].str.replace('[^\w\s]','')
df['new'] = df['new_column'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

pos_reviews =  df[df["Rating"] == 1]
print(pos_reviews.shape)

neg_reviews =  df[df["Rating"] == 0]
print(neg_reviews.shape)


count = Counter()
for text in pos_reviews['new']:
    for word in text.split(" "):
        count[word] =count[word] + 1
list1 = sorted(count, key = count.get, reverse = True)
list1 = list1[:50]

count2 = Counter()
for text in neg_reviews['new']:
    for word in text.split(" "):
        count2[word] =count2[word] +1

list2 = sorted(count2, key = count2.get, reverse = True)
list2= list2[:50]

print("Positive")
temp3 = [item for item in list1 if item not in list2]
print(temp3)


print("Negative")
temp3 = [item for item in list2 if item not in list1]
print(temp3)


