import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import emoji
from googletrans import Translator
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
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

vectorizer=CountVectorizer(stop_words='english', max_features=1000, min_df = 2)
x_train_bow=vectorizer.fit_transform(X2.values.astype('U'))
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
vectorized_X_bow = tfidf_transformer.fit_transform(x_train_bow)
df_bow_sklearn = pd.DataFrame(vectorized_X_bow.toarray(),columns=vectorizer.get_feature_names())
print(df_bow_sklearn.head)


pos_review =  df[df["Rating"] == 1]
print(pos_review.shape)

neg_review =  df[df["Rating"] == 0]
print(neg_review.shape)

CVals= []
accuracyList2 = []
errorList2 =[]
for C in range(1,21):
    logistic_model = LogisticRegression(C = C, penalty='l2', max_iter=1000)
    #score the model
    scores = cross_val_score(logistic_model, df_bow_sklearn , X1, cv=5, scoring='f1')
    print("Accuracy:", scores)
    CVals.append(C)
    accuracyList2.append(np.array(scores).mean())
    errorList2.append(np.array(scores).std())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(
    "f1 scores vs penalty value C of Logistic Regression model with 5000 features and min_df= 2")
ax.errorbar(CVals, accuracyList2, label='Mean Error', yerr=errorList2)
ax.set_xlabel("C")
ax.set_ylabel("f1 score")
ax.legend()
plt.show()



##confusion matrix and ROC

X_train, X_test, y_train, y_test = train_test_split(df_bow_sklearn, X1, test_size=0.33)

tupleList = []
knn = LogisticRegression(penalty='l2',C=1, max_iter=1000)
knn.fit(X_train, y_train)
importance = knn.coef_[0]
plt.plot()
for m,v in enumerate(importance): 
    tupleList.append((X_train.columns[m],v))
sortedtupleList = sorted(
    tupleList ,
    key=lambda t: t[1]
)
posSort = sortedtupleList[-10:]
negSort = sortedtupleList[:10]

plt.bar(range(len(posSort)), [val[1] for val in posSort ], align='center')
plt.xticks(range(len(posSort )), [val[0] for val in posSort])
plt.xticks(fontsize=7,rotation=70)
plt.title("Feature importance in logistic Regression for Positive sentiment")
plt.legend(["Coefficients"])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Coefficients of different features")
plt.legend(["coefficient"]) 
plt.bar([x for x in range(len(importance))], importance) 
plt.show() 
y_pred= knn.predict(X_test) 
y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test,y_pred)*100
precision = precision_score(y_test,y_pred)*100
recall = recall_score(y_test,y_pred)*100
cm = confusion_matrix(y_test, y_pred)
print("Logististic regression")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("AUC",auc)
print("CM",cm)



knn = DummyClassifier(strategy='uniform')
knn.fit(X_train, y_train)
y_pred= knn.predict(X_test) 
y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr2, tpr2, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test,y_pred)*100
precision = precision_score(y_test,y_pred)*100
recall = recall_score(y_test,y_pred)*100
cm = confusion_matrix(y_test, y_pred)
print("Random Model")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("AUC",auc)
print("CM",cm)

knn = DummyClassifier(strategy='most_frequent')
knn.fit(X_train, y_train)
y_pred= knn.predict(X_test) 
y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr3, tpr3, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test,y_pred)*100
precision = precision_score(y_test,y_pred)*100
recall = recall_score(y_test,y_pred)*100
cm = confusion_matrix(y_test, y_pred)
print("Most frequent model")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("AUC",auc)
print("CM",cm)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("ROC curves for different models")
ax.plot(fpr, tpr, color='green')
ax.plot(fpr2, tpr2, color='orange')
ax.plot(fpr3, tpr3, color='red')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
plt.legend(["kNN", "Logistic Regression","Random Model", "Most Frequent Model"])
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
        count[word] +=1
#list1 = count.most_common(50)
list1 = sorted(count, key = count.get, reverse = True)
list1 = list1[:50]

count2 = Counter()
for text in neg_reviews['new']:

    
    for word in text.split(" "):
        count2[word] +=1
#list2 = count2.most_common(50)
list2 = sorted(count2, key = count2.get, reverse = True)
list2= list2[:50]

print("Positive")
temp3 = [item for item in list1 if item not in list2]
print(temp3)

print("Negative")
temp3 = [item for item in list2 if item not in list1]
print(temp3)


