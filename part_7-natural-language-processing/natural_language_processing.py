# Importing the libraries
# Cleaning the texts
import re
from warnings import simplefilter

import numpy as np
import pandas as pd
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def aplica_nltk(df):
	ps = PorterStemmer()  # steaming é pegar a raiz da palavra.
	stopword_set = set(stopwords.words('english'))
	corpus_ = []
	for i in range(0, len(df['Review'])):
		review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
		review = review.lower()
		review = review.split()
		# set() é usado para deixar a busca mais rápida.
		review = [ps.stem(word) for word in review if word not in stopword_set]
		# as palavras são novamente unidas em uma string ao invés de lista
		review = ' '.join(review)
		corpus_.append(review)
	return corpus_


# function to get cross validation scores
def get_cv_scores(model):
	scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
	print('CV Mean: ', np.mean(scores))
	print('STD: ', np.std(scores))
	print('\n')


def model_performance(model, X_test):
	y_pred = model.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	tn = cm[0, 0]
	fn = cm[1, 0]
	tp = cm[1, 1]
	fp = cm[0, 1]
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * precision * recall / (precision + recall)
	print(f'Accuracy: {accuracy}')
	print(f'Precision: {precision}')
	print(f'Recall: {recall}')
	print(f'F1_score: {f1_score}')
	return [accuracy, precision, recall, f1_score]


# Importing the dataset
dataset = pd.read_csv('C:\git\Machine-Learning-A-Z-Udemy\data_files\Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = aplica_nltk(dataset)
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB

# define models
naive_bayes = GaussianNB()
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
KNeighbors = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
kernel = SVC(kernel='rbf', random_state=0)
LogisticRegression = sklearn.linear_model.LogisticRegression(random_state=0)
RandomForest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
SVC = sklearn.svm.SVC(kernel='linear', random_state=0)
XGBC = XGBClassifier(n_jobs=-1, max_depth=15, scale_pos_weight=0.8)
models = [naive_bayes, tree, KNeighbors, kernel, LogisticRegression, RandomForest, SVC, XGBC]
models_names = ['naive_bayes', 'tree', 'KNeighbors', 'kernel', 'LogisticRegression', 'RandomForest', 'SVC', 'XGBC']
models_dic = OrderedDict(zip(models_names, models))

# loop through list of models
dic = {}
for model_name, model in models_dic.items():
	print(model_name)
	model.fit(X_train, y_train)
	dic[model_name] = model_performance(model, X_test)

df = pd.DataFrame(dic, index=['accuracy', 'precision', 'recall', 'f1_score']).transpose()
