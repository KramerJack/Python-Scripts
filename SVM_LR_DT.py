
##PACKAGES AND LIBRARIES 
#FFull Porject
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

#FOR SVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

#For Naive Bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
#--MultiNB not used
# from sklearn.naive_bayes import MultinomialNB
#from sklearn.model_selection import train_test_split

#For Decision Tree
from sklearn import tree
import pydotplus
from IPython.display import Image
#from sklearn.metrics import classification_report, accuracy_score
#from sklearn.model_selection import train_test_split
import graphviz 
from sklearn import metrics


##END PACKAGES AND LIBRAIRES

#START DATA IMPORT AND DISCOVERY

#loading the dataset into a df
data = pd.read_csv("INSERT FILE PATH TO CREDIT CARD FRAUD CSV")
data = pd.DataFrame(data)
data.head()
data.info()

#Eploratory Data Analysis
data.isnull().values.any() # No null values retursn false

#Histogram of Class records
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title('Transaction Class Distrobution')
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show() 

#Get the fraud and the normal dataset
fraud = data[data['Class']==1]
normal = data[data['Class']==0]
print(fraud.shape,normal.shape)
# Need to analyze more amount of info from the transaction data
#How different are the amount of money used in different transaction classes?
fraud.Amount.describe()
normal.Amount.describe()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by Class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of transactions')
plt.xlim((0,20000))
plt.yscale('log')
plt.show();

#Check if fraudelent transactions occur more often during certain time frames
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (inseconds)')
plt.ylabel('Amount')
plt.show()

#END DATA IMPORT AND DISCOVERY

#START DATA PREPERATION

#Create a sample set
data1 = data.sample(frac = 0.1,random_state=1)
data1.shape

Fraud = data1[data1['Class']==1]
Valid = data1[data1['Class']==0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print("Fraud Cases : {}".format(len(Fraud)))
print('Valid Cases : {}'.format(len(Valid)))

#Correlation
#may need to re improt seaborn as sns
#import seaborn as sns
#get correlations of each features is dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

#create independent and Dependent Features
columns = data1.columns.tolist()
#filter the columns to remove data we dont want
columns = [c for c in columns if c not in ["Class"]]
#Store the variable we are predicting
target = "Class"
#define a random state
state = np.random.RandomState(42)
X = data1[columns] #This is for the SVM
Y = data1[target] #This is for the SMV
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1])
print(X.shape)
print(Y.shape)

#creating testing and training data set. = for the NB and Decision Tree model it better to use the full data set
X1 = data[columns]
Y1 = data[target]
X1_outliers = state.uniform(low=0, high=1, size=(X1.shape[0], X1.shape[1])
print(X1.shape)
print(Y1.shape)

#END DATA PREPERATION

#START SVM MODEL
#define the outlier detection methods
classifiers = {
    
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), contamination=outlier_fraction,random_state=state, verbose=0),
    
    "Lotacl Outlier Fractor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=outlier_fraction),

    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, max_iter=-1, random_state=state)
}

type(classifiers)


n_outliers = len(Fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #fir the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
#repshape the predicition values to 0 for Valid transaction, 1 for Fraud transaction

y_pred[y_pred == 1] = 0
y_pred[Y-pred == -1] = 1
n_errors = (y_pred != Y).sum()
#run classification metrics
print("{}: {}".format(clf_name,n_errors))
print("Accuracy Score :")
print(accuracy_score(Y,y_pred))
print("Classification Report :")
print(classification_report(Y,y_pred))

#Evaluation SVM
##Confussion Matrix
confusion_matrix(Y, y_pred)
#AUCROC Curve
fpr, tpr, thresholds = metrics.roc_curve(Y, y_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for SMV Fraud Classification')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Possitive Rate (Sensitivity)')
plt.grid(True)
plt.show()

#END SVM MODEL

#START NAIVE BAYES MODEL

#Crating the train and test populations 33% in testing data set. for Naive Bayes and Decision Tree
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = .33, random_state = 17)

#NB1 BernoulliNB
BernNB = BernoulliNB(binarize = 0.025) # use either 0.025 0.1 or True
BernNB.fit(X1_train, Y1_train)
print(BernNB)

Y1_expect = Y1_test
Y1_pred = BernNB.predict(X1_test)
print(accuracy_score(Y1_expect, Y1_pred))

#BernNB Evalutation
#Confusion Matrix
confusion_matrix(Y1_expect, Y1_pred)
#AUCROC Curve
fpr, tpr, thresholds = metrics.roc_curve(Y1_expect, Y1_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Beroulli Naives bays Fraud Classification')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Possitive Rate (Sensitivity)')
plt.grid(True)
plt.show()

#NB2 GaussianNB
GausNB = GaussianNB()
GausNB.fit(X1_train, Y1_train)
print(GausNB)

Y1_pred = GausNB.predict(X1_test)
print(accuracy_score(Y1_expect, Y1_pred))

#GausNB Evaluation
#Confusion Maxtrix
confusion_matrix(Y1_expect, Y1_pred)
#AUCROC Curve
fpr, tpr, thresholds = metrics.roc_curve(Y1_expect, Y1_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Gaussian Naive Bayes Fraud Classification')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Possitive Rate (Sensitivity)')
plt.grid(True)
plt.show()

#END NAIVE BAYES MODEL

#START DECISON TREE
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = .33)
clf = tree.DecisionTreeClassifier()
clf_train = clf.fit(X1_train, Y1_train)

#Create decision tree
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(X1.columns.values), rounded=True, filled=True)
print(dot_data)

#accuracy prediction
Y1_pred = clf.predict(X1_test)
print("Accuracy:",metrics.accuracy_score(Y1_test, Y1_pred))

#export Tree
graph = pydotplus.graph_from_dot_data(dot_data)
#show graph
Image(graph.create_png())

#Decision Tree Evaluation
#Confusion Matrix
confusion_matrix(Y1_test, Y1_pred)
#AUCROC Curve
fpr, tpr, thresholds = metrics.roc_curve(YY1_test, Y1_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Decision Tree Fraud Classification')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Possitive Rate (Sensitivity)')
plt.grid(True)
plt.show()

#END DECISION TREE



