print("............................................................................................")
print("Extracting Data (Bachelor Of Science(Forensic) )") #data extraction
import pandas as pd
datasetC=pd.read_csv("project.csv")
options = ['Bachelor of Science (Forensic Science)'] 
dataset = datasetC[datasetC['MHRDName'].isin(options)] 
print("Size of extracted dataset ",dataset.shape)# getting basic information from data
print("\n\n");  


##################################################################################################
#############################################preprocessing########################################

print("\n\n Preprocessing.......")
print(dataset.describe())
print(dataset.iloc[:10  ,:])
print("Number Of Missing  Data In Every Feature ")
import numpy as np
print(pd.isnull(dataset).sum()) # getting null values

# removing rows with less than 30 % information (features)
print("\n\n As Dataset is Comparatively small  dropping only those tuples which has less than 30 % information  ")
dataset.dropna(thresh=dataset.shape[1]*.30,inplace=True,axis=0)

# imputing null values 
print("\n Imputing values in others ")
from sklearn.impute import SimpleImputer 
si=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
dataset=pd.DataFrame(si.fit_transform(dataset))

# data is now compelete
print("\n\n Number Of Missing  Data In Every Feature ")
import numpy as np
print(pd.isnull(dataset).sum()) 

# Label encoding data with nunique values <=30 (Categorical)
print('\n\n Label Encoding Non Continuous Data :')
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder();
for i in range(dataset.shape[1]):
    if(dataset.iloc[:,i].nunique()<=30):
        print(i)
        dataset.iloc[:,i]=lb.fit_transform( dataset.iloc[:,i])
        

# dropping data with only 1 data like every student is from forencis science
print('\n\n Removing all single valued columns.....')
column=[]
for i in range(dataset.shape[1]):
    if(dataset.iloc[:,i].nunique()==1):
        print(i)
        column.append(i)
        
dataset.drop(columns=column,inplace=True,axis=1)


# scalling remain values
print("\n\n Standard Scalling Dataset...... Except Categorical Values")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
for i in range(dataset.shape[1]):
    if(dataset.iloc[:,i].nunique()>30):
        d=dataset.iloc[:,i].values.reshape(-1,1)
        print(d.shape)
        d=sc.fit_transform(d)
        dataset.iloc[:,i]=d.reshape(366)


# heat map for r2 score very close to zero
print("pruning useless features using corellation matrix")
from matplotlib import pyplot as plt
import seaborn as sns
cm=np.corrcoef(dataset.values.T)
sns.heatmap(cm,annot=True)


for i in range(dataset.shape[1]):
    print(dataset.iloc[:,i].nunique())

    

columns=[19,17,16,15,13] # results fro  corellation matrix
datafinal=dataset.drop(columns=columns,inplace=False,axis=1) #dropping above selected values
for i in range(datafinal.shape[1]):
    print(datafinal.iloc[:,i].nunique())

# seperating data from target
print("\n\n Seperating Target...........")
target=datafinal.iloc[:,3]
datafinal.drop(columns=[3],inplace=True,axis=1) #removing target (grades of students)

for i in range(datafinal.shape[1]):
    print(datafinal.iloc[:,i].nunique())

print(target.shape)# getting final dataset shape
print(datafinal.shape)

###########################################################################################################################
########################################### Classification ##################################################################


print("\n\n Selecting best classifier for data")
print("\n\n Naive bayes")

label=[]
accuracy=[]# used at last for comparing diffeerent algorithms

#splitting data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(datafinal,target,test_size=0.2,random_state=4)
print(xtrain.shape)
print(ytrain.shape)

# starting with Naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
label.append('Naive Bayes')
nb.fit(xtrain,ytrain)
npPredict=nb.predict(xtest)

from sklearn.metrics import confusion_matrix, accuracy_score # testing how it worked
print(confusion_matrix(npPredict,ytest))
print(accuracy_score(npPredict,ytest))# accuracy score ..
accuracy.append(accuracy_score(npPredict,ytest))

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
Dtc=DecisionTreeClassifier(criterion='entropy',max_depth=50)
Dtc.fit(xtrain,ytrain)
DtcPredict=Dtc.predict(xtest)
label.append('Decision Tree')


print(confusion_matrix(DtcPredict,ytest))
print(accuracy_score(DtcPredict,ytest))
accuracy.append(accuracy_score(DtcPredict,ytest))
#observations hyper paremeters----- gini (49) ,max depth -100->(43) ,max depth(70)->44 ,max depth(49), using entropy ->( 50)


# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier( p=[1,2],metric='manhattan')
knn.fit(xtrain,ytrain)
knnPredict=knn.predict(xtest)
label.append('Knn')

#result 
print(confusion_matrix(knnPredict,ytest))
print(accuracy_score(knnPredict,ytest))
accuracy.append(accuracy_score(knnPredict,ytest))

# hyper parameter tuning for knn minkowski < manhattan distance

# SVM
from sklearn.svm import SVC
svm= SVC(decision_function_shape='ovo')
svm.fit(xtrain,ytrain)
svmPredict=svm.predict(xtest)
label.append('SVM')

#result
print(confusion_matrix(svmPredict,ytest))
print(accuracy_score(svmPredict,ytest))
accuracy.append(accuracy_score(svmPredict,ytest))

# hyperparameter tuning result using decision_function_shape='ovo' because of multiclass classification

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lg= LogisticRegression(multi_class='auto',solver='lbfgs')
lg.fit(xtrain,ytrain)
lgPredict=lg.predict(xtest)
label.append('Log. Regression')



print(confusion_matrix(lgPredict,ytest))
print(accuracy_score(lgPredict,ytest))#result
accuracy.append(accuracy_score(lgPredict,ytest))
#hyper parameter tuning -solver->warn << solver->auto


#voting classifier
from sklearn.ensemble import VotingClassifier
vc=VotingClassifier(estimators=[('GaussianNB',nb),('DecisionTree',Dtc),('LogisticRegression',lg)], voting='hard')
vc.fit(xtrain,ytrain)
vcPredict=vc.predict(xtest)
label.append('Vot. Classifier')

print(confusion_matrix(vcPredict,ytest)) # result
print(accuracy_score(vcPredict,ytest))
accuracy.append(accuracy_score(vcPredict,ytest))

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier
bg=BaggingClassifier(n_estimators=12,base_estimator=Dtc,random_state=25)
bg.fit(xtrain,ytrain)
bgPredict=bg.predict(xtest)
label.append('Bagging')

print(confusion_matrix(bgPredict,ytest))
print(accuracy_score(bgPredict,ytest))# result
accuracy.append(accuracy_score(bgPredict,ytest))
# n_estimators 11 gives best accuracy

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
adb=AdaBoostClassifier(base_estimator=Dtc,n_estimators=20)
adb.fit(xtrain,ytrain)
adbPredict=adb.predict(xtest)
label.append('Boosting')

print(confusion_matrix(adbPredict,ytest))
print(accuracy_score(adbPredict,ytest))#result
accuracy.append(accuracy_score(adbPredict,ytest))
# Hyper Parameter Tuning  works better when Decision Trees Are Used  and n_estimators =20


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(criterion='entropy')
rfc.fit(xtrain,ytrain)
rfcPredict=rfc.predict(xtest)
label.append('Random forest')


print(confusion_matrix(rfcPredict,ytest))
print(accuracy_score(rfcPredict,ytest))#result
accuracy.append(accuracy_score(rfcPredict,ytest))
#hyper parameters tuning works better in this case with criteration ->entropy

###############################################################################################################################
#####################################################result#############################################################################3
#plotting Accuracy of Each Classifier using bar plot
y_pos = np.arange(len(accuracy))
plt.bar(y_pos,height=accuracy)
plt.xticks(y_pos, label)
plt.show()

