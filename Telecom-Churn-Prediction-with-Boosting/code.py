# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 
df=pd.read_csv(path)

# Code starts here
X=df.iloc[:,1:len(df.columns)-1]
y=df.loc[:,'Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)



# --------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train['TotalCharges']=X_train['TotalCharges'].replace(' ',np.NaN)
X_test['TotalCharges']=X_test['TotalCharges'].replace(' ',np.NaN)
X_train['TotalCharges']=pd.to_numeric(X_train['TotalCharges'])
X_test['TotalCharges']=pd.to_numeric(X_test['TotalCharges'])
X_train['TotalCharges']=X_train['TotalCharges'].fillna((X_train['TotalCharges'].mean()))
X_test['TotalCharges']=X_test['TotalCharges'].fillna((X_test['TotalCharges'].mean()))
#print(X_train.isnull().sum())
cat_cols=list(set(X_train.columns)-set(X_train._get_numeric_data().columns))
le=LabelEncoder()
for col in cat_cols:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

y_train.replace({'No':0, 'Yes':1},inplace=True)
y_test.replace({'No':0, 'Yes':1},inplace=True)

    



# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
#print("--"*25,"X_train")
#print(X_train)
#print("--"*25,"X_test")
#print(X_test)
#print("--"*25,"y_train")
#print(y_train)
#print("--"*25,"y_test")
#print(y_test)
ada_model=AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred=ada_model.predict(X_test)
ada_score=accuracy_score(y_test,y_pred)
ada_cm=confusion_matrix(y_test,y_pred)
ada_cr=classification_report(y_test,y_pred)
print("Accuracy score :",ada_score)
print("Cdafusion Matrix :",ada_cm)
print("Classification Report :",ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_clf=XGBClassifier(random_state=0)
xgb_clf.fit(X_train,y_train)
y_pred=xgb_clf.predict(X_test)
xgb_score=accuracy_score(y_test,y_pred)
xgb_cm=confusion_matrix(y_test,y_pred)
xgb_cr=classification_report(y_test,y_pred)
print('Accuracy Score',xgb_score)
print('XGBclassifier confusion matrix',xgb_cm)
print('XGBClassifier classification report',xgb_cr)
#Grid Search CV
clf_model=GridSearchCV(estimator=xgb_clf,param_grid=parameters)
clf_model.fit(X_train,y_train)
y_pred=clf_model.predict(X_test)
clf_score=accuracy_score(y_test,y_pred)
clf_cm=confusion_matrix(y_test,y_pred)
clf_cr=classification_report(y_test,y_pred)
print("Accuracy Score of GridsearchCv",clf_score)
print('Confusion matrix for Gridsearchcv',clf_cm)
print('Classifcation report for Gridsearchcv',clf_cr)


