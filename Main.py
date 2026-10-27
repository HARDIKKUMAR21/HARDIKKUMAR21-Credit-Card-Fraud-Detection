"Importing important Libraries"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

"Loading Dataset using Pandas"
df=pd.read_csv('/content/drive/MyDrive/Files for Data Analysis/creditcard.csv')

"Exploratory Data Analysis"
# Using ProfileReport for better Understanding of the Data

from ydata_profiling import ProfileReport
profile=ProfileReport(df,title='EDA Report',explorative=True)
profile.to_file('EDA2.html')

df.info()

# Checking the number of fraud transactions and legit transactions
df['Class'].value_counts()

'''This Data is highly Unbalanced

0--> Legit transaction

1--> Fraud transaction'''

# Seperating the data for analysis

legit=df[df['Class']==0]
fraud=df[df['Class']==1]

'''Under-Sampling

Build a Sample Dataset Containing similar distribution for normal transaction and and Fraudulent transaction

Number of Fraudulent transaction->492

'''

legit_sample=legit.sample(n=492)

# Concatnating Both the DataFrame
df2=pd.concat([legit_sample,fraud],axis=0)

legit2=df2[df2['Class']==0]
fraud2=df2[df2['Class']==1]

# Splitting the data into independent and dependent features
x=df2.iloc[:,:-1]
y=df2.iloc[:,-1]

# Splitting the data into training and testing set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=42)

"Model Training"
model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

"Model Evaluation"
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

"Saving the model"

# To Save the Model
pickle.dump(model,open('model.pkl','wb'))

#To Load the model
rand_f=pickle.load(open('model.pkl','rb'))