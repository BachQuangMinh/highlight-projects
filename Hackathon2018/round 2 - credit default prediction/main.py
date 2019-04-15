import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os.path
CEED=10

# preprocessing steps, selecting and calculating necessary columns and transform the data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(BASE_DIR, 'data/train.csv')
df = pd.read_csv(filename)

# # take a first look at the data 
# print(df.describe())
# print(df.std())

# correlation checking
import seaborn as sns
# compute the correlation matrix
g = sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")

# prepare the data
df=df.drop([169],0) # this row contain an unusually low 'Accounts & Notes Receivable' value

# based on the observation of the confusion matrix in the report, we choose the following features
select_feature = ['Gross Profit','Revenue','Pretax Income','Total Assets','Total Liabilities',
                  'Total Current Assets','Total Current Liabilities','Inventories',
                  'Interest Expense','Short-Term Borrowings','Cash & Near Cash Items',
                  'Accounts & Notes Receivable']

X=df[select_feature]
y=df['default']



#Find selection feature 
import itertools
results = []
max_acu = -1
good_ft = None
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=CEED)
resu = pd.DataFrame(columns=['list', 'true'])
for j in range(5,7):
    for i in itertools.combinations(select_feature, j):
        #Train 
        XX_train = X_train[list(i)]
        clf_0 = LogisticRegression(class_weight='balanced').fit(XX_train, y_train)
        
        XX_test = X_test[list(i)]
        pred_test = clf_0.predict(XX_test)
        
        cnf_matrix = confusion_matrix(y_test, pred_test)
        
        results.append([list(i),cnf_matrix[1][1]])
        resu = resu.append({'list':list(i), 'true':cnf_matrix[1][1]}, ignore_index=True)
        
        point = float(cnf_matrix[1][1]/(cnf_matrix[1][0]+cnf_matrix[1][1]))
        print(point, max_acu)
        
        if point > max_acu:
            max_acu = point
            good_ft = i

print(max_acu)
print(good_ft)
'''
0.9
('Gross Profit', 'Revenue', 'Pretax Income', 'Total Assets', 'Total Current Assets')
'''

filename = os.path.join(BASE_DIR, 'data/train.csv')        
dataset = pd.read_csv(filename)
filename = os.path.join(BASE_DIR, 'data/test.csv')
predict = pd.read_csv(filename)

select_feature = ['Gross Profit', 'Revenue', 'Pretax Income', 'Total Assets',
                  'Total Current Assets']

feature_train = dataset[select_feature]
target_train  = dataset['default']

feature_pred = predict[select_feature]
target_pred  = None


clf_final = LogisticRegression(class_weight='balanced').fit(feature_train, target_train)

target_pred = clf_final.predict(feature_pred)

predict['default'] = target_pred 

predict.to_csv('./output/output_test.csv')




