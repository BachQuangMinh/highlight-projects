# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:14:05 2018

@author: DELL
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import itertools
#confusion matrix plot function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(9*1.5,3*1.5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#set work dir
dir='C:\\Users\\DELL\\Google Drive\\JVN couse materials\\Projects\\Practice projects\\airbnb'
import os 
os.chdir(dir)

#import the data and preprocess the data
data=pd.read_excel('reviewed.xlsx')

#how the review score distribute
plt.figure(figsize=(4*1.5,3*1.5))
plt.hist(data.loc[:,'Review Scores Rating'], bins=30, density=True)
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Density')
plt.show()

Manhattan_reviewscore=data[data.loc[:,'Neighbourhood ']=='Manhattan'].loc[:,'Review Scores Rating']
Bronx_reviewscore=data[data.loc[:,'Neighbourhood ']=='Bronx'].loc[:,'Review Scores Rating']
Queens_reviewscore=data[data.loc[:,'Neighbourhood ']=='Queens'].loc[:,'Review Scores Rating']
Brooklyn_reviewscore=data[data.loc[:,'Neighbourhood ']=='Brooklyn'].loc[:,'Review Scores Rating']
StatenIsland_reviewscore=data[data.loc[:,'Neighbourhood ']=='Staten Island'].loc[:,'Review Scores Rating']

plt.figure(figsize=(9*1.5,3*1.5))
plt.boxplot([Manhattan_reviewscore,
             Brooklyn_reviewscore,
             Queens_reviewscore,
             Bronx_reviewscore,
             StatenIsland_reviewscore,
             data.loc[:,'Review Scores Rating']],vert=False)
plt.title('Review Scores Distribution Overview')
plt.xlabel('Review Scores Rating')
plt.ylabel('Neighbourhood')
plt.yticks(range(1,7), ('Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'StatenIsland','Overall'))
plt.show()

reviewscoredescribeneighbourhood=pd.DataFrame({'Manhattan':Manhattan_reviewscore.describe(),
                            'Brooklyn':Brooklyn_reviewscore.describe(),
                            'Queens':Queens_reviewscore.describe(),
                            'Bronx':Bronx_reviewscore.describe(),
                            'StatenIsland':StatenIsland_reviewscore.describe(),
                            'Overall':data.loc[:,'Review Scores Rating'].describe()})
cols = reviewscoredescribeneighbourhood.columns.tolist()
cols = ['Brooklyn', 'Bronx', 'Manhattan', 'Queens', 'StatenIsland', 'Overall']
reviewscoredescribeneighbourhood=reviewscoredescribeneighbourhood[cols]

Apartment_reviewscore=data[data.loc[:,'Property Type']=='Apartment'].loc[:,'Review Scores Rating']
House_reviewscore=data[data.loc[:,'Property Type']=='House'].loc[:,'Review Scores Rating']
Loft_reviewscore=data[data.loc[:,'Property Type']=='Loft'].loc[:,'Review Scores Rating']

a=['Apartment','House','Loft']
tempt=data
for i in a:
    Others_reviewscore=tempt[tempt.loc[:,'Property Type']!=i]
    tempt=Others_reviewscore   
Others_reviewscore=Others_reviewscore.loc[:,'Review Scores Rating']    

plt.figure(figsize=(9*1.5,3*1.5))
plt.boxplot([House_reviewscore,
             Apartment_reviewscore,
             Loft_reviewscore,
             Others_reviewscore,
             data.loc[:,'Review Scores Rating']],vert=False)
plt.title('Review Scores Distribution by Property Type')
plt.xlabel('Review Scores Rating')
plt.ylabel('Property Type')
plt.yticks(range(1,6), ('House', 'Apartment', 'Loft', 'Others','Overall'))
plt.show()

reviewdescribePropertytype=pd.DataFrame({'Apartment':Apartment_reviewscore.describe(),
                            'House':House_reviewscore.describe(),
                            'Loft':Loft_reviewscore.describe(),
                            'Others':Others_reviewscore.describe(),
                            'Overall':data.loc[:,'Review Scores Rating'].describe()})
cols = ['Apartment','House','Loft','Others','Overall']
reviewdescribePropertytype=reviewdescribePropertytype[cols]

Entire_reviewscore=data[data.loc[:,'Room Type']=='Entire home/apt'].loc[:,'Review Scores Rating']
Private_reviewscore=data[data.loc[:,'Room Type']=='Private room'].loc[:,'Review Scores Rating']
Shared_reviewscore=data[data.loc[:,'Room Type']=='Shared room'].loc[:,'Review Scores Rating']

plt.figure(figsize=(9*1.5,3*1.5))
plt.boxplot([Entire_reviewscore,
             Private_reviewscore,
             Shared_reviewscore,
             data.loc[:,'Review Scores Rating']],vert=False)
plt.title('Review Scores Distribution by Room Type')
plt.xlabel('Review Scores Rating')
plt.ylabel('Room Type')
plt.yticks(range(1,5), ('Entire home/apt', 'Private room', 'Shared room', 'Overall'))
plt.show()

reviewscoredescribeRoomtype=pd.DataFrame({'Entire home/apt':Entire_reviewscore.describe(),
                            'Private room':Private_reviewscore.describe(),
                            'Shared room':Shared_reviewscore.describe(),
                            'Overall':data.loc[:,'Review Scores Rating'].describe()})
cols = ['Entire home/apt','Private room','Shared room','Overall']
reviewscoredescribeRoomtype=reviewscoredescribeRoomtype[cols]