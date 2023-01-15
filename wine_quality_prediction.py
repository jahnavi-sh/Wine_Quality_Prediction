#understanding the problem statement - 
#we will predict the quality of wine on the basis of given features.  

#volatile acidity - gaseous acids present in wine
#fixed acidity - primary fixed acids found in wine are tartaric, succinic, citric and malic
#residual sugar - amount of sugar left after fermentation 
#citric acid - it is weak organic acid, found in citric fruit naturally 
#chlorides - amount of salt present in wine
#total sulphur dioxide - it is used for prevention of spoilage of wine by oxidation and micro-organisms
#pH - it is used for checking acidity
#density 
#sulphates - added sulfites preserve freshness and protect wine from oxidation, and bacteria 
#alcohol - percentage of wine present in wine 
#quality - score between 0 to 10 

#our task is to build a machine learning algorithm to automate  the process of wine quality prediction, machine learning saves 
#both resources and time for winemaking businesses

#workflow 
#1. wine data 
#2. data analysis - check which features are relevant 
#3. data preprocessing 
#4. train test split 
#5. model used - random forest model
#6. model evaluation 


#load libraries 

#linear algebra 
import numpy as np          #for working with arrays 

#data preprocessing and exploration 
import pandas as pd         #library for data handling

#data visualisation 
import matplotlib.pyplot as plt
import seaborn as sns       

#model development and evaluation. 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#load dataset into pandas dataframe 
wine_dataset = pd.read_csv(r'wine_data.csv')

#explore the data. get familiar with it 

#view first five rows 
wine_dataset.head()

#view number of rows and columns 
wine_dataset.shape
#there are 1599 rows (1600 data points) and 12 columns 

#get insight into statistical measures of the data
wine_dataset.describe()

#check missing values 
wine_dataset.isnull().sum()
#the dataset doesn't have any missing values 

#data analysis and visualisation 
#number of values for each quality 
sns.catplot(x='quality', data=wine_dataset, kind='count')

#volatile acidity vs quality 
plot = plt.figure(figsize=(5,5))
sns.barplot (x='quality', y='volatile acidity', data = wine_dataset)
#it can be observed that if the volatile acidity is more, then quality of wine is less and vice versa. 
#therefore, from the barplot, it can conclude that volatile acidity and quality of wine are inversely proportional

#citric acid vs quality 
plot = plt.figure(figsize=(5,5))
sns.barplot (x='quality', y='citric acid', data = wine_dataset)
#it can be observed that if the citric acid value is high, the wine quality is high and vice versa.
#therefore, from the barplot, it can be concluded that citric acid value and wine quality are directly proportional.  

#finding correlation 
correlation = wine_dataset.corr()
#constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.if', annot=True, annot_kws={'size':0}, cmap='Blues')

#data preprocessing 
#separate data and label 
X = wine_dataset.drop('Quality', axis=1)
#label binarization 
Y = wine_dataset['Quality'].apply(lambda y_values: 1 if y_values>=7 else 0)

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

#train model 
#random forest classifier 
model = RandomForestClassifier()
model.fit(X_train, Y_train)

#evaluate model 
#accuracy score 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print ("test data accuracy", test_data_accuracy)
#the accuracy score is 92.5% 