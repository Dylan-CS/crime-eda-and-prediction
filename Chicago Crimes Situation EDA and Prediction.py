#!/usr/bin/env python
# coding: utf-8

# <p style = "font-size : 42px; color : #393e46 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #00adb5; border-radius: 5px 5px;"><strong>Chicago Crimes Situation EDA and Prediction</strong></p>

# <img style="margin-left: 10%; float: center;border:5px solid #ffb037;width:80%; max-height: 80%;" src="https://www.gannett-cdn.com/-mm-/8feaaab14d7b65ae58708871aa2b081b20258548/c=0-48-4034-2324/local/-/media/USATODAY/test/2013/12/14//1387050313000-XXX-chicago-murder-capital09.JPG?width=1024&height=643">
# 

# <a id = '0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Table of Contents</strong></p> 
# 
# * [EDA](#2.0)
#     * [1. How does crime types vary over time and location?](#2.1)
#     * [2. Whether the incident was domestic-related as defined by the Illinois Domestic Violence Act?](#2.2)
#     * [3. Weekly distribution of crime incidents.](#2.3)
#     * [4. Analysis of crime situation related to Location type.](#2.4)
#     
# * [Data Pre Processing](#3.0)
# * [Model Building](#4.0)
#     * [Logistic Regression](#4.1)
#     * [Knn](#4.2)
#     * [Decision Tree Classifier](#4.3)
#     * [Random Forest Classifier](#4.4)
#     * [Ada Boost Classifier](#4.5)
#     * [Gradient Boosting Classifier](#4.6)
#     * [XgBoost](#4.7)
#     * [Cat Boost Classifier](#4.8)
#     * [Extra Trees Classifier](#4.9)
#     * [LGBM Classifier](#4.10)
# 
# * [Models Comparison](#5.0)
# 

# ### Note: Trainning for these 10 models may take more than 5 hours;
# **Configuration:  
# RAM: 16GB  
# CPU: i7-8750H CPU @2.2GHZ  
# Graphics card: GTX1060(desktop)**

# In[39]:


# pip install missingno
# pip install catboost
# pip install lightgbm
# pip install plotly


# In[40]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

import folium
from folium.plugins import HeatMap
import plotly.express as px

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 32)


# In[41]:


# reading data
df = pd.read_csv('Chicago_Crimes_DataSet/Chicago_Crimes_2012_to_2017.csv')
df.head()


# In[42]:


df.describe()


# In[43]:


df.info()


# In[44]:


# checking for null values 

null = pd.DataFrame({'Null Values' : df.isna().sum(), 'Percentage Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})
null


# In[45]:


# filling null values with zero

df.fillna(0, inplace = True)


# In[46]:


# visualizing null values
msno.bar(df)
plt.show()


# In[47]:


# # adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time
# filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
# df[filter]

# df = df[~filter]
df


# <a id = '2.0'></a>
# <p style = "font-size : 40px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Exploratory Data Analysis (EDA)</strong></p> 

# <a id = '2.1'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>1.How does crime types vary over time and location?</strong></p> 

# In[48]:


# Group by crime type and year to count occurrences
grouped = df.groupby(['Year','Primary Type']).size()

# Calculate the percentage of each crime type for each year
grouped_pct = grouped.groupby(level=1).apply(lambda x: 100 * x / float(x.sum()))

# Plot a stacked bar chart of the percentages
fig, ax = plt.subplots(figsize=(20, 10))
grouped_pct.unstack().plot(kind='bar', stacked=True, ax=ax)

# Set chart attributes
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of Crimes')
ax.set_title('Crime Types Over the Years')

# Show the chart
plt.show()


# In[49]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

df1 = df

# Load GeoJSON file
chicago = gpd.read_file('Chicago_Crimes_DataSet/chicago_community_areas.geojson')

# Match area_numbe in GeoJSON with Community Area in CSV
df1['area_numbe'] = df['Community Area'].astype(int).astype(str)
df1 = df.merge(chicago, on='area_numbe')

# Calculate total number of crime incidents for each Community Area
crime_count = df.groupby('area_numbe').size().reset_index(name='crime_count')

# Merge results with GeoJSON and display on map
chicago = chicago.merge(crime_count, on='area_numbe')
ax = chicago.plot(column='crime_count', cmap='OrRd', legend=True, figsize=(15, 15))

# Add labels for each area
for idx, row in chicago.iterrows():
    plt.annotate(text=row['crime_count'], xy=row['geometry'].centroid.coords[0], horizontalalignment='center')

# Show map
plt.axis('off')
plt.show()


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>Community No. 25, which is the Austin community, has the highest crime rate, try to avoid lightning.</strong></p> 

# <a id = '2.2'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>2.whether the incident was domestic-related as defined by the Illinois Domestic Violence Act</strong></p> 

# In[50]:


import pandas as pd
import geopandas as gpd

# Load dataset
df2 = df

# Match area_numbe in GeoJSON with Community Area in CSV
chicago = gpd.read_file('Chicago_Crimes_DataSet/chicago_community_areas.geojson')
df2['area_numbe'] = df['Community Area'].astype(int).astype(str)
df2 = df2.merge(chicago, on='area_numbe')

# Calculate total number of crime incidents and domestic-related incidents for each Community Area
crime_count = df2.groupby('area_numbe').size().reset_index(name='crime_count')
domestic_count = df2.loc[df['Domestic'] == True].groupby('area_numbe').size().reset_index(name='domestic_count')

# Merge results with GeoJSON and calculate domestic crime ratio
chicago = chicago.merge(crime_count, on='area_numbe')
chicago = chicago.merge(domestic_count, on='area_numbe', how='left').fillna(0)
chicago['domestic_ratio'] = chicago['domestic_count'] / chicago['crime_count'] * 100

# Sort and print top three areas
top_3 = chicago[['community', 'domestic_ratio']].sort_values('domestic_ratio', ascending=False).head(3)
print(top_3)

# Display on map
ax = chicago.plot(column='domestic_ratio', cmap='OrRd', legend=True, figsize=(15, 15))
ax.set_title("Percentage of Domestic Related Crimes by Community Area in Chicago")
ax.set_axis_off()

for idx, row in chicago.iterrows():
    ax.annotate(text=str(int(row['domestic_ratio'])) + '%', xy=row['geometry'].centroid.coords[0], 
                horizontalalignment='center', verticalalignment='center')


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>Based on the analysis of the crime data in Chicago, the community areas with the highest percentage of domestic-related crimes are Forest Glen, Gage Park, and Irving Park, with domestic ratios of 18.09%, 17.85%, and 17.44% respectively. These findings suggest that domestic violence is a significant issue in these areas and that policymakers and law enforcement agencies may need to implement targeted interventions to address this problem. Further research may be needed to investigate the underlying factors contributing to the high rates of domestic violence in these communities and to develop effective strategies for prevention and intervention.</strong></p> 

# <a id = '2.3'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>3.Weekly distribution of crime incidents</strong></p> 

# In[51]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df3 = df

# Convert the Date column to datetime format
df3['Date'] = pd.to_datetime(df3['Date'])

# Extract the day of the week from the Date column
df3['DayOfWeek'] = df3['Date'].dt.day_name()

# Group the crimes by day of the week and count the number of crimes on each day
crimes_by_day = df3.groupby('DayOfWeek')['ID'].count()

# Set the figure size
fig, ax = plt.subplots(figsize=(10, 10))

# Plot a pie chart of the distribution
ax.pie(crimes_by_day, labels=crimes_by_day.index, autopct='%1.1f%%')
ax.set_title('Distribution of Crimes by Day of the Week')

# Show the plot
plt.show()

# Print the percentage of crimes on each day
print('Crimes by Day of the Week:\n', crimes_by_day)


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>The distribution of crimes by day of the week shows that the number of crimes committed is relatively evenly spread throughout the week. The highest number of crimes occur on Saturdays, followed closely by Fridays and Wednesdays. Sundays have the lowest number of crimes reported. However, the differences in the number of crimes committed on each day of the week are relatively small. The results suggest that there is no significant correlation between the day of the week and the occurrence of crimes. Further analysis is needed to investigate the potential factors that contribute to the variations in the number of crimes reported on different days of the week.</strong></p> 

# <a id = '2.4'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>4.Analysis of crime situation related to Location type.</strong></p> 

# In[14]:


import pandas as pd
import plotly.express as px

# Load the dataset
df4 = pd.read_csv('Chicago_Crimes_DataSet/Chicago_Crimes_2012_to_2017.csv')

# Count the number of crimes by location description
crime_counts = df4['Location Description'].value_counts()

# Create a dataframe with the location descriptions and counts
crime_df = pd.DataFrame({'Location Description': crime_counts.index, 'Count': crime_counts.values})

# Calculate the percentage of crimes in each location
crime_df['Percentage'] = crime_df['Count'] / sum(crime_df['Count']) * 100

# Create the Treemap figure
fig = px.treemap(crime_df, path=['Location Description'], values='Count', color='Percentage',height=700,
                 color_continuous_scale='spectral', labels={'Count': 'Number of Crimes', 'Percentage': 'Percentage of Total Crimes'})

# Update the text position and font size
fig.update_traces(textposition='middle center', textfont=dict(size=10), texttemplate='%{value}<br>%{label}')

# Show the figure
fig.show()


# In[15]:


# Sort the dataframe by percentage in descending order
crime_df = crime_df.sort_values(by='Percentage', ascending=False)

# Select only the top five rows
top_five = crime_df[['Location Description', 'Percentage']].head(5)

# Print the top five locations by percentage
print(top_five.to_string(index=False))


# <li style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>The table shows the top five location descriptions where crimes occur in the dataset. The "Street" category has the highest percentage of crimes, accounting for 22.7% of all crimes. "Residence" and "Apartment" are the second and third most common locations for crimes, with percentages of 16.0% and 12.7%, respectively. "Sidewalk" ranks fourth, accounting for 11.1% of crimes. Finally, "Other" is the fifth most common location for crimes, with a percentage of 3.8%. These results suggest that crimes in this dataset are most likely to occur in outdoor public spaces such as streets and sidewalks, followed by residential areas.</strong></li>

# <a id = '3.0'></a>
# <p style = "font-size : 40px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Data Pre Processing</strong></p> 

# In[16]:


df = pd.read_csv('Chicago_Crimes_DataSet/Chicago_Crimes_2012_to_2017.csv')
null = pd.DataFrame({'Null Values' : df.isna().sum(), 'Percentage Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})
df.fillna(0, inplace = True)
# Merge Crime Dataset with Economy DataSet

# get the economy data
poverty = pd.read_excel('Chicago_Economy_DataSet/poverty.xlsx')
median_house_income = pd.read_excel('Chicago_Economy_DataSet/median_house_income.xlsx')
black = pd.read_excel('Chicago_Economy_DataSet/black.xlsx')
naturecitizenship = pd.read_excel('Chicago_Economy_DataSet/citizenship.xlsx')
education = pd.read_excel('Chicago_Economy_DataSet/education.xlsx')
health_insurance = pd.read_excel('Chicago_Economy_DataSet/health_insurance.xlsx')
hispanic_latino = pd.read_excel('Chicago_Economy_DataSet/hispanic_latino.xlsx')
noncitizenship = pd.read_excel('Chicago_Economy_DataSet/noncitizenship.xlsx')
owner_occupied = pd.read_excel('Chicago_Economy_DataSet/owner_occupied.xlsx')

merged = pd.merge(df, poverty, on='Community Area')
merged1 = pd.merge(df, median_house_income, on='Community Area')
merged2 = pd.merge(df, black, on='Community Area')
merged3 = pd.merge(df, naturecitizenship, on='Community Area')
merged4 = pd.merge(df, health_insurance, on='Community Area')
merged5 = pd.merge(df, hispanic_latino, on='Community Area')
merged6 = pd.merge(df, noncitizenship, on='Community Area')
merged7 = pd.merge(df, owner_occupied, on='Community Area')
merged8 = pd.merge(df, education, on='Community Area')

df1 = merged
df1['poverty'] = merged.lookup(merged.index,merged['Year'])
df1['median_house_income'] = merged1.lookup(merged1.index,merged1['Year'])
df1['black'] = merged2.lookup(merged2.index,merged2['Year'])
df1['naturecitizenship'] = merged3.lookup(merged3.index,merged3['Year'])
df1['health_insurance'] = merged4.lookup(merged4.index,merged4['Year'])
df1['hispanic_latino'] = merged5.lookup(merged5.index,merged5['Year'])
df1['noncitizenship'] = merged6.lookup(merged6.index,merged6['Year'])
df1['owner_occupied'] = merged7.lookup(merged7.index,merged7['Year'])
df1['education'] = merged8.lookup(merged8.index,merged8['Year'])

df1.to_csv('test.csv')
data = pd.read_csv('test.csv')
data = data.drop(columns =['2012','2013','2014','2015','2016','2017','IUCR',"Description"],axis=1)


# In[17]:


from sklearn.preprocessing import LabelEncoder

object_features = ["Case Number", "Date", "Block", "Primary Type", 
                    "Location Description", "FBI Code",
                    "Updated On", "Location"]
for feature in object_features:
    data[feature]=LabelEncoder().fit_transform(data[feature])
    
data.head()


# In[18]:


plt.figure(figsize = (36, 20))

corr = data.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()


# In[19]:


correlation = data.corr()['Primary Type'].abs().sort_values(ascending = False)
correlation


# In[20]:


# dropping columns that are not useful(which is low than 0.01)

useless_col = ['owner_occupied','Case Number','Unnamed: 0.1', 'Date', 'Unnamed: 0', 'Longitude','X Coordinate', 'ID','Latitude']

data.drop(useless_col, axis = 1, inplace = True)


# In[21]:


data.var()


# In[22]:


# normalizing numerical variables

from sklearn.preprocessing import MinMaxScaler

# create a scaler object
scaler = MinMaxScaler()

# select the columns to be scaled
cols_to_scale = ['Block', 'Location Description', 'Beat', 'District', 'Ward', 
                 'Community Area', 'FBI Code', 'Y Coordinate','Updated On', 
                 'Year', 'Location','poverty',	'median_house_income',	'black',
                 'naturecitizenship'	,'health_insurance'	,'hispanic_latino',
                 'noncitizenship',	'education']

for feature in cols_to_scale:
    data[feature] = data[feature].replace(to_replace="NaN", value=np.NaN)
    data[feature] = data[feature].fillna(data[feature].median(skipna=True))

# fit and transform the selected columns using the scaler object
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

# print the normalized dataframe
data


# In[23]:


data.var()


# In[24]:


y = data ['Primary Type']
X = data.drop(['Primary Type'], axis=1)
X.shape,y.shape


# In[25]:


# splitting data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# <a id = '4.0'></a>
# <p style = "font-size : 45px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Model Building</strong></p> 

# <a id = '4.1'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Logistic Regression</strong></p> 

# In[26]:


# Fit the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
conf = confusion_matrix(y_test, y_pred_lr)
clf_report = classification_report(y_test, y_pred_lr)

print(f"Accuracy Score of Logistic Regression is : {acc_lr}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.2'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>KNN</strong></p> 

# In[27]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
conf = confusion_matrix(y_test, y_pred_knn)
clf_report = classification_report(y_test, y_pred_knn)

print(f"Accuracy Score of KNN is : {acc_knn}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.3'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Decision Tree Classifier</strong></p> 

# In[28]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

acc_dtc = accuracy_score(y_test, y_pred_dtc)
conf = confusion_matrix(y_test, y_pred_dtc)
clf_report = classification_report(y_test, y_pred_dtc)

print(f"Accuracy Score of Decision Tree is : {acc_dtc}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.4'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Random Forest Classifier</strong></p> 

# In[29]:


rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

y_pred_rd_clf = rd_clf.predict(X_test)

acc_rd_clf = accuracy_score(y_test, y_pred_rd_clf)
conf = confusion_matrix(y_test, y_pred_rd_clf)
clf_report = classification_report(y_test, y_pred_rd_clf)

print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.5'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Ada Boost Classifier</strong></p> 

# In[30]:


ada = AdaBoostClassifier(base_estimator = dtc)
ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)

acc_ada = accuracy_score(y_test, y_pred_ada)
conf = confusion_matrix(y_test, y_pred_ada)
clf_report = classification_report(y_test, y_pred_ada)

print(f"Accuracy Score of Ada Boost Classifier is : {acc_ada}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.6'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Gradient Boosting Classifier</strong></p> 

# In[31]:


gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

acc_gb = accuracy_score(y_test, y_pred_gb)
conf = confusion_matrix(y_test, y_pred_gb)
clf_report = classification_report(y_test, y_pred_gb)

print(f"Accuracy Score of Ada Boost Classifier is : {acc_gb}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.7'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>XgBoost Classifier</strong></p> 

# In[32]:


xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
conf = confusion_matrix(y_test, y_pred_xgb)
clf_report = classification_report(y_test, y_pred_xgb)

print(f"Accuracy Score of Ada Boost Classifier is : {acc_xgb}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.8'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Cat Boost Classifier</strong></p> 

# In[33]:


cat = CatBoostClassifier(iterations=100)
cat.fit(X_train, y_train)

y_pred_cat = cat.predict(X_test)

acc_cat = accuracy_score(y_test, y_pred_cat)
conf = confusion_matrix(y_test, y_pred_cat)
clf_report = classification_report(y_test, y_pred_cat)


# In[34]:


print(f"Accuracy Score of Ada Boost Classifier is : {acc_cat}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.9'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Extra Trees Classifier</strong></p> 

# In[35]:


etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

y_pred_etc = etc.predict(X_test)

acc_etc = accuracy_score(y_test, y_pred_etc)
conf = confusion_matrix(y_test, y_pred_etc)
clf_report = classification_report(y_test, y_pred_etc)

print(f"Accuracy Score of Ada Boost Classifier is : {acc_etc}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.10'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>LGBM Classifier</strong></p> 

# In[36]:


lgbm = LGBMClassifier(learning_rate = 1)
lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm.predict(X_test)

acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
conf = confusion_matrix(y_test, y_pred_lgbm)
clf_report = classification_report(y_test, y_pred_lgbm)

print(f"Accuracy Score of Ada Boost Classifier is : {acc_lgbm}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '5.0'></a>
# <p style = "font-size : 45px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Models Comparison</strong></p> 

# In[37]:


models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier',
             'Gradient Boosting Classifier', 'XgBoost', 'Cat Boost', 'Extra Trees Classifier', 'LGBM'],
    'Score' : [acc_lr, acc_knn, acc_dtc, acc_rd_clf, acc_ada, acc_gb, acc_xgb, acc_cat, acc_etc, acc_lgbm]
})


models.sort_values(by = 'Score', ascending = False)


# In[38]:


px.bar(data_frame = models.sort_values(by = 'Score', ascending = False), 
       x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', title = 'Models Comparison')


# <p style = "font-size : 30px; color : #03506f ; font-family : 'Comic Sans MS'; "><strong>We got accuracy score of 97% which is quite impresive.</strong></p> 

# In[ ]:




