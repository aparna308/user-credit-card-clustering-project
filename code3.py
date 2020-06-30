# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:58:00 2020

@author: dell
"""

# =============================================================================
#  Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. 
#  The dataset uses 18 columns to define various aspects of the customer's spending habits.
#  We will develop a customer segmentation to define marketing strategy. 
#  We will use K Means Clustering to divide the customers into various groups and then use pca to reduce the 18 features to 2 features to visualize the clusters.
#  Dataset has the following columns-
#  CUSTID : Identification of Credit Card holder (Categorical)
#  BALANCE : Balance amount left in their account to make purchases (
#  BALANCEFREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
#  PURCHASES : Amount of purchases made from account
#  ONEOFFPURCHASES : Maximum purchase amount done in one-go
#  INSTALLMENTSPURCHASES : Amount of purchase done in installment
#  CASHADVANCE : Cash in advance given by the user
#  PURCHASESFREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
#  ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
#  PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
#  CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
#  CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
#  PURCHASESTRX : Numbe of purchase transactions made
#  CREDITLIMIT : Limit of Credit Card for user
#  PAYMENTS : Amount of Payment done by user
#  MINIMUM_PAYMENTS : Minimum amount of payments made by user
#  PRCFULLPAYMENT : Percent of full payment paid by user
#  TENURE : Tenure of credit card service for user
# =============================================================================
 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings(action="ignore")

data = pd.read_csv('clustring_dataset.csv')
print(data.shape)
#(8950, 18)
data.head(n=10)
# =============================================================================
#  CUST_ID      BALANCE  ...  PRC_FULL_PAYMENT  TENURE
# 0  C10001    40.900749  ...          0.000000      12
# 1  C10002  3202.467416  ...          0.222222      12
# 2  C10003  2495.148862  ...          0.000000      12
# 3  C10004  1666.670542  ...          0.000000      12
# 4  C10005   817.714335  ...          0.000000      12
# 5  C10006  1809.828751  ...          0.000000      12
# 6  C10007   627.260806  ...          1.000000      12
# 7  C10008  1823.652743  ...          0.000000      12
# 8  C10009  1014.926473  ...          0.000000      12
# 9  C10010   152.225975  ...          0.000000      12
# =============================================================================

data.describe()
# =============================================================================
#             BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT       TENURE
# count   8950.000000        8950.000000  ...       8950.000000  8950.000000
# mean    1564.474828           0.877271  ...          0.153715    11.517318
# std     2081.531879           0.236904  ...          0.292499     1.338331
# min        0.000000           0.000000  ...          0.000000     6.000000
# 25%      128.281915           0.888889  ...          0.000000    12.000000
# 50%      873.385231           1.000000  ...          0.000000    12.000000
# 75%     2054.140036           1.000000  ...          0.142857    12.000000
# max    19043.138560           1.000000  ...          1.000000    12.000000
# =============================================================================

# =============================================================================
# We should have information on variability or dispersion of the data.
# A boxplot is a graph that gives you a good indication of how the values in the data are spread out.
# A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”). 
# It can tell you about your outliers and what their values are. 
# It can also tell you if your data is symmetrical, how tightly your data is grouped, and if and how your data is skewed.
# =============================================================================
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=data)
#We have many outliners in our dataset.
data.dtypes
# =============================================================================
# CUST_ID                              object
# BALANCE                             float64
# BALANCE_FREQUENCY                   float64
# PURCHASES                           float64
# ONEOFF_PURCHASES                    float64
# INSTALLMENTS_PURCHASES              float64
# CASH_ADVANCE                        float64
# PURCHASES_FREQUENCY                 float64
# ONEOFF_PURCHASES_FREQUENCY          float64
# PURCHASES_INSTALLMENTS_FREQUENCY    float64
# CASH_ADVANCE_FREQUENCY              float64
# CASH_ADVANCE_TRX                      int64
# PURCHASES_TRX                         int64
# CREDIT_LIMIT                        float64
# PAYMENTS                            float64
# MINIMUM_PAYMENTS                    float64
# PRC_FULL_PAYMENT                    float64
# TENURE                                int64
# 
# =============================================================================
#counting the number of missing values in our dataset
data.isna().sum()
# =============================================================================
# CUST_ID                               0
# BALANCE                               0
# BALANCE_FREQUENCY                     0
# PURCHASES                             0
# ONEOFF_PURCHASES                      0
# INSTALLMENTS_PURCHASES                0
# CASH_ADVANCE                          0
# PURCHASES_FREQUENCY                   0
# ONEOFF_PURCHASES_FREQUENCY            0
# PURCHASES_INSTALLMENTS_FREQUENCY      0
# CASH_ADVANCE_FREQUENCY                0
# CASH_ADVANCE_TRX                      0
# PURCHASES_TRX                         0
# CREDIT_LIMIT                          1
# PAYMENTS                              0
# MINIMUM_PAYMENTS                    313
# PRC_FULL_PAYMENT                      0
# TENURE                                0
# =============================================================================
#there is 1 missing value in CREDIT_LIMIT and 313 missing values in MINIMUM_PAYMENTS.

#Since there are too many missing values, we should not drop those rows, instead we replace the missing values by mean of that column.
data.loc[(data['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].mean()
data.loc[(data['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=data['CREDIT_LIMIT'].mean()
# =============================================================================
# By dropping outliers we can lose many rows as there are too many outliers in dataset. 
# So making ranges to deal with extreme values.
# 
# =============================================================================

columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
        'PAYMENTS', 'MINIMUM_PAYMENTS']

for c in columns:    
    Range=c+'_RANGE'
    data[Range]=0        
    data.loc[((data[c]>0)&(data[c]<=500)),Range]=1
    data.loc[((data[c]>500)&(data[c]<=1000)),Range]=2
    data.loc[((data[c]>1000)&(data[c]<=3000)),Range]=3
    data.loc[((data[c]>3000)&(data[c]<=5000)),Range]=4
    data.loc[((data[c]>5000)&(data[c]<=10000)),Range]=5
    data.loc[((data[c]>10000)),Range]=6
    
columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
         'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

for c in columns:    
    Range=c+'_RANGE'
    data[Range]=0
    data.loc[((data[c]>0)&(data[c]<=0.1)),Range]=1
    data.loc[((data[c]>0.1)&(data[c]<=0.2)),Range]=2
    data.loc[((data[c]>0.2)&(data[c]<=0.3)),Range]=3
    data.loc[((data[c]>0.3)&(data[c]<=0.4)),Range]=4
    data.loc[((data[c]>0.4)&(data[c]<=0.5)),Range]=5
    data.loc[((data[c]>0.5)&(data[c]<=0.6)),Range]=6
    data.loc[((data[c]>0.6)&(data[c]<=0.7)),Range]=7
    data.loc[((data[c]>0.7)&(data[c]<=0.8)),Range]=8
    data.loc[((data[c]>0.8)&(data[c]<=0.9)),Range]=9
    data.loc[((data[c]>0.9)&(data[c]<=1.0)),Range]=10

columns=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']  

for c in columns:    
    Range=c+'_RANGE'
    data[Range]=0
    data.loc[((data[c]>0)&(data[c]<=5)),Range]=1
    data.loc[((data[c]>5)&(data[c]<=10)),Range]=2
    data.loc[((data[c]>10)&(data[c]<=15)),Range]=3
    data.loc[((data[c]>15)&(data[c]<=20)),Range]=4
    data.loc[((data[c]>20)&(data[c]<=30)),Range]=5
    data.loc[((data[c]>30)&(data[c]<=50)),Range]=6
    data.loc[((data[c]>50)&(data[c]<=100)),Range]=7
    data.loc[((data[c]>100)),Range]=8

data.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY',  'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT' ], axis=1, inplace=True)

X= np.asarray(data)
#Before applying KMeans we should always normalizing input array.

scale = StandardScaler()
X = scale.fit_transform(X)
X.shape

#Using the elbow method the determine the optimal number of clusters.     
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 40):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 40), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#Looking at the graph, the no. of cluster should be between 6 to 10. We will use 6 clusters.

#Using k-means++ to stop avoid random initialisation trap.Labels is our dependent variable. It will denote which cluster a customer belongs too.
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
labels = kmeans.fit_predict(X)

#Using PCA to transform data to 2 dimensions for visualization.
dist = 1 - cosine_similarity(X)
pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)

x, y = X_PCA[:, 0], X_PCA[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple'}

names = {0: 'Cluster 1', 
         1: 'Cluster 2', 
         2: 'Cluster 3', 
         3: 'Cluster 4', 
         4: 'Cluster 5',
         5: 'Cluster 6'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
plt.show()

