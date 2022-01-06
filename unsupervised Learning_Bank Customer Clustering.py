#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[104]:


from matplotlib.pyplot import figure as fig
from scipy.stats import linregress
# Essentials:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# t-SNE visualization
from sklearn.manifold import TSNE

# imputation
from sklearn.impute import KNNImputer

# Scaling
from sklearn.preprocessing import StandardScaler

# PCA
from sklearn.decomposition import PCA

# K-means for Clustering
from sklearn.cluster import KMeans

# cluster metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

# Silhouette Visualizer


# # Read tha data

# In[105]:


data= pd.read_csv('bankmarketing.csv')


# In[106]:


data.head()


# In[107]:


data.info()


# In[108]:


data.columns


# # EDA

# In[109]:


data = data.drop(['y'], axis=1)


# In[110]:


data.columns


# In[111]:


bank_cust = data[['age','job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','day_of_week','poutcome']]
bank_cust.head()


# In[112]:


bank_cust['age_bin'] = pd.cut(bank_cust['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                              labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
bank_cust  = bank_cust.drop('age',axis = 1)
bank_cust.head()


# In[113]:


bank_cust.shape


# In[114]:


bank_cust.describe().T


# In[115]:


bank_cust.info()


# In[116]:


data.dtypes


# In[117]:


#data['age_bin'].value_counts()


# In[118]:


#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()

#data = bank_cust.apply(le.fit_transform)
#data.head()


# In[119]:


data.month.unique()


# In[120]:


#Exploratory Data Analysis
#Import ploting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
colors = ['#DB1C18','#DBDB3B','#51A2DB']
sns.set(palette=colors, font='Serif', style='white', rc={'axes.facecolor':'whitesmoke', 'figure.facecolor':'whitesmoke'})


# In[121]:


#Univariated data analysis
data.count()


# In[122]:


fig, ax = plt.subplots(nrows=4,ncols=3, figsize=(15,8), constrained_layout=True)
plt.suptitle("Univariated Data Analyis")
ax=ax.flatten()
int_cols= data.select_dtypes(exclude='object').columns
for x, i in enumerate(int_cols):
    sns.histplot(data[i], ax=ax[x], kde=True, color=colors[2])
    


# In[123]:


#Model Building
# First we will keep a copy of data
data_copy = data.copy()


# # scaled_data

# In[124]:


from sklearn.preprocessing import StandardScaler


# In[125]:


scaler= StandardScaler()
dt = data.select_dtypes(include=np.number)
dt


# In[126]:


scaled_data= scaler.fit_transform(dt)


# In[127]:


pd.DataFrame(scaled_data)


# # Models

# # K_means

# In[128]:


#How to choose k
##Elbow method
#Sum of squares of distances of points from corresponding cluster centroid (inertia) should be small


# In[129]:


from  sklearn.cluster import KMeans


# In[130]:


import matplotlib.pyplot as plt


# In[131]:


import seaborn as sns


# In[132]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[133]:


sse ={}
for k in range(1,8):
        km = KMeans(n_clusters =k)
        print(k)
        km.fit(dt)
        km.inertia_
#print(sse)


# In[134]:


"kmeans" == "Kmeans"


# In[135]:


sse ={}
for k in range(1,8):
        Kmeans = KMeans(n_clusters =k, random_state=55)
        Kmeans.fit(dt)
        sse[k] = Kmeans.inertia_
        
plt.title("The Elbe Method") 
plt.xlabel("k")
plt.ylabel("SSE")
sns.pointplot(x=list(sse.keys()), y = list(sse.values()))


# In[136]:


# best number of clustteres is 3


# In[137]:


model= KMeans(n_clusters=4, random_state=32)


# In[138]:


model.fit(scaled_data)


# In[139]:


model.labels_


# In[140]:


new_df= pd.DataFrame(dt)
new_df                    


# In[141]:


data["labels"]= model.labels_
data


# In[142]:


data.groupby("labels").agg({"mean", "count"}) 


# In[ ]:


#km=KMeans(n_clusters = 4, random_state=42)
#labels= km.fit_predict()
#data["cluster"]=labels


# In[ ]:


#data["cluster"].value_counts()


# In[ ]:


#from mpl_toolkits.mplot3d import Axes3D
#ig = plt.figure(figsize = (20,10))
#ax = fig.add_subplot(111, projection = "3d")
#ax.scatter(data["age"][data.labels == 0], data["loan"][data.labels == 0], c = "blue", s =60)
#ax.scatter(data["age"][data.labels == 1], data["loan"][data.labels == 1], c = "red", s =60)


#ax.view_init(30,185)
#plt.xlabel("job")
#plt.ylabel("marital")
#plt.show()


# In[ ]:


#centers =pd.DataFrame (km.cluster_centers_,columns= data.columns)


# In[ ]:


#centers.to_clipboard()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


ss = StandardScaler()


# In[ ]:


data = ss.fit_transform(new_df)


# In[ ]:


data


# In[ ]:


from kmodes.kmodes import KModes
#Using K-Mode with "Cao" initialization
km_cao = KModes(n_clusters=4, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(data)


# In[ ]:


# Predicted Clusters
fitClusters_cao


# In[ ]:


km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(data)


# In[ ]:


fitClusters_huang


# In[ ]:


cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)


# In[ ]:


y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)


# In[ ]:


## Choosing K=2
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(data)


# In[ ]:


fitClusters_cao


# In[144]:


data = data_copy.reset_index()


# In[145]:


clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([data, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)


# In[146]:


combinedDf.head()


# In[147]:


combinedDf.job.value_counts()


# In[148]:


#Cluster Identification
cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
cluster_2 = combinedDf[combinedDf['cluster_predicted'] == 2]
cluster_3 = combinedDf[combinedDf['cluster_predicted'] == 3]
cluster_0.info()


# In[149]:


cluster_1.info()


# In[150]:


cluster_2.info()


# In[151]:


cluster_3.info()


# In[155]:


#job
plt.subplots(figsize = (25,8))
sns.countplot(x=combinedDf['job'],order=combinedDf['job'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[98]:


# Marital
plt.subplots(figsize = (5,5))
sns.countplot(x=combinedDf['marital'],order=combinedDf['marital'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[84]:


# Education
plt.subplots(figsize = (15,5))
sns.countplot(x=combinedDf['education'],order=combinedDf['education'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[85]:


# Default
f, axs = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x=combinedDf['default'],order=combinedDf['default'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])
sns.countplot(x=combinedDf['housing'],order=combinedDf['housing'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])
sns.countplot(x=combinedDf['loan'],order=combinedDf['loan'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[2])

plt.tight_layout()
plt.show()


# In[86]:


f, axs = plt.subplots(1,2,figsize = (15,5))
sns.countplot(x=combinedDf['month'],order=combinedDf['month'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])
sns.countplot(x=combinedDf['day_of_week'],order=combinedDf['day_of_week'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])

plt.tight_layout()
plt.show()


# # Principal Component Analysis (PCA)
# 

# In[ ]:


from sklearn.datasets import load_breast_cancer


# In[ ]:


breast = load_breast_cancer()
breast_data = breast.data


# In[ ]:


breast_data.shape


# # affinity propagation clustering
# 

# In[ ]:


from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
# define dataset
new_df = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = AffinityPropagation(damping=0.9)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()


# # Hiererarchy

# In[ ]:


import os # new
from pathlib import Path


# In[ ]:


from sklearn.linear_model import SGDClassifier


# In[ ]:


import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
plt.figure(figsize=(10,7))
plt.title("Dendrograms")
dend =shc.dendrogram(shc.linkage(new_df,method='ward'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




