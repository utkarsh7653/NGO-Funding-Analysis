#!/usr/bin/env python
# coding: utf-8

# In[18]:


#Importing the necessary libraries required for this project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[12]:


df = pd.read_csv(r'C:\Users\utkar\Downloads\Foldfer for output\country-data.csv')


# In[13]:


print(df)


# In[14]:


#Viewing the first five values to explore the data briefly
df.head()


# In[15]:


#checking the distribution of the variables respectively
df.describe()


# In[16]:


#removing cells with null values
df = df.dropna()


# In[17]:


df.head()


# In[20]:


#Checking the data types of the variables
df.info()


# In[21]:


#Checking correlation across all the variables excluding country name as that is non numerical
df_subset = df.iloc[:,1:]
df_subset.corr()


# In[28]:


#Based upon the correlation analysis plotting relationship b/w variables that showed high correlation values
#Plotting relationship between export and import 
sns.scatterplot(data=df,x='exports',y='imports', color='blue')
plt.xlabel('Export of Goods/Services')
plt.ylabel('Import of Goods/Services')
plt.title('Relation b/w export and import')
plt.grid(True)
plt.show()


# In[29]:


#Plotting relationship between mortality rate and total fertility
sns.scatterplot(data=df,x='child_mort',y='total_fer', color='blue')
plt.xlabel('child_mortality_rate')
plt.ylabel('total_fer')
plt.title('Relation b/w child mortality rate and total fertility')
plt.grid(True)
plt.show()


# In[30]:


#Plotting relationship between income and life expectancy
sns.scatterplot(data=df,x='income',y='life_expec', color='blue')
plt.xlabel('income')
plt.ylabel('life expectancy')
plt.title('Relation b/w income and life expectancy')
plt.grid(True)
plt.show()


# In[33]:


pip install geopandas


# In[34]:


pip install gdal fiona pyproj shapely


# In[35]:


import geopandas as gpd


# In[127]:


#Observing distribution child mortality rate around the world
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.merge(df, how='left', left_on='name', right_on='country')
world.plot(column='child_mort',cmap='OrRd', legend=True, figsize=(15,10))
plt.title('child mortality rate choropleth map')
plt.show()


# In[39]:


#Now seeing countries having the highest child mortality so NGO knows which country needs attention
mean_mortality = df.groupby('country')['child_mort'].mean()
top_countries = mean_mortality.nlargest(10)
plt.figure(figsize=(10,6))
top_countries.plot(kind='bar',color='skyblue')
plt.title('Top Countries with Highest Child Mortality Rate')
plt.xlabel('Country')
plt.ylabel('Mean Child Mortality')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[40]:


from sklearn.preprocessing import LabelEncoder


# In[41]:


label_encoder = LabelEncoder()
df['Country_Label'] = label_encoder.fit_transform(df['country'])
print(df)


# In[42]:


df.head()


# In[63]:


#Deleting this to perform clustering and PCA
del df[df.columns[0]]
df


# In[94]:


#Standardizing and scaling
from sklearn.decomposition import PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# In[95]:


pca = PCA(n_components=2) #PCA will reduce dimensions to 2
principal_components = pca.fit_transform(scaled_data) #transform scaled data to principal components

principal_df = pd.DataFrame(data=principal_components,columns=['PC1','PC2']) #Stores transformed data into PC1 and PC2
print(principal_df)


# In[98]:


import plotly.express as px
pca = PCA(n_components=9)  # Adjust the number of components as needed
pca_df = pca.fit_transform(scaled_data) 
features = range(pca.n_components)
#Shows explained variance of each principal component
fig = px.bar(x=features, y=pca.explained_variance_) 
fig.update_layout(title='PCA features')
fig.update_xaxes(title_text='Features')
fig.update_yaxes(title_text='Variance')
fig.show()


# In[99]:


#Understanding the total variance of the data by looking deeply into variance of the number of principal components
cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
fig_cumulative = px.line(x=features, y=cumulative_explained_variance, labels={'x': 'Principal Components', 'y': 'Cumulative Explained Variance'})
fig_cumulative.update_layout(title='Cumulative Explained Variance by Principal Components')
fig_cumulative.show()

# Plot the principal components
principal_df = pd.DataFrame(data=pca_df, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
plt.figure(figsize=(8, 6))
plt.scatter(principal_df['PC1'], principal_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Sample Data')
plt.grid()
plt.show()


# In[102]:


from sklearn.cluster import KMeans
#Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Applying K Means Clustering
kmeans = KMeans(n_clusters=2,random_state=42)
kmeans.fit(scaled_data)

#Getting cluster labels
cluster_labels = kmeans.labels_

#Adding cluster labels to dataframe
df['Cluster'] = cluster_labels

print(df)


# In[106]:


#Cluster distribution among different relations
sns.pairplot(df,hue='Cluster')
plt.show()


# In[117]:


df.head()


# In[118]:


df = pd.read_csv(r'C:\Users\utkar\Downloads\Foldfer for output\country-data.csv')
df.head()


# In[119]:


#Similar cluster countries
df[df['Cluster']==0]['country'].values


# In[120]:


df[df['Cluster']==1]['country'].values


# In[123]:


world_with_clusters = world.merge(df,how='left',left_on='name',right_on='country')


# In[126]:


# Define colormap for clusters
#Cluster in red need more attention by the NGO
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['blue', 'red'])

# Plot choropleth map with legend
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
world_with_clusters.plot(column='Cluster', cmap=cmap, legend=True, ax=ax)
ax.set_title('Choropleth Map with Cluster 0 and Cluster 1')
plt.show()


# In[ ]:




