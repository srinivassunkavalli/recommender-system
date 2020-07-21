#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[40]:


data=pd.read_csv(r"C:\Users\dell\Desktop\book.csv", encoding='latin-1')


# In[41]:


data


# In[42]:


data=data.drop('Unnamed: 0',axis=1)


# In[44]:


data.columns=['User_ID','Book_Title', 'Book_Rating']


# In[45]:


data


# In[46]:


data['Book_Title'].unique().shape


# In[47]:


data['Book_Rating'].unique().shape


# In[48]:


ratings = pd.DataFrame(data.groupby('Book_Title')['Book_Rating'].mean())


# In[49]:


ratings


# In[50]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ratings['Book_Rating'].hist(bins=50)


# In[51]:


ratings['number_of_ratings'] = data.groupby('Book_Title')['Book_Rating'].count()


# In[52]:


ratings['number_of_ratings'].hist(bins=60)


# In[53]:


import seaborn as sns
sns.jointplot(x='Book_Rating', y='number_of_ratings', data=ratings)


# In[60]:


movie_matrix = data.pivot_table(index='User_ID', columns='Book_Title', values='Book_Rating').fillna(0)
movie_matrix


# In[61]:


from scipy.sparse import csr_matrix


# In[64]:


l=csr_matrix(movie_matrix.values)


# In[65]:


from sklearn.neighbors import NearestNeighbors


# In[66]:


model_knn=NearestNeighbors(algorithm='brute',
    metric='cosine')


# In[69]:


model_knn.fit(l)


# In[70]:


movie_matrix.shape


# In[73]:


query_index=np.random.choice(movie_matrix.shape[0])


# In[74]:


distances, indices = model_knn.kneighbors(movie_matrix.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)


# In[75]:


movie_matrix.head()


# In[76]:


for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_matrix.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_matrix.index[indices.flatten()[i]], distances.flatten()[i]))

