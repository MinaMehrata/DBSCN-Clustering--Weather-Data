#!/usr/bin/env python
# coding: utf-8

# ### DBSCN-Weathert CLustering
# 

# ### About the dataset
# 
# 		
# <h4 align = "center">
# Environment Canada    
# Monthly Values for July - 2015	
# </h4>
# <html>
# <head>
# <style>
# table {
#     font-family: arial, sans-serif;
#     border-collapse: collapse;
#     width: 100%;
# }
# 
# td, th {
#     border: 1px solid #dddddd;
#     text-align: left;
#     padding: 8px;
# }
# 
# tr:nth-child(even) {
#     background-color: #dddddd;
# }
# </style>
# </head>
# <body>
# 
# <table>
#   <tr>
#     <th>Name in the table</th>
#     <th>Meaning</th>
#   </tr>
#   <tr>
#     <td><font color = "green"><strong>Stn_Name</font></td>
#     <td><font color = "green"><strong>Station Name</font</td>
#   </tr>
#   <tr>
#     <td><font color = "green"><strong>Lat</font></td>
#     <td><font color = "green"><strong>Latitude (North+, degrees)</font></td>
#   </tr>
#   <tr>
#     <td><font color = "green"><strong>Long</font></td>
#     <td><font color = "green"><strong>Longitude (West - , degrees)</font></td>
#   </tr>
#   <tr>
#     <td>Prov</td>
#     <td>Province</td>
#   </tr>
#   <tr>
#     <td>Tm</td>
#     <td>Mean Temperature (°C)</td>
#   </tr>
#   <tr>
#     <td>DwTm</td>
#     <td>Days without Valid Mean Temperature</td>
#   </tr>
#   <tr>
#     <td>D</td>
#     <td>Mean Temperature difference from Normal (1981-2010) (°C)</td>
#   </tr>
#   <tr>
#     <td><font color = "black">Tx</font></td>
#     <td><font color = "black">Highest Monthly Maximum Temperature (°C)</font></td>
#   </tr>
#   <tr>
#     <td>DwTx</td>
#     <td>Days without Valid Maximum Temperature</td>
#   </tr>
#   <tr>
#     <td><font color = "black">Tn</font></td>
#     <td><font color = "black">Lowest Monthly Minimum Temperature (°C)</font></td>
#   </tr>
#   <tr>
#     <td>DwTn</td>
#     <td>Days without Valid Minimum Temperature</td>
#   </tr>
#   <tr>
#     <td>S</td>
#     <td>Snowfall (cm)</td>
#   </tr>
#   <tr>
#     <td>DwS</td>
#     <td>Days without Valid Snowfall</td>
#   </tr>
#   <tr>
#     <td>S%N</td>
#     <td>Percent of Normal (1981-2010) Snowfall</td>
#   </tr>
#   <tr>
#     <td><font color = "green"><strong>P</font></td>
#     <td><font color = "green"><strong>Total Precipitation (mm)</font></td>
#   </tr>
#   <tr>
#     <td>DwP</td>
#     <td>Days without Valid Precipitation</td>
#   </tr>
#   <tr>
#     <td>P%N</td>
#     <td>Percent of Normal (1981-2010) Precipitation</td>
#   </tr>
#   <tr>
#     <td>S_G</td>
#     <td>Snow on the ground at the end of the month (cm)</td>
#   </tr>
#   <tr>
#     <td>Pd</td>
#     <td>Number of days with Precipitation 1.0 mm or more</td>
#   </tr>
#   <tr>
#     <td>BS</td>
#     <td>Bright Sunshine (hours)</td>
#   </tr>
#   <tr>
#     <td>DwBS</td>
#     <td>Days without Valid Bright Sunshine</td>
#   </tr>
#   <tr>
#     <td>BS%</td>
#     <td>Percent of Normal (1981-2010) Bright Sunshine</td>
#   </tr>
#   <tr>
#     <td>HDD</td>
#     <td>Degree Days below 18 °C</td>
#   </tr>
#   <tr>
#     <td>CDD</td>
#     <td>Degree Days above 18 °C</td>
#   </tr>
#   <tr>
#     <td>Stn_No</td>
#     <td>Climate station identifier (first 3 digits indicate   drainage basin, last 4 characters are for sorting alphabetically).</td>
#   </tr>
#   <tr>
#     <td>NA</td>
#     <td>Not Available</td>
#   </tr>
# 
# 
# </table>
# 
# </body>
# </html>
# 
#  

# ### 1- Load the dataset
# We will import the .csv then we creates the columns for year, month and day.

# In[12]:


import csv
import pandas as pd
import numpy as np

filename='weather-stations20140101-20141231.csv'

#Read csv
pdf = pd.read_csv(filename)
pdf.head(5)


# ### 2-Cleaning
# Lets remove rows that dont have any value in the __Tm__ field.

# In[13]:


pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)
pdf.head(5)


# ### 3-Visualization
# Visualization of stations on map using basemap package. 

# In[14]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (14,10)

llon=-140
ulon=-50
llat=40
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To collect data based on stations        

xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()

#Visualization1
for index,row in pdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
   my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()


# ### 4- Clustering of stations based on their location i.e. Lat & Lon

# In[15]:


from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)


# In[16]:


set(labels)


# ### 5- Visualization of clusters based on location
# Now, we can visualize the clusters using basemap:

# In[17]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))


# ### 6- Clustering of stations based on their location, mean, max, and min Temperature
# In this section,rerun DBSCAN, but this time on a 5-dimensional dataset:

# In[18]:


from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)


# ### 7- Visualization of clusters based on location and Temperture
# 

# In[19]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

