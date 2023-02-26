############################################################################################################
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

######### Carga de datos ###################################
uber = pd.read_csv("./Datos/uber-raw-data-apr14.csv", sep=",")
print(uber.head(5))
print(len(uber))

###### Eliminamos los campos Data/Time, base porque no aportan nada
uber.drop(["Date/Time","Base"],axis=1,inplace=True)
print(uber.head(5))

## Verificamos los valores NULL
print(uber.isnull().sum())
xs = uber.values ## Lo convertimos en array para podre trabajarlo
print(xs)

######################################################################
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(xs[:,0],xs[:,1])
plt.show()

################ Aplicamos el Escalamiento de las variables ######
ms = MinMaxScaler()
X = ms.fit_transform(xs)
print(X)

####################################################################
df_uber = pd.DataFrame(X,columns=uber.columns)
print(df_uber.head(5))

#### Volvemos a Visualizar
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(X[:,0],X[:,1])
plt.show()

######################## ANALISIS DEL CODO PARA DETERMINAR EL NUMERO DE K ##################################
from sklearn.metrics import euclidean_distances, silhouette_score
from sklearn.cluster import KMeans

coef_inertia_uber =[]
for i in range(2,11):
    kmeans_model=KMeans(n_clusters=i,
                       init="k-means++",
                       max_iter=300,
                       n_init=10,
                       random_state=2022)
    kmeans_model.fit(df_uber)
    coef_inertia_uber.append(kmeans_model.inertia_)

#print(coef_inertia_uber)
####################### GRAFICAMOS EL CODO ###############################
plt.plot(range(2,11),
        coef_inertia_uber)
plt.show()

### Indentificar Clusters Kmeans
kmeans_model=KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=2022)
kmeans_model=kmeans_model.fit(df_uber)
y_kmeans= kmeans_model.predict(df_uber) + 1
print(y_kmeans)

data_values = uber.values
plt.scatter(data_values[y_kmeans==1,0], data_values[y_kmeans==1,1], s=10,  c="red", label="Ruta A")
plt.scatter(data_values[y_kmeans==2,0], data_values[y_kmeans==2,1], s=10,  c="green", label="Ruta B")
plt.scatter(data_values[y_kmeans==3,0], data_values[y_kmeans==3,1], s=10,  c="blue", label="Ruta c")
plt.legend()
plt.show()

################ Prediccion de ruta #########################################################
Allison = pd.DataFrame({"Lat":[40.7690],"Lon":[-73.9548]})
sol_allison = kmeans_model.predict(ms.transform(Allison)) +1; print(sol_allison)

Manuel = pd.DataFrame({"Lat":[40.7688],"Lon":[-73.9612]})
Chof = kmeans_model.predict(ms.transform(Manuel)) +1; print(Chof)