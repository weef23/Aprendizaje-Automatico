################## Tecnicas de Balanceo de los datos ###########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

### Importacion de datos
data = pd.read_csv("./Datos/Grid v1.csv", sep=",")
print(data.head(5))

### La data anterior presenta el numero de reclamos por cada registro
### rsrp indicador de la señal
### La variable rsrp solo puede traer valores negativos
data = data.loc[(data["rsrp"] < 0)]
print(data.head(3))
### Reseteamos los index
data = data.reset_index(drop=True)
print(data.head())

### Sacamos las estadisticas de en base a la variable incidente para determinar el balance entre clases
print(pd.value_counts(data["incidentes"]))
### Reducimos las categorias a 2, 0 sin incidentes 1 con incidentes
data["incidentes"] = data["incidentes"].replace([1,2,3,4,5,6,7],1)
### Volvemos a imprimir la distribucion de clases, si dividemos el numero de cada clase entre el total y lo multiplicamos
### Por 100 obtenemos el porcentaje de cada clase, vemos un desbalanceo bastante alto.
print(pd.value_counts(data["incidentes"])/len(data) * 100)
### El Balanceo de los datos unicamente aplica a los datos de entrenamiento no aplica a los datos de testeo
features = data[["rsrp", "redireccion", "cqi", "incidentes"]] ## Seleccion de variables Pandas.
print(features.head(5))

#### En ML es muy importante  separar las varibles predictoras de la variable a predecir
x = features.iloc[:, 0:3].values
y = features.iloc[:, -1].values #variable a balancear
## Aplicamos un Split de los datos, eso lo podemos hacer con la funcion train_test_split, test_size corresponde a
## A la particion asignada al set de pruebas, randon_state sirve para establecer una semilla
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2023)

### Balanceo de datos usando UnderSampling , extrae una muestra de la clase dominante o mayoritaria
### El parametro sampling_strategy representa el procentaje que representara la clase minoritaria
under = NearMiss(sampling_strategy=0.8, version=2)
x_under, y_under = under.fit_resample(x_train, y_train)
## Ahora la data esta mucho mas balanceada y la clase minoritaria representa el 80% de la clase mayoritaria.
print(pd.value_counts(y_under))

#### Tecnica de Oversampling, la tecnica de undersampling toma una muestra reducida de la clase mayaroritaria,
### La tecnica de Oversampling lo que hace es toma la clase minoritaria y la pone al nivel de la clase mayoritaria
### Para ello hace replica de la data.
over = RandomOverSampler(sampling_strategy=0.8)
x_over, y_over = over.fit_resample(x_train,y_train)
print(pd.value_counts(y_over))

### Ambos metodos tienen desventajas, el primero puede eliminar casos potencialmente importantes
### que puedan explicar la clase mayoritaria, el puede sobreestimar caso no tan importantes.

## Smote and Tomek, este es un metodo intermedio que realiza undersampling y oversampling a la vez
st = SMOTETomek(sampling_strategy=0.8, random_state=2023)
x_st, y_st = st.fit_resample(x_train,y_train)
print(pd.value_counts(y_st))