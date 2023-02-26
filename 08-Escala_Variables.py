import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
########################################################################################################
df = pd.read_csv("Datos/data_credit.csv", sep=",", encoding="ISO-8859-1")
print(df.head(5))
######### SEPARAMOS LAS VARIABLES DE LA VARIABLE A PREDECIR
x = df.iloc[:, 0:15].values
y = df.iloc[:, 15].values
######### HACEMOS UN SPLIT A LOS DATOS
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=2023)
############### SACAMOS EL MAXIMO Y MINIMO, ESTO ES PORQUE INTENTAMOS SIMULAR EL METODO MIN MAX
datos = [23, 21, 18, 32, 30, 26]
min_data = 18
max_data = 32
datos_norm=[]
######################## NORMALIZACION MIN MAX DE FORMA MANUAL ###################################
for i in range(0, len(datos)):
    d = (datos[i]-min_data)/(max_data-min_data)
    datos_norm.append(d)

##################################################################################################
df_caso = pd.DataFrame({"datos": datos, "datos_norm": datos_norm})
print(df_caso.head(5))

################### PYTHON IMPLEMENTA LA  LIBRERIA preprocessing PARA QUE IMPLEMENTA EL MIN MAX SCALER
minmax=MinMaxScaler()
## LA CLASE MINMAXScaler implementa el metodo de escalamiento min max
x_train_norm= minmax.fit_transform(x_train)

