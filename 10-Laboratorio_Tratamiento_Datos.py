#################### LABORATORIO SOBRE TRATAMIENTO DE LA INFORMACION ########################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

#### PASO NUMERO 1 CARGAMOS LA DATA QUE VAMOS A UTILIZAR EN ESTE CASO DATA_FUGA.csv
datos_fuga = pd.read_csv("Datos/DATA_FUGA.csv", sep=",") ## LA DATA VIENE SEPARADA POR COMAS
print(datos_fuga.head(3))

#### PASO NUMERO 2 ANALISIS DE COMPLETITUD DE LOS DATOS

## La funcion is null nos devuelve todos los nulos, con la funcion sum obtenemos el numero de nulos
## Al dividirlo entre la cantidad de datos y multiplicarlo por 100 obrenmos los nulos
nulos = datos_fuga.isnull().sum()/len(datos_fuga) *100
print(nulos)

### Vemos que los datos que tienen mas del 30% de nulos son FLG_VEH_SF, FLG_CONV_SF, REC_AGENTE_TD
### Hagamos el particionamiento de lo datos y veamos si el comportamiento se mantiene
### ELIMINAMOS LA VARIABLE REC_AGENTE_TD porque casi todos sus valores estan nulos
##datos_fuga = datos_fuga.drop("REC_AGENTE_TD")
datos_fuga = datos_fuga.drop(['REC_AGENTE_TD'], axis=1)
print(datos_fuga.info())
#### PASO NUMERO 3 IMPUTACION DE LOS DATOS
## El set de datos que tenemos combina Combina variables tanto cualitativas como cuantitativas
## Esto implica que no podemos simplementa aplicar un solo metodo de imputacion por lo que vamos a Tratar de reorganizar
## El dataframe
tipoDatos=datos_fuga.columns.groupby(datos_fuga.dtypes)
print(tipoDatos)
### Lo Anterior nos devuelve un diccionario con los tipos de datos que tenemos Enteros, Flotantes y Objetos
## Separamos aquellos campos que son Objetos y aquellos que no lo son
columns_object = list(tipoDatos[np.dtype("O")])
#print(columns_object)
columns_number = list(tipoDatos[np.dtype("int64")]) + list(tipoDatos[np.dtype("float64")])
#print(columns_number)
###################################################################################################
df_objetos = datos_fuga.loc[:, columns_object]
df_number = datos_fuga.loc[:, columns_number]
###################################################################################################
df_fuga = pd.concat([df_number, df_objetos], axis=1)
#print(df_fuga.head(3))
########################## SEPARAMOS LA VARIABLE TARGET DEL RESTO DE VARIABLES ###################
x = df_fuga.loc[:, df_fuga.columns != "TARGET_MODEL2"]
y = df_fuga.TARGET_MODEL2

columns_number.remove("TARGET_MODEL2")
################## SEPARAMOS LA DATA DE ENTRENAMIENTO DE LA DATA DE PRUEBAS ######################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=20, stratify=y)
######### Aplicamos la imputacion de los datos a la data de entrenamiento
imp_media = SimpleImputer(missing_values=np.nan, strategy="mean")
imp_moda = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
########## APLICAMOS LA IMPUTACION POR MODA Y MEDIA EN LA DATA DE ENTRENAMIENTO
x_train[columns_number] = imp_media.fit_transform(x_train[columns_number])
x_train[columns_object] = imp_moda.fit_transform(x_train[columns_object])
nulos = x_train.isnull().sum()/len(x_train) *100
print(nulos)
########################################################################################################################
x_train.plot.scatter(x="ANT_CLIENTE", y ="INGRESO_BRUTO_M1")
plt.show()
############### APLIQUE EL Z_Score Al x_train y determines cuantos valores quedan
### Aplicamos el z-score
#z_score = x_train[columns_number].apply(stats.zscore)
z_score = np.abs(stats.zscore(x_train[columns_number].values))
x_train_zscore = x_train[(z_score < 3).all(axis=1)]
print(x_train_zscore)

### PASO NUMERO 4 DISCRETIZACION DE VARIABLES

### Aplicamos la discretizacion de la variable INGRESO_BRUTO_M1
amplitud = KBinsDiscretizer(n_bins=18, encode="ordinal", strategy="uniform")
## Aplicamos la discretizacion
x_train_uniform = pd.DataFrame(amplitud.fit_transform(x_train[["INGRESO_BRUTO_M1"]]))
## Validamos la cantidad de valores por categoria
print(x_train_uniform.value_counts(dropna=True))

### Discretizacion por Quantile
cuartil = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
x_train_quantile = pd.DataFrame(cuartil.fit_transform(x_train[["EDAD"]]))
print(x_train_quantile.value_counts(dropna=True))

### PASO NUMERO 5 BALANCEO DE DATOS
### APLICAMOS UNDERSAMPLING A LOS DATOS  PARA EQUILIBRAR LAS CLASES

### Primero con UnderSampling
us = RandomUnderSampler(sampling_strategy=0.8, random_state=20)
x_under, y_under = us.fit_resample(x_train, y_train)
print(pd.value_counts(y_under))

### Ahora con Oversampling
over = RandomOverSampler(sampling_strategy=0.8,random_state=20)
x_over, y_over = over.fit_resample(x_train, y_train)
print(pd.value_counts(y_over))
