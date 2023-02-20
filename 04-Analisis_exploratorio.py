###### En este ejemplo se visulizaran todas las tecnicas de preprocesamiento de los datos
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


### Cargamos los datos que vamos a preprocesar, es importante definir el separador para obtener el df correctamente
mark2 = pd.read_csv("./Datos/MarketingDirecto_2.csv", sep=";")

### La funcion Head puede mostrarnos los primeros 5 valores
print(mark2.head(5))
### De la siguiente forma podemos extraer los nulos
nulos = mark2.isnull().sum()/len(mark2) *100
print(nulos)

### Ejemplo de imputacion de faltantes usando una regresion lineal
edad = [25, 35, 28, 29, 29]
peso = [75, 65, 68, "NA", 72]
## Usamos todos los datos que tienen valores completos para predecir los faltantes
edad = [25, 35, 28, 29]
peso = [75, 65, 68, 72]
## Convertimos los vectores en un solo data frame, nota usamos un diccionario para definirlo
df = pd.DataFrame({"edad":edad, "peso":peso})
print(df)

##Importamos la regresion lineal

lm = LinearRegression()
x_train = df[["edad"]]
y_train = df[["peso"]]

lm.fit(x_train,y_train)
## Generamos el modelo lineal para generar la prediccion esta es una ecuacion tipo
## Y= mx + b, b es el intercept de la ecuacion
print(lm.intercept_)
## Ahoral la pendiente es decir m
print(lm.coef_)

## Ahora predecimos el valor faltante
y = list(lm.intercept_ + lm.coef_*29) ## Con esto predecimos el valor
print(f"El valor NA es {y}")
## Imputacion utilizanod el SimplerImputer, Esta clase nos perimite imputar usando Media, Mediana, Moda
from sklearn.impute import  SimpleImputer

impt_media = SimpleImputer(missing_values=np.nan, strategy="mean")
impt_mediana = SimpleImputer(missing_values=np.nan, strategy="median")
impt_moda = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

## Para aplicarlo simplemente lo hacemos de la siguiente forma
## Nota el Fit transform solo se utiliza para los datos de training
## Para los datos de testing usamos transform para que no nos cambie los datos
mark2[["Historial"]] = impt_moda.fit_transform(mark2[["Historial"]])
print(mark2.head(5))
mark2[["Edad"]] = impt_moda.fit_transform(mark2[["Edad"]])
print(mark2.head(5))

### Imputacion usando metodo supervisado, En esta parte vamos a usar lo el metodo supervisado visto anteriormente
### Para usar la regresion para tratar de predecir los valores, es posible realizar un analisis de variables, en este
## Caso lo omitimos porque ya tenemos identificadas las variables

## Lo primero es cargar los datos
mark2=pd.read_csv("./Datos/MarketingDirecto_2.csv", sep=";")
## Luego hacer el analisis de completitud de los datos, anteriormente tambien vimos como convertir eso a porcentajes
print(mark2.isnull().sum())
## Lo convertimos en porcentajes
nulos = mark2.isnull().sum()/len(mark2) *100
print(nulos)

## El primer paso es determinar cuales son los valores nulos, la funcion loc es muy util en pandas es usada para
## Indexas usando etiquetas
es_nulo = pd.isna(mark2.loc[:,"Monto"])
## Con esto procedemos extraer todos aquellos valores con montos nulos en un dataframe aparte
df_nulos = mark2.loc[es_nulo]
print(df_nulos.head(5))
## Ahora hacemos lo propio con los valores no nulos
## El replace lo que hace es tomar los valores nulos e invertirlos, ya que ahora queremos lo contrario
## Traer los valores que estan totalmente completos
no_esnulo = es_nulo.replace({True:False, False:True})

## Traemos los valores completos
df_data = mark2.loc[no_esnulo]
print(df_data.head(5))

## Ahora tomaremos dos variables, la variable que deseamos imputar, y la variable que usaremos para al imputacion.
## Para seleccionar la veriable que queremos usar para la prediccion podemos hacer un anlisis de regresion o bien
## Usar un analisis de correlacion entre las variables, si vemos una variable con mas 0.5 de correlacion podemos
## usar esa variable, en este caso ya tenemos las variables salario y monto

## Usamos los valores completos para entrenar el modelo
x= df_data[["Salario"]]
y= df_data[["Monto"]]
## Estos son los datos que queremos imputar
x_test = df_nulos[["Salario"]]
y_test = df_nulos[["Monto"]]

## Usaremos el modelo de regresion como en el ejemplo anterior para hacer la imputacion
regresion = LinearRegression()
regresion.fit(x,y)

### Procedemos con la imputacion usando el metodo predict
df_nulos[["Monto"]]=np.round(regresion.predict(x_test),0)
print(df_nulos.head())

## Ahora procedemos a concatenar los dos dataframe, los imputados con los datos completos
## Primero reseteamos los indices, se resetean los indices para poderlos integrar
df_nulos = df_nulos.reset_index(drop=True)
df_data = df_data.reset_index(drop=True)
## Procedemos con la concatenacion
df_imputados = pd.concat([df_nulos,df_data],axis=0)

## Validemos que los datos se hayan imputados
nulos = df_imputados.isnull().sum()/len(mark2) *100
print(nulos)

## La variable anterior usamos una regresion lineal para hacer la imputacion de los datos, ahora queremos imputar
## El Historial pero para eso necesitamos usar un algoritmo que soporte variables cualitativas

## De la misma forma extraemos los valores nulos que queremos imputar
es_nulo = pd.isna(df_imputados.loc[:,"Historial"])
df_nulos = df_imputados.loc[es_nulo]
print(df_nulos.head(5))

## Estraemos los datos completos
es_nulo2=es_nulo.replace({True:False, False:True})
df_data=df_imputados.loc[es_nulo2]
print(df_data.head(5))

## Igual que en el ejemplo anterior usamos los datos completos para predecir los datos

y=df_data[["Historial"]]
x=df_data[["Edad","Genero","Ecivil"]]

### Por que se seleccionaron las variables anteriores para ello generamos una tabla de contingencia

print("Variables de cotingencias \n:")
print(pd.crosstab(df_data["Genero"], df_data["Historial"]))
print(pd.crosstab(df_data["Edad"], df_data["Historial"]))
print(pd.crosstab(df_data["Ecivil"], df_data["Historial"]))

## Antes de hacer cualquier cosa es necesario codificar las variable Edad Genero y ECivil, de lo contrario no
## Funcionara completamente
d = defaultdict(LabelEncoder)

## Usamos la funcion apply y una expresion lambda para aplicar la transformacion a cada columna
fit = x.apply(lambda x: d[x.name].fit(x))
x = x.apply(lambda x: d[x.name].transform(x))
print(x.head(5))

## Extraemos las variables con los valores nulos
y_test=df_nulos[["Historial"]]
x_test=df_nulos[["Edad","Genero","Ecivil"]]

## Hacemos la transformacion
x_test=x_test.apply(lambda x: d[x.name].transform(x))
print(x_test.head(5))

## Entrenamos el arbol con los valores completos
arboles = DecisionTreeClassifier(random_state=2023)
arboles.fit(x, y)

y_test = arboles.predict(x_test)

df_nulos["Historial"] =y_test
print(df_nulos.head())
#### Procedemos con la parte final concatenar ambos df
df_nulos = df_nulos.reset_index(drop=True)
df_data = df_data.reset_index(drop=True)
## Procedemos con la concatenacion
df_imputados = pd.concat([df_nulos,df_data],axis=0)
## Validemos que los datos se hayan imputados
nulos = df_imputados.isnull().sum()/len(mark2) *100
print(nulos)