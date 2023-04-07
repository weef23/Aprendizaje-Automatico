#### Laboratorio Modelos Supervisados  Intermedio

### Importacion de Librerias
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
import numpy as np
import model_evaluation_utils as meu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

## Paso numero 1 Cargar los datos
vinos = pd.read_csv("./Datos/wines_types.csv")
print(vinos.info())
### Validacion de datos Nulos, Vemos que la data esta libre de nulos y lista para trabajar.
nulos = vinos.isnull().sum()/len(vinos) *100
print(nulos)

## Codificacion de variable objetico white= 1 y red=2, usamos la funcion replace para hacer la codificacion
vinos["wine_type"]= vinos["wine_type"].replace({"white":1,"red":2})
print(vinos.head(3))

## Paso numero 2 Particionamiento de los datos
x = vinos.iloc[:,0:11].values
y = vinos.iloc[:,11].values

## Procedemos con el particionamiento de los datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 20)
print(pd.value_counts(y_train))

## Paso numero 3 balanceo de los datos, Para este proceso vamos a aplicar el metodo SMOTETomek el cual combina
## Under Sampling con Oversampling
st = SMOTETomek(sampling_strategy=0.8, random_state=20)
## A continuacion se aplica el balanceo de los datos
x_t_balanced, y_t_balanced = st.fit_resample(x_train,y_train)
print(pd.value_counts(y_t_balanced))

## Entrenamos una regresion logistica
model_logiR = LogisticRegression()
model_logiR.fit(x_t_balanced, y_t_balanced)

### Entrenamiento del modelo
pcorte=0.6
probabilidades=model_logiR.predict_proba(x_train)
print(probabilidades)
probs = probabilidades[:, 1]
print(probs)

## El siguiente paso es aplicar la evaluacion para el conjunto de testing/trainintg
probs_train = pd.DataFrame(probs, columns=["Probabilidades"])
probs_train["Prediccion"] = np.where(probs_train["Probabilidades"] > pcorte, 2, 1)

### A partir del corte generado por el modelo se crean las predicciones
print(probs_train.head())

## Ahora a hecer la prueba con el set de prueas completo usando el modelo ya entrenado
probabilidades=model_logiR.predict_proba(x_test)

## La clase positiva es Vino rojo asi que por ello tenemos que tomar la segunda columna
probs = probabilidades[:,1]
probs_test = pd.DataFrame(probs,columns=["Probabilidades"])
probs_test["Prediccion"]=np.where(probs_test["Probabilidades"]>pcorte,#condición
                                   2,#de cumplir la condición
                                   1)# de no complir la condición

print(probs_test.head())

#### Evaluacion de Metricas usando la data train
labels_names = [1, 2]
print(meu.display_model_performance_metrics(y_train, probs_train["Prediccion"], labels_names))
meu.plot_model_roc_curve(model_logiR,x_train, y_train)

## Evaluamos con la test
labels_names = [1, 2]
print(meu.display_model_performance_metrics(y_test, probs_test["Prediccion"], labels_names))
meu.plot_model_roc_curve(model_logiR, x_test, y_test)

### Uso del Analisis discriminante, Declaramos la clase LinearDiscriminantAnalysis para crear el modelo
lda = LinearDiscriminantAnalysis()
print(len(y_t_balanced))

## Entrenamos el modelo con la data Balanceada
lda.fit(x_t_balanced,y_t_balanced)

### Obtenemos las predicciones con el mismo punto de corte
probabilidades=lda.predict_proba(x_train)
print(probabilidades)
probs = probabilidades[:,1]
probs_train_lda = pd.DataFrame(probs,columns=["Probabilidades"])
probs_train_lda["Prediccion"]=np.where(probs_train_lda["Probabilidades"]>pcorte, 2, 1)
print(probs_train_lda)

## Aplicamos el mismo procedimiento para los datos de testeo
probabilidades=lda.predict_proba(x_test)
print(probabilidades)
probs = probabilidades[:,1]
probs_test_lda = pd.DataFrame(probs,columns=["Probabilidades"])
probs_test_lda["Prediccion"] = np.where(probs_test_lda["Probabilidades"] > pcorte, 2, 1)
print(probs_test_lda)