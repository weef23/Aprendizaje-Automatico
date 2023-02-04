### El clasificador KNN es uno de los mas comunes y los mas basicos clasificadores utilizados en ML
### En realidad este clasificador lo que hace es generar una clasificacion muy sencilla basandose
## En la cercania de los puntos vecinos circundantes. Esta basado en instancias y no es un generalizador.

## Importamos las librerias
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (KFold, cross_val_score)
from sklearn.model_selection import cross_val_predict
import numpy as np

## Lo primero que tenemos que hacer en es  cargar la data
## Basicamente lo que estamos cargando es el dataser iris que contiene datos especificos las flores de iris
## En X tenemos los predictores dentro de los cuales tenemos datos como el largo y ancho del sepalo
## Largo y ancho del petalo
x, y = load_iris(return_X_y=True)
### Lo que procedemos ahora es a hacer un Split de los datos para poder hacer el entrenamiento y pruebas
## Con el metodo train_test_split basicamente lo que estamos haciendo es dividiendo el dataset en 70%
## Entrenamiento y 30 % para las pruebas.
x_train, x_test, y_train, y_test = train_test_split(x,y, stratify= y, test_size=0.7, random_state=42)
## KNN es un algoritmo basado en distancia por lo tanto siempre es buena idea normalizar el dataset.
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
## Ahora entrenamos el clasificador con nuestra data de test.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

## Una vez entrenado queremos validar el modelo tanto en la fase de entrenamiento como en pruebas
## En teoria tenemos un modelo con una muy buena precision de 0.93
print('Precision del modelo con el set de Entrenamiento: {:.2f}'
     .format(knn.score(x_train, y_train)))
print('Presicion del modelo con el set de pruebas: {:.2f}'
     .format(knn.score(x_test, y_test)))

## Sin embargo la mejor forma de determinar si de verdad esta funcionando es generando una matriz de confusion
## Con la funcion predict podemos generar nuestra prediccion
pred = knn.predict(x_test)
## Generamos la matriz de confusion de nuestro clasificador
print(confusion_matrix(y_test, pred))
## Ahora veamos que la presicion y el recall del clasificador
print(classification_report(y_test, pred))
### KNN Utilizando Validacion Cruzada, la validacion cruzada nos permite hacer mas eficiente el entrenamiento
### De los modelos, la idea de utilizar distintos conjuntos de validacion es reducir en Sobreentrenamiento
## La Validacion cruzada nos permite crear n conjuntos a partir del conjunto de entrenamiento y en cada
## Ciclo dejar un conjunto para entrenamiento.
## Con la funcion cross_val_score le decimos que genere 5 set de datos distintos, de los cual de cada 10 muestras
## Habra una para validacion y 9 para entrenar

## Con el Kfold definios el numero de sets que vamos a utilizar
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(knn, x, y, cv=kfold, scoring='accuracy')
print(cv_scores)
## Esto nos muestra como realmente se comportaria con distintos sets aleatoreos nos ayuda a construir
## a ver con exactitud como se comportaria nuestro modelo. Esto no entrena nuestro modelo sino que valida su verdadera
## Capacidad.
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
########################################################################################################################

