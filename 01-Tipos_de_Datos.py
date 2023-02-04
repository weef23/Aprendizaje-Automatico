from sklearn.preprocessing import QuantileTransformer
import numpy as np
######################## Tipos de datos en Python #################################################
### Los tipos de datos son muy importante en el aprendizaje no supervisado
### Esto debido a que muchas de las funciones que ofrecen librerias como sklearn, keras etc.
### Exigen parametros con tipos de datos muy especificos.
### Ejemplo de numeros real

####################################################################################################
numero_real = 11.3
## La funcion type es muy util para entender el tipo de datos con el que tratamos.
print(type(numero_real))

## Numeros complejos o imaginarios, A continuacion se muestra la representacion de un numero complejo.
numero_Complejo = 11.3 + 7j
print(type(numero_Complejo))

## A continuacion se muestra un ejemplo de un preprocesamiento en usando sklearn
## Primero vamos a crear un arreglo usando numpyarray
rng = np.random.RandomState(0)
## Con la siguiente funcion normal vamos a generar una distribucion normal la funcion recibe los siguientes parametros:
## loc es un float o un arreglo de float, y representa El centro de la distribucion osea la media.
## este valor como vemos recibe uno o varios decimales, no puede recibir una cadena por ejemplo
## scale corresponde a la desviacion estandar es decir el ancho de la muestra recibe uno o mas float
## size representa la forma de salida recibe una tupla de valores enteros
## A su vez la funcion retorna un ndarray
x = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
print(x)
### Devuelve un ndarray
print(type(x))

## A continuacion tenemos la funcion QuantileTransformer la funcion recibe como parametros principales
## n_quantiles el cual es un numero entero
## random_state tambien es un numero entero
qt = QuantileTransformer(n_quantiles=10, random_state=0)
qt.fit_transform(x)

### Tipos de datos en estadisticas
### En estadisticas se manejan tipos de datos categoricos, y tipos de datos numericos.
##  Los tipos de datos categoricos  manejan categorias, los numericos manejan magnitudes
