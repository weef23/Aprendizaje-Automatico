#################### Analisis de datos con Numpy ######################################
### Numpy es la libreria mas importante utilizada para trabajar con algebra lineal
import numpy as np

### creacion de un arreglo usando numpy
lista = [1, 2, 3, 4]

## Una de las ventajas que tiene numpy es que podemos crear un arreglo
## desde casi cualquier estructura de datos
arreglo = np.array(lista)
print(arreglo)
## Matriz multidimensional con numpy
matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matriznp = np.array(matriz)
print(matriz)
## Podemos definir un arreglo tambien en un rango
rango = np.arange(0, 10) ## arreglo con valores entre 0 y 10
print(rango)
