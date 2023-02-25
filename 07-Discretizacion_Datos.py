################# LA DISCRETIZACION CONSISTE EN CONVERTIR VARIABLES CONTINUAS EN DISCRETAS ############################
################# ESTO ES NECESARION EN MODELOS COMO NAIVE BAYES O ARBOLES ############################################
from sklearn.preprocessing import KBinsDiscretizer
import math as m
import pandas as pd


import numpy as np

#### EXISTEN DISTINTOS TIPOS DE DISCRETIZACION DE LOS DATOS
data = pd.read_csv("Datos/data.csv") ## Cargamos los datos
print(data.head(3))
### EL primer paso es eliminar la columna sin nombre ya que nos sirve para nada
data.drop(columns="Unnamed: 0", axis=1, inplace=True)
print(data.head(3))
#### UNA FORMA INTERESANTE DE HACER LA DISCRETIZACION ES MENDIANTE EL RANGO DE UNA VARIABLE
### EJEMPLO TOMAMOS LA VARIABLE CQI Y CALCULAMOS EL RANGO
minimo = data["cqi"].min() ### Valor minimo
maximo = data["cqi"].max() ### Valor maximo
rango = maximo - minimo ### Rango de variables
print(f"El minimo es {minimo} , el maximo es {maximo} el rango es {rango}")
### SUPONGAMOS QUE QUEREMOS DISCRETIZAR LA VARIABLE EN 4 CATEGORIAS
interv = rango/4 + minimo
### CATEGORIA 1
c1inf = minimo
c1sup = interv
### Categoria 2
interv = 2*rango/4 + minimo
c2inf = c1sup
c2sup = interv
## Categoria 3
interv = 3*rango/4 + minimo
c3inf =c2sup
c3sup = interv
## Categoria 4
c4inf = c3sup
c4sup = maximo
#######################################################################################################################
print(f"La categoria 1 <--- {c1inf} - {c1sup}")
print(f"La categoria 2 <--- {c2inf} - {c2sup}")
print(f"La categoria 3 <--- {c3inf} - {c3sup}")
print(f"La categoria 4 <--- {c4inf} - {c4sup}")
#### EN EL METODO ANTERIOR USAMOS UN METODO DE RANGOS DE IGUAL AMPLITUD DE TAL FORMA QUE CADA UNO TENGA IGUAL AMPLITUD
############# DETERMINAR EL NUMERO DE CATEGORIAS #################################################
#### EXISTE UN TECNICA LLAMADA REGLA DE STURGES LA CUAL SIGUE LA SIGUIENTE FORMULA  1 + log2(n)
sturges = round(1 + m.log2(len(data)), 0);
print(f"El numero de categorias es {sturges}")
amplitud = rango/sturges
print(f"La amplitud para cada categoria es {amplitud}")
### NOTA SI NOSOTROS TENEMOS VALORES ATIPICOS NO SE RECOMIENDA APLICAR
##### USO DE KBINSDISCRETIZER

#### PARAMETROS n_bins número de categorías de la variable, encode #son de naturaleza ordinal,
### Estrategia  uniform= discretización por intervalos de igual amplitud
amplitud = KBinsDiscretizer(n_bins=12, encode="ordinal", strategy="uniform")
### Anteriormente configuramos los parametros del KBinDiscretizer
data["cqi_inter_amplitud"] = amplitud.fit_transform(data[["cqi"]])
print(data.head(3))
### TAMBIEN EXISTE EL METODO DE DISCRETIZACION USANDO QUARTILES, EN ESTE CASO TOMAMOS COMO REFERENCIA
### LOS CUARTILES PARA LA CREACION DE LAS CATEGORIAS, EN ESTE CASO EL NUMERO DE BINES SERA 4
### Y LA ESTRATEGIA ES QUANTILE
cuartil = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
data["cqi_cuartil"] = cuartil.fit_transform(data[["cqi"]])
print(data.head(3))
####### DISCRETIZACION POR KMEAS, CONSISTE EN USAR EL ALGORITMO DE KMEAS PARA DISCRETIZAR DE DATOS #####################
##### ESTA TECNINCA UTILIZA EL AGRUPAMIENTO PARA MEDIANTE LA CREACION DE CENTRO SELECCIONAR Y AGRUPARA AQUELLOS ########
##### DATOS MAS CERCANOS ENTRE SI

#### VEAMOS EL SIGUIENTE EJEMPLO, EN LA SIGUIENTE LISTA TENEMOS UNA LISTA DE EDADES QUE CONVERIMOS EN UN DATAFRAME
datos = [23, 25, 22, 29, 36, 41, 38, 18]
df = pd.DataFrame(datos, columns=["edad"])
#### La Clase KBinsDiscretizer soporta la discretiacion por kmeans en este caso crearemos dos categorias.
kmeans = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="kmeans")
## LE SUMAMOS 1 PORQUE PYTHON EMPIEZA A ENUMERAR LAS CATEGORIAS A PARTIR DE 0
### Kmeans tambien se ve afectado por los valores atipicos
df["edad_kmeans"] = kmeans.fit_transform(df[["edad"]])+1
print(df.head(3))
data["cqi_kmeans"] = kmeans.fit_transform(data[["cqi"]])
print(data.head(3))
############## DISCRETIZACION POR ENTROPIA ################################################################
############## LA DISCRETIZACION POR ENTROPIA ESTA BASADA EN ARBOLES DE DECISION ##########################
###### EL PRIMER PASO PARA LA DISCRETIZACION POR ENTROPIA ES EL SIGUIENTE #################################
r = 4000
nr = 6000
t = 10000
### LA IMPUREZA SE CALCULA EN BASE AL NUMERO DE RECLAMOS Y AL NUMERO DE NO RECLAMOS
impureza= abs(((r/t)*m.log(r/t))+((nr/t)*m.log(nr/t)))
print(impureza)
#### VEMOS QUE LA IMPUREZA INICIAL ES 0.67 AHORA TRATAMOS DE REDUCIR LA IMPUREZA USANDO UN PUNTO DE CORTE
#### LA IMPUREZA SE CALCULA USANDO LA FOMULA ANTERIOR PERO COMO ESTOS SON DATOS SIMULADOS VAMOS SUPONER LO SIGUIENTE
#### La impureza inicial del árbol es 0.673 y la impureza final usando el punto de corte "30 años" es 0.672
#### La impureza inicial del árbol es 0.673 y la impureza final usando el punto de corte "38 años" es 0.35
#### La impureza inicial del árbol es 0.673 y la impureza final usando el punto de corte "40 años" es 0.27

#### EN ESTE CASO TOMAMOS EL PUNTO DE CORTE CON MENOR IMPUREZA SIEMPRE, A CONTINUACION SE MUESTRA LA FORMA EN QUE
### SE CALCULA LA IMPUREZA FINAL USANDO EL PUNTO DE CORTE

########## AHORA SUPONEMOS QUE TOMAMOS EL PUNTO DE CORTE DE 30
t = 3500 ## El total de clientes de 18 a 30 años es 3500
r = 1500 ## De estos 3500 solamente 1500 son los que reclaman
nr = 2000 ## No reclaman

## La impureza debe ser ponderada para ello dividimos el total de esta categoria y por el total de los datos
## eso lo multiplicamos por la impureza y nos da la impureza real
impureza1 = abs(((r/t)*m.log(r/t))+((nr/t)*m.log(nr/t)))
print(impureza * 0.35)

### Hacemos lo mismo con la otra clase
t=6500
r=2500
nr=4000

impureza2=abs(((r/t)*m.log(r/t))+((nr/t)*m.log(nr/t)));

print(impureza2 * 0.65)
## La suma de ambas nos da la impureza ponderada
suma_i=(impureza1*0.35)+(impureza2*0.65);
print(suma_i)

import os
print(os.getcwd())