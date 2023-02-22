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
data["cqi_cuartil"]= cuartil.fit_transform(data[["cqi"]])
print(data.head(3))