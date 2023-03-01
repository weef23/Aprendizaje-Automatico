################################ LABORATORIO TRATAMIENTO DE VALORES ######################################
import pandas as pd
import numpy as np

#### PASO NUMERO 1 CARGA DE VARIABLES
dataAusentismo = pd.read_spss("Datos/AusentismoPres2011.sav")
print(dataAusentismo.info())

### SELECCION DE VARIABLES
data_columns= ["porc_hogares_sin_medios", "alfabetismo", "porc_2_NBI", "IDH", "GINI"]
dataAusentismo = dataAusentismo[data_columns]
print(dataAusentismo.head(3))