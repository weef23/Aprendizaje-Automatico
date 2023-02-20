import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### El tratamiento de los datos Atipicos es muy importante ya que puede afectar procesos como el de clusterizacion
### Es por ello que se debe aplicar un tratamiento a esos datos
data_credit = pd.read_csv("./Datos/data_credit.csv", sep=",", encoding="ISO-8859-1")
print(data_credit.head(5))

sns.boxplot(x=data_credit.camt.values)
plt.show()

### IRQ Score
### La funcion quantile nos devuelve cada uno de los cuartiles
Q1 = data_credit["camt"].quantile(0.25)
print(f"Primer Cuartil es {Q1}")

Q2 = data_credit["camt"].quantile(0.50)
print(f"Primer Cuartil es {Q2}")

Q3 = data_credit["camt"].quantile(0.75)
print(f"Primer Cuartil es {Q3}")

### Simetria
der = Q3 - Q2
izq = Q2 - Q1
print(f"La simetria por al izquierda es {izq} la simetria por la derecha es {der}")
#Rango Intercuartilico
IRQ = Q3 - Q1
print(f"El rango intercuartilico es {IRQ}")

############### Deteccion de Atipicos #############################
noutizq = sum(data_credit.camt.value < (Q1 - (1.5*IRQ)))
derizq = sum(data_credit.camt.value < (Q3 + (1.5*IRQ)))

########### Utilizando el Z-Score para la deteccion de Anomalias
from scipy import  stats

zscore = np.abs(stats.zscore(data_credit.camt.value))
k = 3 ## Este criterio es un estandar
print(np.where(zscore>=k)[0])

#df_outlier =

