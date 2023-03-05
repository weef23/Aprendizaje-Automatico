################################ LABORATORIO TRATAMIENTO DE VALORES ######################################
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
#### PASO NUMERO 1 CARGA DE VARIABLES
dataAusentismo = pd.read_csv("Datos/AusentismoPres2011.csv", sep=",")
print(dataAusentismo.info())

### SELECCION DE VARIABLES
data_columns= ["porc_hogares_sin_medios", "alfabetismo", "porc_2_NBI", "IDH", "GINI"]
dataAusentismo = dataAusentismo[data_columns]
print(dataAusentismo.head(3))

### PASO 1 Validar nulos
nulos = dataAusentismo.isnull().sum()/len(dataAusentismo) *100
print(nulos)
imputer_number = SimpleImputer(strategy="mean")
### OBTENEMOS LAS COLUMNAS DEL DATAFRAME
columns_names = dataAusentismo.columns.values
dataAusentismo[columns_names] = imputer_number.fit_transform(dataAusentismo[columns_names])
nulos = dataAusentismo.isnull().sum()/len(dataAusentismo) *100
print(nulos)

### PASO 2 ESCALA DE VALORES
std=StandardScaler()
dataAusentismo_std = std.fit_transform(dataAusentismo.values)
print(dataAusentismo_std)

### PASO 3 OBTENER MATRIZ DE COVARIANZA
cov_mat = np.cov(dataAusentismo_std.T)
## Obtenemos los Autovalores
autovalores, autovectores = np.linalg.eig(cov_mat)
print(np.round(autovalores,2))

##### APLICACION DE UN PCA
pca = PCA(n_components=2, random_state=20)
dataAusentismo_std_pca = pca.fit_transform(dataAusentismo_std)
x_train_pca_df = pd.DataFrame(dataAusentismo_std_pca,
                             columns=["PC1","PC2"])

print(x_train_pca_df.head())

### Aplicando Analisis Factorial
factor = FactorAnalysis(n_components=2, random_state=20)
fa = FactorAnalysis(n_components=2)
x_train_fa = fa.fit_transform(dataAusentismo_std)
x_train_fa_df = pd.DataFrame(x_train_fa, columns=["FA1", "FA2"])
x_train_fa_df.corr(method="pearson")
print(x_train_fa_df.head())