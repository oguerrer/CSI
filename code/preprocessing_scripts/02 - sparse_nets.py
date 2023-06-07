import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
import scipy.stats as st
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

warnings.simplefilter("ignore")
sparsebn = importr('sparsebn')

home =  os.getcwd()[:-4]
os.chdir(home+'/code/')


# ## Installing R packages (optional)
# import rpy2.robjects.packages as rpackages
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)
# packnames = ('sparsebnUtils')
# from rpy2.robjects.vectors import StrVector
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))






df = pd.read_csv(home+'Data/Final/KEN/final_KEN_national_normed_imputed.csv')
colYears = np.array([c for c in df.columns if c.isnumeric()])

S = (df[colYears].values[:,1::] - df[colYears].values[:,0:-1]).T

nr, nc = S.shape
Sr = robjects.r.matrix(S, nrow=nr, ncol=nc)
robjects.r.assign("S", Sr)

robjects.r('''
    data <- sparsebnData(S, type = "continuous")
    dags.estimate <- estimate.dag(data)
    dags.param <- estimate.parameters(dags.estimate, data=data)
    selected.lambda <- select.parameter(dags.estimate, data=data)
    dags.final.net <- dags.estimate[[selected.lambda]]
    dags.final.param <- dags.param[[selected.lambda]]
    adjMatrix <- as(dags.final.param$coefs, "matrix")
    ''')
    
A = robjects.globalenv['adjMatrix']

edges = []
for i, rowi in df.iterrows():
    for j, rowj in df.iterrows():
        if A[i,j] != 0:
            edges.append((i, j, A[i,j]))

dff = pd.DataFrame(edges, columns=['From', 'To', 'Weight'])   
dff.to_csv(home+'Data/Final/KEN/network_KEN_national.csv', index=False)


 
 

 

 

 

 

 

 

