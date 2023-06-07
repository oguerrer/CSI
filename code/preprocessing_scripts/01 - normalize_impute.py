import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, warnings
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")


home =  os.getcwd()[:-4]
os.chdir(home)


df = pd.read_csv(home+'Data/Final/KEN/final_KEN.csv')
df = df[[c for c in df.columns if not c.isnumeric() or (c.isnumeric() and int(c)>=2000)]]
df = df[df['Geographical Level']=='National']

colYears = np.array([c for c in df.columns if c.isnumeric()])
years = colYears.astype(float)
years_indices = df.columns.isin(colYears)

min_obs = 10

new_rows = []
for index, row in df.iterrows():
    
    observations = np.where(~row[colYears].isnull())[0]
    missing_values = np.where(row[colYears].isnull())[0]
    new_row = row.values.copy()
               
    vals = row[colYears].values.copy()
    
    if np.sum(vals.astype(str)!='nan')>=min_obs and np.nanmax(vals) != np.nanmin(vals) and len(set(vals)) > 3:
    
        vals = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
    
        x = years[observations]
        y = vals[observations]
        X = x.reshape(-1, 1)
    
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)
    
        x_pred = years.reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        
        vals[missing_values] = y_pred[missing_values]
        new_row[years_indices] = vals
            
        new_rows.append(new_row)
        

    
dff = pd.DataFrame(new_rows, columns=df.columns)
dff.to_csv(home+'Data/Final/KEN/final_KEN_national_normed_imputed.csv', index=False)
    























































