# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_excel('dados//processados_GSR.xlsx', sheet_name='Planilha1')

x = df.iloc[:, 3:3003]

#x = x.apply(zscore, axis=1)
x = x.div(x.max(axis=1), axis=0)

x  = x.values

classe = df.iloc[:, 1:2].values
classe = np.reshape(classe, -1)


lda = LinearDiscriminantAnalysis(n_components=2, solver='svd',store_covariance=True, shrinkage=None)
previsores = lda.fit(x, classe).transform(x)

'''
target_names = ['1', '2', '3','4','5','6','7','8']
# X_r2[y == i, 0]  X_r2[y == i, 1] Zero ou um  corresponde ao as linhas da coluna

plt.figure(figsize=(15, 10), dpi=80)
colors = ['red', 'green', 'blue','black', 'brown', 'purple', 'pink', 'yellow']
for color, i, target_name in zip(colors, [1, 2, 3, 4, 5, 6, 7, 8], target_names):
    plt.scatter(previsores[classe == i, 0], previsores[classe == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')
plt.show()
'''
