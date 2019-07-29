# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_excel('dadosecg_02_1020pronto.xlsx', sheet_name='Planilha1')

x = df.iloc[:, 3:1023]

#x.to_excel("delta.xlsx") 

#x = x.apply(zscore, axis=1)
#x = x.div(x.max(axis=1), axis=0)

x = x.values

classe = df.iloc[:, 1:2].values
classe = np.reshape(classe, -1)


lda_ecg = LinearDiscriminantAnalysis(n_components=2, solver='svd',store_covariance=True, shrinkage=None)
previsores = lda_ecg.fit(x, classe).transform(x)

pickle.dump(lda_ecg, open('lda_ecg.sav', 'wb'))


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

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.001, random_state=0)

'''
from sklearn.svm import SVC
gsr_classificador3 = SVC(kernel = 'rbf', random_state = 1, C = 3.0)

gsr_classificador3.fit(previsores_treinamento, classe_treinamento)
previsoes = gsr_classificador3.predict(previsores_teste)
'''

from sklearn.naive_bayes import GaussianNB
ecg_classificador = GaussianNB()
ecg_classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = ecg_classificador.predict(previsores_teste)


pickle.dump(ecg_classificador, open('ecg_classificador.sav', 'wb'))


from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)



dfeeg = pd.read_excel('Amostras//amostra_ID2_VD1_UI3.xlsx', sheet_name = 'Sheet1')


sinaldeltat = dfeeg.iloc[3:4, :]

tamanhoamostra = sinaldeltat.size


aa = 3002
i = 2
print("\nTESTE EEG INICIADO\n")
while aa <= tamanhoamostra :
    
    sinaldelta = dfeeg.iloc[3:4, i:aa]
    sinaldelta = sinaldelta.div(sinaldelta.max(axis=1), axis=0)
    sinaldelta = sinaldelta.values
    sinaldelta = sinaldelta.astype(float)
    
    delta = lda_delta3.transform(sinaldelta)
    previsoesdelta = delta_classificador.predict(delta)
    print("delta", previsoesdelta)
    
    i= i+500
    aa = aa+500
    
print("\nTESTE EEG FIM\n")




