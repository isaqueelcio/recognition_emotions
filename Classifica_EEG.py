# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import pickle



dfeeg = pd.read_excel('Amostras//amostra_ID1_VD2_UI4.xlsx', sheet_name = 'Sheet1')

#####Delta
lda_deltaa = pickle.load(open('classificador//lda//lda_delta3.sav', 'rb'))
delta_classificador = pickle.load(open('classificador//class//delta_classificador3.sav', 'rb'))
#####Final Delta

#####highAlpha
lda_highAlphaa = pickle.load(open('classificador//lda//lda_highAlpha3.sav', 'rb'))
highAlpha_classificador = pickle.load(open('classificador//class//highAlpha_classificador3.sav', 'rb'))
#####Final highAlpha

########highBeta
lda_highBetaa = pickle.load(open('classificador//lda//lda_highBeta3.sav', 'rb'))
highBeta_classificador = pickle.load(open('classificador//class//highBeta_classificador3.sav', 'rb'))

#######Fim highBeta

###########lda_lowAlpha
lda_lowAlphaa = pickle.load(open('classificador//lda//lda_lowAlpha3.sav', 'rb'))
lowAlpha_classificador = pickle.load(open('classificador//class//lowAlpha_classificador3.sav', 'rb'))

##########lda_lowAlpha fim

#################lowBeta
lda_lowBetaa = pickle.load(open('classificador//lda//lda_lowBeta3.sav', 'rb'))
lowBeta_classificador = pickle.load(open('classificador//class//lowBeta_classificador3.sav', 'rb'))
#################lowBeta fim

#################lowGamma
lda_lowGamma = pickle.load(open('classificador//lda//lda_lowGamma3.sav', 'rb'))
lowGamma_classificador = pickle.load(open('classificador//class//lowGamma_classificador3.sav', 'rb'))
#################lowGamma

################ midGamma
lda_midGamma = pickle.load(open('classificador//lda//lda_midGamma3.sav', 'rb'))
midGamma_classificador = pickle.load(open('classificador//class//midGamma_classificador3.sav', 'rb'))
################# fim midGamma 

################ Theta
lda_theta = pickle.load(open('classificador//lda//lda_theta3.sav', 'rb'))
theta_classificador = pickle.load(open('classificador//class//theta_classificador3.sav', 'rb'))
################ Fim Theta 




sinaldeltat = dfeeg.iloc[3:4, 2:]

tamanhoamostra = sinaldeltat.size

dfeeg = dfeeg.iloc[:, 2:]
#800 ou 3000
aa = 3000
i = 0
print("\nTESTE EEG INICIADO\n")
while aa <= tamanhoamostra:
    #Delta
    sinaldelta = dfeeg.iloc[3:4, i:aa]
    sinaldelta = sinaldelta.div(sinaldelta.max(axis=1), axis=0)
    sinaldelta = sinaldelta.values
    sinaldelta = sinaldelta.astype(float)
    
    delta = lda_deltaa.transform(sinaldelta)
    previsoesdelta = delta_classificador.predict(delta)
    #Final delta
    
    sinalhighAlpha = dfeeg.iloc[6:7, i:aa]
    sinalhighAlpha = sinalhighAlpha.div(sinalhighAlpha.max(axis=1), axis=0)
    sinalhighAlpha = sinalhighAlpha.values
    sinalhighAlpha = sinalhighAlpha.astype(float)
    
    highAlpha = lda_highAlphaa.transform(sinalhighAlpha)
    previsoeshighAlpha = highAlpha_classificador.predict(highAlpha)
    
    #highBeta
    sinalhighBeta = dfeeg.iloc[8:9, i:aa]
    sinalhighBeta = sinalhighBeta.div(sinalhighBeta.max(axis=1), axis=0)
    sinalhighBeta = sinalhighBeta.values
    sinalhighBeta = sinalhighBeta.astype(float)
    
    highBeta = lda_highBetaa.transform(sinalhighBeta)
    previsoeshighBeta = highBeta_classificador.predict(highBeta)
    #highBeta fim
    
    #lowAlpha
    sinallowAlpha = dfeeg.iloc[5:6, i:aa]
    sinallowAlpha = sinallowAlpha.div(sinallowAlpha.max(axis=1), axis=0)
    sinallowAlpha = sinallowAlpha.values
    sinallowAlpha = sinallowAlpha.astype(float)
    
    lowAlpha = lda_lowAlphaa.transform(sinallowAlpha)
    previsoeslowAlpha = lowAlpha_classificador.predict(lowAlpha)
    
    #lowAlpha fim
    
    #lowBeta
    sinallowBeta = dfeeg.iloc[7:8, i:aa]
    sinallowBeta = sinallowBeta.div(sinallowBeta.max(axis=1), axis=0)
    sinallowBeta = sinallowBeta.values
    sinallowBeta = sinallowBeta.astype(float)
    
    lowBeta = lda_lowBetaa.transform(sinallowBeta)
    previsoeslowBeta = lowBeta_classificador.predict(lowBeta)
    
    #lowBeta fim
    
    ##lowGamma
    sinallowGamma = dfeeg.iloc[9:10, i:aa]
    sinallowGamma = sinallowGamma.div(sinallowGamma.max(axis=1), axis=0)
    sinallowGamma = sinallowGamma.values
    sinallowGamma = sinallowGamma.astype(float)
    
    lowGamma = lda_lowGamma.transform(sinallowGamma)
    previsoeslowGamma = lowGamma_classificador.predict(lowGamma)
    
    ##lowGamma fim
    
    # midGamma
    sinalmidGamma = dfeeg.iloc[10:11, i:aa]
    sinalmidGamma = sinalmidGamma.div(sinalmidGamma.max(axis=1), axis=0)
    sinalmidGamma = sinalmidGamma.values
    sinalmidGamma = sinalmidGamma.astype(float)
    
    midGamma = lda_midGamma.transform(sinalmidGamma)
    previsoesmidGamma = midGamma_classificador.predict(lowGamma)
    
    # midGamma fim 
    
    # Theta
    sinaltheta = dfeeg.iloc[4:5, i:aa]
    sinaltheta = sinaltheta.div(sinaltheta.max(axis=1), axis=0)
    sinaltheta = sinaltheta.values
    sinaltheta = sinaltheta.astype(float)
    
    theta = lda_theta.transform(sinaltheta)
    previsoestheta = midGamma_classificador.predict(theta)
    
    # Theta fim 

             
    
    total = (previsoesdelta + previsoeshighAlpha + previsoeshighBeta + previsoeslowAlpha + previsoeslowBeta + previsoeslowGamma + 
             previsoesmidGamma + previsoestheta) / 8
    
    print("delta", previsoesdelta ,"highAlpha", previsoeshighAlpha, "highBeta", previsoeshighBeta, "lowAlpha", previsoeslowAlpha,"lowBeta", previsoeslowBeta
          , "lowGamma", previsoeslowGamma, "midGamma", previsoesmidGamma, "theta", previsoestheta)
    
    #print("Total", total)
    
    i= i+500
    aa = aa+500
    total = 0
    
print("\nTESTE EEG FIM\n")

################ GSR
'''
lda_gsr = pickle.load(open('classificador//lda//lda_gsr.sav', 'rb'))
gsr_classificador = pickle.load(open('classificador//class//gsr_classificador3.sav', 'rb'))
'''
lda_gsr = pickle.load(open('classificador//lda_gsr.sav', 'rb'))
gsr_classificador = pickle.load(open('classificador//gsr_classificador.sav', 'rb'))


################ Fim GSR 


#800 ou 3000


gsraa = 3000
gsri = 0



print("\n Teste GSR inicio")

while gsraa <= tamanhoamostra:
    #GSR
    sinalgsr = dfeeg.iloc[13:14, gsri:gsraa]
    
    #sinalgsr = sinalgsr.div(sinalgsr.max(axis=1), axis=0)
    
    sinalgsr = sinalgsr.values
    sinalgsr = sinalgsr.astype(float)
    
    gsr = lda_gsr.transform(sinalgsr)
    previsoesgsr = gsr_classificador.predict(gsr)
    #Final GSR
    
    print("GSR", previsoesgsr)
    
    gsri= gsri+500
    gsraa = gsraa+500
    total = 0


print("\n Teste GSR Fim")
    
lda_ecg = pickle.load(open('classificador//lda_ecg.sav', 'rb'))
ecg_classificador = pickle.load(open('classificador//ecg_classificador.sav', 'rb'))

ecgaa = 3000
ecgi = 0



print("\n Teste GSR inicio")

while ecgaa <= tamanhoamostra:
    #GSR
    sinalecg = dfeeg.iloc[14:15, gsri:gsraa]
    
    #sinalgsr = sinalgsr.div(sinalgsr.max(axis=1), axis=0)
    
    sinalecg = sinalecg.values
    sinalecg = sinalecg.astype(float)
    
    ecg = lda_ecg.transform(sinalecg)
    previsoesecg = gsr_classificador.predict(ecg)
    #Final GSR
    
    print("GSR", previsoesdelta)
    
    gsri= gsri+500
    gsraa = gsraa+500
    total = 0


print("\n Teste GSR Fim")





 