from scipy.io import arff
import pandas as pd
import numpy as np
import prenosne_funkcije as pf
from nevronska_mreza import Nevronska_mreza

podatki = arff.loadarff('iris.arff')[0]
#podatki = arff.loadarff('breast.arff')[0]
#podatki = arff.loadarff('diabetes.arff')[0]
podatki = pd.DataFrame(podatki)
podatki = np.asarray(podatki)

slovar =  {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
#slovar =  {b'tested_negative':0, b'tested_positive':1}
#slovar = {b'0':0, b'1':1, b'2':2, b'3':3, b'4':4, b'5':5, b'6':6, b'7':7, b'8':8, b'9':9}
podatki = podatki.T
prevedeno = np.copy(podatki[-1])
for k, v in slovar.items(): prevedeno[podatki[-1]==k] = v
podatki[-1] = prevedeno
podatki = podatki.T

eta_train_test = 0.3
np.random.shuffle(podatki)
test, train =  np.split(podatki, [int(len(podatki)*eta_train_test)])
print(slovar.items())
moja_mreza = Nevronska_mreza(4, 5, 3, prenosna_funkcija=pf.ReLU)
moja_mreza.ucenje(train, 0.001, 20, test)

