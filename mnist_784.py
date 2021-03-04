from scipy.io import arff
import pandas as pd
import numpy as np
from nevronska_mreza import Nevronska_mreza

podatki = arff.loadarff('mnist_784.arff')[0]
podatki = pd.DataFrame(podatki)
podatki = np.asarray(podatki)

slovar = {b'0':0, b'1':1, b'2':2, b'3':3, b'4':4, b'5':5, b'6':6, b'7':7, b'8':8, b'9':9}
podatki = podatki.T
prevedeno = np.copy(podatki[-1])
for k, v in slovar.items(): prevedeno[podatki[-1]==k] = v
podatki[-1] = prevedeno
podatki = podatki.T

eta_train_test = 0.2
np.random.shuffle(podatki)
podatki, po = np.split(podatki, [int(len(podatki)*0.01)])
test, train =  np.split(podatki, [int(len(podatki)*eta_train_test)])

moja_mreza = Nevronska_mreza(784, 30, 10)
moja_mreza.ucenje(train, 1, 30, test)

