from scipy.io import arff
import pandas as pd
import numpy as np
from nevronska_mreza import Nevronska_mreza
from sklearn.model_selection import train_test_split

data = arff.loadarff('mnist_784.arff')
data = pd.DataFrame(data[0])
data = np.asarray(data)

slovar = {b'0':0, b'1':1, b'2':2, b'3':3, b'4':4, b'5':5, b'6':6, b'7':7, b'8':8, b'9':9}
data = data.T
prevedeno = np.copy(data[-1])
for k, v in slovar.items(): prevedeno[data[-1]==k] = v
data[-1] = prevedeno
data = data.T

train, test = train_test_split(data, test_size=0.2)

moja_mreza = Nevronska_mreza(784, 30, 10)

moja_mreza.ucenje(train, 4.9, 30, test)

#test_set_dolzina = len(test)
#pravilno = 0
#stanje = 1
#for podatek in test:
#    napoved = moja_mreza.napovej(podatek)
#    rezultat = podatek[-1]
#    if napoved == rezultat:
#        pravilno += 1
#    print('Napoved: {0}, Rezultat: {1}, Natančnost: {2}, Stanje: {3}/{4}'.format(
#        napoved, rezultat, (pravilno/test_set_dolzina), stanje, test_set_dolzina))
#    stanje += 1



