from random import random
import prenosne_funkcije as pf

class Nevronska_mreza:
    def __init__(self, st_vhodni, st_skriti, st_izhod, prenosna_funkcija=pf.sigmoid):
        self.mreza = []
        self.st_vhodni = st_vhodni
        self.st_skriti = st_skriti
        self.st_izhodni = st_izhod
        self.prenosna_funkcija = prenosna_funkcija
        self.mreza.append([{'utezi': [random() for i in range(st_vhodni + 1)]} for j in range(st_skriti)])
        self.mreza.append([{'utezi': [random() for i in range(st_skriti + 1)]} for j in range(st_izhod)])

    def aktivacijska_funkcija(self, utezi, sloj):
        bias = utezi[-1]
        aktivacija = bias
        for i in range(len(utezi) - 1):
            aktivacija += utezi[i] * sloj[i]
        return aktivacija

    def odvod(self, x):
        return self.prenosna_funkcija(x, D=True)

    def razsirjanje_naprej(self, vhod):
        for sloj in self.mreza:
            nov_vhod = []
            for nevron in sloj:
                nevron['izhod'] = self.prenosna_funkcija(self.aktivacijska_funkcija(nevron['utezi'], vhod))
                nov_vhod.append(nevron['izhod'])
            vhod = nov_vhod
        return vhod

    def povratno_razsirjanje(self, pricakovane_vrednosti):
        for i in reversed(range(len(self.mreza))):
            sloj = self.mreza[i]
            napake = []
            if i < len(self.mreza) - 1:
                for j in range(len(sloj)):
                    napaka = 0
                    for nevron in self.mreza[i + 1]:
                        napaka += (nevron['utezi'][j] * nevron['delta'])
                    napake.append(napaka)
            else:
                for j in range(len(sloj)):
                    nevron = sloj[j]
                    napake.append(pricakovane_vrednosti[j] - nevron['izhod'])
            for j in range(len(sloj)):
                nevron = sloj[j]
                nevron['delta'] = napake[j] * self.odvod(nevron['izhod'])

    def posodobitev_utezi(self, vhod, ucni_koeficient):
        for i in range(len(self.mreza)):
            vhod = vhod[:-1]
            if i != 0:
                vhod = [nevron['izhod'] for nevron in self.mreza[i - 1]]
            for nevron in self.mreza[i]:
                for j in range(len(vhod)):
                    nevron['utezi'][j] += ucni_koeficient * nevron['delta'] * vhod[j]
                nevron['utezi'][-1] += ucni_koeficient * nevron['delta']

    def online_ucenje_klasifikacija(self, ucni_podatki, ucni_koeficient, st_epoch, st_izhod):
        for epoch in range(st_epoch):
            napaka = 0
            for podatek in ucni_podatki:
                izhod = self.razsirjanje_naprej(podatek)
                pricakovane_vrednosti = [0 for i in range(st_izhod)]
                pricakovane_vrednosti[podatek[-1]] = 1
                napaka += sum([(ucni_podatki[i][-1] - izhod[i])**2 for i in range(len(pricakovane_vrednosti))])
                self.povratno_razsirjanje(pricakovane_vrednosti)
                self.posodobitev_utezi(podatek, ucni_koeficient)

    def napovej(self, vhod):
        izhod = self.razsirjanje_naprej(vhod)
        return izhod.index(max(izhod))



