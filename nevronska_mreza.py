from random import random
import numpy as np
import prenosne_funkcije as pf

class Nevronska_mreza:
    def __init__(self, st_vhodni, st_skriti, st_izhodni, prenosna_funkcija=pf.sigmoid):
        self.st_vhodni = st_vhodni
        self.st_skriti = st_skriti
        self.st_izhodni = st_izhodni
        self.prenosna_funkcija = prenosna_funkcija
        self.sloj1_utezi = np.array([[random() for i in range(st_vhodni+1)] for j in range(st_skriti)])
        self.sloj2_utezi = np.array([[random() for i in range(st_skriti+1)] for j in range(st_izhodni)])
        self.sloj1_delta = np.array([random() for j in range(st_skriti)])
        self.sloj2_delta = np.array([random() for j in range(st_izhodni)])
        self.izhod = np.zeros(st_izhodni)
        self.skriti = np.zeros(st_skriti)
        self.vhod = np.zeros(st_vhodni)

    def aktivacijska_funkcija(self, utezi, sloj):
        bias = utezi[-1]
        return np.dot(utezi[:-1], sloj) + bias

    def prenos_aktivacija(self, i, sloj, st_sloj):
        if st_sloj == 1:
            return self.prenosna_funkcija(self.aktivacijska_funkcija(self.sloj1_utezi[i], sloj))
        else:
            return self.prenosna_funkcija(self.aktivacijska_funkcija(self.sloj2_utezi[i], sloj))

    def odvod(self, x):
        return self.prenosna_funkcija(x, D=True)

    def razsirjanje_naprej(self, vhod):
        self.vhod = np.array(vhod)
        self.skriti = np.array([self.prenos_aktivacija(i, self.vhod, 1) for i in range(self.st_skriti)])
        self.izhod = np.array([self.prenos_aktivacija(i, self.skriti, 2) for i in range(self.st_izhodni)])
        return self.izhod

    def povratno_razsirjanje(self, pricakovane_vrednosti):
        self.sloj2_delta = (np.array(pricakovane_vrednosti) - np.array(self.izhod)) * self.odvod(self.izhod)
        matrika_delta = np.array([self.sloj2_delta for i in range(self.st_skriti)])
        self.sloj1_delta = np.dot(matrika_delta, self.sloj2_utezi.T[:-1].T).sum(axis=0)

    def posodobitev_utezi(self, podatek, ucni_koeficient):
        self.vhod=podatek[:-1]
        utezi1 = self.sloj1_utezi.T[:-1].T + ucni_koeficient *  np.tensordot(self.sloj1_delta, self.vhod, axes=0)
        bias1 = self.sloj1_utezi.T[-1] + ucni_koeficient * self.sloj1_delta
        utezi2 = self.sloj2_utezi.T[:-1].T + ucni_koeficient * np.tensordot(self.sloj2_delta, self.skriti, axes=0)
        bias2 = self.sloj2_utezi.T[-1] + ucni_koeficient * self.sloj2_delta
        self.sloj1_utezi = np.concatenate((utezi1.T, np.array([bias1])), axis=0).T
        self.sloj2_utezi = np.concatenate((utezi2.T, np.array([bias2])), axis=0).T

    def ucenje(self, ucni_podatki, ucni_koeficient, st_epoch, test):
        for epoch in range(st_epoch):
            napaka = 0
            for podatek in ucni_podatki:
                izhod = self.razsirjanje_naprej(podatek[:-1])
                pricakovane_vrednosti = np.zeros(self.st_izhodni)
                pricakovane_vrednosti[podatek[-1]] = 1
                napaka += np.sum((pricakovane_vrednosti - izhod)**2)
                self.povratno_razsirjanje(pricakovane_vrednosti)
                self.posodobitev_utezi(podatek, ucni_koeficient)
                pravilno = 0
                for podatek_t in test:
                    napoved = self.napovej(podatek_t)
                    rezultat = podatek_t[-1]
                    if napoved == rezultat:
                        pravilno += 1
            print('>epoch=%d, natančnost=%d, kvadratna_napaka=%.3f' % (epoch, pravilno/len(test), napaka))

    def napovej(self, vhod):
        izhod = list(self.razsirjanje_naprej(vhod[:-1]))
        return izhod.index(max(izhod))

