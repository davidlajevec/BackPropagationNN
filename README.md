# BackPropagationNN
Projekt implementacije nevronske mreže z vzvratnim razširjanjem napake pri predmetu Računalništvo na Fakulteti za matematiko in fiziko v Ljubjani. Nevronska mreža ima en skrit sloj in se uči na način [online](https://en.wikipedia.org/wiki/Online_machine_learning). Kot prenosna funkcija je bil uporabljen **sigmoid**

<h3>Uporaba</h3>
Za uporabo so potrebni moduli: Numpy, Pandas in Scipy.io 
<p>&nbsp;</p>
Mrežo ustvarimo z: 
<code>
mreza = Nevronska_mreza(stevilo_vhodnih_nevronov, stevilo_skritih_nevronov, stevilo_izhodnih_nevronov)
</code>
<p>&nbsp;</p>
Mrežo naučimo na določenem podatkovnem naboru z: 
<code>mreza.ucenje(podatkovni_nabor_ucenje, ucni_koeficient, st_epoch, podatkovni_nabor_test)</code>
<p>&nbsp;</p>
Za napoved glede na vhodni podatek z že naučeno mrežo uporabimo:
<code>mreza.napovej(vhodni_podatek)</code> Pri čemer mora biti vhodni podatek seznam z dolžino 784. 

<h3>Uspešnost</h3>
Algoritem je bil ovrednoten na podatkovnem naboru [MNIST_784](https://www.openml.org/d/554). Pri tem je bil enkrat uporabljena kot prenosna funkcija uporabljen sigmoid drugič pa tanh. Učni koeficient(ang. [learning_rate](https://en.wikipedia.org/wiki/Learning_rate)) je znašal 3,1. Določen je bil z uporabo metode [Trial and error](https://en.wikipedia.org/wiki/Trial_and_error). V skritem sloju smo imeli 30 nevronov. V tabeli so predstavljeni rezultati za vsak epoch posebej. Testni nabor podatkov, ki je bil preverjen po vsakem epochu je zajemal 20% naključno izbranih podatkov.

| epoch | % pravilnih napovedi |   MSE  |
|:-----:|:--------------------:|:------:|
|   1   |         66.86        | 0.4789 |
|   2   |         68.75        | 0.4512 |
|   3   |         69.52        | 0.4418 |
|   4   |         71.15        | 0.4198 |
|   5   |         75.64        | 0.3809 |
|   6   |         77.89        | 0.3517 |
|   7   |         78.34        | 0.3488 |
|   8   |         78.66        | 0.3460 |
|   9   |         79.23        | 0.3407 |
|   10  |         79.31        | 0.3400 |
|   11  |         79.53        | 0.3349 |
|   12  |         79.16        | 0.3454 |
|   13  |         78.92        | 0.3481 |
|   14  |         78.64        | 0.3486 |
|   15  |         78.74        | 0.3443 |
|   16  |         78.84        | 0.3416 |
|   17  |         78.26        | 0.3485 |
|   18  |         77.10        | 0.3593 |
|   19  |         77.35        | 0.3676 |
|   20  |         76.25        | 0.3829 |
 
<h3>Viri</h3>

* [https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)
* [http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent](http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent)
* [https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [https://en.wikipedia.org/wiki/Online_machine_learning](https://en.wikipedia.org/wiki/Online_machine_learning)
* [https://en.wikipedia.org/wiki/Learning_rate](https://en.wikipedia.org/wiki/Learning_rate)
*  

