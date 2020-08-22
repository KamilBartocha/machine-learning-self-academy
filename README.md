# Intern allegro

Allegro 2020 internship task. Allegro xsummer experience

Twoim zadaniem jest wytrenowanie klasyfikatora binarnego na podzbiorze zbioru MNIST, w którym wyróżniamy klasy (cyfry 0 i 1 mają zostać wyłączone ze zbioru)
- Liczby pierwsze (2,3,5,7)
- Liczby złożone (4,6,8,9)

Napisz wydajną implementację modelu **regresji logistycznej** trenowanego algorytmem ***SGD z momentum***. Cały proces trenowania musisz napisać samodzielnie, 
w języku Python, korzystając z biblioteki numpy. Na potrzeby zadania niedozwolone jest korzystanie z gotowych implementacji optimizerów i modeli oraz bibliotek 
do automatycznego różniczkowania funkcji (np. Tensorflow, pytorch, autograd).
Dobierz hiperparametry tak, aby uzyskać jak najlepszy wynik na zbiorze walidacyjnym.
Wyciągnij i zapisz wnioski z przeprowadzonych eksperymentów. 


Kryteria jakie przyjęliśmy przy ocenianiu zadań to: wybór funkcji kosztu oraz poprawność wyliczania gradientów, poprawność generowania zbioru danych, uwzględnienie 
momentum w SGD, wydajność implementacji (notacja wektorowa), czytelność oraz ustrukturyzowane kodu, no i finalnie - to jak wyczerpująca była analiza 
przeprowadzonych eksperymentów.
