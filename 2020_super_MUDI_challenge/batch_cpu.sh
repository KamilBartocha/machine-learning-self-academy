#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J CNN1
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=50GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=72:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A mriml 
## Specyfikacja partycji
#SBATCH --partition=plgrid


srun /net/people/plgbartocha/anaconda3/bin/python cnn_main_iso_pro1.py 


