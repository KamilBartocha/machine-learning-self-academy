#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J Isoextraction
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=25GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=1:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A mriml
## Specyfikacja partycji
#SBATCH --partition=plgrid


srun /net/PATH/anaconda3/bin/python extract_iso_pro.py

