from random import *
import operator as op
from collections import Counter
import csv


def dist_euclidienne(p,q):
    somme = 0
    for i in range(len(p)):
        somme += (p[i]-q[i])**2
    return sqrt(somme)
    

def dist_minkowski(p,q,n):
    somme = 0
    for i in range(len(p)):
        somme += (abs(p[i]-q[i]))**n
    if (i == 3) : somme += 20*((abs(p[i]-q[i]))**n)
    #We rise the value added by the features n°3, giving less importance. 
    return somme**(1/n)


def Mean(LISTE):
    nb_vrb = len(LISTE[0])
    liste_mean = []
    for vrb in range(nb_vrb):
        moyenne = 0
        for i in range(len(LISTE)): moyenne += LISTE[i][vrb]
        moyenne /= len(LISTE)
        liste_mean.append(moyenne)
    return liste_mean
        
    
class KNN():
    
    def __init__(self, Individus, Labels, k=3):
        self.Individus = Individus # Corresponding to vector X
        self.Labels = Labels # Corresponding to vector Y
        self.k = k
        label_names = set(self.Labels) #All different labels
        
        self.Label_dico = {}
        compteur = 0
        for i in label_names: 
            self.Label_dico[i] = compteur
            compteur += 1
    
    
    def prediction(self, X):
        return [self.algorithm_euc(x) for x in X]
    
    
    def prediction_mkwsk(self, X):
        n = 4
        while(n % 2 != 0 and n <= 0):
            n = eval(input("Choose n : Minkowski division order \n"))
        return [self.algorithm_minkowski(x,n) for x in X]
    
    
    def algorithm_euc(self, x):
        distance = [dist_euclidienne(x, elm) for elm in self.Individus]
        distance = enumerate(distance)
        distance = sorted(distance,key=op.itemgetter(1))
        indice = [elm[0] for elm in distance[1:self.k+1]]
        voisins = [self.Labels[elm] for elm in indice]
        return Counter(voisins).most_common(1)[0][0]
    
    
    def algorithm_minkowski(self, x, n = -1):
        if (n == -1):
            while(n % 2 != 0 and n <= 0):
                n = eval(input("Choose n : Minkowski division order \n"))       
        distance = [dist_minkowski(x, elm, n) for elm in self.Individus]
        distance = enumerate(distance)
        # enumerate create a tuple of each element of the list with the associate index 
        distance = sorted(distance,key=op.itemgetter(1))
        # Distance are sorted in ascending order
        indice = [elm[0] for elm in distance[0:self.k]]
        # distance[1:self.k+1] to ignore the first element
        # which corresponds to itself
        voisins = [self.Labels[elm] for elm in indice]
        return Counter(voisins).most_common(1)[0][0]
    
        
    def confusion_matrix(self):
        taille = len(self.Label_dico)
        somme = 0
        matrix_Cfsn = [[0 for i in range(taille)] 
                       for j in range(taille)]
# Choose minkwoski or euclidienne prediction
        Liste_predictions = self.prediction_mkwsk(self.Individus)
        
        for i in range(len(self.Labels)):
            ligne_reelle = self.Label_dico[self.Labels[i]]
            colonne_estimée = self.Label_dico[Liste_predictions[i]]
            matrix_Cfsn[ligne_reelle][colonne_estimée] += 1
        
        for i in range(taille):
            somme += matrix_Cfsn[i][i] #diagonals sum
        precision = somme/len(self.Labels)
            
        print("\t          Real Label / Predicted Label ")
        for cle in self.Label_dico.keys():
            print("%13s" % cle, end=" ")
        print("")
        for ligne in matrix_Cfsn : 
            for colonne in ligne :
                print("%12i" % colonne,end=" ")
            print("")
        print("\nAccuracy = ", precision)
        
        
    def Avg_prm(self):
        l = []
        for label in self.Label_dico:
            first = self.Labels.index(label)
            last = len(self.Labels) - self.Labels[::-1].index(label)
            M = self.Individus[first:last]
            l.append(Mean(M).append(label))
            
           
        
# Function to split the dataset in two parts        
def separation(echantillon, label):
    
    connu = set(label)            
    echantillon_train = []
    label_train = []
    
    n = eval(input("Choose a percentage for the training set : \n"))
    n /= 100
    
    for elm in connu :
        first = label.index(elm)
        last = len(label) - label[::-1].index(elm)
        nombre = last - first
        L = [i for i in range(first,last)]
        sep = sample(L,int(n*nombre))    
        for i in sep:
            echantillon_train.append(echantillon[i])
            label_train.append(label[i])
    
    return echantillon_train, label_train