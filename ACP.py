from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def z_score(X: np.array):
    """ Normalize the dataset """
    means = np.mean(X, axis=0)
    sds = np.std(X, axis=0)
    X = (X - means) / sds
    X[np.isnan(X)] = 0
    return X


def composantes_calculus(X: np.array, s, vect_p):
    """Calcul des composantes"""
    F = np.zeros((X.shape[0], s))
    for i in range(s):
        F[:, i] = X.dot(vect_p[i][1])
    return F


def corelation_var(X: np.array, F):
    """Corellations variables composantes"""
    C_axe = np.zeros((X.shape[1], F.shape[1]))
    for i in range(F.shape[1]):
        for j in range(X.shape[1]):
            C_axe[j, i], _ = pearsonr(X[:, j], F[:, i])
    return C_axe


def plot_var(coord_var):
    """ Plot les vecteurs des variables"""
    fig, axe = plt.subplots()
    if coord_var.shape[1] == 2:
        plt.axhline(y=0, color='black')
        plt.axvline(x=0, color='black')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        axe.quiver(np.zeros((1, coord_var.shape[0])), np.zeros(
            (1, coord_var.shape[0])), coord_var[:, 0], coord_var[:, 1], angles='xy', scale_units='xy', scale=1)
        cercle = plt.Circle((0, 0), 1, fill=False)
        axe.set_aspect(1)
        axe.add_artist(cercle)
    elif coord_var.shape[1] == 3:
        p = np.linspace(-2, 2, 100)
        z = np.zeros((100, 1))
        ax = fig.add_subplot(projection='3d')
        z = np.zeros((1, coord_var.shape[0]))
        ax.quiver(z, z, z, coord_var[:, 0], coord_var[:, 1], coord_var[:, 2])
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
    plt.title("Représentation des variables")
    plt.show()


def plot_individu(F):
    """"Plot des individu dans la nouvelle dimension"""

    if F.shape[1] >= 4:
        print("Aucune représentation possible en dimension > 3")
        return
    plt.grid(True, which='both')
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')

    if F.shape[1] == 1:
        plt.scatter(F[:,0], np.zeros((F.shape[0])))
        for i, pt in enumerate(F[:,0]):
            plt.annotate(f'I{i+1}', (pt, 0))
    elif F.shape[1] == 2:
        plt.scatter(F[:, 0], F[:, 1])
        for i in range(F.shape[0]):
            plt.annotate(f'I{i+1}', (F[i][0], F[i][1]))
    elif F.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(F[:, 0], F[:, 1], F[:, 2])
        for i in range(F.shape[0]):
            ax.text(F[i][0], F[i][1], F[i][2], f'I{i+1}')

    plt.title("Nuage des individus")
    plt.show()
    
    
def reconstitution_matrice(X, F, val_and_vect):
    V = F/np.array([sqrt(v[0]) for v in val_and_vect[:F.shape[1]]])
    X_ = np.zeros(X.shape)
    for i in range(F.shape[1]):
        X_ += sqrt(val_and_vect[i][0])*np.array([V[:,i]]).T.dot(np.array([val_and_vect[i][1]]))
    return X_

def main(X: np.array):
    choix = input("1.Taux de conservation\n2.Valeur de S\n")
    if choix == "1":
        p = float(input("Entrez le taux d'inertie: "))
        a = z_score(X)
        print("Matrice centrée réduite: \n", a)
        input()
        R = a.T.dot(a)/a.shape[0]
        valeur_p, vecteur_p = np.linalg.eig(R)
        val_and_vect = []
        for i in range(valeur_p.shape[0]):
            val_and_vect.append((valeur_p[i], vecteur_p[:, i]))
        val_and_vect.sort(key=lambda x: x[0], reverse=True)
        trace = sum([e[0] for e in val_and_vect])
        s = 0
        val_consider = 0
        for val_p in val_and_vect:
            val_consider += val_p[0]
            s += 1
            if val_consider*100/trace > p:
                break   
        
        F = composantes_calculus(a, s, val_and_vect)
        plot_individu(F)

        print("Composantes principales: \n", F)
        input()

        F2 = corelation_var(a, F)
        print("Correlation matrice composante: \n", F2)
        input()
        plot_var(F2)
        input()
        
        print("Matrice reconstititué: \n")
        print(reconstitution_matrice(a, F, val_and_vect))
        
    elif choix == "2":
        s = int(input("Entrez la dimension de réduction: "))
        if(s > min(X.shape)):
            print(
                f"Valeur plus grande que min(I, K) = min{X.shape}={min(X.shape)}")
            return
        a = z_score(X)
        print("Matrice centrée réduite: \n", a)
        input()
        R = np.matmul(a.T, a)/a.shape[0]
        valeur_p, vecteur_p = np.linalg.eig(R)
        val_and_vect = []
        for i in range(valeur_p.shape[0]):
            val_and_vect.append((valeur_p[i], vecteur_p[:, i]))
        val_and_vect.sort(key=lambda x: x[0], reverse=True)

        F = composantes_calculus(a, s, val_and_vect)
        plot_individu(F)

        print("Composantes principales: \n", F)
        input()

        F2 = corelation_var(a, F)
        print("Correlation matrice composante: \n", F2)
        input()
        plot_var(F2)

        print(
            f"Pourcentage d'inertie: {sum([e[0] for e in val_and_vect[:s]])/sum([e[0] for e in val_and_vect])*100}%")
        input()
        print("Matrice reconstititué: \n")
        print(reconstitution_matrice(a, F, val_and_vect))
        

if __name__ == '__main__':
    a = np.array([[90, 140, 6.0], [60, 85, 5.9], [75, 135, 6.1],
                 [70, 145, 5.8], [85, 130, 5.4], [70, 145, 5.0]])
    
    main(a)
