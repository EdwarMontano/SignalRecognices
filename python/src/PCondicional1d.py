import numpy as np 


def datosAleatorios(cantmuestras,promedio, varianza):
    # ND1=400; % Numero vectores 
    # mu1 = 5; % Promedio
    # SIG1 = 1.5; % Varianza
    # r1 = mvnrnd(mu1,SIG1,ND1); %data set Clase 1
    # minr1=min(r1);maxr1=max(r1);
    mu = np.array([5, 5])  # mean of the distribution
    SIG = np.array([[1.5 , 0], [0, 1.5]])  # covariance matrix of the distribution
    datos =  np.random.multivariate_normal(mu, SIG, 400)
    # datos  = np.random.normal(cantmuestras, varianza*promedio)
    minimo=1
    maximo=5
    return datos,minimo,maximo



if __name__=='__main__':
    r1,_,_=datosAleatorios(400,5,1.5)
    print(len(r1))