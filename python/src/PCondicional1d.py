import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)

class DataseTwoClass1D:

    def __init__(self,promedio1,promedio2,varianza1,varianza2,muestras):
        self.promedio1 = promedio1
        self.promedio2 = promedio2
        self.varianza1 = varianza1
        self.varianza2 = varianza2
        self.muestras  = muestras

    def generarData(self):        
        mu = np.array([self.promedio1, self.promedio2])  # mean of the distribution
        SIG = np.array([[self.varianza1 , 0], [0, self.varianza2]])  # covariance matrix of the distribution
        datos =  np.random.multivariate_normal(mu, SIG, self.muestras)
        r1=datos[:,0]
        r2=datos[:,1]
        minimo=r1.min()
        maximo=5
        return r1,r2



if __name__=='__main__':
    dataset=DataseTwoClass1D(5,7,1.5,2.5,400)#promedio1,promedio2,varianza1,varianza2,muestras
    r1,r2=dataset.generarData()
    print(r1.min())
    ax = fig.add_subplot(gs[0, :])
    ax.plot(r1,0.1*np.ones(400),'|',xunits=0.2)
    ax.plot(r2,-0.1*np.ones(400),'+',xunits=0.2)
    ax.set_ylabel('')
    ax.set_xlabel('X')

    # %% DENSIDADES DE PROBABILIDAD CONDICIONAL DE CLASE p(x|w)
    # %p(x|w1)
    # % numero bins y particionamiento eje horizontal
    # %Nbin=ND1/20;
    Nbin=ND1/10;
    delta1=(maxr1-minr1)/Nbin; %longitud bin
    eje1=[minr1: delta1 : maxr1-delta1];
    pxw1a=hist(r1,Nbin)/(delta1*ND1); % Se divide por delta para obtener la densidad de probabilidad

    pxw1 = medfilt1(pxw1a,5); %;%filtro mediana

    %pxw1 =pxw1a;
    subplot(2,2,3);
    plot(eje1,pxw1,'b');title('p(x|w1)'); xlabel('x');

    %p(x|w2)
    % n�mero bins y particionamiento eje horizontal
    %Nbin=ND2/20;
    Nbin=ND2/10;
    delta2=(maxr2-minr2)/Nbin;%longitud bin
    eje2=[minr2: delta2 : maxr2-delta2];
    pxw2a=hist(r2,Nbin)/(delta2*ND2);

    pxw2 = medfilt1(pxw2a,5);

    %pxw2=pxw2a;
    subplot(2,2,4); 
    plot(eje2,pxw2,'r');title('p(x|w2)'); xlabel('x');


    for i in range(2):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(np.arange(1., 0., -0.1) * 2000., np.arange(1., 0., -0.1))
        # ax.set_ylabel('YLabel1 %d'i+1)
        ax.set_xlabel(f'p(x|w%d)' %(i+1))
        
        if i == 0:
            ax.tick_params(axis='x', rotation=55)
    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

    plt.show()

