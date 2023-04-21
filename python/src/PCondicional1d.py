import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt





# np.random.seed(0)
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(3, 2)

class DataseTwoClass1D:

    def __init__(self,promedio1,promedio2,varianza1,varianza2,muestras):
        self.promedio1 = promedio1
        self.promedio2 = promedio2
        self.varianza1 = varianza1
        self.varianza2 = varianza2
        self.muestras  = muestras

    def generarData(self):
        np.random.seed(1)       
        mu = np.array([self.promedio1, self.promedio2])  # mean of the distribution
        SIG = np.array([[self.varianza1 , 0], [0, self.varianza2]])  # covariance matrix of the distribution
        datos =  np.random.multivariate_normal(mu, SIG, self.muestras)
        r1=datos[:,0]
        r2=datos[:,1]
        return r1,r2



if __name__=='__main__':
    dataset=DataseTwoClass1D(5,7,1.5,2.5,4000)#promedio1,promedio2,varianza1,varianza2,muestras
    r1,r2=dataset.generarData()
    print(r1.min())
    ax = fig.add_subplot(gs[0, :])
    ax.plot(r1,0.1*np.ones(dataset.muestras),'|')
    ax.plot(r2,-0.1*np.ones(dataset.muestras),'+')
    plt.ylim([-0.2,0.2])
    ax.set_ylabel('')
    ax.set_xlabel('X')


    for i in range(2):
        if i == 0:
            ax = fig.add_subplot(gs[1, i])
            ax.set_xlabel('p(x|w%d)' % (i+1))
            ND1=dataset.muestras
            Nbin=ND1/10;
            delta1=(r1.max()-r1.min())/Nbin
            eje1=np.linspace(r1.min(), r1.max()-delta1, num=int(Nbin))
            n,bins=np.histogram(r1,bins=int(Nbin))
            pxw1a=n/(delta1*ND1)
            pxw1=medfilt(pxw1a, kernel_size=5)
            ax.plot(eje1, pxw1)
            # ax.set_ylabel('YLabel1 %d'i+1)
            ax.tick_params(axis='x', rotation=55)
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(eje1, 100*pxw1)
            plt.hist(r1,bins=int(Nbin) ,color='blue', alpha=0.2)

        elif i == 1:
            ax = fig.add_subplot(gs[1, i])
            ax.set_xlabel('p(x|w%d)' % (i+1))
            ND2=dataset.muestras
            Nbin2=ND2/10;
            delta2=(r2.max()-r2.min())/Nbin
            eje2=np.linspace(r2.min(), r2.max()-delta2, num=int(Nbin2))
            n2,bins2=np.histogram(r2,bins=int(Nbin))
            pxw2a=n2/(delta2*ND2)
            pxw2=medfilt(pxw2a, kernel_size=5)
            ax.plot(eje2, pxw2,color='orange')
            ax = fig.add_subplot(gs[2, 1])
            ax.plot(eje2, 100*pxw2, color='orange')
            plt.hist(r2,bins=int(Nbin2) ,color='orange', alpha=0.2)


            # ax.set_ylabel('YLabel1 %d'i+1)
            ax.tick_params(axis='x', rotation=55)
        
        
    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

    plt.show()


