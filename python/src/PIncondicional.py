import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import make_interp_spline





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
    dataset=DataseTwoClass1D(5,7,1.5,2.5,400)#promedio1,promedio2,varianza1,varianza2,muestras
    r1,r2=dataset.generarData()
    ax = fig.add_subplot(gs[0, :])
    ax.plot(r1,0.1*np.ones(dataset.muestras),'|')
    ax.plot(r2,-0.1*np.ones(dataset.muestras),'+')
    plt.ylim([-0.2,0.2])
    ax.set_ylabel('')
    ax.set_xlabel('X')



    for i in range(2):
        if i == 0:
            ax = fig.add_subplot(gs[1, :])
            ax.set_xlabel('p(x|w%d)' % (i+1))
            R=np.concatenate((r1, r2))
            ND1=dataset.muestras
            Nbin=ND1/10
            delta1=(R.max()-R.min())/Nbin
            eje1=np.linspace(R.min(), R.max()-delta1, num=int(Nbin))
            n,bins=np.histogram(R,bins=int(Nbin))
            pxw1a=n/(delta1*len(R))
            pxw1=medfilt(pxw1a, kernel_size=3)            
            ax.plot(eje1, pxw1)
            # ax.set_ylabel('YLabel1 %d'i+1)
            ax.tick_params(axis='x', rotation=55)
            ax = fig.add_subplot(gs[2, :])
            ax.plot(eje1, 100*pxw1)
            plt.hist(r1,bins=int(Nbin) ,ec="yellow",color='blue', alpha=0.2)

        
        
    fig.align_labels()  

    plt.show()


