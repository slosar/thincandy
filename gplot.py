#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pyccl as ccl
import thincandy 

def main():
    #zmins, zmaxs, fs8s, fs8errs =
    fig, ax = plt.subplots(figsize=(10,4))

    tc = thincandy.ThinCandy()
    tc.getModels()
    plot_model(ax,tc.CBase, 'k:', "$\Lambda$CDM")
    plot_model(ax,tc.Copen, 'r--', "o$\Lambda$CDM")
    plot_model(ax,tc.Cw, 'g-.', "wCDM")
    plot_model(ax,tc.Cmnu, 'b-', "wCDM")

    EdSP = tc.CBaseP
    EdSP['Omega_c']=1-EdSP['Omega_b']
    plot_model(ax,ccl.Cosmology(**EdSP), 'm:', "Einstein de Sitter")
    

    for zmin,zmax, fs8,fs8err in np.loadtxt('data/fs8.txt',usecols=(1,2,3,4)):
        print(zmin,zmax,fs8,fs8err, 0.5*(zmin+zmax))
        ax.errorbar(0.5*(zmin+zmax),fs8, yerr=fs8err, xerr=(zmax-zmin)/2,fmt='-',color='k',lw=1)
        ax.errorbar(0.5*(zmin+zmax),fs8, yerr=fs8err,fmt='o',color='k',lw=2)

    for zmin,zmax,val,errplus, errmin in np.loadtxt('data/wl.txt',usecols=(1,2,3,4,5)):
        ## value is (Om/0.3)^1/2*sigma8
        baseval = np.sqrt(tc.CBase['Omega_m']/0.3)*ccl.sigma8(tc.CBase)
        print (val,baseval,errplus)
        zeff = np.sqrt((1+zmin)*(1+zmax))-1
        fs8eff = (ccl.growth_rate(tc.CBase,1/(1+zeff))*
                  ccl.sigma8(tc.CBase)*ccl.growth_factor(tc.CBase,1/(1+zeff)) )
        fact = fs8eff/baseval
        val*=fact
        errplus*=fact
        errmin*=fact
        ax.errorbar(zeff,val, yerr=([errmin],[errplus]),
                    xerr=([zeff-zmin],[zmax-zeff]),fmt='c',lw=1)
        ax.errorbar(zeff,val, yerr=([errmin],[errplus]),fmt='oc',lw=2)
        
        #ax.add_patch(Rectangle((zmin,val-errmin), zmax-zmin, errmin+errplus,color='blue',alpha=0.5))
        

        
    ax.set_xlim(0,2.5)
    ax.set_xlabel("$z$",fontsize=14)
    ax.set_ylim(0.2,0.7)
    ax.set_ylabel("$f\,\sigma_8$",fontsize=14)
    plt.legend()
    plt.savefig('gplot.pdf')
    plt.show()


    
def plot_model(ax,C, fmt, name):
    zar=np.linspace(0,2.5,100)
    aar=1/(1+zar)
    f = ccl.growth_rate (C,aar)
    s8 = ccl.sigma8(C)*ccl.growth_factor(C,aar)
    ax.plot(zar,f*s8,fmt,label=name)
    
if __name__=="__main__":
    main()

    
