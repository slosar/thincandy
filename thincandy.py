#!/usr/bin/env python
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt

def rd_cuesta_approx(obh2, ocbh2, onuh2, Nnu):
    if (abs(Nnu-3) > 0.1):
        print("ERROR, Tony Cuesta says: 'not in this ceral box.'")
        print("Nnu=", Nnu)
        stop()
    return 55.154 / (ocbh2**0.25351 * (obh2)**0.12807 * np.exp(
        (onuh2+0.0006)**2.0/0.1176**2))

def rd(C):
    h2=C['h']**2
    obh2 = C['Omega_b']*h2
    ocbh2 = (C['Omega_b']+C['Omega_c'])*h2
    return rd_cuesta_approx (obh2,ocbh2,0.06/93.4,3.04)

def aperp (C):
    aperp = ccl.comoving_angular_distance(C,aar) / rd(C)
    aperp0 = ccl.comoving_angular_distance(CBase,aar) / rd(CBase)
    return aperp / aperp0


def apar (C):
    apar = 1/ccl.h_over_h0(C,aar) / rd(C)
    apar0 = 1/ccl.h_over_h0(CBase,aar) / rd(CBase)
    return apar / apar0

def aiso (C):
    ## these things actually do commute
    return aperp(C)**(2/3)*apar(C)**(1/3)

def sn(C):
    ## there are fixed (1+z)**2 factors here for lum distance, which cancel out
    sn = ccl.comoving_angular_distance(C,aar) 
    sn0 = ccl.comoving_angular_distance(CBase,aar) 
    return (sn/sn.mean()) / (sn0 / sn0.mean())
    

def fs8 (C):
    fs8 = ccl.growth_rate(C,aar) * ccl.sigma8(C)
    fs80 = ccl.growth_rate(CBase,aar) * ccl.sigma8(CBase)
    return fs8 / fs80

def shearpower (C,z):
    tracer = ccl.WeakLensingTracer(C,  dndz=(zar, np.exp( -(z-zar)**2/(2*0.1**2))))
    ell = 0.2 * ccl.comoving_angular_distance(C,1/(1+z))
    return ccl.angular_cl(C, tracer, tracer, ell)

def shearshear (C):
    sp = np.array([shearpower(C,z) for z in zar])
    sp0 = np.array([shearpower(CBase,z) for z in zar])
    return sp/sp0
    
    
    


# Planck best fit
Obh2 = 0.02233
Och2 = 0.1198
lnttAs = 3.043
h = 67.32/100.
ns = 0.9652

CBaseP = {
    'Omega_c' :Och2/h/h,
    'Omega_b' : Obh2/h/h,
    'h' : h,
    'n_s':  ns,
    'A_s' : np.exp(lnttAs)/10e10*np.pi**2 # Eh??
}

CBase = ccl.Cosmology(**CBaseP)
print (ccl.sigma8(CBase), rd(CBase))
Copen = ccl.Cosmology(**CBaseP, Omega_k=-0.1)
Cw = ccl.Cosmology(**CBaseP, w0=-0.7)
Cwa = ccl.Cosmology(**CBaseP, w0=-0.9, wa=+0.5)

zar=np.linspace (0,2,100)
aar = 1/(1+zar)

plt.figure(figsize=(10,6), dpi=200)
for i,(fun,name) in enumerate([(aiso,'isotropic BAO'),
                               (aperp, 'transverse BAO'),
                               (apar, 'radial BAO'),
                               (sn, 'SN distance '),
                               (fs8, 'fsigma8'),
                               (shearshear, 'WL shear auto')]):
    plt.subplot(2,3,i+1)
    plt.plot(zar,fun(CBase),'k-')
    plt.plot(zar,fun(Copen),'r:',label='OLCDM')
    plt.plot(zar,fun(Cw),'g--', label='wCDM')
    plt.plot(zar,fun(Cwa),'b-.', label='wwaCDM')
    if (i==0):
        plt.legend()
    plt.xlabel("z")
    plt.ylabel('$\Delta$ '+name)

plt.tight_layout()
plt.savefig('thincandy.png')
plt.show()




