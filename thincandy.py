#!/usr/bin/env python
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from  scipy.interpolate import interp1d

def rd_cuesta_approx(obh2, ocbh2, onuh2, Nnu):
    if (abs(Nnu-3) > 0.1):
        print("ERROR, Tony Cuesta says: 'not in this ceral box.'")
        print("Nnu=", Nnu)
        stop()
    return 55.154 / (ocbh2**0.25351 * (obh2)**0.12807 * np.exp(
        (onuh2+0.0006)**2.0/0.1176**2))



class ThinCandy:

    def __init__(self):
        self.Nl=100
        self.Nr=50
        self.zar=np.hstack((np.linspace (0,3,self.Nl),np.linspace(1100,1200,self.Nr)))
        self.aar = 1/(1+self.zar)

    
        # Planck best fit
        self.Obh2 = 0.02233
        self.Och2 = 0.1198
        self.lnttAs = 3.043
        self.h = 67.32/100.
        self.ns = 0.9652

        self.CBaseP = {
            'Omega_c' : self.Och2/self.h**2,
            'Omega_b' : self.Obh2/self.h**2,
            'h' : self.h,
            'n_s':  self.ns,
            'm_nu': 0.06,
            'A_s' : np.exp(self.lnttAs)/10e10*np.pi**2 # Eh??
        }
        self.CBase = ccl.Cosmology(**self.CBaseP)

        self.sndata = np.loadtxt('data/pantheon.txt')
        self.snz = self.sndata[:,0]
        self.snval = self.sndata[:,2]
        self.snerr = self.sndata[:,3]
        self.baodata = np.loadtxt('data/bao.txt', usecols=(1,2,3,4,5,6,7))
        

    def bgplot(self):
        print (ccl.sigma8(self.CBase), self.rd(self.CBase))
        print ('getting open')
        self.Copen = self.GetCosmo ({'Omega_k':-0.044})
        print ('getting cw')
        self.Cw = self.GetCosmo ({'w0':-1.58})
        print ('getting cmnu')
        self.Cmnu = self.GetCosmo ({'m_nu':0.26})
        #self.Copen=self.CBase
        #self.Cw=self.CBase
        #self.Cmnu=self.CBase

        #[(aiso,'isotropic BAO'), (aperp, 'transverse BAO'), (apar,
        #'radial BAO'), (sn, 'SN distance '), (fs8, 'fsigma8'),
        # (shearshear, 'WL shear auto')]):




        f,axl = plt.subplots(4,2, facecolor='w',
                             gridspec_kw={'width_ratios':(3,1),'hspace':0.0,'wspace':0.05},figsize=(10,8))

        print (axl)
        for i,((fun,name),(axl,axr)) in enumerate(zip ([(self.aiso,'BAO $\\bigcirc$'),
                                                      (self.aperp, 'BAO $\perp$'),
                                                      (self.apar, 'BAO $\parallel$'),
                                                      (self.sn, 'SN')],axl)):
            
            
            for model,label,style in [(self.CBase,'LCDM','k--'), (self.Copen,'OLCDM','r:'), (self.Cw,'wCDM','g-.'),
                                (self.Cmnu, '$\\nu$CDM','b-')]:
                vals=fun(model)
                axl.plot(self.zar[:self.Nl],vals[:self.Nl],style,label=label)
                axr.plot(self.zar[-self.Nr:],vals[-self.Nr:],style)

            if "perp" in name:
                for zb,iso,isoe,perp,perpe,par,pare in self.baodata:
                    if perp>0:
                        norm = ccl.comoving_angular_distance(self.CBase,1/(1+zb)) / self.rd(self.CBase)
                        #print (zb,norm,perp,perpe,perp/norm,perpe/norm)
                        axl.errorbar(zb,perp/norm,yerr=perpe/norm,fmt='k.')
                    axr.errorbar(1150,1,yerr=0.000605211, fmt='k.')
            elif "par" in name:
                for zb,iso,isoe,perp,perpe,par,pare in self.baodata:
                    if par>0:
                        norm = 299792.45/(ccl.h_over_h0(self.CBase,1/(1+zb))*self.CBase['H0'] * self.rd(self.CBase))
                        #print (zb,norm,par,pare,par/norm,pare/norm)
                        axl.errorbar(zb,par/norm,yerr=pare/norm,fmt='k.')
            elif "circ" in name:
                for zb,iso,isoe,perp,perpe,par,pare in self.baodata:
                    if iso>0:
                        print (self.rd(self.CBase))
                        normperp = ccl.comoving_angular_distance(self.CBase,1/(1+zb)) / self.rd(self.CBase)
                        normpar = 299792.45/(ccl.h_over_h0(self.CBase,1/(1+zb))*self.CBase['H0'] * self.rd(self.CBase))
                        norm = (zb*normpar)**(1/3)*normperp**(2/3)
                        axl.errorbar(zb,iso/norm,yerr=isoe/norm,fmt='k.')
            
            elif "SN" in name:
                axl.errorbar(self.snz,self.snval,yerr=self.snerr,fmt='k.')
            if (i==0):
                axl.legend(fontsize=8)
            
            #plt.xlabel("z")
            axl.set_ylabel(name)
            axl.spines['right'].set_visible(False)
            axr.spines['left'].set_visible(False)
            #axl.yaxis.tick_left()
            #axl.tick_params(labelright='off')
            axl.tick_params(axis='x',direction='inout', length=5)
            axr.tick_params(axis='x',direction='inout', length=5)
            axl.patch.set_alpha(0.0)
            axr.patch.set_alpha(0.0)

            axl.set_xlim(0.0,3.1)
            axr.set_xlim(1100,1200)
            axr.set_xticks([1120,1200])

            if (i<3):
                axl.set_ylim(0.85,1.15)
                axl.set_yticks([0.90,1.0,1.1])
            else:
                axl.set_ylim(0.8,1.2)
                axl.set_yticks([0.9,1.0,1.1])
                
            if (i==0):
                axr.set_ylim(0.995,1.005)
                axr.set_yticks([0.997,1.0,1.003])
            elif (i==1):
                axr.set_ylim(0.995,1.005)
                axr.set_yticks([0.997,1.0,1.003])
            elif (i==2):
                axr.set_ylim(0.995,1.005)
                axr.set_yticks([0.997,1.0,1.003])
            elif (i==3):
                axr.set_ylim(0.8,1.2)
                axr.set_yticks([0.9,1.0,1.1])

            
            if (i<3):
                for l in axl.get_xticklabels():
                    l.set_visible(False)
                for l in axr.get_xticklabels():
                    l.set_visible(False)
            for l in axr.get_yticklabels():
                l.set_visible(False)
            axr.get_yaxis().tick_right()#set_visible(False)            


            d=0.05
            kwargs = dict(transform=axl.transAxes, color='k', clip_on=False)
            axl.plot((1-d/5,1+d/5), (-d,+d), **kwargs)
            axl.plot((1-d/5,1+d/5),(1-d,1+d), **kwargs)
            kwargs = dict(transform=axr.transAxes, color='k', clip_on=False)
            axr.plot((-d*3/5,+d*3/5), (-d,+d), **kwargs)
            axr.plot((-d*3/5,+d*3/5),(1-d,1+d), **kwargs)
            if (i==3):
                axl.set_xlabel("$z$")


        #plt.tight_layout()
        plt.savefig(f'thincandy_bg.pdf')


        plt.show()


    def lpars(self,C):
        str='h=%1.2f $\\Omega_m=%1.2f$ '%(C['h'],C['Omega_m'])
        return str

    def GetCosmo (self,addict):
        ## find the correct h, to kee distance t
        acmb=1/(1150.)
        target = ccl.comoving_angular_distance(self.CBase,acmb) / self.rd(self.CBase)
        def _C(h):
            CP = {
                'Omega_c' : self.Och2/h**2,
                'Omega_b' : self.Obh2/h**2,
                'h' : h,
                'n_s':  self.ns,
                'm_nu': 0.06,
                'A_s' : np.exp(self.lnttAs)/10e10*np.pi**2 # Eh??
            }
            CP.update(addict)
            print (CP)
            return ccl.Cosmology(**CP)
        def _t(h):
            C=_C(h) 
            d=ccl.comoving_angular_distance(C,acmb) / self.rd(C)
            return d-target
        print (_t(0.3), _t(1.0))
        hout = so.bisect(_t,0.3,1.0)#,xtol=1e-5,rtol=1e-2)
        return _C(hout)


    def rd(self,C):
        h2=C['h']**2
        obh2 = C['Omega_b']*h2
        ocbh2 = (C['Omega_b']+C['Omega_c'])*h2
        return rd_cuesta_approx (obh2,ocbh2,0.06/93.4,3.04)

    def aperp (self,C):
        aperp = ccl.comoving_angular_distance(C,self.aar) / self.rd(C)
        aperp0 = ccl.comoving_angular_distance(self.CBase,self.aar) / self.rd(self.CBase)
        return aperp / aperp0

    def apar (self,C):
        apar = 1/ccl.h_over_h0(C,self.aar)/C['h'] / self.rd(C)
        apar0 = 1/ccl.h_over_h0(self.CBase,self.aar)/self.CBase['h'] / self.rd(self.CBase)
        return apar / apar0

    def aiso (self,C):
        ## these things actually do commute
        return self.aperp(C)**(2/3)*self.apar(C)**(1/3)


    def sn(self,C):
        ## there are fixed (1+z)**2 factors here for lum distance, which cancel out
        sn = ccl.comoving_angular_distance(C,self.aar)
        sn0 = ccl.comoving_angular_distance(self.CBase,self.aar)
        ratio = sn/sn0
        fact = (interp1d(self.zar,ratio)(self.snz)/self.snerr**2).sum() / (1/self.snerr**2).sum()

        #now normalize ratio so that error weighted is one
        ratio /= fact
        
        return (ratio)


    def fs8 (self,C):
        fs8 = ccl.growth_rate(C,self.aar) * ccl.sigma8(C)
        fs80 = ccl.growth_rate(self.CBase,self.aar) * ccl.sigma8(self.CBase)
        return fs8 / fs80

    def shearpower (self,C,z):
        tracer = ccl.WeakLensingTracer(C,  dndz=(self.zar, np.exp( -(z-self.zar)**2/(2*0.1**2))))
        ell = 0.2 * ccl.comoving_angular_distance(C,1/(1+z))
        return ccl.angular_cl(C, tracer, tracer, ell)

    def shearshear (self,C):
        sp = np.array([shearpower(C,z) for z in self.zar])
        sp0 = np.array([shearpower(self.CBase,z) for z in self.zar])
        return sp/sp0


if __name__=="__main__":
    tc=ThinCandy()
    tc.bgplot()
    

    


