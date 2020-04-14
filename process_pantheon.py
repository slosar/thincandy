#!/usr/bin/env python
# process pantheon from cosmomc, we are not including it
import numpy as np
import scipy.linalg as la
import thincandy as tc
import pyccl as ccl
import matplotlib.pyplot as plt

zcmb, mb,mberr = np.loadtxt("Pantheon/lcparam_full_long_zhel.txt", skiprows=1,usecols=[1,4,5],unpack=True)
N=len(zcmb)
cov = np.loadtxt("Pantheon/sys_full_long.txt",skiprows=1).reshape((N,N))
cov += np.diag(mberr**2)
assert (cov[3,2]==cov[2,3]) ## sanity

## Ok, now we will subtract magnitude in standard cosmology so that we can bin numbers than don't vary much across the inb

t=tc.ThinCandy()
D = ccl.comoving_angular_distance(t.CBase,1/(1+zcmb))*(1+zcmb)
mu = 5*np.log10(D)
mb -= mu


dz = 0.1
print ("zmax=",zcmb.max())
#bins = np.digitize(zcmb,list(np.linspace(0,0.1,20))+list(np.logspace(np.log10(0.1),np.log10(zcmb.max()+0.1),20)))
binl = list(np.arange(0.1,1.1,0.15))+list(np.arange(1.2,2.5,0.3))
bins = np.digitize(zcmb,binl)
Nbins = bins.max()

out=[]
for bin in range(Nbins):
    w = np.where(bins==bin)[0]
    if (len(w)==0):
        continue
    ccov = cov[np.ix_(w,w)]
    iccov = la.inv(ccov)
    zmean = np.dot(iccov,zcmb[w]).sum()/iccov.sum()
    curmb = np.dot(iccov,mb[w]).sum()/iccov.sum()
    errmb = 1/np.sqrt(iccov.sum())
    out.append([zmean, len(w), curmb, errmb])
out = np.array(out)

## take out the mean, so that we jump around 0 mag difference
z,Nl,dm,dme = out.T

dm -= (dm/dme**2).sum()/(1/dme**2).sum()
D = 10**(0.2*(dm))
De = D*np.log(10.0)*0.2*dme

print ((((D-1)/De)**2).sum(),len(D))



with open('data/pantheon.txt','w') as outf:
    for line in zip(z,Nl,D,De):
        outf.write("%g %i %g %g\n"%(line))
