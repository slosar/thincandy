#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
import pyccl as ccl
import thincandy


def main():
    # zmins, zmaxs, fs8s, fs8errs =
    fig, ax = plt.subplots(figsize=(10, 3))

    tc = thincandy.ThinCandy()
    basefs8 = plot_model(ax, tc.CBase, tc.CBase, "k:", "$\Lambda$CDM")
    if True:
        tc.getModels()
        plot_model(ax, tc.CBase, tc.Copen, "r--", "o$\Lambda$CDM")
        plot_model(ax, tc.CBase, tc.Cw, "g-.", "wCDM")
        plot_model(ax, tc.CBase, tc.Cmnu, "b-", "$\\nu$CDM")
        # plot_model(ax,tc.CBase,tc.Cmnuless, 'b:', "massles $\\nu$CDM")

    EdSP = tc.CBaseP
    EdSP["Omega_c"] = 1 - EdSP["Omega_b"]
    EdSP["sigma8"] = ccl.sigma8(tc.CBase)
    del EdSP["A_s"]
    plot_model(ax, tc.CBase, ccl.Cosmology(**EdSP), "m:", "EdS CDM")

    for zmin, zmax, fs8, fs8err in np.loadtxt("data/fs8.txt", usecols=(1, 2, 3, 4)):
        print(zmin, zmax, fs8, fs8err, 0.5 * (zmin + zmax))
        zeff = 0.5 * (zmin + zmax)
        fs8 /= basefs8(zeff)
        fs8err /= basefs8(zeff)
        ax.errorbar(
            zeff, fs8, yerr=fs8err, xerr=(zmax - zmin) / 2, fmt="-", color="k", lw=1
        )
        ax.errorbar(zeff, fs8, yerr=fs8err, fmt="o", color="k", lw=2)

    # for zmin,zmax,val,errplus, errmin in np.loadtxt('data/wl.txt',usecols=(1,2,3,4,5)):
    #     ## value is (Om/0.3)^1/2*sigma8
    #     baseval = np.sqrt(tc.CBase['Omega_m']/0.3)*ccl.sigma8(tc.CBase)
    #     print (val,baseval,errplus)
    #     zeff = np.sqrt((1+zmin)*(1+zmax))-1
    #     fs8eff = (ccl.growth_rate(tc.CBase,1/(1+zeff))*
    #               ccl.sigma8(tc.CBase)*ccl.growth_factor(tc.CBase,1/(1+zeff)) )
    #     fact = fs8eff/baseval
    #     val*=fact
    #     errplus*=fact
    #     errmin*=fact
    #     ax.errorbar(zeff,val, yerr=([errmin],[errplus]),
    #                 xerr=([zeff-zmin],[zmax-zeff]),fmt='c',lw=1)
    #     ax.errorbar(zeff,val, yerr=([errmin],[errplus]),fmt='oc',lw=2)

    #     #ax.add_patch(Rectangle((zmin,val-errmin), zmax-zmin, errmin+errplus,color='blue',alpha=0.5))

    ax.set_xlim(0, 2.5)
    ax.set_xlabel("$z$", fontsize=14)
    ax.set_ylim(0.4, 1.6)
    ax.set_ylabel("$f\,\sigma_8 / [f\,\sigma_8]_{\\rmfid}$", fontsize=14)
    plt.legend(fontsize=12, ncol=5, frameon=False, loc="upper center")
    plt.tight_layout()
    plt.savefig("output/thincandy_gr.pdf")
    plt.show()


def plot_model(ax, CBase, C, fmt, name):
    zar = np.linspace(0, 2.5, 100)
    aar = 1 / (1 + zar)
    f = ccl.growth_rate(C, aar)
    s8 = ccl.sigma8(C) * ccl.growth_factor(C, aar)

    f0 = ccl.growth_rate(CBase, aar)
    s80 = ccl.sigma8(CBase) * ccl.growth_factor(CBase, aar)

    ax.plot(zar, (f * s8) / (f0 * s80), fmt, label=name)
    return interp1d(zar, (s80 * f0))


if __name__ == "__main__":
    main()
