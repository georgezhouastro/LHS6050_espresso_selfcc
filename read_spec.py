import os,sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def read_spec(fitsname):
    hdul = fits.open(fitsname)
    flux = hdul['SCIDATA'].data
    wave = hdul['WAVEDATA_VAC_BARY'].data


    return [wave,flux]

def read_header(fitsname,return_full=False):
    hdul = fits.open(fitsname)
    rv = hdul[0].header['HIERARCH ESO QC CCF RV']
    rverr = hdul[0].header['HIERARCH ESO QC CCF RV ERROR']
    fwhm = hdul[0].header['HIERARCH ESO QC CCF FWHM']
    bjd = hdul[0].header['HIERARCH ESO QC BJD']
    drift = 0 #hdul[0].header['HIERARCH ESO QC DRIFT DET0 MEAN']*480

    if not return_full:
        return bjd,rv,rverr,fwhm,drift

    if return_full:
        fwhmerr=hdul[0].header['HIERARCH ESO QC CCF FWHM ERROR']
        flux_asym=hdul[0].header['HIERARCH ESO QC CCF FLUX ASYMMETRY ']
        flux_asym_err=hdul[0].header['HIERARCH ESO QC CCF FLUX ASYMMETRY ERROR']
        bis=hdul[0].header['HIERARCH ESO QC CCF BIS SPAN']
        bis_err=hdul[0].header['HIERARCH ESO QC CCF BIS SPAN ERROR']
        return bjd,rv,rverr,fwhm,fwhmerr,flux_asym,flux_asym_err,bis,bis_err,drift

    

if __name__ == "__main__":
    #fitsname = "2024-10-03/r.ESPRE.2024-10-04T04:25:35.056_S2D_A.fits"
    #fitsname = "2024-12-26/r.ESPRE.2024-12-27T02:33:02.581_S2D_A.fits"
    #fitsname = "2024-10-03/r.ESPRE.2024-10-04T04:25:35.056_S2D_SKYSUB_A.fits"
    fitsname = "2024-10-03/r.ESPRE.2024-10-04T04:25:35.056_S2D_TELL_CORR_A.fits"
    read_header(fitsname)
    wave,flux = read_spec(fitsname)
    

    for i in range(len(wave)):
        print(i,np.nanmedian(wave[i]))
        plt.plot(wave[i],flux[i])
    plt.show()
