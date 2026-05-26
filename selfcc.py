import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import emcee
import crosscorrFFT
import read_spec
from astropy.io import fits as pyfits


# Constants
c = 299792.458

telluric = np.loadtxt("telluric.dat")
telluric[:,0] *= 10

def air_to_vac(w_air):
    """
    Converts air wavelength to vacuum wavelength in Angstroms (A).
    Formula from Morton (1991), often used in astronomical spectroscopy.
    """
    s = 10**4 / w_air
    n = 1.0 + 1e-8 * (6432.8 + (2949810.0 / (146.0 - s**2)) + (25540.0 / (41.0 - s**2)))
    wavelength_vac = w_air * n
    return wavelength_vac

telluric[:,0] = air_to_vac(telluric[:,0])

def convolve_to_resolution(wave, flux, R):
    """
    Convolves a spectrum to a given resolution R = lambda / delta_lambda.
    """
    lambda_min, lambda_max = np.min(wave), np.max(wave)
    d_ln_lambda = 1.0 / (3.0 * R) 
    ln_wave_new = np.arange(np.log(lambda_min), np.log(lambda_max), d_ln_lambda)
    wave_new = np.exp(ln_wave_new)

    f_interp = interp1d(wave, flux, kind='linear', fill_value="extrapolate")
    flux_log = f_interp(wave_new)

    sigma_pixels = (1.0 / R) / d_ln_lambda / 2.355
    flux_convolved = gaussian_filter1d(flux_log, sigma_pixels)

    f_final = interp1d(wave_new, flux_convolved, kind='linear', fill_value="extrapolate")
    return f_final(wave)

telluric[:,1] = convolve_to_resolution(telluric[:,0], telluric[:,1], R=130000)

def telluricmask(wave, berv):
    if np.min(wave) > 5800 and np.max(wave) < 9497.0:
        telluric_wave = telluric[:,0] + telluric[:,0] * (berv/c)
        telluric_flux = interpolate.splrep(telluric_wave, telluric[:,1], k=1)
        telluric_flux = interpolate.splev(wave, telluric_flux, ext=1)
        telluric_mask = telluric_flux < 0.95
        return telluric_mask
    else:
        telluric_mask = wave != wave
        return telluric_mask

def normalise(spec, niter=1, sigma_low=0.05, deg=1):
    x = np.arange(len(spec))
    mask = spec == spec
    spec_iter = spec[mask]
    x_iter = x.copy()[mask]

    i = 0
    while i < niter:
        fit = np.polyfit(x_iter, spec_iter, deg)
        fit = np.polyval(fit, x_iter)

        mask = spec_iter - fit > sigma_low * np.std(spec_iter - fit)
        spec_iter = spec_iter[mask]
        x_iter = x_iter[mask]
        i += 1

    fit = np.polyfit(x_iter, spec_iter, deg)
    fit = np.polyval(fit, x)

    if np.min(fit) > 0:
        spec /= fit
    else:
        max_prefit = np.max(fit)
        spec -= fit
        spec += max_prefit

        mask = spec == spec
        spec_iter = spec[mask]
        x_iter = x[mask]

        fit = np.polyfit(x_iter, spec_iter, 1)
        fit = np.polyval(fit, x_iter)

        mask = spec_iter - fit > sigma_low * np.std(spec_iter - fit)
        spec_iter = spec_iter[mask]
        x_iter = x_iter[mask]

        fit = np.polyfit(x_iter, spec_iter, 1)
        fit = np.polyval(fit, x)

        spec /= fit

    return spec

def combine(filelist):
    fitstemplate = read_spec.read_spec(filelist[0])
    dataout_1 = [[],[]]
    for order in np.arange(len(fitstemplate[0])):
        dataout_1[1].append([])
        dataout_1[0].append(fitstemplate[0][order])
        
    for i in np.arange(len(filelist)):
        path = filelist[i]
        if os.path.exists(path):
            try:
                fits = read_spec.read_spec(path)
                berv = pyfits.open(path)[0].header['HIERARCH ESO QC BERV']

                for order in range(len(fits[0])):
                    wave = fits[0][order]
                    indx = np.argsort(wave)
                    mask = fits[1][order] == fits[1][order]
                    flux_spline = interpolate.splrep(wave[mask][indx], fits[1][order][mask][indx], k=1)
                    flux = interpolate.splev(fitstemplate[0][order], flux_spline, ext=0)
                    mask = flux == 0
                    flux[mask] = np.nan
                    flux = normalise(flux, niter=1, sigma_low=0.05, deg=4)

                    if np.min(wave) > 5800:
                        telmask = telluricmask(wave, berv)
                        flux[telmask] = 1

                    dataout_1[1][order].append(flux)

            except None:
                pass

    averagespec = []
    for order in range(len(fitstemplate[0])):
        order_flux = np.array(dataout_1[1][order])
        print(order_flux)
        order_flux = np.nanmedian(order_flux, axis=0)
        averagespec.append(np.transpose(np.array([fitstemplate[0][order], order_flux])))

    return averagespec

def apodize(x, ap=50):
    func = np.ones(len(x))
    xpos = np.arange(len(x))

    mask = xpos < ap
    func[mask] = 0.5 * np.cos((xpos[mask] + ap) * 2 * np.pi / (ap * 2)) + 0.5
    mask = np.max(xpos) - xpos < ap
    func[mask] = 0.5 * np.cos((np.max(xpos) - ap - xpos[mask]) * 2 * np.pi / (ap * 2)) + 0.5

    return func * x

def cross_correlate_order(observation, template, berv=0, ap=50.):
    observation[:, 1] = normalise(observation[:, 1], niter=1, sigma_low=0.05, deg=5)
    observation[:, 1] = 1 - observation[:, 1]
    template[:, 1] = 1 - template[:, 1]
    observation = observation[300:-300]

    if np.min(observation[:,0]) > 5800:
        telmask = telluricmask(observation[:,0], berv)
        observation[:,1][telmask] *= 0

    observation[:, 1] = apodize(observation[:, 1], ap=500)
    template[:, 1] = apodize(template[:, 1], ap=500)

    drv, cc = crosscorrFFT.crosscorrRV_fft_linear(
        observation[:, 0], observation[:, 1],
        template[:, 0], template[:, 1],
        rvmin=-50, rvmax=50, drv=0.01
    )

    return drv, cc

def fitrv(filelist):
    average_spec = combine(filelist)
    fitstemplate = read_spec.read_spec(filelist[0])
    vsini = 5
    rvout = []

    def runfits(i):
        try:
            path = filelist[i]
            if os.path.exists(path):
                fits = read_spec.read_spec(path)
                berv = pyfits.open(path)[0].header['HIERARCH ESO QC BERV']
                bjd, rv, rverr, vsini, drift = read_spec.read_header(path)
                specorderrv = [bjd]
                
                if True: 
                    for order in np.arange(30, len(average_spec)):
                        template = average_spec[order]
                        mask = template[:,1] == template[:,1]
                        mask *= template[:,1] != 0
                        template = template[mask]
                        observation = np.transpose([fits[0][order], fits[1][order]])
                        mask = observation[:,1] == observation[:,1]
                        mask *= observation[:,1] != 0
                        observation = observation[mask]

                        drv, cc = cross_correlate_order(observation, template, berv=berv)

                        polymask = np.abs(drv - (rv)) > vsini * 2
                        cc -= np.polyval(np.polyfit(drv[polymask], cc[polymask], 1), drv)

                        v0 = drv[np.argmax(cc)]
                        velmask = np.abs(drv - v0) < vsini * 2
                        velmask *= cc > 0.3 * np.nanmax(cc)
                        err = 0.001 * np.nanstd(cc[~velmask])

                        def lnlike(x0):
                            f = x0[2] * np.exp(-1 * (drv[velmask] - x0[0])**2 / (2 * x0[1]**2))
                            residual = f - cc[velmask]
                            prob = -0.5 * np.nansum(residual**2 / err**2)
                            if prob == prob:
                                return prob
                            else:
                                return -1 * np.inf

                        nwalkers = 100
                        p0 = []
                        while len(p0) < nwalkers:
                            pi = [np.random.normal(v0, 1.),
                                  np.random.normal(vsini/2.55, 1.),
                                  np.random.normal(np.max(cc) - np.median(cc), 0.1 * (np.max(cc) - np.median(cc)))]
                            
                            if np.abs(lnlike(pi)) < np.inf:
                                p0.append(pi)
                            else:
                                print("Bad walker")
                            
                        ndim = len(p0[0])

                        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, threads=1)
                        pos, _, _ = sampler.run_mcmc(p0, 200)
                        sampler.reset()
                        sampler.run_mcmc(pos, 200)

                        chain = sampler.flatchain
                        specorderrv.append(np.median(chain[:, 0]))

                np.savetxt(filelist[i]+".ccf.rv", np.array(specorderrv))

        except None:
            pass

    import multiprocessing
    processes = []
    for i in range(len(filelist)):
        p = multiprocessing.Process(target=runfits, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    for i in range(len(filelist)):
        rv_file = f"{filelist[i]}.ccf.rv"
        if os.path.exists(rv_file):
            rvout.append(np.loadtxt(rv_file))

    return rvout

def mmd(x):
    stdev = np.nanstd(x)
    mask = np.abs(x - np.nanmedian(x)) < 5 * stdev
    x = x[mask]
    return 1.35 * np.nanmedian(np.abs(x - np.nanmedian(x))) / np.sqrt(float(len(x)))

def analyseRVs(filelist):
    headerlist = ["bjd", "selfrv", "selfrv_err", "maskrv", "maskrv_err", "fwhm", "fwhm_err", "flux_asym", "flux_asym_err", "bis", "bis_err"]

    ccrvs = np.loadtxt(filelist[0]+".ccf.rv")
    orderi = 1
    ordere = len(ccrvs)

    rvarray = []
    for i in range(len(filelist)):
        ccrvs = np.loadtxt(filelist[i]+".ccf.rv")
        ccfrow = np.abs(ccrvs[orderi:ordere] - np.nanmedian(ccrvs[orderi:ordere]))
        rvarray.append(ccfrow)

    rvarray = np.array(rvarray)
    print(rvarray)
    print(np.shape(rvarray))
    stdmask = np.std(rvarray, axis=0)
    plt.plot(stdmask)
    plt.show()
    mask = stdmask < 0.02
    print(stdmask, len(stdmask))

    orderlist = np.arange(len(mask))
    print('orders', orderlist[mask])

    rvlist = []
    weights = 1 / stdmask[mask]
    
    for i in range(len(filelist)):
        ccrvs = np.loadtxt(filelist[i]+".ccf.rv")
        bjd, maskrv, maskrverr, fwhm, fwhmerr, flux_asym, flux_asym_err, bis, bis_err, drift = read_spec.read_header(filelist[i], return_full=True)
        rv = np.median(ccrvs[orderi:ordere][mask])

        mask_data = ccrvs[orderi:ordere][mask]
        mask_weights = weights

        sorter = np.argsort(mask_data)
        mask_data = mask_data[sorter]
        mask_weights = mask_weights[sorter]

        cumulative_weight = np.cumsum(mask_weights)
        cutoff = 0.5 * np.sum(mask_weights)

        rv_median = mask_data[np.searchsorted(cumulative_weight, cutoff)]

        deviations = np.abs(mask_data - rv_median)

        dev_sorter = np.argsort(deviations)
        sorted_devs = deviations[dev_sorter]
        sorted_dev_weights = mask_weights[dev_sorter]

        cum_dev_weight = np.cumsum(sorted_dev_weights)
        cutoff_dev = 0.5 * np.sum(sorted_dev_weights)

        w_mad = sorted_devs[np.searchsorted(cum_dev_weight, cutoff_dev)]

        rverr = 1.2533 * w_mad / np.sqrt(np.sum(mask))

        rvlist.append([ccrvs[0], rv, rverr, maskrv, maskrverr, fwhm, fwhmerr, flux_asym, flux_asym_err, bis, bis_err])

    rvlist = np.array(rvlist)
    print(len(rvlist))
    plt.errorbar(rvlist[:,0], rvlist[:,1] - np.nanmedian(rvlist[:,1]), rvlist[:,2], marker='o', capsize=0, linestyle='none', color='k')
    plt.errorbar(rvlist[:,0], rvlist[:,3] - np.nanmedian(rvlist[:,3]), rvlist[:,4], marker='o', capsize=0, linestyle='none', color='r')

    print(np.nanstd(rvlist[:,1]), np.nanmean(rvlist[:,2]), np.nanstd(rvlist[:,3]), np.nanmean(rvlist[:,4]))

    df = pd.DataFrame(data=rvlist, columns=headerlist)
    df.to_csv("LHS6050.rv.csv", index=False)

    plt.show()

if __name__ == "__main__":
    filelist = glob.glob("telcor/*S2D_TELL_CORR_A.fits")
    fitrv(filelist)
    analyseRVs(filelist)
