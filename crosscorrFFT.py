import numpy as np
from scipy.interpolate import interp1d

c = 299792.458  # km/s

def crosscorrRV_fft_linear(wave, flux, templ_wave, templ_flux,
                           rvmin=-50, rvmax=50, drv=0.01,
                           mode="doppler", kind="linear", pad_factor=2):
    """
    Fourier-domain cross-correlation for RV measurement.

    Parameters
    ----------
    wave : 1D array
        Observed wavelengths [same units as templ_wave].
    flux : 1D array
        Observed fluxes.
    templ_wave : 1D array
        Template wavelengths.
    templ_flux : 1D array
        Template fluxes.
    rvmin, rvmax : float
        Minimum and maximum radial velocity to return (km/s).
    drv : float
        Velocity sampling step (km/s).
    mode : {"doppler","lin"}
        If "doppler", shift by v/c multiplicative; if "lin", shift by constant ?w.
    kind : str
        Interpolation kind for resampling (passed to scipy.interpolate).
    pad_factor : int
        Zero-padding factor to minimize wrap-around in the FFT.

    Returns
    -------
    rv_grid : 1D array
        Radial velocities (km/s) corresponding to the CCF.
    ccf : 1D array
        Cross-correlation function evaluated on rv_grid.
    """
    # 1) Define log-lambda grid
    lnwave = np.log(wave)
    dln = np.min(np.diff(lnwave))
    N = int(np.ceil((rvmax - rvmin)/drv)) + 1
    # grid size: pad to next power of two times pad_factor
    M = int(2**np.ceil(np.log2(len(wave)))) * pad_factor

    # build uniform ln? grid centered on median
    ln_center = np.median(lnwave)
    ln_grid = ln_center + dln * (np.arange(M) - M//2)

    # 2) Interpolate both spectra onto ln? grid
    interp_obs  = interp1d(lnwave, flux,       kind=kind,
                           bounds_error=False, fill_value=0.0)
    interp_temp = interp1d(np.log(templ_wave), templ_flux,
                           kind=kind, bounds_error=False, fill_value=0.0)

    f_obs  = interp_obs(ln_grid)
    f_temp = interp_temp(ln_grid)

    # 3) FFT multiply and inverse-FFT
    F_obs  = np.fft.fft(f_obs)
    F_temp = np.fft.fft(f_temp)
    cc_raw = np.fft.ifft(F_obs * np.conj(F_temp))

    # shift zero lag to center
    cc_shifted = np.fft.fftshift(cc_raw).real

    # 4) Build RV grid
    dv = c * dln
    lag_inds = np.arange(-M//2, M//2)
    rv_full = lag_inds * dv

    # 5) Extract the desired rv range
    mask = (rv_full >= rvmin) & (rv_full <= rvmax)
    rv_grid = rv_full[mask]
    ccf     = cc_shifted[mask]
    
    # 6) Linearise the grid
    rv_grid = np.arange(rvmin, rvmax, drv)
    ccf = np.interp(rv_grid, rv_full, cc_shifted)

    return rv_grid, ccf
