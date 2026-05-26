LHS6050 ESPRESSO Radial Velocity Pipeline (selfcc)


A script to extract RVs from ESPRESSO S2D spectra of LHS 6050. It generates a high-SNR master template from the observations, performs order-by-order FFT cross-correlation.

Requirements
Requires Python 3
pandas,matplotlib,scipy,emcee,astropy

Usage
Ensure telluric.dat and the telcor/ directory (containing the 65 .fits files) are in the same folder as selfcc.py, then run:
python selfcc.py

Note: The script uses multiprocessing and may take some time to finish.

Outputs
telcor/*.ccf.rv: Intermediate text files containing the order-by-order RVs for each spectrum.

LHS6050.rv.csv: The final aggregated catalog containing Barycentric Julian Dates (bjd), pipeline RVs (selfrv), header RVs (maskrv), and their respective errors/asymmetry metrics.
