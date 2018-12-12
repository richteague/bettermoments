"""
Collapse a FITS image cube down to a velocity map using one of the methods
implemented in bettermoments. This file should only contain basic collapsing
functions (if they are easily implemented in NumPy), more complex functions
should be located in bettermoments.py.
"""

import argparse
import numpy as np
import scipy.constants as sc
from astropy.io import fits


def collapse_quadratic(velax, data, dV=None, rms=None, N=5):
    """
    Collapse the cube using the quadratic method.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        dV (Optional[float]): Doppler width of the line. If specified will be
            used to smooth the data prior to the quadratic fit.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.

    Returns:
        v0 (ndarray): Line center in the same units as velax.
        dv0 (ndarray): Uncertainty on v0 in the same units as velax.
        Fnu (ndarray): Line peak in the same units as the data.
        dFnu (ndarray): Uncertainty in Fnu in the same units as the data.
    """

    try:
        from bettermoments.methods import quadratic
    except ImportError:
        raise ImportError("Cannot find `methods`.")

    # Check the shape.
    if data.shape[0] != velax.size:
        raise ValueError("Must have the zeroth axis as velocity.")

    # Estimate the uncertainty.
    if rms is None:
        rms = _estimate_RMS(data, N=N)
        rms = np.nanmean(rms)

    # Convert linewidth units and make sure positive.
    chan = abs(np.diff(velax).mean())
    if dV is not None:
        dV /= chan

    return quadratic(data, x0=velax[0], dx=chan, uncertainty=rms, linewidth=dV)


def collapse_maximum(velax, data, rms=None, N=5):
    """
    Coallapses the cube by taking the velocity of the maximum intensity pixel
    along the spectral axis. This is the 'ninth' moment in CASA's immoments
    task. Can additionally return the peak value ('eighth moment').

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.

    Returns:
        v0 (ndarray): Line center in the same units as velax.
        dv0 (ndarray): Uncertainty on v0 in the same units as velax. Will be
            the channel width.
        Fnu (ndarray): If return_peak=True. Line peak in the same units as
            the data.
        dFnu (ndarray): If return_peak=True. Uncertainty in Fnu in the same
            units as the data. Will be the RMS calculate from the first and
            last channels.
    """
    v0 = np.take(velax, np.argmax(data, axis=0))
    dv0 = np.ones(v0.shape) * abs(np.diff(velax).mean())
    Fnu = np.nanmax(data, axis=0)
    if rms is None:
        dFnu = _estimate_RMS(data, N=N)
    else:
        dFnu = rms * np.ones(v0.shape)
    return v0, dv0, Fnu, dFnu


def collapse_first(velax, data, clip=None, rms=None, N=5, mask=None):
    """
    Collapses the cube using the intensity weighted average velocity (or first
    moment map).

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        clip (Optional[float]): Clip any pixels below this SNR level.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        mask (Optional[ndarray]): A boolean or integeter array masking certain
            pixels to be excluded in the fitting. Can either be a full 3D mask,
            a 2D channel mask, or a 1D spectrum mask.

    Returns:
        v0 (ndarray): Line center in the same units as velax.
    """

    # Calculate the weights.
    weight_noise = 1e-20 * np.random.randn(data.size).reshape(data.shape)
    weights = np.where(np.isfinite(data), data, weight_noise)
    no_signal = np.sum(data, axis=0) > 0.0
    weights = np.where(no_signal[None, :, :], weights, weight_noise)

    # Apply the SNR clipping.
    if clip is not None:
        Fnu, dFnu = collapse_maximum(velax, data, N=N)[2:]
        if rms is None:
            rms = _estimate_RMS(data, N=N)
        dFnu = rms * np.ones(Fnu.shape)
        snrmask = (Fnu / dFnu >= clip)
        weights = np.where(snrmask, weights, weight_noise)

    # Mask the data.
    if mask is not None:
        if mask.shape == data.shape:
            weights = np.where(mask, weights, weight_noise)
        elif mask.shape == data[0].shape:
            weights = np.where(mask[None, :, :], weights, weight_noise)
        elif mask.shape == velax.shape:
            weights = np.where(mask[:, None, None], weights, weight_noise)
        else:
            raise ValueError("Unknown shape for the mask.")

    # Calculate the intensity weighted average velocity. Apply the mask for
    # regions with no signal in them, or clipped.
    velax_cube = np.ones(data.shape) * velax[:, None, None]
    v0 = np.average(velax_cube, weights=weights, axis=0)
    mask = np.logical_and(no_signal, np.sum(weights) > 1e-10)
    return np.where(mask, v0, np.nan)


def _get_cube(path):
    """Return the data and velocity axis from the cube."""
    return _get_data(path), _get_velax(path)


def _get_data(path):
    """Read the FITS cube. Should remove Stokes axis if attached."""
    return np.squeeze(fits.getdata(path))


def _get_velax(path):
    """Read the velocity axis information."""
    return _read_velocity_axis(fits.getheader(path))


def _read_rest_frequency(header):
    """Read the rest frequency in [Hz]."""
    try:
        nu = header['restfreq']
    except KeyError:
        try:
            nu = header['restfrq']
        except KeyError:
            nu = header['crval3']
    return nu


def _read_velocity_axis(header):
    """Wrapper for _velocityaxis and _spectralaxis."""
    if 'freq' in header['ctype3'].lower():
        specax = _read_spectral_axis(header)
        nu = _read_rest_frequency(header)
        velax = (nu - specax) * sc.c / nu
    else:
        velax = _read_spectral_axis(header)
    return velax


def _read_spectral_axis(header):
    """Returns the spectral axis in [Hz] or [m/s]."""
    specax = (np.arange(header['naxis3']) - header['crpix3'] + 1.0)
    return header['crval3'] + specax * header['cdelt3']


def _estimate_RMS(data, N=5):
    """Return the estimated RMS in the first and last N channels."""
    x1, x2 = np.percentile(np.arange(data.shape[2]), [25, 75])
    y1, y2 = np.percentile(np.arange(data.shape[1]), [25, 75])
    x1, x2, y1, y2, N = int(x1), int(x2), int(y1), int(y2), int(N)
    rms = np.nanstd([data[:N, y1:y2, x1:x2], data[-N:, y1:y2, x1:x2]])
    return rms * np.ones(data[0].shape)


def _save_array(original_path, new_path, array):
    """Use the header from `original_path` to save a new FITS file."""
    header = fits.getheader(original_path)

    # Remove pesky values.
    for key in ['history']:
        try:
            header.pop(key, None)
        except KeyError:
            pass

    # Update the spectral axis.
    header['naxis3'] = 1
    header['cdelt3'] = np.nanstd(array)
    header['crval3'] = np.nanmean(array)
    header['crpix3'] = 1e0

    fits.writeto(new_path, array, header, overwrite=True)


def main():

    # Define all the parser arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to the FITS cube.')
    parser.add_argument('-method', default='quadratic',
                        help='Method used to collapse cube.')
    parser.add_argument('-clip', default=5.0, type=float,
                        help='Mask values below this SNR.')
    parser.add_argument('-fill', default=np.nan, type=float,
                        help='Fill value for masked pixels.')
    parser.add_argument('-dV', default=None, type=float,
                        help='Linewidth used to smooth data.')
    parser.add_argument('-rms', default=None, type=float,
                        help='Estimated RMS noise from a line free channel.')
    parser.add_argument('-N', default=5, type=int,
                        help='Number of end channels to use to estimate RMS.')
    parser.add_argument('--silent', action='store_true',
                        help='Run silently.')
    args = parser.parse_args()

    # Read in the cube [Jy/beam] and velocity axis [m/s].

    data, velax = _get_cube(args.path)
    v0, dv0, Fnu, dFnu = None, None, None, None

    # Collapse the cube with the approrpriate method.

    if args.method.lower() == 'quadratic':
        out = collapse_quadratic(velax=velax, data=data, dV=args.dV,
                                 rms=args.rms, N=args.N)
        v0, dv0, Fnu, dFnu = out

    elif args.method.lower() == 'maximum':
        out = collapse_maximum(velax=velax, data=data, rms=args.rms, N=args.N)
        v0, dv0, Fnu, dFnu = out

    elif args.method.lower() == 'first':
        if args.mask is not None:
            args.mask = _get_data(args.mask)
            if args.mask.shape != data.shape:
                raise ValueError("Mismatch in data and mask shape.")
        v0 = collapse_first(velax=velax, data=data, clip=args.clip,
                            rms=args.rms, N=args.N, mask=args.mask)
    else:
        raise ValueError("Unknown method.")

    # Mask the data. If no uncertainties are found for dFnu, use the RMS.

    if args.clip is not None:

        if not args.silent:
            print("Masking maps.")

        if Fnu is None:
            signal = collapse_maximum(velax, data)[2]
        else:
            signal = Fnu
        if dFnu is None:
            noise = collapse_maximum(velax, data)[3]
        else:
            noise = dFnu

        rndm = abs(1e-10 * np.random.randn(noise.size).reshape(noise.shape))
        mask = np.where(noise != 0.0, noise, rndm)
        mask = np.where(np.isfinite(mask), mask, rndm)
        mask = np.where(np.isfinite(signal), signal, 0.0) / mask >= args.clip

        v0 = np.where(mask, v0, args.fill)
        if dv0 is not None:
            dv0 = np.where(mask, dv0, args.fill)
        if Fnu is not None:
            Fnu = np.where(mask, Fnu, args.fill)
        if dFnu is not None:
            dFnu = np.where(mask, dFnu, args.fill)

    # Save the files.

    if not args.silent:
        print("Saving maps.")
    if v0 is not None:
        _save_array(args.path, args.path.replace('.fits', '_v0.fits'), v0)
    if dv0 is not None:
        _save_array(args.path, args.path.replace('.fits', '_dv0.fits'), dv0)
    if Fnu is not None:
        _save_array(args.path, args.path.replace('.fits', '_Fnu.fits'), Fnu)
    if dFnu is not None:
        _save_array(args.path, args.path.replace('.fits', '_dFnu.fits'), dFnu)


if __name__ == '__main__':
    main()
