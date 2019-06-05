"""
Collapse a FITS image cube down to a velocity map using one of the methods
implemented in bettermoments. This file should only contain basic collapsing
functions, wrappers of more complex functions located in methods.py.
"""

import argparse
import numpy as np
import scipy.constants as sc
from astropy.io import fits
from scipy.ndimage.filters import convolve1d


def collapse_quadratic(velax, data, linewidth=None, rms=None, N=5, axis=0):
    """
    Collapse the cube using the quadratic method. Will return the line center,
    v0, and the uncertainty on this, as well as the line peak, Fnu, and the
    uncertainty on that.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        linewidth (Optional[float]): Doppler width of the line. If specified
            will be used to smooth the data prior to the quadratic fit.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        v0 (ndarray): Line center in the same units as velax.
        dv0 (ndarray): Uncertainty on v0 in the same units as velax.
        Fnu (ndarray): Line peak in the same units as the data.
        dFnu (ndarray): Uncertainty in Fnu in the same units as the data.
    """
    from bettermoments.methods import quadratic
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    if linewidth > 0.0:
        linewidth = abs(linewidth / chan / np.sqrt(2.))
    else:
        linewidth = None
    return quadratic(data, x0=velax[0], dx=chan, linewidth=linewidth,
                     uncertainty=np.ones(data.shape)*rms, axis=axis)


def collapse_zeroth(velax, data, rms=None, N=5, threshold=None, mask=None,
                    axis=0):
    """
    Collapses the cube by integrating along the spectral axis. It will return
    the integrated intensity along the spectral axis, I0, and the associated
    uncertainty, dI0.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        I0 (ndarray): Integrated intensity along provided axis.
        dI0 (ndarray): Uncertainty on I0 in the same units as I0.
    """
    from bettermoments.methods import integrated_intensity
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    return integrated_intensity(data=data, dx=abs(chan), threshold=threshold,
                                rms=rms, mask=_read_mask(mask, data),
                                axis=axis)


def collapse_maximum(velax, data, rms=None, N=5, axis=0):
    """
    Coallapses the cube by taking the velocity of the maximum intensity pixel
    along the spectral axis. This is the 'ninth' moment in CASA's immoments
    task. Can additionally return the peak value ('eighth moment'). This will
    return the line center, v0, and its ucertainty, along with the like peak,
    Fnu, and its uncertainty, dFnu.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along.

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
    from bettermoments.methods import peak_pixel
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    v0, dv0, Fnu = peak_pixel(data=data, x0=velax[0], dx=chan, axis=axis)
    dFnu = rms * np.ones(v0.shape)
    return v0, dv0, Fnu, dFnu


def collapse_first(velax, data, rms=None, N=5, threshold=None, mask=None,
                   axis=0):
    """
    Collapses the cube using the intensity weighted average velocity (or first
    moment map). For a symmetric line profile this will be the line center.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        mask (Optional[ndarray]): A boolean or integeter array masking certain
            pixels to be excluded in the fitting. Can either be a full 3D mask,
            a 2D channel mask, or a 1D spectrum mask.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        v0 (ndarray): Intensity weighted average velocity.
        dv0 (ndarray): Uncertainty in the intensity weighted average velocity.
    """
    from bettermoments.methods import intensity_weighted_velocity as first
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    return first(data=data, x0=velax[0], dx=chan, rms=rms, threshold=threshold,
                 mask=_read_mask(mask, data), axis=axis)

def collapse_second(velax, data, rms=None, N=5, threshold=None, mask=None,
                    axis=0):
    """
    Collapses the cube using the intensity-weighted average velocity dispersion
    (or second moment). For a symmetric line profile this will be a measure of
    the line width.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        mask (Optional[ndarray]): A boolean or integeter array masking certain
            pixels to be excluded in the fitting. Can either be a full 3D mask,
            a 2D channel mask, or a 1D spectrum mask.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        dV (ndarray): Intensity weighted velocity dispersion.
        ddV (ndarray): Uncertainty in the intensity weighted velocity
            disperison.
    """
    from bettermoments.methods import intensity_weighted_dispersion as second
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    return second(data=data, x0=velax[0], dx=chan, rms=rms,
                  threshold=threshold, mask=_read_mask(mask, data), axis=axis)


def collapse_width(velax, data, linewidth=0.0, rms=None, N=5, threshold=None,
                   mask=None, axis=0):
    """
    Returns an effective width, a rescaled ratio of the integrated intensity
    and the line peak. For a Gaussian line profile this would be the Doppler
    width. This should be more robust against noise than second moment maps.
    This returns all four parameters.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        linewidth (Optional[float]): Doppler width of the line. If specified
            will be used to smooth the data prior to the quadratic fit.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        mask (Optional[ndarray]): A boolean or integeter array masking certain
            pixels to be excluded in the fitting. Can either be a full 3D mask,
            a 2D channel mask, or a 1D spectrum mask.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        I0 (ndarray): Integrated intensity along provided axis.
        dI0 (ndarray): Uncertainty on I0 in the same units as I0.
        v0 (ndarray): Line center in the same units as velax.
        dv0 (ndarray): Uncertainty on v0 in the same units as velax. Will be
            the channel width.
        Fnu (ndarray): Line peak in the same units as the data.
        dFnu (ndarray): Uncertainty in Fnu in the same units as the data. Will
            be the RMS calculate from the first and last channels.
        dV (ndarray): Effective velocity dispersion.
        ddV (ndarray): Uncertainty on the velocity dispersion.
    """
    from bettermoments.methods import integrated, quadratic
    mask = _read_mask(mask, data)
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    I0, dI0 = integrated(data=data, dx=chan, uncertainty=rms,
                         threshold=threshold*rms, mask=mask, axis=axis)
    if linewidth > 0.0:
        linewidth = abs(linewidth / chan / np.sqrt(2.))
    else:
        linewidth = None
    v0, dv0, Fnu, dFnu = quadratic(data, x0=velax[0], dx=chan,
                                   uncertainty=np.ones(data.shape)*rms,
                                   linewidth=linewidth, axis=axis)
    dV = I0 / Fnu / np.sqrt(np.pi)
    ddV = dV * np.hypot(dFnu / Fnu, dI0 / I0)
    return I0, dI0, v0, dv0, Fnu, dFnu, dV, ddV


def _get_cube(path):
    """Return the data and velocity axis from the cube."""
    return _get_data(path), _get_velax(path)


def _get_data(path):
    """Read the FITS cube. Should remove Stokes axis if attached."""
    data = np.squeeze(fits.getdata(path))
    return np.where(np.isfinite(data), data, 0.0)


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
    x1, x2 = np.percentile(np.arange(data.shape[2]), [45, 55])
    y1, y2 = np.percentile(np.arange(data.shape[1]), [45, 55])
    x1, x2, y1, y2, N = int(x1), int(x2), int(y1), int(y2), int(N)
    rms = np.nanstd([data[:N, y1:y2, x1:x2], data[-N:, y1:y2, x1:x2]])
    return rms * np.ones(data[0].shape)


def _read_mask(mask, data):
    """Read in the mask and make sure it is the same shape as the data."""
    if mask:
        mask = _get_data(mask)
        if mask.shape != data.shape:
            raise ValueError("Mismatch in mask and data shape.")
    else:
        mask = None
    return mask


def _verify_data(data, velax, rms=None, N=5, axis=0):
    """Veryify the data shape and read in image properties."""
    if data.shape[axis] != velax.size:
        raise ValueError("Must collapse along the spectral axis!")
    if rms is None:
        rms = _estimate_RMS(data=data, N=N)
    chan = np.diff(velax).mean()
    return rms, chan


def _collapse_beamtable(path):
    """Returns the median beam from the CASA beam table if present."""
    header = fits.getheader(path)
    if header.get('CASAMBM', False):
        beam = fits.open(path)[1].data
        beam = np.median([b[:3] for b in beam.view()], axis=0)
        return beam[0] / 3600., beam[1] / 3600., beam[2]
    try:
        return header['bmaj'], header['bmin'], header['bpa']
    except KeyError:
        return abs(header['cdelt1']), abs(header['cdelt2']), 0.0


def _write_header(path, bunit):
    """Write a new header for the saved file."""
    header = fits.getheader(path, copy=True)
    new_header = fits.PrimaryHDU().header
    new_header['SIMPLE'] = True
    new_header['BITPIX'] = -64
    new_header['NAXIS'] = 2
    beam = _collapse_beamtable(path)
    new_header['BMAJ'] = beam[0]
    new_header['BMIN'] = beam[1]
    new_header['BPA'] = beam[2]
    if bunit is not None:
        new_header['BUNIT'] = bunit
    else:
        new_header['BUNIT'] = header['BUNIT']
    for i in [1, 2]:
        for val in ['NAXIS', 'CTYPE', 'CRVAL', 'CDELT', 'CRPIX', 'CUNIT']:
            key = '%s%d' % (val, i)
            if key in header.keys():
                new_header[key] = header[key]
    try:
        new_header['RESTFRQ'] = header['RESTFRQ']
    except KeyError:
        new_header['RESTFREQ'] = header['RESTFREQ']
    try:
        new_header['SPECSYS'] = header['SPECSYS']
    except KeyError:
        pass
    new_header['COMMENT'] = 'made with bettermoments'
    return new_header


def _save_array(original_path, new_path, array, overwrite=True, bunit=None):
    """Use the header from `original_path` to save a new FITS file."""
    header = _write_header(original_path, bunit)
    fits.writeto(new_path, array.astype(float), header, overwrite=overwrite,
                 output_verify='silentfix')


def main():

    # Define all the parser arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to the FITS cube.')
    parser.add_argument('-method', default='quadratic',
                        help='Method used to collapse cube. Current available '
                             'methods are: quadratic, maximum, zeroth, first, '
                             'second and width.')
    parser.add_argument('-clip', default=5.0, type=float,
                        help='Mask values below this SNR.')
    parser.add_argument('-fill', default=np.nan, type=float,
                        help='Fill value for masked pixels. Default is NaN.')
    parser.add_argument('-linewidth', default=0.0, type=float,
                        help='Linewidth in m/s used to smooth data.'
                             'For best results, use a linewidth comprable to '
                             'the intrinsic linewidth.')
    parser.add_argument('-rms', default=None, type=float,
                        help='Estimated RMS noise from a line free channel. '
                             'Same units as the brightness unit.')
    parser.add_argument('-N', default=5, type=int,
                        help='Number of end channels to use to estimate RMS.')
    parser.add_argument('-mask', default='',
                        help='Path to the mask FITS cube. Must have the same '
                             'shape as the input data.')
    parser.add_argument('-axis', default=0, type=int,
                        help='Axis to collapse the cube along. Default is 0.')
    parser.add_argument('-downsample', default=1, type=int,
                        help='Downsample the data by this factor.')
    parser.add_argument('-overwrite', default=True, type=bool,
                        help='Overwrite existing files with the same name.')
    parser.add_argument('--nomask', action='store_true',
                        help='Clip the final moment map using the provided '
                             '`clip` value. Default is True.')
    parser.add_argument('--silent', action='store_true',
                        help='Run silently.')
    args = parser.parse_args()
    args.method = args.method.lower()

    # Read in the cube [Jy/beam] and velocity axis [m/s].

    data, velax = _get_cube(args.path)
    I0, dI0, dV, ddV = None, None, None, None
    v0, dv0, Fnu, dFnu = None, None, None, None

    # If resampled is requested, average over the data and convolve with a top
    # hat function to minimic the channelization of ALMA.

    if args.downsample > 1:
        N = int(args.downsample)
        data = np.array([np.nanmean(data[i*N:(i+1)*N], axis=0)
                         for i in range(int(velax.size / N))])
        kernel = np.ones(N) / float(N)
        data = convolve1d(data, kernel, mode='reflect', axis=args.axis)
        velax = np.array([np.average(velax[i*N:(i+1)*N])
                          for i in range(int(velax.size / N))])

    # Collapse the cube with the approrpriate method.

    if not args.silent:
        print("Calculating maps.")

    if args.method == 'quadratic':
        out = collapse_quadratic(velax=velax, data=data, N=args.N,
                                 axis=args.axis, linewidth=args.linewidth,
                                 rms=args.rms)
        v0, dv0, Fnu, dFnu = out

    elif (args.method == 'maximum' or args.method == 'eighth'):
        out = collapse_maximum(velax=velax, data=data, rms=args.rms, N=args.N,
                               axis=args.axis)
        v0, dv0, Fnu, dFnu = out

    elif args.method == 'second':
        dV, ddV = collapse_second(velax=velax, data=data, threshold=args.clip,
                                  rms=args.rms, N=args.N, mask=args.mask,
                                  axis=args.axis)

    elif args.method == 'first':
        v0, dv0 = collapse_first(velax=velax, data=data, threshold=args.clip,
                                 rms=args.rms, N=args.N, mask=args.mask,
                                 axis=args.axis)

    elif args.method == 'zeroth':
        out = collapse_zeroth(velax=velax, data=data, threshold=args.clip,
                              rms=args.rms, N=args.N, mask=args.mask,
                              axis=args.axis)
        I0, dI0 = out

    elif args.method == 'width':
        out = collapse_width(velax=velax, data=data, threshold=args.clip,
                             rms=args.rms, N=args.N, mask=args.mask,
                             axis=args.axis, linewidth=args.linewidth)
        I0, dI0, v0, dv0, Fnu, dFnu, dV, ddV = out

    else:
        raise ValueError("Unknown method.")

    # Mask the data. If no uncertainties are found for dFnu, use the RMS.

    if args.clip > 0.0 and not args.nomask:

        if not args.silent:
            print("Masking maps.")

        if Fnu is None:
            signal = collapse_maximum(velax=velax, data=data)[2]
        else:
            signal = Fnu
        if dFnu is None:
            noise = collapse_maximum(velax=velax, data=data)[3]
        else:
            noise = dFnu

        rndm = abs(1e-10 * np.random.randn(noise.size).reshape(noise.shape))
        mask = np.where(noise != 0.0, noise, rndm)
        mask = np.where(np.isfinite(mask), mask, rndm)
        mask = np.where(np.isfinite(signal), signal, 0.0) / mask >= args.clip
    else:
        mask = np.moveaxis(np.ones(data.shape), args.axis, 0)
        mask = np.ones(mask[0].shape)

    # Save the files.

    if not args.silent:
        print("Saving maps.")

    if I0 is not None:
        I0 = np.where(mask, I0, args.fill)
        _save_array(args.path, args.path.replace('.fits', '_I0.fits'), I0,
                    overwrite=args.overwrite, bunit='Jy/beam m/s')
    if dI0 is not None:
        dI0 = np.where(mask, abs(dI0), args.fill)
        _save_array(args.path, args.path.replace('.fits', '_dI0.fits'), dI0,
                    overwrite=args.overwrite, bunit='Jy/beam m/s')

    if v0 is not None:
        v0 = np.where(mask, v0, args.fill)
        _save_array(args.path, args.path.replace('.fits', '_v0.fits'), v0,
                    overwrite=args.overwrite, bunit='m/s')
    if dv0 is not None:
        dv0 = np.where(mask, abs(dv0), args.fill)
        _save_array(args.path, args.path.replace('.fits', '_dv0.fits'), dv0,
                    overwrite=args.overwrite, bunit='m/s')

    if Fnu is not None:
        Fnu = np.where(mask, Fnu, args.fill)
        _save_array(args.path, args.path.replace('.fits', '_Fnu.fits'), Fnu,
                    overwrite=args.overwrite)
    if dFnu is not None:
        dFnu = np.where(mask, abs(dFnu), args.fill)
        _save_array(args.path, args.path.replace('.fits', '_dFnu.fits'), dFnu,
                    overwrite=args.overwrite)

    if dV is not None:
        dV = np.where(mask, dV, args.fill)
        _save_array(args.path, args.path.replace('.fits', '_dV.fits'), dV,
                    overwrite=args.overwrite, bunit='m/s')
    if ddV is not None:
        ddV = np.where(mask, abs(ddV), args.fill)
        _save_array(args.path, args.path.replace('.fits', '_ddV.fits'), ddV,
                    overwrite=args.overwrite, bunit='m/s')


if __name__ == '__main__':
    main()
