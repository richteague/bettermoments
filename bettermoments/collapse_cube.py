"""
Collapse a data cube down to a summary statistic using various methods. Now
returns statistical uncertainties for all statistics.

TODO:
    - Deal with the fact we're using three different convolution routines.
"""

import argparse
import numpy as np
import multiprocessing
from astropy.io import fits
import scipy.constants as sc


def _get_bunits(path):
    """Return the dictionary of units."""
    bunits = {}
    flux_unit = fits.getheader(path)['bunit']
    bunits['M0'] = '{} m/s'.format(flux_unit)
    bunits['dM0'] = '{} m/s'.format(flux_unit)
    bunits['M1'] = 'm/s'
    bunits['dM1'] = 'm/s'
    bunits['M2'] = 'm/s'
    bunits['dM2'] = 'm/s'
    bunits['M8'] = '{}'.format(flux_unit)
    bunits['dM8'] = '{}'.format(flux_unit)
    bunits['M9'] = 'm/s'
    bunits['dM9'] = 'm/s'
    bunits['v0'] = 'm/s'
    bunits['dv0'] = 'm/s'
    bunits['Fnu'] = '{}'.format(flux_unit)
    bunits['dFnu'] = '{}'.format(flux_unit)
    bunits['dV'] = 'm/s'
    bunits['ddV'] = 'm/s'
    bunits['gv0'] = bunits['v0']
    bunits['gFnu'] = bunits['Fnu']
    bunits['gdV'] = bunits['dV']
    bunits['gtau'] = ''
    bunits['dgv0'] = bunits['gv0']
    bunits['dgFnu'] = bunits['gFnu']
    bunits['dgdV'] = bunits['gdV']
    bunits['dgtau'] = ''
    bunits['ghv0'] = bunits['v0']
    bunits['ghFnu'] = bunits['Fnu']
    bunits['ghdV'] = bunits['dV']
    bunits['ghh3'] = ''
    bunits['ghh4'] = ''
    bunits['dghv0'] = bunits['gv0']
    bunits['dghFnu'] = bunits['gFnu']
    bunits['dghdV'] = bunits['gdV']
    bunits['dghh3'] = bunits['ghh3']
    bunits['dghh4'] = bunits['ghh4']
    bunits['mask'] = 'bool'
    return bunits


def load_cube(path):
    """Return the data and velocity axis from the cube."""
    return _get_data(path), _get_velax(path), _get_bunits(path)


def _get_data(path, fill_value=0.0):
    """Read the FITS cube. Should remove Stokes axis if attached."""
    data = np.squeeze(fits.getdata(path))
    return np.where(np.isfinite(data), data, fill_value)


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


def estimate_RMS(data, N=5):
    """Return the estimated RMS in the first and last N channels."""
    x1, x2 = np.percentile(np.arange(data.shape[2]), [25, 75])
    y1, y2 = np.percentile(np.arange(data.shape[1]), [25, 75])
    x1, x2, y1, y2, N = int(x1), int(x2), int(y1), int(y2), int(N)
    rms = np.nanstd([data[:N, y1:y2, x1:x2], data[-N:, y1:y2, x1:x2]])
    return rms


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
        try:
            new_header['RESTFREQ'] = header['RESTFREQ']
        except KeyError:
            new_header['RESTFREQ'] = 0.0
    try:
        new_header['SPECSYS'] = header['SPECSYS']
    except KeyError:
        pass
    new_header['COMMENT'] = 'made with bettermoments'
    return new_header


def _collapse_beamtable(path):
    """Returns the median beam from the CASA beam table if present."""
    header = fits.getheader(path)
    if header.get('CASAMBM', False):
        try:
            beam = fits.open(path)[1].data
            beam = np.max([b[:3] for b in beam.view()], axis=0)
            return beam[0] / 3600., beam[1] / 3600., beam[2]
        except IndexError:
            print('WARNING: No beam table found despite CASAMBM flag.')
            return abs(header['cdelt1']), abs(header['cdelt2']), 0.0
    try:
        return header['bmaj'], header['bmin'], header['bpa']
    except KeyError:
        return abs(header['cdelt1']), abs(header['cdelt2']), 0.0


def _get_pix_per_beam(path):
    """Returns the number of pixels per beam FWHM."""
    bmaj, _, _ = _collapse_beamtable(path)
    return bmaj / abs(fits.getheader(path)['cdelt1'])


def save_to_FITS(original_path, new_path, array, overwrite=True, bunit=None):
    """Use the header from `original_path` to save a new FITS file."""
    header = _write_header(original_path, bunit)
    fits.writeto(new_path, array.astype(float), header, overwrite=overwrite,
                 output_verify='silentfix')


def _save_smoothed_data(data, args):
    """Save the smoothed data for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'smoothed data used for moment map creation'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-smooth {}'.format(args.smooth)
    header['COMMENT'] = '-polyorder {}'.format(args.polyorder)
    new_path = args.path.replace('.fits', '_smoothed_data.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_mask(data, args):
    """Save the combined mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'mask used for moment map creation'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-lastchannel {}'.format(args.lastchannel)
    header['COMMENT'] = '-firstchannel {}'.format(args.firstchannel)
    header['COMMENT'] = '-mask {}'.format(args.mask)
    header['COMMENT'] = '-clip {}'.format(args.clip)
    header['COMMENT'] = '-smooththreshold {}'.format(args.smooththreshold)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = args.path.replace('.fits', '_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_channel_count(data, args):
    """Save the number of channels used in each pixel."""
    header = fits.getheader(args.path, copy=True)
    header['BUNIT'] = 'channels'
    header['COMMENT'] = 'number of channels used in each pixel'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-lastchannel {}'.format(args.lastchannel)
    header['COMMENT'] = '-firstchannel {}'.format(args.firstchannel)
    header['COMMENT'] = '-mask {}'.format(args.mask)
    header['COMMENT'] = '-clip {}'.format(args.clip)
    header['COMMENT'] = '-smooththreshold {}'.format(args.smooththreshold)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = args.path.replace('.fits', '_channel_count.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_threshold_mask(data, args):
    """Save the smoothed data for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined threshold mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-clip {}'.format(args.clip)
    header['COMMENT'] = '-smooththreshold {}'.format(args.smooththreshold)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = args.path.replace('.fits', '_threshold_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_channel_mask(data, args):
    """Save the user-defined channel mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined channel mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-lastchannel {}'.format(args.lastchannel)
    header['COMMENT'] = '-firstchannel {}'.format(args.firstchannel)
    new_path = args.path.replace('.fits', '_channel_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_user_mask(data, args):
    """Save the user-defined velocity mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-mask {}'.format(args.mask)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = args.path.replace('.fits', '_user_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def smooth_data(data, smooth=0, polyorder=0):
    """
    Smooth the input data with a kernel of a width ``smooth``. If ``polyorder``
    is provided, will smooth with a Savitzky-Golay filter, while if
    ``polyorder=0``, the default, then only a top-hat kernel will be used. From
    experimentation, ``smooth=5`` with ``polyorder=3``provides a good result
    for noisy, but spectrally resolved data.

    ..warning::
        When smoothing low resolution data, this can substantially alter the
        line profile, so measurements must be taken with caution.

    Args:
        data (array): Data to smooth.
        smooth (optional[int]): The width of the kernel for smooth in number of
            channels.
        polyorder (optional[int]): Polynomial order for the Savitzky-Golay
            filter. This must be smaller than ``smooth``. If not provided, the
            smoothing will only be a top-hat filter.
        silent (bool): Whether to print the processes.

    Returns:
        smoothed_data (array): A smoothed copy of ``data``.
    """
    assert data.ndim == 3, "Data must have 3 dimensions to smooth."
    if smooth > 1:
        if polyorder > 0:
            from scipy.signal import savgol_filter
            smooth += 0 if smooth % 2 else 1
            smoothed_data = savgol_filter(data, smooth, polyorder=polyorder,
                                          mode='wrap', axis=0)
        else:
            from scipy.ndimage import uniform_filter1d
            a = uniform_filter1d(data, smooth, mode='wrap', axis=0)
            b = uniform_filter1d(data[::-1], smooth, mode='wrap', axis=0)[::-1]
            smoothed_data = np.mean([a, b], axis=0)
    else:
        smoothed_data = data.copy()
    return smoothed_data


def get_channel_mask(data, firstchannel=0, lastchannel=-1, user_mask=None):
    """
    Returns the channel mask (a mask for the zeroth axis) based on a first and
    last channel. A ``chan_mask`` can also be provided for more complex masks,
    however be warned that the ``firstchannel`` and ``lastchannel`` will always
    take precedence over ``chan_mask``.

    Args:
        data (array): The data array to use for masking.
        firstchanenl (optional[int]): The first channel to include. Defaults to
            the first channel.
        lastchannel (optional[int]): The last channel to include. Defaults to
            the last channel. This can be both a positive value, or a negative
            value following the normal indexing conventions, i.e. ``-1``
            describes the last channel.
        user_mask (optional[array]): A 1D array with size ``data.shape[0]``
            detailing which channels to include in the moment map creation.

    Returns:
        channel_mask (array): A mask array the same shape as ``data``.
    """
    channels = np.arange(data.shape[0])
    channel_mask = np.ones(data.shape[0]) if user_mask is None else user_mask
    assert channel_mask.shape == channels.shape
    lastchannel = channels[lastchannel] if lastchannel < 0 else lastchannel
    assert 0 <= firstchannel < lastchannel <= data.shape[0]
    channel_mask = np.where(channels >= firstchannel, channel_mask, 0)
    channel_mask = np.where(channels <= lastchannel, channel_mask, 0)
    return np.where(channel_mask[:, None, None], np.ones(data.shape), 0.0)


def get_user_mask(data, user_mask_path=None):
    """
    Returns a mask based on a user-provided file. All positive values are
    included in the mask.

    Args:
        data (array): The data array to mask.
        user_mask_path (optional[str]): Path to the FITS cube containing the
            user-defined mask.

    Returns:
        user_mask (array): A mask array the same shape as ``data``.
    """
    if user_mask_path is None:
        user_mask = np.ones(data.shape)
    else:
        user_mask = np.where(_get_data(user_mask_path) > 0, 1.0, 0.0)
    assert user_mask.shape == data.shape
    return user_mask.astype('float')


def get_threshold_mask(data, clip=None, smooth_threshold_mask=0,
                       noise_channels=5):
    """
    Returns a mask based on a sigma-clip to the input data. The most standard
    approach would be to use ``clip=3`` to mask out all pixels with intensities
    :math:`|I| \leq 3\sigma`. If you wanted to specify an asymmetric criteria
    then you can provide a tuple, ``clip=(-2, 3)`` which would mask out all
    pixels where :math:`-2\sigma \leq I \leq 3\sigma`.

    [Some discussion on the smooth_threshold_mask coming...]

    Args:
        data (array): The data array to mask.
        clip (optional[float/tuple]): The sigma clip to apply. If a single
            value is provided, this is taken to be a symmetric mask. If a tuple
            if provided, this is taking as a minimum and maximum clip value.
        smooth_threshold_mask (optional[float]): Convolution kernel FWHM in
            pixels.
        noise_channels (optional[int]): Number of channels at the start and end
            of the velocity axis to use for estimating the noise.

    Returns:
        threshold_mask (array): A mask array the same shape as ``data``.
    """

    # No clipping required.

    if clip is None:
        return np.ones(data.shape)

    # Define the clippng range.

    clip = np.atleast_1d(clip)
    clip = np.array([-clip[0], clip[0]]) if clip.size == 1 else clip
    assert np.all(clip != 0.0), "Use `clip=None` to not use a threshold mask."

    # If we are making a Frankenmask, we must first smooth the cube to both
    # lower the background noise and extend the range of the real emission.
    # After the smoothing, we devide through by the RMS to generate a SNR mask.

    assert smooth_threshold_mask >= 0.0
    if smooth_threshold_mask > 0.0:
        from scipy.ndimage import gaussian_filter
        SNR = [gaussian_filter(c, sigma=smooth_threshold_mask) for c in data]
        SNR = np.array(SNR)
    else:
        SNR = data.copy()
    SNR /= estimate_RMS(SNR, noise_channels)

    # Return the mask.

    return np.logical_or(SNR < clip[0], SNR > clip[-1]).astype('float')


def get_combined_mask(user_mask, threshold_mask, channel_mask, combine='and'):
    """
    Return the combined user, threshold and channel masks, ``user_mask``,
    ``threshold_mask`` and ``velo_mask``, respectively. The user and threshold
    masks can be combined either through ``AND`` or ``OR``, which is controlled
    through the ``combine`` argument. This defaults to ``AND``, such that all
    mask requirements are met.

    Args:
        user_mask (array): User-defined mask from ``get_user_mask``.
        threshold_mask (array): Threshold mask from ``get_threshold_mask``.
        channel_mask (array): Channel mask from ``get_channel_mask``.

    Returns:
        combined_mask (array): A combined mask.

    """
    assert combine in ['and', 'or'], "Unknown `combine`: {}.".format(combine)
    combine = np.logical_and if combine == 'and' else np.logical_or
    combined_mask = combine(combine(user_mask, threshold_mask), channel_mask)
    return combined_mask.astype('float')


def main():

    # Parse all the command line arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to the FITS cube to collapse.')
    parser.add_argument('-method', default='quadratic',
                        help='Method used to collapse cube.')
    parser.add_argument('-smooth', default=0, type=int,
                        help='Width of filter to smooth spectrally.')
    parser.add_argument('-rms', default=None, type=float,
                        help='Estimated uncertainty on each pixel.')
    parser.add_argument('-processes', default=-1, type=int,
                        help='Number of process to use for analytical fits.')
    parser.add_argument('-noisechannels', default=5, type=int,
                        help='Number of end channels to use to estimate RMS.')
    parser.add_argument('-mask', default=None,
                        help='Path to the mask FITS cube.')
    parser.add_argument('-firstchannel', default=0, type=int,
                        help='First channel to use when collapsing cube.')
    parser.add_argument('-lastchannel', default=-1, type=int,
                        help='Last channel to use when collapsing cube.')
    parser.add_argument('-clip', default=None, nargs='*', type=float,
                        help='Mask absolute values below this SNR.')
    parser.add_argument('-smooththreshold', default=0.0, type=float,
                        help='Kernel in beam FWHM to smooth threshold map.')
    parser.add_argument('-combine', default='and',
                        help='How to combine the masks if provided.')
    parser.add_argument('-polyorder', default=0, type=int,
                        help='Polynomial order to use for SavGol filtering.')
    parser.add_argument('-outname', default=None, type=str,
                        help='Filename prefix for the saved images.')
    parser.add_argument('--nooverwrite', action='store_false',
                        help='Do not overwrite files.')
    parser.add_argument('--silent', action='store_true',
                        help='Do not see how the sausages are made.')
    parser.add_argument('--returnmask', action='store_true',
                        help='Return the masked used as a FITS file.')
    parser.add_argument('--debug', action='store_true',
                        help='Return all intermediate products to help debug.')

    args = parser.parse_args()

    # Check they all make sense.

    if args.noisechannels < 1:
        raise ValueError("`noisechannels` must an integer greater than 1.")

    args.combine = args.combine.lower()
    if args.combine not in ['and', 'or']:
        raise ValueError("`combine` must be `and` or `or`.")

    if not args.silent:
        import warnings
        warnings.filterwarnings("ignore")

    if args.processes == -1:
        args.processes = multiprocessing.cpu_count()

    # Read in the data and the user-defined mask.
    # If nothing is provided, include all pixels.

    if not args.silent:
        print("Loading up data...")
    data, velax, bunits = load_cube(args.path)

    # Load up the user-defined mask.

    if not args.silent:
        print("Loading up user-defined mask...")
    user_mask = get_user_mask(data=data, user_mask_path=args.mask)
    if args.debug:
        _save_user_mask(user_mask, args)

    # Define the velocity mask based on first and last channels. If nothing is
    # provided, use all channels. A more extensive version is possible for the
    # non-command line version.

    if not args.silent:
        print("Defining channel-based mask...")
    channel_mask = get_channel_mask(data=data,
                                    firstchannel=args.firstchannel,
                                    lastchannel=args.lastchannel)
    if args.debug:
        _save_channel_mask(channel_mask, args)

    # Smooth the data in the spectral dimension. Uses by default a uniform
    # (boxcar) filter. If a `polyorder` is provided, assumes the user wants a
    # Savitzky-Golay filter. In this case, extend all even window sizes by one
    # to make sure it is an odd number.

    if not args.silent:
        print("Smoothing the data...")
    data = smooth_data(data=data,
                       smooth=args.smooth,
                       polyorder=args.polyorder)
    if args.debug:
        _save_smoothed_data(data, args)

    # Calculate the RMS based on the first and last `noisechannels`, which is 5
    # by default. TODO: Test if there's a better way of doing this...

    if not args.silent:
        print("Estimating noise in the data...")
    if args.rms is None:
        args.rms = estimate_RMS(data, args.noisechannels)
        if not args.silent:
            print("Estimated RMS: {:.2e}.".format(args.rms))

    # Define the threshold mask. This includes the spatial smoothing of the
    # data for create Frankenmasks.

    if not args.silent:
        print("Calculating threshold-based mask...")
    threshold_mask = get_threshold_mask(data=data,
                                        clip=args.clip,
                                        smooth_threshold_mask=args.smooththreshold,
                                        noise_channels=args.noisechannels)
    if args.debug:
        _save_threshold_mask(threshold_mask, args)

    # Combine the masks and apply to the data.

    if not args.silent:
        print("Masking the data...")
    combined_mask = get_combined_mask(user_mask=user_mask,
                                      threshold_mask=threshold_mask,
                                      channel_mask=channel_mask,
                                      combine=args.combine)
    if args.returnmask or args.debug:
        _save_mask(combined_mask, args)
    if args.debug:
        _save_channel_count(np.sum(combined_mask, axis=0), args)
    masked_data = data.copy() * combined_mask

    # Reverse the direction if the velocity axis is decreasing.

    if np.diff(velax).mean() < 0:
        masked_data = masked_data[::-1]
        velax = velax[::-1]

    # Calculate the moments.

    if not args.silent:
        print("Calculating maps...")

    tosave = {}

    if args.method == 'zeroth':
        from .methods import collapse_zeroth
        M0, dM0 = collapse_zeroth(velax=velax,
                                  data=masked_data,
                                  rms=args.rms)
        tosave['M0'], tosave['dM0'] = M0, dM0

    elif args.method == 'first':
        from .methods import collapse_first
        M1, dM1 = collapse_first(velax=velax,
                                 data=masked_data,
                                 rms=args.rms)
        tosave['M1'], tosave['dM1'] = M1, dM1

    elif args.method == 'second':
        from .methods import collapse_second
        M2, dM2 = collapse_second(velax=velax,
                                  data=masked_data,
                                  rms=args.rms)
        tosave['M2'], tosave['dM2'] = M2, dM2

    elif args.method == 'eighth':
        from .methods import collapse_eighth
        M8, dM8 = collapse_eighth(velax=velax,
                                  data=masked_data,
                                  rms=args.rms)
        tosave['M8'], tosave['dM8'] = M8, dM8

    elif args.method == 'ninth':
        from .methods import collapse_ninth
        M9, dM9 = collapse_ninth(velax=velax,
                                 data=masked_data,
                                 rms=args.rms)
        tosave['M9'], tosave['dM9'] = M9, dM9

    elif args.method == 'maximum':
        from .methods import collapse_maximum
        temp = collapse_maximum(velax=velax,
                                data=masked_data,
                                rms=args.rms)
        tosave['M8'], tosave['dM8'] = temp[:2]
        tosave['M9'], tosave['dM9'] = temp[2:]

    elif args.method == 'quadratic':
        from .methods import collapse_quadratic
        temp = collapse_quadratic(velax=velax,
                                  data=masked_data,
                                  rms=args.rms)
        tosave['v0'], tosave['dv0'] = temp[:2]
        tosave['Fnu'], tosave['dFnu'] = temp[2:]
        if args.clip is not None:
            temp = tosave['Fnu'] / tosave['dFnu'] >= max(args.clip)
            temp = np.where(temp, 1.0, np.nan)
            tosave['v0'] = tosave['v0'] * temp
            tosave['dv0'] = tosave['dv0'] * temp
            tosave['Fnu'] = tosave['Fnu'] * temp
            tosave['dFnu'] = tosave['dFnu'] * temp

    elif args.method == 'width':
        from .methods import collapse_width
        dV, ddV = collapse_width(velax=velax,
                                 data=masked_data,
                                 ms=args.rms)
        tosave['dV'], tosave['ddV'] = dV, ddV

    elif args.method == 'gaussian':
        from .methods import collapse_gaussian
        print("Using {} CPUs.".format(args.processes))
        temp = collapse_gaussian(velax=velax,
                                 data=masked_data,
                                 rms=args.rms,
                                 chunks=args.processes,
                                 mcmc=None)
        tosave['gv0'], tosave['dgv0'] = temp[:2]
        tosave['gdV'], tosave['dgdV'] = temp[2:4]
        tosave['gFnu'], tosave['dgFnu'] = temp[4:]

    elif args.method == 'gaussthick':
        from .methods import collapse_gaussthick
        print("Using {} CPUs.".format(args.processes))
        temp = collapse_gaussthick(velax=velax,
                                   data=masked_data,
                                   rms=args.rms,
                                   chunks=args.processes,
                                   mcmc=None)
        tosave['gv0'], tosave['dgv0'] = temp[:2]
        tosave['gdV'], tosave['dgdV'] = temp[2:4]
        tosave['gFnu'], tosave['dgFnu'] = temp[4:6]
        tosave['gtau'], tosave['dgtau'] = temp[6:]

    elif args.method == 'gausshermite':
        from .methods import collapse_gausshermite
        print("Using {} CPUs.".format(args.processes))
        temp = collapse_gausshermite(velax=velax,
                                     data=masked_data,
                                     rms=args.rms,
                                     chunks=args.processes,
                                     mcmc=None)
        tosave['ghv0'], tosave['dghv0'] = temp[:2]
        tosave['ghdV'], tosave['dghdV'] = temp[2:4]
        tosave['ghFnu'], tosave['dghFnu'] = temp[4:6]
        tosave['ghh3'], tosave['dghh3'] = temp[6:8]
        tosave['ghh4'], tosave['dghh4'] = temp[8:]
    else:
        raise ValueError("Unknown method.")

    # Save as FITS files.

    if not args.silent:
        print("Saving maps...")

    for map_name in tosave.keys():
        if args.outname is None:
            outname = args.path.replace('.fits', '_{}.fits'.format(map_name))
        else:
            outname = args.outname.replace('.fits', '')
            outname += '_{}.fits'.format(map_name)
        save_to_FITS(args.path, outname, tosave[map_name],
                     overwrite=args.nooverwrite, bunit=bunits[map_name])


if __name__ == '__main__':
    main()
