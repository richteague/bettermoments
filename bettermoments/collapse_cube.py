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


def collapse_gaussian(velax, data, rms=None, threshold=3.0, N=5, axis=0):
    """
    Collapse the cube by fitting Gaussians to each pixel.

    To help the fitting, which is done with ``scipy.optimize.curve_fit`` which
    utilises non-linear least squares, the
    :func:`bettermoments.collapse_cube.quadratic` method is first run to obtain
    line centers and peaks, while the :func:`bettermoments.collapse_cube.width`
    method is used to estimate the line width. These values are further used to
    locate the pixels which will be fit, i.e. those which have ``Fnu / dFnu >=
    threshold``.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the zeroth axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If ``None`` is specified,
            this will be calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along, by
            default ``axis=0``.

    Returns:
        ``gv0`` (`ndarray`), ``gdv0`` (`ndarray`), ``gdV`` (`ndarray`), ``gddV`` (`ndarray`), ``gFnu`` (`ndarray`), ``gdFnu`` (`ndarray`):
            ``gv0``, the line center in the same units as ``velax`` with
            ``gdv0`` as the uncertainty on ``v0`` in the same units as
            ``velax``. ``gdV`` is the Doppler linewidth of the Gaussian fit in
            the same units as ``velax`` with uncertainty ``gddV``. ``gFnu`` is
            the line peak in the same units as the ``data`` with associated
            uncertainties, ``gdFnu``.
    """
    # Rorate the axis if necessary.
    if axis != 0:
        raise NotImplementedError("Can only collapse along the zeroth axis.")

    # Verfify the data and calculate the noise.
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)

    # Calculate the initial guesses.
    v0, _, Fnu, _ = collapse_quadratic(velax=velax, data=data, rms=rms, N=N,
                                       axis=axis)
    dV, _ = collapse_width(velax=velax, data=data, rms=rms, N=N,
                           threshold=threshold, axis=axis)
    Fnu = np.where(abs(Fnu) / rms > threshold, Fnu, np.nan)

    # Fit the gaussians.
    from bettermoments.methods import gaussian
    return gaussian(data=data, specax=velax, uncertainty=rms, axis=axis,
                    v0=v0, Fnu=Fnu, dV=dV)


def collapse_gausshermite(velax, data, rms=None, threshold=3.0, N=5, axis=0):
    """
    Collapse the cube by fitting a Hermite expansion of a Gaussians to each
    pixel. This allows for a flexible line profile that purely a Gaussian,
    where the ``h3`` and ``h4`` terms quantify the skewness and kurtosis of the
    line as in `_van der Marel & Franx (1993)`_.

    To help the fitting, which is done with ``scipy.optimize.curve_fit`` which
    utilises non-linear least squares, the
    :func:`bettermoments.collapse_cube.quadratic` method is first run to obtain
    line centers and peaks, while the :func:`bettermoments.collapse_cube.width`
    method is used to estimate the line width. These values are further used to
    locate the pixels which will be fit, i.e. those which have ``Fnu / dFnu >=
    threshold``.

    .. _van der Marel & Franx (1993): https://ui.adsabs.harvard.edu/abs/1993ApJ...407..525V/abstract

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the zeroth axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If ``None`` is specified,
            this will be calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along, by
            default ``axis=0``.

    Returns:
        ``ghv0`` (`ndarray`), ``dghv0`` (`ndarray`), ``ghdV`` (`ndarray`), ``dghdV`` (`ndarray`), ``ghFnu`` (`ndarray`), ``dghFnu`` (`ndarray`), ``ghh3`` (`ndarray`), ``dghh3`` (`ndarray`), ``ghh4`` (`ndarray`), ``dghh4`` (`ndarray`):
            ``ghv0``, the line center in the same units as ``velax`` with
            ``dghv0`` as the uncertainty on ``ghv0`` in the same units as
            ``velax``. ``ghdV`` is the Doppler linewidth of the Gaussian fit in
            the same units as ``velax`` with uncertainty ``dghdV``. ``ghFnu``
            is the line peak in the same units as the ``data`` with associated
            uncertainties, ``dghFnu``. ``ghh3`` and ``ghh4`` describe the
            skewness and kurtosis of the line, respectively, with uncertainties
            ``dghh3`` and ``dghh4``.
    """
    # Rotate the axis if necessary.
    if axis != 0:
        raise NotImplementedError("Can only collapse along the zeroth axis.")

    # Verfify the data and calculate the noise.
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)

    # Calculate the initial guesses.
    v0, _, Fnu, _ = collapse_quadratic(velax=velax, data=data, rms=rms, N=N,
                                       axis=axis)
    dV, _ = collapse_width(velax=velax, data=data, rms=rms, N=N,
                           threshold=threshold, axis=axis)
    Fnu = np.where(abs(Fnu) / rms > threshold, Fnu, np.nan)

    # Fit the gaussians.
    from bettermoments.methods import gausshermite
    return gausshermite(data=data, specax=velax, uncertainty=rms, axis=axis,
                        v0=v0, Fnu=Fnu, dV=dV)


def collapse_quadratic(velax, data, rms=None, N=5, axis=0):
    """
    Collapse the cube using the quadratic method presented in `Teague &
    Foreman-Mackey (2018)`_. Will return the line center, ``v0``, and the
    uncertainty on this, ``dv0``, as well as the line peak, ``Fnu``, and the
    uncertainty on that, ``dFnu``. This provides the sub-channel precision of
    :func:`bettermoments.collapse_cube.collapse_first` with the robustness to
    noise from :func:`bettermoments.collapse_cube.collapse_ninth`.

    .. _Teague & Foreman-Mackey (2018): https://iopscience.iop.org/article/10.3847/2515-5172/aae265

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the zeroth axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If ``None`` is specified,
            this will be calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along, by
            default ``axis=0``.

    Returns:
        ``v0`` (`ndarray`), ``dv0`` (`ndarray`), ``Fnu`` (`ndarray`), ``dFnu`` (`ndarray`):
            ``v0``, the line center in the same units as ``velax`` with ``dv0``
            as the uncertainty on ``v0`` in the same units as ``velax``.
            ``Fnu`` is the line peak in the same units as the
            ``data`` with associated uncertainties, ``dFnu``.
    """
    from bettermoments.methods import quadratic
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    return quadratic(data, x0=velax[0], dx=chan,
                     uncertainty=np.ones(data.shape)*rms, axis=axis)


def collapse_zeroth(velax, data, rms=None, N=5, threshold=None, mask_path=None,
                    axis=0):
    r"""
    Collapses the cube by integrating along the spectral axis. It will return
    the integrated intensity along the spectral axis, ``M0``, and the
    associated uncertainty, ``dM0``. Following `Teague (2019)`_ these are
    calculated by,

    .. math::
        M_0 = \sum_{i}^N I_i \, \Delta v_{{\rm chan},\,i}

    and

    .. math::
        M_0 = \sqrt{\sum_{i\,(I_i > 0)}^N \sigma_i^2 \cdot \Delta v_{{\rm chan},\,i}^2}

    where :math:`\Delta v_i` and :math:`I_i` are the chanenl width and flux
    density at the :math:`i^{\rm th}` channel, respectively and the sum goes
    over the whole ``axis``.

    .. _Teague (2019): https://iopscience.iop.org/article/10.3847/2515-5172/ab2125

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        mask_path (Optional[str]): Path to a file containing a boolean mask,
            either stored as `.FITS` or `.npy` file.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        ``M0`` (`ndarray`), ``dM0`` (`ndarray`):
            ``M0``, the integrated intensity along provided axis and ``dM0``,
            the uncertainty on ``M0`` in the same units as ``M0``.
    """
    from bettermoments.methods import integrated_intensity
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    return integrated_intensity(data=data, dx=abs(chan), threshold=threshold,
                                rms=rms, mask_path=mask_path, axis=axis)


def collapse_maximum(velax, data, rms=None, N=5, axis=0):
    """
    A wrapper returning the result of both
    :func:`bettermoments.collapse_cube.collapse_eighth` and
    :func:`bettermoments.collapse_cube.collapse_ninth`.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        ``M8`` (`ndarray`), ``dM8`` (`ndarray`), ``M9`` (`ndarray`), ``dM9`` (`ndarray`):
            The peak value, ``M8``, and the associated uncertainty, ``dM8``.
            The velocity value of the peak value, ``M9``, and the associated
            uncertainty, ``dM9``.
    """
    from bettermoments.methods import peak_pixel
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    M9, dM9, M8 = peak_pixel(data=data, x0=velax[0], dx=chan, axis=axis)
    dM8 = rms * np.ones(M8.shape)
    return M8, dM8, M9, dM9


def collapse_eighth(data, rms=None, N=5, axis=0):
    """
    Take the peak value along the provided axis. The uncertainty is the RMS
    noise of the image.

    Args:
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        ``M8`` (`ndarray`), ``dM8`` (`ndarray`):
            The peak value, ``M8``, and the associated uncertainty, ``dM8``.
    """
    from bettermoments.methods import peak_pixel
    velax = np.ones(data.shape[int(axis)])
    rms, _ = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    _, _, M8 = peak_pixel(data=data, x0=0.0, dx=0.0, axis=axis)
    dM8 = rms * np.ones(M8.shape)
    return M8, dM8


def collapse_ninth(velax, data, rms=None, N=5, axis=0):
    """
    Take the velocity of the peak intensity along the provided axis. The
    uncertainty is half the channel width.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        ``M9`` (`ndarray`), ``dM9`` (`ndarray`):
            The velocity value of the peak value, ``M9``, and the associated
            uncertainty, ``dM9``.
    """
    from bettermoments.methods import peak_pixel
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    M9, dM9, _ = peak_pixel(data=data, x0=velax[0], dx=chan, axis=axis)
    return M9, dM9


def collapse_first(velax, data, rms=None, N=5, threshold=None, mask_path=None,
                   axis=0):
    r"""
    Collapses the cube using the intensity weighted average velocity (or first
    moment map). For a symmetric line profile this will be the line center,
    however for highly non-symmetric line profiles, this will not give a
    meaningful result. Following `Teague (2019)`_, the line center is given by,

    .. math::
        M_1 = \frac{\sum_i^N I_i v_i}{\sum_i^N I_i}

    where :math:`v_i` and :math:`I_i` are the velocity and flux density at the
    :math:`i^{\rm th}` channel, respectively and the sum goes over the whole
    ``axis``. In addition, the uncertainty is given by,

    .. math::
        \delta M_1 = \sqrt{\sum_{i\,(I_i > 0)}^N \sigma_i^2 \cdot (v_i - M_1)^2}

    where :math:`\sigma_i` is the rms noise.

    .. _Teague (2019): https://iopscience.iop.org/article/10.3847/2515-5172/ab2125

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the zeroth axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        mask_path (Optional[str]): Path to a file containing a boolean mask,
            either stored as `.FITS` or `.npy` file.
        axis (Optional[int]): Spectral axis to collapse the cube along. By
            default this is the zeroth axis.

    Returns:
        ``M1`` (`ndarray`), ``dM1`` (`ndarray`):
            ``M1``, the intensity weighted average velocity in units of
            ``velax`` and ``dM1``, the uncertainty in the intensity weighted
            average velocity with same units as ``v0``.
    """
    from bettermoments.methods import intensity_weighted_velocity as first
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    return first(data=data, x0=velax[0], dx=chan, rms=rms, threshold=threshold,
                 mask_path=mask_path, axis=axis)


def collapse_second(velax, data, rms=None, N=5, threshold=None, mask_path=None,
                    axis=0):
    r"""
    Collapses the cube using the intensity-weighted average velocity dispersion
    (or second moment). For a symmetric line profile this will be a measure of
    the line width. Following `Teague (2019)`_ this is calculated by,

    .. math::
        M_2 = \sqrt{\frac{\sum_i^N I_i (v_i - M_1)^2}{{\sum_i^N I_i}}}

    where :math:`M_1` is the first moment and :math:`v_i` and :math:`I_i` are
    the velocity and flux density at the :math:`i^{\rm th}` channel,
    respectively. The uncertainty is given by,

    .. math::
        \delta M_2 &= \frac{1}{2 M_2} \cdot \sqrt{\sum_{i\,(I_i > 0)}^N \sigma_i^2 \cdot \big[(v_i - M_1)^2 - M_2^2\big]^2}

    where :math:`\sigma_i` is the rms noise in the :math:`i^{\rm th}` channel.

    .. _Teague (2019): https://iopscience.iop.org/article/10.3847/2515-5172/ab2125

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If none is specified, will be
            calculated from the first and last N channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        mask_path (Optional[str]): Path to a file containing a boolean mask,
            either stored as `.FITS` or `.npy` file.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        ``M2`` (ndarray), ``dM2`` (ndarray):
            ``M2`` is the intensity weighted velocity dispersion with units of
            ``velax``.  ``dM2`` is the unceratinty of ``M2`` in the same units.
    """
    from bettermoments.methods import intensity_weighted_dispersion as second
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    return second(data=data, x0=velax[0], dx=chan, rms=rms,
                  threshold=threshold, mask_path=mask_path, axis=axis)


def collapse_width(velax, data, linewidth=0.0, rms=None, N=5, threshold=None,
                   mask_path=None, axis=0):
    r"""
    Returns an effective width, a rescaled ratio of the integrated intensity
    and the line peak. For a Gaussian line profile this would be the Doppler
    width as the total intensity is given by,

    .. math::
        M_0 = \sum_{i}^N I_i \, \Delta v_{{\rm chan},\,i}

    where :math:`\Delta v_i` and :math:`I_i` are the chanenl width and flux
    density at the :math:`i^{\rm th}` channel. If the line profile is Gaussian,
    then equally

    .. math::
        M_0 = \sqrt{\pi} \times F_{\nu} \times \Delta V

    where :math:`F_{\nu}` is the peak value of the line and :math:`\Delta V` is
    the Doppler width of the line. As :math:`M_0` and :math:`F_{\nu}` are
    readily calculated using
    :func:`bettermoments.collapse_cube.collapse_zeroth` and
    :func:`bettermoments.collapse_cube.collapse_quadratic`, respectively,
    :math:`\Delta V` can calculated through :math:`\Delta V = M_{0} \, / \,
    (\sqrt{\pi} \, F_{\nu})`. This should be more robust against noise than
    second moment maps.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (Optional[float]): Noise per pixel. If ``None`` is specified, this
            will be calculated from the first and last ``N`` channels.
        N (Optional[int]): Number of channels to use in the estimation of the
            noise.
        threshold (Optional[float]): Clip any pixels below this RMS value.
        mask_path (Optional[str]): Path to a file containing a boolean mask,
            either stored as `.FITS` or `.npy` file.
        axis (Optional[int]): Spectral axis to collapse the cube along.

    Returns:
        ``dV`` (`ndarray`), ``ddV`` (`ndarray`):
            The effective velocity dispersion, ``dV`` and ``ddV``, the
            associated uncertainty.
    """
    from bettermoments.methods import integrated_intensity, quadratic
    rms, chan = _verify_data(data, velax, rms=rms, N=N, axis=axis)
    I0, dI0 = integrated_intensity(data=data, dx=abs(chan),
                                   threshold=threshold, rms=rms,
                                   mask_path=mask_path, axis=axis)
    if linewidth > 0.0:
        linewidth = abs(linewidth / chan / np.sqrt(2.))
    else:
        linewidth = None
    _, _, Fnu, dFnu = quadratic(data, x0=velax[0], dx=chan,
                                uncertainty=np.ones(data.shape)*rms,
                                axis=axis)
    dV = I0 / Fnu / np.sqrt(np.pi)
    ddV = dV * np.hypot(dFnu / Fnu, dI0 / I0)
    return dV, ddV


def _get_cube(path):
    """Return the data and velocity axis from the cube."""
    return _get_data(path), _get_velax(path), _get_bunits(path)


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


def _verify_data(data, velax, rms=None, N=5, axis=0):
    """Veryify the data shape and read in image properties."""
    if data.shape[axis] != velax.size:
        raise ValueError("Must collapse along the spectral axis!")
    if rms is None:
        rms = _estimate_RMS(data=data, N=N)
    chan = np.diff(velax).mean()
    return rms, chan


def _estimate_RMS(data, N=5):
    """Return the estimated RMS in the first and last N channels."""
    x1, x2 = np.percentile(np.arange(data.shape[2]), [45, 55])
    y1, y2 = np.percentile(np.arange(data.shape[1]), [45, 55])
    x1, x2, y1, y2, N = int(x1), int(x2), int(y1), int(y2), int(N)
    rms = np.nanstd([data[:N, y1:y2, x1:x2], data[-N:, y1:y2, x1:x2]])
    return rms * np.ones(data[0].shape)


def _collapse_beamtable(path):
    """Returns the median beam from the CASA beam table if present."""
    header = fits.getheader(path)
    if header.get('CASAMBM', False):
        try:
            beam = fits.open(path)[1].data
            beam = np.median([b[:3] for b in beam.view()], axis=0)
            return beam[0] / 3600., beam[1] / 3600., beam[2]
        except IndexError:
            print('WARNING: No beam table found despite CASAMBM flag.')
            return abs(header['cdelt1']), abs(header['cdelt2']), 0.0
    try:
        return header['bmaj'], header['bmin'], header['bpa']
    except KeyError:
        return abs(header['cdelt1']), abs(header['cdelt2']), 0.0


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
    bunits['dgv0'] = bunits['gv0']
    bunits['dgFnu'] = bunits['gFnu']
    bunits['dgdV'] = bunits['gdV']
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
    return bunits


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
                             'second, width, gaussian and gausshermite.')
    parser.add_argument('-clip', default=5.0, type=float,
                        help='Mask values below this SNR.')
    parser.add_argument('-fill', default=np.nan, type=float,
                        help='Fill value for masked pixels. Default is NaN.')
    parser.add_argument('-smooth', default=0.0, type=float,
                        help='The width of the filter for a second-order '
                             'Savitzky-Golay filter. While reducing the noise '
                             'this filter will dilute Fnu values.')
    parser.add_argument('-rms', default=None, type=float,
                        help='Estimated RMS noise from a line free channel. '
                             'Same units as the brightness unit.')
    parser.add_argument('-N', default=5, type=int,
                        help='Number of end channels to use to estimate RMS.')
    parser.add_argument('-mask', default=None,
                        help='Path to the mask FITS cube. Must have the same '
                             'shape as the input data.')
    parser.add_argument('-axis', default=0, type=int,
                        help='Axis to collapse the cube along. Default is 0.')
    parser.add_argument('-downsample', default=1, type=int,
                        help='Downsample the data by this factor.')
    parser.add_argument('-outname', default=None, type=str,
                        help='Name of the output file.')
    parser.add_argument('-overwrite', default=True, type=bool,
                        help='Overwrite existing files with the same name.')
    parser.add_argument('--nomask', action='store_true',
                        help='Clip the final moment map using the provided '
                             '`clip` value. Default is True.')
    parser.add_argument('--silent', action='store_true',
                        help='Run silently.')
    parser.add_argument('--warnings', action='store_true',
                        help='Show the warnings.')
    args = parser.parse_args()
    args.method = args.method.lower()

    # Hide warnings. Maybe a bad idea?

    if not args.warnings:
        import warnings
        warnings.filterwarnings("ignore")

    # Read in the cube and the units.

    data, velax, bunits = _get_cube(args.path)
    tosave = {}

    # If resampled is requested, average over the data and convolve with a top
    # hat function to minimic the channelization of ALMA.

    if args.downsample > 1:
        if not args.silent:
            print("Down-sampling data...")
        N = int(args.downsample)
        data = np.array([np.nanmean(data[i*N:(i+1)*N], axis=0)
                         for i in range(int(velax.size / N))])
        kernel = np.ones(N) / float(N)
        data = convolve1d(data, kernel, mode='reflect', axis=args.axis)
        velax = np.array([np.average(velax[i*N:(i+1)*N])
                          for i in range(int(velax.size / N))])

    # Presmooth the data with a Gaussian filter. Use astropy to handle NaNs.

    if args.smooth > 0.0:
        if not args.silent:
            print("Smoothing data...")
        from scipy.signal import savgol_filter
        width = int(args.smooth / abs(np.diff(velax)[0]))
        width = width + 1 if not width % 2 else width
        data = savgol_filter(data, width, polyorder=2,
                             mode='wrap', axis=0)

    # Collapse the cube with the approrpriate method.

    if not args.silent:
        print("Calculating maps...")

    if args.method == 'zeroth':
        M0, dM0 = collapse_zeroth(velax=velax, data=data, threshold=args.clip,
                                  rms=args.rms, N=args.N, mask_path=args.mask,
                                  axis=args.axis)
        tosave['M0'], tosave['dM0'] = M0, dM0

    elif args.method == 'first':
        M1, dM1 = collapse_first(velax=velax, data=data, threshold=args.clip,
                                 rms=args.rms, N=args.N, mask_path=args.mask,
                                 axis=args.axis)
        tosave['M1'], tosave['dM1'] = M1, dM1

    elif args.method == 'second':
        M2, dM2 = collapse_second(velax=velax, data=data, threshold=args.clip,
                                  rms=args.rms, N=args.N, mask_path=args.mask,
                                  axis=args.axis)
        tosave['M2'], tosave['dM2'] = M2, dM2

    elif args.method == 'eighth':
        M8, dM8 = collapse_eighth(data=data, rms=args.rms, N=args.N,
                                  axis=args.axis)
        tosave['M8'], tosave['dM8'] = M8, dM8

    elif args.method == 'ninth':
        M9, dM9 = collapse_ninth(velax=velax, data=data, rms=args.rms,
                                 N=args.N, axis=args.axis)
        tosave['M9'], tosave['dM9'] = M9, dM9

    elif args.method == 'maximum':
        M8, dM8, M9, dM9 = collapse_maximum(velax=velax, data=data,
                                            rms=args.rms, N=args.N,
                                            axis=args.axis)
        tosave['M8'], tosave['dM8'] = M8, dM8
        tosave['M9'], tosave['dM9'] = M9, dM9

    elif args.method == 'quadratic':
        v0, dv0, Fnu, dFnu = collapse_quadratic(velax=velax, data=data,
                                                N=args.N, axis=args.axis,
                                                rms=args.rms)
        tosave['v0'], tosave['dv0'] = v0, dv0
        tosave['Fnu'], tosave['dFnu'] = Fnu, dFnu

    elif args.method == 'width':
        dV, ddV = collapse_width(velax=velax, data=data, threshold=args.clip,
                                 rms=args.rms, N=args.N, mask_path=args.mask,
                                 axis=args.axis, linewidth=args.linewidth)
        tosave['dV'], tosave['ddV'] = dV, ddV

    elif args.method == 'gaussian':
        temp = collapse_gaussian(velax=velax, data=data, rms=args.rms,
                                 threshold=args.clip, N=args.N, axis=args.axis)
        tosave['gv0'], tosave['dgv0'] = temp[:2]
        tosave['gdV'], tosave['dgdV'] = temp[2:4]
        tosave['gFnu'], tosave['dgFnu'] = temp[4:]

    elif args.method == 'gausshermite':
        temp = collapse_gausshermite(velax=velax, data=data, rms=args.rms,
                                     threshold=args.clip, N=args.N,
                                     axis=args.axis)
        tosave['ghv0'], tosave['dghv0'] = temp[:2]
        tosave['ghdV'], tosave['dghdV'] = temp[2:4]
        tosave['ghFnu'], tosave['dghFnu'] = temp[4:6]
        tosave['ghh3'], tosave['dghh3'] = temp[6:8]
        tosave['ghh4'], tosave['dghh4'] = temp[8:]
    else:
        raise ValueError("Unknown method.")

    # Mask the data. If no uncertainties are found for dFnu, use the RMS.

    if args.clip > 0.0 and not args.nomask:
        if not args.silent:
            print("Masking maps...")
        signal, noise = collapse_eighth(data=data)
        rndm = abs(1e-10 * np.random.randn(noise.size).reshape(noise.shape))
        mask = np.where(noise != 0.0, noise, rndm)
        mask = np.where(np.isfinite(mask), mask, rndm)
        mask = np.where(np.isfinite(signal), signal, 0.0) / mask >= args.clip
    else:
        mask = np.moveaxis(np.ones(data.shape), args.axis, 0)
        mask = np.ones(mask[0].shape)

    # Save the files.

    if not args.silent:
        print("Saving maps...")

    for map_name in tosave.keys():
        map_file = np.where(mask, tosave[map_name], args.fill)
        outname = args.outname if args.outname is not None else args.path
        outname = outname + '.fits' if outname[:-5] == '.fits' else outname
        outname = outname.replace('.fits', '_{}.fits'.format(map_name))
        bunit = bunits[map_name]
        _save_array(args.path, outname, map_file,
                    overwrite=args.overwrite, bunit=bunit)


if __name__ == '__main__':
    main()
