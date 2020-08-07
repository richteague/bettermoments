"""
Collapse a data cube down to a summary statistic using various methods. Now
returns statistical uncertainties for all statistics.

TODO:
    - Deal with the fact we're using three different convolution routines.
"""

import argparse
import numpy as np
from tqdm import tqdm
from astropy.io import fits
import scipy.constants as sc


# -- Standard Moment Maps -- #


def collapse_zeroth(velax, data, rms):
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
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M0`` (`ndarray`), ``dM0`` (`ndarray`):
            ``M0``, the integrated intensity along provided axis and ``dM0``,
            the uncertainty on ``M0`` in the same units as ``M0``.
    """
    chan = np.diff(velax).mean()
    npix = np.sum(data != 0.0, axis=0)
    M0 = np.trapz(data, dx=chan, axis=0)
    dM0 = chan * rms * npix**0.5 * np.ones(M0.shape)
    if M0.shape != data[0].shape:
        raise ValueError("`data` not collapsed correctly." +
                         " Expected shape: {},".format(data[0].shape) +
                         " returned shape: {}.".format(M0.shape))
    if M0.shape != dM0.shape:
        raise ValueError("Mismatch in `M0` and `dM0` shapes: "
                         "{} and {}.".format(M0.shape, dM0.shape))
    return M0, dM0


def collapse_first(velax, data, rms):
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
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M1`` (`ndarray`), ``dM1`` (`ndarray`):
            ``M1``, the intensity weighted average velocity in units of
            ``velax`` and ``dM1``, the uncertainty in the intensity weighted
            average velocity with same units as ``v0``.
    """
    chan = np.diff(velax).mean()
    vpix = chan * np.arange(data.shape[0]) + velax[0]
    vpix = vpix[:, None, None] * np.ones(data.shape)

    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data != 0.0, abs(data), weights)
    M1 = np.average(vpix, weights=weights, axis=0)
    dM1 = (vpix - M1[None, :, :]) * rms / np.sum(weights, axis=0)
    dM1 = np.sqrt(np.sum(dM1**2, axis=0))

    npix = np.sum(data != 0.0, axis=0)
    M1 = np.where(npix >= 1.0, M1, np.nan)
    dM1 = np.where(npix >= 1.0, dM1, np.nan)

    if M1.shape != data[0].shape:
        raise ValueError("`data` not collapsed correctly." +
                         " Expected shape: {},".format(data[0].shape) +
                         " returned shape: {}.".format(M1.shape))
    if M1.shape != dM1.shape:
        raise ValueError("Mismatch in `M1` and `dM1` shapes: "
                         "{} and {}.".format(M1.shape, dM1.shape))
    return M1, dM1


def collapse_second(velax, data, rms):
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
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M2`` (ndarray), ``dM2`` (ndarray):
            ``M2`` is the intensity weighted velocity dispersion with units of
            ``velax``.  ``dM2`` is the unceratinty of ``M2`` in the same units.
    """
    chan = np.diff(velax).mean()
    vpix = chan * np.arange(data.shape[0]) + velax[0]
    vpix = vpix[:, None, None] * np.ones(data.shape)

    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data != 0.0, abs(data), weights)

    M1 = collapse_first(velax=velax, data=data, rms=rms)[0]
    M1 = M1[None, :, :] * np.ones(data.shape)
    M2 = np.sum(weights * (vpix - M1)**2, axis=0) / np.sum(weights, axis=0)
    M2 = np.sqrt(M2)

    dM2 = ((vpix - M1)**2 - M2**2) * rms / np.sum(weights, axis=0)
    dM2 = np.sqrt(np.sum(dM2**2, axis=0)) / 2. / M2

    npix = np.sum(data != 0.0, axis=0)
    M2 = np.where(npix >= 1.0, M2, np.nan)
    dM2 = np.where(npix >= 1.0, dM2, np.nan)

    if M2.shape != data[0].shape:
        raise ValueError("`data` not collapsed correctly." +
                         " Expected shape: {},".format(data[0].shape) +
                         " returned shape: {}.".format(M2.shape))
    if M2.shape != dM2.shape:
        raise ValueError("Mismatch in `M2` and `dM2` shapes: "
                         "{} and {}.".format(M2.shape, dM2.shape))
    return M2, dM2


def collapse_eighth(velax, data, rms):
    """
    Take the peak value along the provided axis. The uncertainty is the RMS
    noise of the image.

    Args:
        velax (ndarray): Velocity axis of the cube. Not needed.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M8`` (`ndarray`), ``dM8`` (`ndarray`):
            The peak value, ``M8``, and the associated uncertainty, ``dM8``.
    """
    M8 = np.max(data, axis=0)
    dM8 = rms * np.ones(M8.shape)
    return M8, dM8


def collapse_ninth(velax, data, rms):
    """
    Take the velocity of the peak intensity along the provided axis. The
    uncertainty is half the channel width.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M9`` (`ndarray`), ``dM9`` (`ndarray`):
            The velocity value of the peak value, ``M9``, and the associated
            uncertainty, ``dM9``.
    """
    M9 = velax[np.argmax(data, axis=0)]
    dM9 = 0.5 * abs(np.diff(velax).mean())
    return M9, dM9


# -- Line Profile Fitting -- #


def collapse_gaussian(velax, data, rms):
    r"""
    Collapse the cube by fitting Gaussians to each pixel,

    .. math::
        I(v) = F_{\nu} \times \exp \left[ -\frac{(v-v_0)^2}{\Delta V^2} \right]

    To help the fitting, which is done with ``scipy.optimize.curve_fit`` using
    a non-linear least squares fit, the
    :func:`bettermoments.collapse_cube.quadratic` method is first run to obtain
    line centers and peaks, while the :func:`bettermoments.collapse_cube.width`
    method is used to estimate the line width. Pixels that return a ``NaN``
    are skipped for the line fitting.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``gv0`` (`ndarray`), ``gdv0`` (`ndarray`), ``gdV`` (`ndarray`), ``gddV`` (`ndarray`), ``gFnu`` (`ndarray`), ``gdFnu`` (`ndarray`):
            ``gv0``, the line center in the same units as ``velax`` with
            ``gdv0`` as the uncertainty on ``v0`` in the same units as
            ``velax``. ``gdV`` is the Doppler linewidth of the Gaussian fit in
            the same units as ``velax`` with uncertainty ``gddV``. ``gFnu`` is
            the line peak in the same units as the ``data`` with associated
            uncertainties, ``gdFnu``.
    """
    from scipy.optimize import curve_fit

    v0, _, Fnu, _ = collapse_quadratic(velax=velax, data=data, rms=rms)
    dV, _ = collapse_width(velax=velax, data=data, rms=rms)
    p0 = np.squeeze([v0, dV, Fnu])

    sigma = rms * np.ones(velax.shape)
    fits = np.ones((6, data.shape[1], data.shape[2])) * np.nan

    with tqdm(total=np.all(np.isfinite(p0), axis=0).sum()) as pbar:
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):

                p0_tmp = p0[:, y, x]
                if any(np.isnan(p0_tmp)):
                    continue

                f0 = data[:, y, x].copy()
                f0 = np.isfinite(f0) & (f0 != 0.0)

                x_tmp = velax[f0]
                y_tmp = data[f0, y, x]
                dytmp = sigma[f0]

                try:
                    popt, covt = curve_fit(_gaussian, x_tmp, y_tmp,
                                           sigma=dytmp, p0=p0_tmp,
                                           absolute_sigma=True,
                                           maxfev=1000000)
                    covt = np.diag(covt)**0.5
                except:
                    popt = np.ones(3) * np.nan
                    covt = np.ones(3) * np.nan

                fits[::2, y, x] = popt
                fits[1::2, y, x] = covt
                pbar.update(1)

    v0, dV0, dV, ddV, Fnu, dFnu = fits
    return v0, dV0, abs(dV), ddV, Fnu, dFnu


def collapse_gaussthick(velax, data, rms, threshold=None):
    r"""
    Collapse the cube by fitting Gaussian to each pixel, including an
    approximation of an optically thick line core,

    .. math::
        I(v) = F_{\nu} \big(1 - \exp(\mathcal{G}(v, v0, \Delta V, \tau))\big)

    where :math:`\mathcal{G}` is a Gaussian function.

    To help the fitting, which is done with ``scipy.optimize.curve_fit`` which
    utilises non-linear least squares, the
    :func:`bettermoments.collapse_cube.quadratic` method is first run to obtain
    line centers and peaks, while the :func:`bettermoments.collapse_cube.width`
    method is used to estimate the line width. Pixels that return a ``NaN``
    are skipped for the line fitting.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``gv0`` (`ndarray`), ``gdv0`` (`ndarray`), ``gdV`` (`ndarray`), ``gddV`` (`ndarray`), ``gFnu`` (`ndarray`), ``gdFnu`` (`ndarray`):
            ``gv0``, the line center in the same units as ``velax`` with
            ``gdv0`` as the uncertainty on ``v0`` in the same units as
            ``velax``. ``gdV`` is the Doppler linewidth of the Gaussian fit in
            the same units as ``velax`` with uncertainty ``gddV``. ``gFnu`` is
            the line peak in the same units as the ``data`` with associated
            uncertainties, ``gdFnu``.
    """
    from scipy.optimize import curve_fit

    v0, _, Fnu, _ = collapse_quadratic(velax=velax, data=data, rms=rms)
    dV, _ = collapse_width(velax=velax, data=data, rms=rms)
    p0 = np.squeeze([v0, dV, Fnu, np.ones(v0.shape)])

    sigma = rms * np.ones(velax.shape)
    fits = np.ones((8, data.shape[1], data.shape[2])) * np.nan

    with tqdm(total=np.all(np.isfinite(p0), axis=0).sum()) as pbar:
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):

                p0_tmp = p0[:, y, x]
                if any(np.isnan(p0_tmp)):
                    continue

                f0 = data[:, y, x].copy()
                f0 = np.isfinite(f0) & (f0 != 0.0)

                x_tmp = velax[f0]
                y_tmp = data[f0, y, x]
                dytmp = sigma[f0]

                try:
                    popt, covt = curve_fit(_gaussian_thick, x_tmp, y_tmp,
                                           sigma=dytmp, p0=p0_tmp,
                                           absolute_sigma=True,
                                           maxfev=1000000)
                    covt = np.diag(covt)**0.5
                except:
                    popt = np.ones(4) * np.nan
                    covt = np.ones(4) * np.nan

                fits[::2, y, x] = popt
                fits[1::2, y, x] = covt
                pbar.update(1)

    v0, dV0, dV, ddV, Fnu, dFnu, tau, dtau = fits
    return v0, dV0, abs(dV), ddV, Fnu, dFnu, tau, dtau


def collapse_gausshermite(velax, data, rms, threshold=None):
    """
    Collapse the cube by fitting a Hermite expansion of a Gaussians to each
    pixel. This allows for a flexible line profile that purely a Gaussian,
    where the ``h3`` and ``h4`` terms quantify the skewness and kurtosis of the
    line as in `van der Marel & Franx (1993)`_.

    To help the fitting, which is done with ``scipy.optimize.curve_fit`` which
    utilises non-linear least squares, the
    :func:`bettermoments.collapse_cube.quadratic` method is first run to obtain
    line centers and peaks, while the :func:`bettermoments.collapse_cube.width`
    method is used to estimate the line width. Pixels that return a ``NaN``
    are skipped for the line fitting.

    .. _van der Marel & Franx (1993): https://ui.adsabs.harvard.edu/abs/1993ApJ...407..525V/abstract

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

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
    from scipy.optimize import curve_fit

    v0, _, Fnu, _ = collapse_quadratic(velax=velax, data=data, rms=rms)
    dV, _ = collapse_width(velax=velax, data=data, rms=rms)
    p0 = np.squeeze([v0, dV, Fnu, np.ones(v0.shape), np.ones(v0.shape)])

    sigma = rms * np.ones(velax.shape)
    fits = np.ones((10, data.shape[1], data.shape[2])) * np.nan

    with tqdm(total=np.all(np.isfinite(p0), axis=0).sum()) as pbar:
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):

                p0_tmp = p0[:, y, x]
                if any(np.isnan(p0_tmp)):
                    continue

                f0 = data[:, y, x].copy()
                f0 = np.isfinite(f0) & (f0 != 0.0)

                x_tmp = velax[f0]
                y_tmp = data[f0, y, x]
                dytmp = sigma[f0]

                try:
                    popt, covt = curve_fit(_gaussian_hermite, x_tmp, y_tmp,
                                           sigma=dytmp, p0=p0_tmp,
                                           absolute_sigma=True,
                                           maxfev=1000000)
                    covt = np.diag(covt)**0.5
                except:
                    popt = np.ones(5) * np.nan
                    covt = np.ones(5) * np.nan

                fits[::2, y, x] = popt
                fits[1::2, y, x] = covt
                pbar.update(1)

    v0, dV0, dV, ddV, Fnu, dFnu, H3, dH3, H4, dH4 = fits
    return v0, dV0, abs(dV), ddV, Fnu, dFnu, H3, dH3, H4, dH4


def _H3(x):
    """Third Hermite polynomial."""
    return (2 * x**3 - 3 * x) * 3**-0.5


def _H4(x):
    """Fourth Hermite polynomial."""
    return (4 * x**4 - 12 * x**2 + 3) * 24**-0.5


def _gaussian_hermite(v, v0, dV, A, h3=0.0, h4=0.0):
    """Gauss-Hermite expanded line profile. ``dV`` is the Doppler width."""
    x = 1.4142135623730951 * (v - v0) / dV
    corr = 1.0 + h3 * _H3(x) + h4 * _H4(x)
    return A * np.exp(-x**2 / 2) * corr


def _gaussian_thick(v, v0, dV, A, tau):
    """Gaussian profile with optical depth. ``dV`` is the Doppler width."""
    tau_profile = _gaussian_hermite(v, v0, dV, tau)
    return A * (1 - np.exp(-tau_profile))


def _gaussian(v, v0, dV, A):
    """Gaussian profile. ``dV`` is the Doppler width."""
    return A * np.exp(-np.power((v - v0) / dV, 2.0))


# -- Non-Traditional Methods -- #


def collapse_quadratic(velax, data, rms):
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
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``v0`` (`ndarray`), ``dv0`` (`ndarray`), ``Fnu`` (`ndarray`), ``dFnu`` (`ndarray`):
            ``v0``, the line center in the same units as ``velax`` with ``dv0``
            as the uncertainty on ``v0`` in the same units as ``velax``.
            ``Fnu`` is the line peak in the same units as the
            ``data`` with associated uncertainties, ``dFnu``.
    """
    from bettermoments.quadratic import quadratic
    chan = np.diff(velax).mean()
    return quadratic(data, x0=velax[0], dx=chan, uncertainty=rms)


def collapse_maximum(velax, data, rms):
    """
    A wrapper returning the result of both
    :func:`bettermoments.collapse_cube.collapse_eighth` and
    :func:`bettermoments.collapse_cube.collapse_ninth`.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M8`` (`ndarray`), ``dM8`` (`ndarray`), ``M9`` (`ndarray`), ``dM9`` (`ndarray`):
            The peak value, ``M8``, and the associated uncertainty, ``dM8``.
            The velocity value of the peak value, ``M9``, and the associated
            uncertainty, ``dM9``.
    """
    M8, dM8 = collapse_eighth(velax=velax, data=data, rms=rms)
    M9, dM9 = collapse_ninth(velax=velax, data=data, rms=rms)
    return M8, dM8, M9, dM9


def collapse_width(velax, data, rms):
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
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``dV`` (`ndarray`), ``ddV`` (`ndarray`):
            The effective velocity dispersion, ``dV`` and ``ddV``, the
            associated uncertainty.
    """
    M0, dM0 = collapse_zeroth(velax=velax, data=data, rms=rms)
    _, _, Fnu, dFnu = collapse_quadratic(velax=velax, data=data, rms=rms)
    dV = M0 / Fnu / np.sqrt(np.pi)
    ddV = dV * np.hypot(dFnu / Fnu, dM0 / M0)
    return abs(dV), abs(ddV)


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


def _estimate_RMS(data, N=5):
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


def _save_array(original_path, new_path, array, overwrite=True, bunit=None):
    """Use the header from `original_path` to save a new FITS file."""
    header = _write_header(original_path, bunit)
    fits.writeto(new_path, array.astype(float), header, overwrite=overwrite,
                 output_verify='silentfix')


def main():

    # Parse all the command line arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to the FITS cube to collapse.')
    parser.add_argument('-method', default='quadratic',
                        help='Method used to collapse cube.')
    parser.add_argument('-smooth', default=0, type=int,
                        help='Width of filter to smooth spectrally.')
    parser.add_argument('-kernel', default='savitzkygolay',
                        help='Kernel type to use for spectral smoothing.')
    parser.add_argument('-rms', default=None,
                        help='Estimated uncertainty on each pixel.')
    parser.add_argument('-noisechannels', default=5,
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
                        help='Width of filter in beam FWHM to smooth threshold map.')
    parser.add_argument('-combine', default='and',
                        help='How to combine the masks if provided.')
    parser.add_argument('--nooverwrite', action='store_false',
                        help='Do not overwrite files.')
    parser.add_argument('--silent', action='store_true',
                        help='Do not see how the sausages are made.')
    parser.add_argument('--returnmask', action='store_true',
                        help='Return the masked used as a FITS file.')
    args = parser.parse_args()

    # Check they all make sense.

    args.kernel = args.kernel.lower()
    if args.kernel not in ['gaussian', 'savitzkygolay']:
        raise ValueError("`kernel` must be `gaussian` or `savitzkygolay`.")

    if not isinstance(args.noisechannels, int) or args.noisechannels < 1:
        raise ValueError("`noisechannels` must an integer greater than 1.")

    args.combine = args.combine.lower()
    if args.combine not in ['and', 'or']:
        raise ValueError("`combine` must be `and` or `or`.")

    if not args.silent:
        import warnings
        warnings.filterwarnings("ignore")

    # Read in the data and the user-defined mask.

    if not args.silent:
        print("Loading up data...")
    data, velax, bunits = _get_cube(args.path)

    if args.mask is None:
        user_mask = np.ones(data.shape)
    else:
        user_mask = _get_data(args.mask)
        user_mask = user_mask.astype('float')

    # Define the velocity mask.

    if args.lastchannel == -1:
        args.lastchannel = data.shape[0]
    velo_mask = np.ones(data.shape)
    velo_mask[args.lastchannel:] = 0.0
    velo_mask[:args.firstchannel] = 0.0

    # Smooth the data in the spectral dimension.

    if args.smooth > 0:
        if not args.silent:
            print("Smoothing data along spectral axis...")
        if args.kernel == 'savitzkygolay':
            from scipy.signal import savgol_filter
            kernel = args.smooth + 1 if not args.smooth % 2 else args.smooth
            data = savgol_filter(data, kernel, polyorder=min(2, kernel - 1),
                                 mode='wrap', axis=0)
        elif args.kernel == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            kernel = args.smooth
            data = gaussian_filter1d(data, kernel, mode='wrap', axis=0)

    # Calculate the RMS.

    if args.rms is None:
        args.rms = _estimate_RMS(data, args.noisechannels)
        if not args.silent:
            print("Estimated RMS: {:.2e}.".format(args.rms))

    # Define the threshold mask.

    if args.clip is not None:
        if len(args.clip) == 1:
            args.clip = [-args.clip[0], args.clip[0]]
        if args.smooththreshold > 0.0:
            if not args.silent:
                print("Smoothing threshold map. May take a while...")
            from astropy.convolution import convolve, Gaussian2DKernel
            kernel = args.smooththreshold * _get_pix_per_beam(args.path)
            kernel = Gaussian2DKernel(kernel)
            noise = []
            with tqdm(total=data.shape[0]) as pbar:
                for c in data.copy():
                    noise += [convolve(c, kernel)]
                    pbar.update(1)
            noise = np.squeeze(noise)
            if args.rms is not None and not args.silent:
                print("WARNING: Convolving threshold mask will reduce RMS")
                print("\t Provided `rms` may over-estimate the true RMS.")
            if noise.shape != data.shape:
                raise ValueError("Incorrect smoothing of threshold mask.")
        else:
            noise = data.copy()

        threshold_mask = np.logical_or(noise / args.rms < args.clip[0],
                                       noise / args.rms > args.clip[-1])
        threshold_mask = threshold_mask.astype('float')
    else:
        threshold_mask = np.ones(data.shape)

    # Combine the masks and apply to the data.

    args.combine = np.logical_and if args.combine == 'and' else np.logical_or
    combined_mask = args.combine(user_mask, threshold_mask) * velo_mask
    masked_data = np.where(combined_mask, data, 0.0)

    # Reverse the direction if the velocity axis is decreasing.

    if np.diff(velax).mean() < 0:
        masked_data = masked_data[::-1]
        velax = velax[::-1]

    # Calculate the moments.

    if not args.silent:
        print("Calculating maps...")

    tosave = {}
    if args.method == 'zeroth':
        M0, dM0 = collapse_zeroth(velax=velax, data=masked_data, rms=args.rms)
        tosave['M0'], tosave['dM0'] = M0, dM0

    elif args.method == 'first':
        M1, dM1 = collapse_first(velax=velax, data=masked_data, rms=args.rms)
        tosave['M1'], tosave['dM1'] = M1, dM1

    elif args.method == 'second':
        M2, dM2 = collapse_second(velax=velax, data=masked_data, rms=args.rms)
        tosave['M2'], tosave['dM2'] = M2, dM2

    elif args.method == 'eighth':
        M8, dM8 = collapse_eighth(velax=velax, data=masked_data, rms=args.rms)
        tosave['M8'], tosave['dM8'] = M8, dM8

    elif args.method == 'ninth':
        M9, dM9 = collapse_ninth(velax=velax, data=masked_data, rms=args.rms)
        tosave['M9'], tosave['dM9'] = M9, dM9

    elif args.method == 'maximum':
        temp = collapse_maximum(velax=velax, data=masked_data, rms=args.rms)
        tosave['M8'], tosave['dM8'] = temp[:2]
        tosave['M9'], tosave['dM9'] = temp[2:]

    elif args.method == 'quadratic':
        temp = collapse_quadratic(velax=velax, data=masked_data, rms=args.rms)
        tosave['v0'], tosave['dv0'] = temp[:2]
        tosave['Fnu'], tosave['dFnu'] = temp[2:]

    elif args.method == 'width':
        dV, ddV = collapse_width(velax=velax, data=masked_data, rms=args.rms)
        tosave['dV'], tosave['ddV'] = dV, ddV

    elif args.method == 'gaussian':
        temp = collapse_gaussian(velax=velax, data=masked_data, rms=args.rms)
        tosave['gv0'], tosave['dgv0'] = temp[:2]
        tosave['gdV'], tosave['dgdV'] = temp[2:4]
        tosave['gFnu'], tosave['dgFnu'] = temp[4:]

    elif args.method == 'gaussthick':
        temp = collapse_gaussthick(velax=velax, data=masked_data, rms=args.rms)
        tosave['gv0'], tosave['dgv0'] = temp[:2]
        tosave['gdV'], tosave['dgdV'] = temp[2:4]
        tosave['gFnu'], tosave['dgFnu'] = temp[4:6]
        tosave['gtau'], tosave['dgtau'] = temp[6:]

    elif args.method == 'gausshermite':
        temp = collapse_gausshermite(velax=velax, data=masked_data, rms=args.rms)
        tosave['ghv0'], tosave['dghv0'] = temp[:2]
        tosave['ghdV'], tosave['dghdV'] = temp[2:4]
        tosave['ghFnu'], tosave['dghFnu'] = temp[4:6]
        tosave['ghh3'], tosave['dghh3'] = temp[6:8]
        tosave['ghh4'], tosave['dghh4'] = temp[8:]
    else:
        raise ValueError("Unknown method.")

    if args.returnmask:
        tosave['mask'] = combined_mask

    # Save as FITS files.

    if not args.silent:
        print("Saving maps...")

    for map_name in tosave.keys():
        outname = args.path.replace('.fits', '_{}.fits'.format(map_name))
        _save_array(args.path, outname, tosave[map_name],
                    overwrite=args.nooverwrite, bunit=bunits[map_name])


if __name__ == '__main__':
    main()
