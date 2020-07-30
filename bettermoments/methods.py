# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np


def peak_pixel(data, x0, dx):
    """
    Returns the velocity of the peak channel for each pixel, and the pixel
    value.

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        x0 (float): The wavelength/frequency/velocity/etc. value for
            the zeroth pixel in the ``axis'' dimension.
        dx (float): The pixel scale of the ``axis'' dimension.

    Returns:
        x_max (ndarray): The centroid of the brightest line along the ``axis''
            dimension in each pixel.
        x_max_sig (ndarray): The uncertainty on ``x_max''.
        y_max (ndarray): The predicted value of the intensity at maximum.
    """
    x_max = np.argmax(data)
    y_max = np.max(data)
    return x0 + dx * x_max, 0.5 * dx, y_max



def gaussian(data, specax, uncertainty=None, axis=0, v0=None, Fnu=None,
             dV=None, curve_fit_kwargs=None):
    """
    Fits a Gaussian line profile to each pixel.

    Initial guesses of the parameters can be proved through the ``v0``, ``Fnu``
    and ``dV`` arguments. If any of these three values are ``np.nan`` for a
    pixel, that pixel is skipped in the fitting.

    The uncertainties are those estimated by ``scipy.optimize.curve_fit`` and
    will not take into account any correlations.

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        specax (ndarray): The spectral axis of the data.
        uncertainty (Optional[ndarray or float]): The uncertainty on the
            intensity given by ``data``. If this is a scalar, all uncertainties
            are assumed to be the same. If this is any arrya, it must have the
            same shape as a channel, i.e. ``data.shape[1:]`` or
            ``data[0].shape``.
        axis (Optional[int]): The axis along which the Gaussian profiles should
            be fit.
        v0 (Optional[ndarray]): Initial guesses of the line centers in same
            units as ``specax``. To mask pixels use ``np.nan`` values.
        Fnu (Optional[ndarray]): Intial guesses of the line peaks in the same
            units as ``data``. To mask pixels use ``np.nan`` values.
        dV (Optional[ndarray]): Initial guesses of the line widths in the same
            units as ``specax``. To mask pixels use ``np.nan`` values.
        curve_fit_kwargs (Optional[dict]): Dictionary of kwargs to pass to
            ``scipy.optimize.curve_fit``.
    """

    # Need to do some data rotation here.
    if axis != 0:
        raise NotImplementedError("Can only collapse along the zeroth axis.")
    assert specax.size == data.shape[0]
    shape = data.shape[1:]

    # Define the velocity axis and uncertainties.
    sigma = np.nanstd(data) if uncertainty is None else uncertainty
    if isinstance(sigma, float):
        sigma = np.ones(data.shape) * sigma
    elif sigma.shape == shape:
        sigma = np.ones(data.shape) * sigma[None, :, :]
    assert sigma.shape == data.shape, "sigma and data do not match shapes"

    # Cycle through initial guesses fitting the data.
    v0 = np.ones(shape=shape) * np.mean(specax) if v0 is None else v0
    assert v0.shape == shape, "Wrong shape in starting ``v0`` values."
    Fnu = np.nanmax(data, axis=0) if Fnu is None else Fnu
    assert Fnu.shape == shape, "Wrong shape in starting ``Fnu`` values."
    dV = np.ones(shape=shape) * 5.0 * np.diff(specax)[0] if dV is None else dV
    assert dV.shape == shape, "Wrong shape in starting ``dV`` values."
    mask = np.all(np.isfinite([v0, Fnu, dV]), axis=0)

    # Set the fitting parameters.
    from scipy.optimize import curve_fit
    params = np.ones(shape=(6, shape[0], shape[1])) * np.nan
    curve_fit_kwargs = {} if curve_fit_kwargs is None else curve_fit_kwargs
    curve_fit_kwargs['maxfev'] = curve_fit_kwargs.pop('maxfev', 100000)

    # Cycle through the pixels applying the fit.
    # Skip over the points which do not have a good initial guess.
    for y in range(shape[0]):
        for x in range(shape[1]):
            if mask[y, x]:
                try:
                    f0 = np.isfinite(data[:, y, x])
                    p0 = [v0[y, x], dV[y, x], Fnu[y, x]]
                    popt, covt = curve_fit(_gaussian_hermite, specax[f0],
                                           data[f0, y, x], p0=p0,
                                           sigma=sigma[f0, y, x],
                                           absolute_sigma=True,
                                           **curve_fit_kwargs)
                    covt = np.diag(covt)**0.5
                except:
                    popt = np.ones(3) * np.nan
                    covt = np.ones(3) * np.nan
                params[::2, y, x] = popt
                params[1::2, y, x] = covt
    return params


def gaussthick(data, specax, uncertainty=None, axis=0, v0=None, Fnu=None,
               dV=None, curve_fit_kwargs=None):
    """
    Fits a Gaussian line profile, including a correction for high optical
    depths to each pixel.

    Initial guesses of the parameters can be proved through the ``v0``, ``Fnu``
    and ``dV`` arguments. If any of these three values are ``np.nan`` for a
    pixel, that pixel is skipped in the fitting.

    The uncertainties are those estimated by ``scipy.optimize.curve_fit`` and
    will not take into account any correlations.

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        specax (ndarray): The spectral axis of the data.
        uncertainty (Optional[ndarray or float]): The uncertainty on the
            intensity given by ``data``. If this is a scalar, all uncertainties
            are assumed to be the same. If this is any arrya, it must have the
            same shape as a channel, i.e. ``data.shape[1:]`` or
            ``data[0].shape``.
        axis (Optional[int]): The axis along which the Gaussian profiles should
            be fit.
        v0 (Optional[ndarray]): Initial guesses of the line centers in same
            units as ``specax``. To mask pixels use ``np.nan`` values.
        Fnu (Optional[ndarray]): Intial guesses of the line peaks in the same
            units as ``data``. To mask pixels use ``np.nan`` values.
        dV (Optional[ndarray]): Initial guesses of the line widths in the same
            units as ``specax``. To mask pixels use ``np.nan`` values.
        curve_fit_kwargs (Optional[dict]): Dictionary of kwargs to pass to
            ``scipy.optimize.curve_fit``.
    """

    # Need to do some data rotation here.
    if axis != 0:
        raise NotImplementedError("Can only collapse along the zeroth axis.")
    assert specax.size == data.shape[0]
    shape = data.shape[1:]

    # Define the velocity axis and uncertainties.
    sigma = np.nanstd(data) if uncertainty is None else uncertainty
    if isinstance(sigma, float):
        sigma = np.ones(data.shape) * sigma
    elif sigma.shape == shape:
        sigma = np.ones(data.shape) * sigma[None, :, :]
    assert sigma.shape == data.shape, "sigma and data do not match shapes"

    # Cycle through initial guesses fitting the data.
    v0 = np.ones(shape=shape) * np.mean(specax) if v0 is None else v0
    assert v0.shape == shape, "Wrong shape in starting ``v0`` values."
    Fnu = np.nanmax(data, axis=0) if Fnu is None else Fnu
    assert Fnu.shape == shape, "Wrong shape in starting ``Fnu`` values."
    dV = np.ones(shape=shape) * 5.0 * np.diff(specax)[0] if dV is None else dV
    assert dV.shape == shape, "Wrong shape in starting ``dV`` values."
    mask = np.all(np.isfinite([v0, Fnu, dV]), axis=0)

    # Set the fitting parameters.
    from scipy.optimize import curve_fit
    params = np.ones(shape=(8, shape[0], shape[1])) * np.nan
    curve_fit_kwargs = {} if curve_fit_kwargs is None else curve_fit_kwargs
    curve_fit_kwargs['maxfev'] = curve_fit_kwargs.pop('maxfev', 100000)

    # Cycle through the pixels applying the fit.
    # Skip over the points which do not have a good initial guess.
    for y in range(shape[0]):
        for x in range(shape[1]):
            if mask[y, x]:
                try:
                    f0 = np.isfinite(data[:, y, x])
                    p0 = [v0[y, x], dV[y, x], Fnu[y, x], 0.0]
                    popt, covt = curve_fit(_gaussian_thick, specax[f0],
                                           data[f0, y, x], p0=p0,
                                           sigma=sigma[f0, y, x],
                                           absolute_sigma=True,
                                           **curve_fit_kwargs)
                    covt = np.diag(covt)**0.5
                except:
                    popt = np.ones(4) * np.nan
                    covt = np.ones(4) * np.nan
                params[::2, y, x] = popt
                params[1::2, y, x] = covt
    params[-2] = np.power(10, params[-2])
    params[-1] *= params[-2] / 0.434
    return params


def gausshermite(data, specax, uncertainty=None, axis=0, v0=None, Fnu=None,
                 dV=None, curve_fit_kwargs=None):
    """
    Fits a Hermite expansion of a Gaussian line profile to each pixel. This
    allows for measures of skewness and kurtosis to be made.

    Initial guesses of the parameters can be proved through the ``v0``, ``Fnu``
    and ``dV`` arguments. If any of these three values are ``np.nan`` for a
    pixel, that pixel is skipped in the fitting.

    The uncertainties are those estimated by ``scipy.optimize.curve_fit`` and
    will not take into account any correlations.
    """

    # Need to do some data rotation here.
    if axis != 0:
        raise NotImplementedError("Can only collapse along the zeroth axis.")
    assert specax.size == data.shape[0]
    shape = data.shape[1:]

    # Define the velocity axis and uncertainties.
    sigma = np.nanstd(data) if uncertainty is None else uncertainty
    if isinstance(sigma, float):
        sigma = np.ones(data.shape) * sigma
    elif sigma.shape == shape:
        sigma = np.ones(data.shape) * sigma[None, :, :]
    assert sigma.shape == data.shape, "sigma and data do not match shapes"

    # Cycle through initial guesses fitting the data.
    v0 = np.ones(shape=shape) * np.mean(specax) if v0 is None else v0
    assert v0.shape == shape, "Wrong shape in starting ``v0`` values."
    Fnu = np.nanmax(data, axis=0) if Fnu is None else Fnu
    assert Fnu.shape == shape, "Wrong shape in starting ``Fnu`` values."
    dV = np.ones(shape=shape) * 5.0 * np.diff(specax)[0] if dV is None else dV
    assert dV.shape == shape, "Wrong shape in starting ``dV`` values."
    mask = np.all(np.isfinite([v0, Fnu, dV]), axis=0)

    # Set the fitting parameters.
    from scipy.optimize import curve_fit
    params = np.ones(shape=(10, shape[0], shape[1])) * np.nan
    curve_fit_kwargs = {} if curve_fit_kwargs is None else curve_fit_kwargs
    curve_fit_kwargs['maxfev'] = curve_fit_kwargs.pop('maxfev', 100000)

    # Cycle through the pixels applying the fit.
    # Skip over the points which do not have a good initial guess.
    for y in range(shape[0]):
        for x in range(shape[1]):
            if mask[y, x]:
                try:
                    f0 = np.isfinite(data[:, y, x])
                    p0 = [v0[y, x], dV[y, x], Fnu[y, x], 0.0, 0.0]
                    popt, covt = curve_fit(_gaussian_hermite, specax[f0],
                                           data[f0, y, x], p0=p0,
                                           sigma=sigma[f0, y, x],
                                           absolute_sigma=True,
                                           **curve_fit_kwargs)
                    covt = np.diag(covt)**0.5
                except:
                    popt = np.ones(5) * np.nan
                    covt = np.ones(5) * np.nan
                params[::2, y, x] = popt
                params[1::2, y, x] = covt
    return params


def _H3(x):
    """Third Hermite polynomial."""
    return (2 * x**3 - 3 * x) * 3**-0.5


def _H4(x):
    """Fourth Hermite polynomial."""
    return (4 * x**4 - 12 * x**2 + 3) * 24**-0.5


def _gaussian_hermite(v, v0, dV, A, h3=0.0, h4=0.0):
    """Gauss-Hermite expanded line profile. ``dV`` is the Doppler width."""
    x = 1.4142135623730951 * (v - v0) / dV
    if h3 != 0.0 or h4 != 0.0:
        corr = 1.0 + h3 * _H3(x) + h4 * _H4(x)
    else:
        corr = 1.0
    return A * np.exp(-x**2 / 2) * corr


def _gaussian_thick(v, v0, dV, A, logtau):
    """Gaussian profile with optical depth. ``dV`` is the Doppler width."""
    tau = _gaussian_hermite(v, v0, dV, np.power(10, logtau))
    return A * (1 - np.exp(-tau))
