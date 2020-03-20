# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["integrated_intensity",
           "intensity_weighted_velocity",
           "intensity_weighted_dispersion",
           "peak_pixel",
           "quadratic",
           "gaussian"]

import numpy as np

try:
    from scipy.ndimage.filters import gaussian_filter1d, _gaussian_kernel1d
except ImportError:
    gaussian_filter1d = None


def _read_mask_path(mask_path, data):
    """Read in the mask and make sure it is the same shape as the data."""
    if mask_path is not None:
        from astropy.io import fits
        extension = mask_path.split('.')[-1].lower()
        if extension == 'fits':
            mask = np.squeeze(fits.getdata(mask_path))
        elif extension == 'npy':
            mask = np.load(mask_path)
        else:
            raise ValueError("Mask must be a `.fits` or `.npy` file.")
        if mask.shape != data.shape:
            raise ValueError("Mismatch in mask and data shape.")
        mask = np.where(np.isfinite(mask), mask, 0.0)
    else:
        mask = np.ones(data.shape)
    return mask.astype('bool')


def _threshold_mask(data, mask, rms=None, threshold=0.0):
    """
    Combines the provided mask with a sigma mask.

    Args:
        data (ndarray): The data cube of intensities or flux densities.
        mask (ndarray): User provided boolean mask from ``_read_mask_path``.
        rms (Optional[ndarray/float]): Either an array or a single value of the
            uncertainty in the image. We assume that the uncertainty is
            constant along the spectrum.
        threshold (Optional[float]): The level of sigma clipping to apply to
            the data (based on the provided uncertainty).

    Returns:
        mask (ndarray): Boolean mask of which pixels to include.
    """
    if rms is None or threshold <= 0.0:
        return mask.astype('bool')
    rms = np.atleast_1d(rms)
    if rms.ndim == 2:
        sigma_mask = abs(data) >= (threshold * rms)[None, :, :]
    else:
        sigma_mask = abs(data) >= threshold * rms
    return np.logical_and(mask, sigma_mask).astype('bool')


def get_intensity_weights(data, mask):
    """
    Returns the weights used for intensity weighted averages. Includes a small
    level of noise so that the weights do not add up to zero along the spectral
    axis.

    Args:
        data (ndarray): The data cube of intensities or flux densities.
        mask (ndarray): The boolean mask of pixels to include.

    Returns:
        weights (ndarray): Array of same shape as data with the non-normalized
            intensity weights with masked regions containing values ~1e-10.
    """
    noise = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    return np.where(mask, abs(data), noise)


def integrated_intensity(data, dx=1.0, rms=None, threshold=0.0, mask_path=None,
                         axis=0):
    """
    Returns the integrated intensity (commonly known as the zeroth moment).

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        dx (Optional[float]): The pixel scale of the ``axis'' dimension.
        rms (Optional[float]): The uncertainty on the intensities given by
            ``data``. All uncertainties are assumed to be the same. If not
            provided, the uncertainty on the centroid will not be estimated.
        threshold (Optional[float]): All pixel values below this value will not
            be included in the calculation of the intensity weighted average.
        mask_path (Optional[ndarray]): A path to a boolean mask to apply to the
            data. Must be in either ``.fits`` or ``.npy`` format and must be in
            the same shape as ``data``.
        axis (Optional[int]): The axis along which the centroid should be
            estimated. By default this will be the zeroth axis.

    Returns:
        m0 (ndarray): The integrated intensity along the ``axis'' dimension in
            each pixel. The units will be [data] * [dx], so typically
            Jy/beam m/s (or equivalently mJy/beam km/s).
        dm0 (ndarray): The uncertainty on ``m0'' if an rms is given, otherwise
            None.
    """
    mask = _read_mask_path(mask_path=mask_path, data=data)
    data = np.moveaxis(data, axis, 0)
    mask = np.moveaxis(mask, axis, 0)
    mask = _threshold_mask(data=data, mask=mask, rms=rms, threshold=threshold)
    npix = np.sum(mask, axis=0)
    m0 = np.trapz(data * mask, dx=dx, axis=0)
    if rms is None:
        return np.where(npix > 1, m0, np.nan)
    dm0 = dx * rms * npix**0.5 * np.ones(m0.shape)
    return np.where(npix > 1, m0, np.nan), np.where(npix > 1, dm0, np.nan)


def intensity_weighted_velocity(data, x0=0.0, dx=1.0, rms=None, threshold=None,
                                mask_path=None, axis=0):
    """
    Returns the intensity weighted average velocity (commonly known as the
    first moment)

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        x0 (Optional[float]): The wavelength/frequency/velocity/etc. value for
            the zeroth pixel in the ``axis'' dimension.
        dx (Optional[float]): The pixel scale of the ``axis'' dimension.
        rms (Optional[float]): The uncertainty on the intensities given by
            ``data``, assumed to be constant along the spectral axis. Can be
            either a 2D map or a single value.
        threshold (Optional[float]): All pixel values below this value will not
            be included in the calculation of the intensity weighted average.
        mask_path (Optional[ndarray]): A path to a boolean mask to apply to the
            data. Must be in either ``.fits`` or ``.npy`` format and must be in
            the same shape as ``data``.
        axis (Optional[int]): The axis along which the centroid should be
            estimated. By default this will be the zeroth axis.

    Returns:
        x_max (ndarray): The centroid of the brightest line along the ``axis''
            dimension in each pixel.
        x_max_sig (ndarray): The uncertainty on ``x_max'' if an uncertainty is
            given, otherwise None.
    """
    mask = _read_mask_path(mask_path=mask_path, data=data)
    data = np.moveaxis(data, axis, 0)
    mask = np.moveaxis(mask, axis, 0)
    mask = _threshold_mask(data=data, mask=mask, rms=rms, threshold=threshold)
    npix = np.sum(mask, axis=0)
    weights = get_intensity_weights(data, mask)
    vpix = dx * np.arange(data.shape[0]) + x0
    vpix = vpix[:, None, None] * np.ones(data.shape)

    # Intensity weighted velocity.
    m1 = np.average(vpix, weights=weights, axis=0)
    if rms is None:
        return np.where(npix > 1, m1, np.nan), None

    # Calculate uncertainty if rms provided.
    dm1 = (vpix - m1[None, :, :]) * rms / np.sum(weights, axis=0)
    dm1 = np.sqrt(np.sum(dm1**2, axis=0))
    return np.where(npix > 1, m1, np.nan), np.where(npix > 1, dm1, np.nan)


def intensity_weighted_dispersion(data, x0=0.0, dx=1.0, rms=None,
                                  threshold=None, mask_path=None, axis=0):
    """
    Returns the intensity weighted velocity dispersion (second moment).
    """

    # Calculate the intensity weighted velocity first.
    m1 = intensity_weighted_velocity(data=data, x0=x0, dx=dx, rms=rms,
                                     threshold=threshold, mask_path=mask_path,
                                     axis=axis)[0]

    # Rearrange the data to what we need.
    mask = _read_mask_path(mask_path=mask_path, data=data)
    data = np.moveaxis(data, axis, 0)
    mask = np.moveaxis(mask, axis, 0)
    mask = _threshold_mask(data=data, mask=mask, rms=rms, threshold=threshold)
    npix = np.sum(mask, axis=0)
    weights = get_intensity_weights(data, mask)
    npix_mask = np.where(npix > 1, 1, np.nan)
    vpix = dx * np.arange(data.shape[0]) + x0
    vpix = vpix[:, None, None] * np.ones(data.shape)

    # Intensity weighted dispersion.
    m1 = m1[None, :, :] * np.ones(data.shape)
    m2 = np.sum(weights * (vpix - m1)**2, axis=0) / np.sum(weights, axis=0)
    m2 = np.sqrt(m2)
    if rms is None:
        return m2 * npix_mask, None

    # Calculate the uncertainties.
    dm2 = ((vpix - m1)**2 - m2**2) * rms / np.sum(weights, axis=0)
    dm2 = np.sqrt(np.sum(dm2**2, axis=0)) / 2. / m2
    return m2 * npix_mask, dm2 * npix_mask


def peak_pixel(data, x0=0.0, dx=1.0, axis=0):
    """
    Returns the velocity of the peak channel for each pixel, and the pixel
    value.

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        x0 (Optional[float]): The wavelength/frequency/velocity/etc. value for
            the zeroth pixel in the ``axis'' dimension.
        dx (Optional[float]): The pixel scale of the ``axis'' dimension.
        axis (Optional[int]): The axis along which the centroid should be
            estimated. By default this will be the zeroth axis.

    Returns:
        x_max (ndarray): The centroid of the brightest line along the ``axis''
            dimension in each pixel.
        x_max_sig (ndarray): The uncertainty on ``x_max''.
        y_max (ndarray): The predicted value of the intensity at maximum.
    """
    x_max = np.argmax(data, axis=axis)
    y_max = np.max(data, axis=axis)
    return x0 + dx * x_max, 0.5 * dx, y_max


def quadratic(data, uncertainty=None, axis=0, x0=0.0, dx=1.0, linewidth=None):
    """
    Compute the quadratic estimate of the centroid of a line in a data cube.

    The use case that we expect is a data cube with spatiotemporal coordinates
    in all but one dimension. The other dimension (given by the ``axis``
    parameter) will generally be wavelength, frequency, or velocity. This
    function estimates the centroid of the *brightest* line along the ``axis''
    dimension, in each spatiotemporal pixel.

    Following Vakili & Hogg we allow for the option for the data to be smoothed
    prior to the parabolic fitting. The recommended kernel is a Gaussian of
    comparable width to the line. However, for low noise data, this is not
    always necessary.

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        uncertainty (Optional[ndarray or float]): The uncertainty on the
            intensities given by ``data``. If this is a scalar, all
            uncertainties are assumed to be the same. If this is an array, it
            must have the same shape as ``data'' and give the uncertainty on
            each intensity. If not provided, the uncertainty on the centroid
            will not be estimated.
        axis (Optional[int]): The axis along which the centroid should be
            estimated. By default this will be the zeroth axis.
        x0 (Optional[float]): The wavelength/frequency/velocity/etc. value for
            the zeroth pixel in the ``axis'' dimension.
        dx (Optional[float]): The pixel scale of the ``axis'' dimension.

    Returns:
        x_max (ndarray): The centroid of the brightest line along the ``axis''
            dimension in each pixel.
        x_max_sig (ndarray or None): The uncertainty on ``x_max''. If
            ``uncertainty'' was not provided, this will be ``None''.
        y_max (ndarray): The predicted value of the intensity at maximum.
        y_max_sig (ndarray or None): The uncertainty on ``y_max''. If
            ``uncertainty'' was not provided, this will be ``None''.

    """
    # Cast the data to a numpy array
    data = np.moveaxis(np.atleast_1d(data), axis, 0)
    shape = data.shape[1:]
    data = np.reshape(data, (len(data), -1))

    # Find the maximum velocity pixel in each spatial pixel
    idx = np.argmax(data, axis=0)

    # Deal with edge effects by keeping track of which pixels are right on the
    # edge of the range
    idx_bottom = idx == 0
    idx_top = idx == len(data) - 1
    idx = np.clip(idx, 1, len(data)-2)

    # Extract the maximum and neighboring pixels
    f_minus = data[(idx-1, range(data.shape[1]))]
    f_max = data[(idx, range(data.shape[1]))]
    f_plus = data[(idx+1, range(data.shape[1]))]

    # Work out the polynomial coefficients
    a0 = 13. * f_max / 12. - (f_plus + f_minus) / 24.
    a1 = 0.5 * (f_plus - f_minus)
    a2 = 0.5 * (f_plus + f_minus - 2*f_max)

    # Compute the maximum of the quadratic
    x_max = idx - 0.5 * a1 / a2
    y_max = a0 - 0.25 * a1**2 / a2

    # Set sensible defaults for the edge cases
    if len(data.shape) > 1:
        x_max[idx_bottom] = 0
        x_max[idx_top] = len(data) - 1
        y_max[idx_bottom] = f_minus[idx_bottom]
        y_max[idx_top] = f_plus[idx_top]
    else:
        if idx_bottom:
            x_max = 0
            y_max = f_minus
        elif idx_top:
            x_max = len(data) - 1
            y_max = f_plus

    # If no uncertainty was provided, end now
    if uncertainty is None:
        return (
            np.reshape(x0 + dx * x_max, shape), None,
            np.reshape(y_max, shape), None,
            np.reshape(2. * a2, shape), None)

    # Compute the uncertainty
    try:
        uncertainty = float(uncertainty) + np.zeros_like(data)
    except TypeError:

        # An array of errors was provided
        uncertainty = np.moveaxis(np.atleast_1d(uncertainty), axis, 0)
        if uncertainty.shape[0] != data.shape[0] or \
                shape != uncertainty.shape[1:]:
            raise ValueError("the data and uncertainty must have the same "
                             "shape")
        uncertainty = np.reshape(uncertainty, (len(uncertainty), -1))

    df_minus = uncertainty[(idx-1, range(uncertainty.shape[1]))]**2
    df_max = uncertainty[(idx, range(uncertainty.shape[1]))]**2
    df_plus = uncertainty[(idx+1, range(uncertainty.shape[1]))]**2

    x_max_var = 0.0625*(a1**2*(df_minus + df_plus) +
                        a1*a2*(df_minus - df_plus) +
                        a2**2*(4.0*df_max + df_minus + df_plus))/a2**4

    y_max_var = 0.015625*(a1**4*(df_minus + df_plus) +
                          2.0*a1**3*a2*(df_minus - df_plus) +
                          4.0*a1**2*a2**2*(df_minus + df_plus) +
                          64.0*a2**4*df_max)/a2**4

    return (
        np.reshape(x0 + dx * x_max, shape),
        np.reshape(dx * np.sqrt(x_max_var), shape),
        np.reshape(y_max, shape),
        np.reshape(np.sqrt(y_max_var), shape))


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
