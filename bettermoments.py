# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["quadratic", "data_from_fits"]

import numpy as np

try:
    from scipy.ndimage.filters import gaussian_filter1d, _gaussian_kernel1d
except ImportError:
    gaussian_filter1d = None


def quadratic(data, uncertainty=None, axis=0, x0=0.0, dx=1.0, linewidth=None):
    """Compute the quadratic estimate of the centroid of a line in a data cube

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
        linewidth (Optional [float]): Estimated standard deviation of the line
            in units of pixels.

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

    # Smooth the data if asked
    truncate = 4.0
    if linewidth is not None:
        if gaussian_filter1d is None:
            raise ImportError("scipy is required for smoothing")
        data = gaussian_filter1d(data, linewidth, axis=0, truncate=truncate)

    # Deal with edge effects by keeping track of which pixels are right on the
    # edge of the range
    idx_bottom = idx == 0
    idx_top = idx == len(data) - 1
    idx = np.clip(idx, 1, len(data)-2)

    # Extract the maximum and neighboring pixels
    get_slice = lambda delta: tuple(range(s) if i != axis else idx + delta  # NOQA
                                    for i, s in enumerate(data.shape))
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
            np.reshape(y_max, shape), None)

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

    # Update the uncertainties for the smoothed data:
    #  sigma_smooth = sqrt(norm * k**2 x sigma_n**2)
    if linewidth is not None:
        # The updated uncertainties need to be updated by convolving with the
        # square of the kernel with which the data were smoothed. Then, this
        # needs to be properly normalized. See the scipy source for the
        # details of this normalization:
        # https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py
        sigma = linewidth / np.sqrt(2)
        lw = int(truncate * linewidth + 0.5)
        norm = np.sum(_gaussian_kernel1d(linewidth, 0, lw)**2)
        norm /= np.sum(_gaussian_kernel1d(sigma, 0, lw))
        uncertainty = np.sqrt(norm * gaussian_filter1d(
            uncertainty**2, sigma, axis=0))

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


def data_from_fits(path, nchan=None):
    """Convenience function to return the data and velocity axis from a FITS
    file in order to pass to quadratic.

    Args:
        path (str): path to the FITS file.
        nchan (Optional[int]): number of channels at the start and at the end
            of the cube used to estimate the uncertainty. If not specified, no
            uncertainty will be estimated.

    Returns:
        data (ndarry): The data cube.
        v0 (float): The wavelength/frequency/velocity/etc. value for the zeroth
            pixel in the velocity / frequency dimension.
        dv (float): The pixel scale of the velocity / frequency axis.
        dx (float): The pixel scale of the spatial axis in arcseconds.
        uncertainty (float or None): The uncertainty on the intensities given
            by ``data``. Assumed to be homogeneous across the dataset.

    """
    from astropy.io import fits

    data = fits.getdata(path)
    header = fits.getheader(path)

    # Calculate the spectral axis.
    dv = header['cdelt3']
    v0 = header['crval3'] + (1.0 - header['crpix3']) * dv
    dx = header['cdelt2'] * 3600.

    # If no nchan is given, return now.
    if nchan is None:
        return (data, v0, dv, dx, None)

    # Estimate the noise in the image.
    nchan = int(nchan)
    uncertainty = np.nanstd([data[:nchan], data[-nchan:]])
    return (data, v0, dv, dx, uncertainty)
