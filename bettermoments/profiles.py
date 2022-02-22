"""
Analytical forms to fit to the spectra. All functions take the velocity axis
as the first argument, then the free parameters. All functions have a ``_cont``
version which includes a baseline offset (the final free parameter).
"""

import numpy as np


def free_params(model_function):
    """Numer of free parameters in each model."""
    return {'gaussian': 3, 'gaussian_cont': 4, 'gaussthick': 4,
            'gaussthick_cont': 5, 'gaussmulti': 6, 'gaussmulti_cont': 7,
            'gausshermite': 5, 'gausshermite_cont': 6}[model_function]


def gaussian(x, *params):
    """
    Gaussian function with Doppler width.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s] and the line peak in [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussian')
    model = params[2] * np.exp(-((x - params[0]) / params[1])**2)
    return model


def gaussian_cont(x, *params):
    """
    Gaussian function with Doppler width.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s], the line peak in [Jy/beam] and the continuum offset in
            [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussian_cont')
    model = gaussian(x, *params[:-1]) + params[-1]
    return model


def gaussmulti(x, *params):
    """
    Multiple Gaussian components with Doppler width.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s] and the line peak in [Jy/beam]. Multiple components are
            added in sequence.

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) % 3 == 0, "wrong number of parameters"
    model = np.zeros(x.size)
    for i in range(len(params) // 3):
        model += gaussian(x, *params[i*3:(i+1)*3])
    return model


def gaussmulti_cont(x, *params):
    """
    Multiple Gaussian components with Doppler width.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s] and the line peak in [Jy/beam]. Multiple components are
            added in sequence. The final parameter is the continuum offset in
            [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) % 3 == 1, "wrong number of parameters"
    model = gaussmulti(x, *params[:-1]) + params[-1]
    return model


def gaussthick(x, *params):
    """
    Gaussian profile with non-negligible optical depth,

    .. math::
        I(v) = I_{\nu} \big(1 - \exp(\mathcal{G}(v, v0, \Delta V, \tau))\big) \, / \, (1 - \exp(-\tau))

    where :math:`\mathcal{G}` is a Gaussian function. Note that :math:`\tau` is
    forced to be non-negative, but negative values will be clipped to 0.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s], the line peak in [Jy/beam] and the optical depth.

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussthick')
    tau = gaussian(x, params[0], params[1], params[3])
    model = params[2] * (1.0 - np.exp(-np.clip(tau, amin=0.0, amax=None)))
    return model / (1.0 - np.exp(-params[3]))


def gaussthick_cont(x, *params):
    """
    The ``gaussthick`` function with continuum offset. See ``gaussthick``
    for more details.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s], the line peak in [Jy/beam], the optical depth and the
            continuum offset in [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussthick_cont')
    tau = gaussian(x, params[0], params[1], params[3])
    model = params[2] * (1.0 - np.exp(-tau)) + params[4]
    return model


def _H3(x):
    """Third Hermite polynomial."""
    return (2 * x**3 - 3 * x) * 3**-0.5


def _H4(x):
    """Fourth Hermite polynomial."""
    return (4 * x**4 - 12 * x**2 + 3) * 24**-0.5


def gausshermite(x, *params):
    """
    Gauss-Hermite expansion with the Doppler width. This allows for a flexible
    line profile that purely a Gaussian, where the ``h3`` and ``h4`` terms
    quantify the skewness and kurtosis of the line as in
    `van der Marel & Franx (1993)`_.

    .. _van der Marel & Franx (1993): https://ui.adsabs.harvard.edu/abs/1993ApJ...407..525V/abstract

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): TBD.

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gausshermite')
    xx = 2**0.5 * (x - params[0]) / params[1]
    model = params[2] * np.exp(-xx**2 / 2)
    model *= 1.0 + params[3] * _H3(xx) + params[4] * _H4(xx)
    return model


def gausshermite_cont(x, *params):
    """
    The ``gausshermite`` function with continuum offset. See ``gausshermite``
    for more details.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): TBD.

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gausshermite_cont')
    xx = 2**0.5 * (x - params[0]) / params[1]
    model = params[2] * np.exp(-xx**2 / 2)
    model *= 1.0 + params[3] * _H3(xx) + params[4] * _H4(xx)
    model += params[5]
    return model
