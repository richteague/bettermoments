# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["quadratic"]

import numpy as np


def quadratic(data, uncertainty=None, axis=0, x0=0.0, dx=1.0):
    """

    """
    # Cast the data to a numpy array
    data = np.atleast_1d(data)

    # Find the maximum velocity pixel in each spatial pixel
    idx = np.argmax(data, axis=axis)

    # Deal with edge effects by keeping track of which pixels are right on the
    # edge of the range
    idx_bottom = idx == 0
    idx_top = idx == len(data) - 1
    idx = np.clip(idx, 1, len(data)-2)

    # Extract the minimum and neighboring pixels

    get_slice = lambda delta: tuple(range(s) if i != axis else idx + delta  # NOQA
                                    for i, s in enumerate(data.shape))
    f_minus = data[get_slice(-1)]
    f_max = data[get_slice(0)]
    f_plus = data[get_slice(1)]

    # Work out the polynomial coefficients
    a1 = 0.5 * (f_plus - f_minus)
    a2 = 0.5 * (f_plus + f_minus - 2*f_max)

    # Compute the maximum of the quadratic
    x_max = idx - 0.5 * a1 / a2

    # Set sensible defaults for the edge cases
    if len(data.shape) > 1:
        x_max[idx_bottom] = 0
        x_max[idx_top] = len(data) - 1
    else:
        if idx_bottom:
            x_max = 0
        elif idx_top:
            x_max = len(data) - 1

    # If no uncertainty was provided, end now
    if uncertainty is None:
        return x0 + dx * x_max, None

    # Compute the uncertainty
    try:
        uncertainty = float(uncertainty)

    except TypeError:
        # An array of errors was provided
        uncertainty = np.atleast_1d(uncertainty)
        if data.shape != uncertainty.shape:
            raise ValueError("the data and uncertainty must have the same "
                             "shape")

        df_minus = uncertainty[get_slice(-1)]**2
        df_max = uncertainty[get_slice(0)]**2
        df_plus = uncertainty[get_slice(1)]**2

        sig2_a1 = df_max + 0.25*(df_minus + df_plus)
        sig2_a2 = 0.25*(df_minus + df_plus)
        sig_a1a2 = 0.25*(df_plus - df_minus)

        x_max_var = (0.5/a2)**2 * sig2_a1
        x_max_var += (0.5*a1/a2**2)**2 * sig2_a2
        x_max_var -= (0.25*a1/a2**3) * sig_a1a2

        return x0 + dx * x_max, dx * np.sqrt(x_max_var)

    else:
        # The uncertainty is a scalar
        x_max_sig = uncertainty*np.sqrt((0.125*a1**2 + 0.375*a2**2)/a2**4)
        # x_max_sig = uncertainty * np.sqrt(1.0 / (8.0 * a2**2) +
        #                                   3.0 * a1**2 / (2.0 * a2**4))
        return x0 + dx * x_max, dx * x_max_sig
