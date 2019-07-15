# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
import scipy.constants as sc
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel

import bettermoments.methods as bm


@pytest.fixture
def mock_data(Nchan=64, Npix=128):
    _, velax, data, vproj = disk_model(Nchan=Nchan, Npix=Npix)
    data = data[:, :-1, :]
    vproj = vproj[:-1, :]
    assert data.shape == (Nchan, Npix-1, Npix)
    assert data.shape[1:] == vproj.shape
    return (velax, data, vproj)


def test_shapes(mock_data):
    velax, data, vproj = mock_data

    # No uncertainties
    x, dx, y, dy = bm.quadratic(data)
    assert x.shape == vproj.shape
    assert dx is None
    assert y.shape == vproj.shape
    assert dy is None

    # With scalar uncertainty
    sigma = 1.0
    x, dx, y, dy = bm.quadratic(data, sigma)
    assert x.shape == vproj.shape
    assert dx.shape == vproj.shape
    assert y.shape == vproj.shape
    assert dy.shape == vproj.shape

    # With full uncertainties
    sigma = np.ones_like(data)
    x, dx, y, dy = bm.quadratic(data, sigma)
    assert x.shape == vproj.shape
    assert dx.shape == vproj.shape
    assert y.shape == vproj.shape
    assert dy.shape == vproj.shape

    # Make sure that everything works with different axes
    old_axis = 0
    for axis in [1, 2]:
        data = np.moveaxis(data, old_axis, axis)
        sigma = np.moveaxis(sigma, old_axis, axis)
        x, dx, y, dy = bm.quadratic(data, sigma, axis=axis)
        assert x.shape == vproj.shape
        assert dx.shape == vproj.shape
        assert y.shape == vproj.shape
        assert dy.shape == vproj.shape
        old_axis = axis


def test_shape_error(mock_data):
    velax, data, vproj = mock_data
    sigma = np.random.rand(*data.shape)

    with pytest.raises(ValueError):
        bm.quadratic(data, sigma[1:])
    with pytest.raises(ValueError):
        bm.quadratic(data, sigma[:, 1:])
    with pytest.raises(ValueError):
        bm.quadratic(data, sigma[:, :, 1:])


def test_constant_uncertainties(mock_data):
    velax, data, vproj = mock_data
    sig1 = 1.0
    x1, dx1, y1, dy1 = bm.quadratic(data, sig1)
    sig2 = sig1 + np.zeros_like(data)
    x2, dx2, y2, dy2 = bm.quadratic(data, sig2)
    assert np.allclose(x1, x2)
    assert np.allclose(dx1, dx2)
    assert np.allclose(y1, y2)
    assert np.allclose(dy1, dy2)


def test_uncertainty_axis(mock_data):
    np.random.seed(42)
    _, data, _ = mock_data
    sigma = np.random.uniform(1e-2, 5e-2, data.size).reshape(data.shape)
    x1, dx1, y1, dy1 = bm.quadratic(data, sigma)

    old_axis = 0
    for axis in [1, 2]:
        data = np.moveaxis(data, old_axis, axis)
        sigma = np.moveaxis(sigma, old_axis, axis)
        x2, dx2, y2, dy2 = bm.quadratic(data, sigma, axis=axis)
        assert np.allclose(x1, x2)
        assert np.allclose(dx1, dx2)
        assert np.allclose(y1, y2)
        assert np.allclose(dy1, dy2)
        old_axis = axis


def test_fortran_order(mock_data):
    np.random.seed(42)
    _, data, _ = mock_data
    sigma = np.random.uniform(1e-2, 5e-2, data.size).reshape(data.shape)
    x1, dx1, y1, dy1 = bm.quadratic(data, sigma)

    data_f = np.array(data, copy=True, order="F")
    x2, dx2, y2, dy2 = bm.quadratic(data_f, sigma)
    assert np.allclose(x1, x2)
    assert np.allclose(dx1, dx2)
    assert np.allclose(y1, y2)
    assert np.allclose(dy1, dy2)

    sigma_f = np.array(sigma, copy=True, order="F")
    x2, dx2, y2, dy2 = bm.quadratic(data_f, sigma_f)
    assert np.allclose(x1, x2)
    assert np.allclose(dx1, dx2)
    assert np.allclose(y1, y2)
    assert np.allclose(dy1, dy2)

    old_axis = 0
    for axis in [1, 2]:
        data_f = np.moveaxis(data_f, old_axis, axis)
        sigma_f = np.moveaxis(sigma_f, old_axis, axis)
        x2, dx2, y2, dy2 = bm.quadratic(data_f, sigma_f, axis=axis)
        assert np.allclose(x1, x2)
        assert np.allclose(dx1, dx2)
        assert np.allclose(y1, y2)
        assert np.allclose(dy1, dy2)
        old_axis = axis


def test_compare_ninth(mock_data):
    _, data, _ = mock_data
    x9 = np.argmax(data, axis=0)
    x = bm.quadratic(data)[0]
    assert np.all(np.abs(x - x9) <= 0.5)


def test_isclose(mock_data):
    velax, data, vproj = mock_data
    x0 = velax[0]
    dx = velax[1] - velax[0]
    x = bm.quadratic(data, x0=x0, dx=dx)[0]

    a1, b1 = vproj.shape[0]//3, 2*vproj.shape[0]//3
    a2, b2 = vproj.shape[1]//3, 2*vproj.shape[1]//3

    assert np.all(np.abs(x[a1:b1, a2:b2] - vproj[a1:b1, a2:b2]) < dx)


def test_units(mock_data):
    np.random.seed(42)
    velax, data, _ = mock_data
    sigma = np.random.uniform(1e-2, 5e-2, data.size).reshape(data.shape)
    x1, dx1, _, _ = bm.quadratic(data, sigma)

    x0 = velax[0]
    dx = velax[1] - velax[0]
    x2, dx2, _, _ = bm.quadratic(data, sigma, x0=x0, dx=dx)

    assert np.allclose(x0 + x1*dx, x2)
    assert np.allclose(dx1*dx, dx2)


# ============================= #
#                               #
# Code for simulating mock data #
#                               #
# ============================= #

def disk_model(inc=30., mstar=1.0, dist=100., Npix=128, r_max=150., vchan=200.,
               Nchan=64, noise=2.0, Tkin0=40., Tkinq=-0.3, mu=28., beam=None):
    """Build an analytical, geometrically thin disk model. The temperature
    profile is a power-law function, Tkin(r) = Tkin0 * (r / 100au)^Tkinq,
    and is used to calculate the linewidth assuming no non-thermal broadening.
    The rotation profile is purely Keplerian around a point source.

    Args:
        inc (float): Inclination of disk in [degrees].
        mstar (float): Mass of central star in [Msun].
        dist (float): Distance to source in [pc].
        Npix (int): Number of pixels for the spatial dimension.
        vchan (float): Width of a velocity channel in [m/s].
        Nchan (int): Number of velocity channels.
        r_max (float): Outer radius of the disk in [au].
        noise (float): Random noise to add the the data in [K]. Note that if
            the cube is convolved, the resulting noise is much less than
            requested.
        Tkin0 (float): Kinetic temperature at 100au.
        Tkinq (float): Gradient of the temperature power-law profile.
        mu (float): Molecular weight of the molecule used for calculating the
            thermal linewidth.
        beam (float): If specified, the FWHM of a circular Gaussian beam to
            convolve the data with.

    Returns:
        axis (ndarray): Spatial axis in [arcsec].
        velax (ndarray): Velocity axis in [m/s].
        data (ndarray): Data cube in [K].
        vproj (ndarray): True projected rotation profile [m/s].

    """

    # Create the axes of the observations. (x, y) in [arcsec], (v) in [km/s].
    # Make the velocity axis at a 10 times resolution and then average down.
    size = 1.5 * r_max / dist
    xgrid = np.linspace(-size, size, Npix)
    ygrid = np.linspace(-size, size, Npix) / np.cos(np.radians(inc))
    velax = vchan * np.arange(-Nchan * 0.5, Nchan * 0.5 + 1, 0.1)

    # Calculate disk midplane coordinates in [au].
    rpnts = np.hypot(ygrid[:, None], xgrid[None, :])
    tpnts = np.arctan2(ygrid[:, None], xgrid[None, :])

    # Keplerian profile in [m/s].
    vrot = np.sqrt(sc.G * mstar * 1.988e30 / (rpnts * sc.au * dist)**1)
    vproj = vrot * np.sin(np.radians(inc)) * np.cos(tpnts)

    # Temperature and linewidth as a powerlaw in [K] and [m/s].
    Tkin = Tkin0 * (rpnts * dist / 100.)**Tkinq
    dV = thermal_width(Tkin, mu=mu)

    # Build the cube and add noise if requested.
    data = gaussian(velax[:, None, None], vproj[None, :, :],
                    Tkin[None, :, :], dV[None, :, :])

    # Resample the data back down.
    data = np.array([np.average(data[c*10:(c+1)*10], axis=0)
                     for c in range(Nchan)])
    velax = np.array([np.average(velax[c*10:(c+1)*10]) for c in range(Nchan)])
    data = np.where(rpnts[None, :, :] > r_max / dist, 0.0, data)

    # Add noise.
    if noise is not None:
        data += noise * np.random.randn(data.size).reshape(data.shape)

    # Convolve the beam if necessary.
    if beam is not None:
        kernel = beam / 2. / np.sqrt(2. * np.log(2.))
        kernel /= np.diff(xgrid).mean()
        kernel = Gaussian2DKernel(kernel)
        data = np.array([convolve_fft(c, kernel) for c in data])

    return xgrid, velax, data, vproj


def gaussian(x, x0, dx, A):
    """Gaussian function."""
    return A * np.exp(-np.power((x - x0) / dx, 2))


def thermal_width(Tkin, mu=28.):
    """Thermal width in [m/s]."""
    return np.sqrt(2. * sc.k * Tkin / mu / sc.m_p)
