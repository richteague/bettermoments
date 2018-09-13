# -*- coding: utf-8 -*-

__all__ = ["disk_model"]

import numpy as np
import scipy.constants as sc
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel


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
    size = 1.5 * r_max / dist
    xgrid = np.linspace(-size, size, Npix)
    ygrid = np.linspace(-size, size, Npix) / np.cos(np.radians(inc))
    velax = vchan * np.arange(-Nchan / 2., Nchan / 2. + 1)

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
    # TODO: Better noise for convolution...
    data = gaussian(velax[:, None, None], vproj[None, :, :],
                    Tkin[None, :, :], dV[None, :, :])
    data = np.where(rpnts[None, :, :] > r_max / dist, 0.0, data)
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
