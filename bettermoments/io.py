"""
All the functions to deal with data I/O.
"""

import scipy.constants as sc
from astropy.io import fits
import numpy as np


# -- READ DATA -- #


def load_cube(path, stokes=0):
    """Return the data and velocity axis from the cube."""
    return _get_data(path, stokes=stokes), _get_velax(path)


def _get_data(path, fill_value=0.0, stokes=0):
    """Read the FITS cube."""
    data = np.squeeze(fits.getdata(path))
    if data.ndim == 4:
        data = data[int(stokes)]
    return np.where(np.isfinite(data), data, fill_value)


def _get_velax(path):
    """Read the velocity axis information."""
    return _read_velocity_axis(fits.getheader(path))


def _get_bunits(path):
    """Return the dictionary of units for each collapse_function result."""
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


# -- WRITE DATA -- #


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


def save_to_FITS(moments, method, path, outname=None, overwrite=True):
    """
    Save the returned fits from ``collapse_{method_name}`` as FITS cubes.
    The filenames will replace the ``.fits`` extension with ``_{param}.fits``.

    Args:
        moments (array): Array of moment values from one of the collapse
            methods.
        method (str): Name of the collapse method used, e.g., ``'zeroth'`` if
            ``collapse_zeroth`` was used.
        path (str): Path of the original data cube to grab header information.
        outname (str): Filename prefix for the saved images. Defaults to the
            path of the provided FITS file.
        overwrite (Optional[bool]): Whether to overwrite files.
    """
    from .methods import collapse_method_products
    moments = np.squeeze(moments)
    assert moments.ndim == 3, "Unexpected shape of `moments`."
    outputs = collapse_method_products(method=method).split(',')
    outputs = [output.strip() for output in outputs]
    assert len(outputs) == moments.shape[0], "Unexpected number of outputs."
    outname = outname or path
    for moment, output in zip(moments, outputs):
        header = _write_header(path=path, bunit=_get_bunits(path)[output])
        fits.writeto(outname.replace('.fits', '') + '_{}.fits'.format(output),
                     moment.astype(float), header, overwrite=overwrite,
                     output_verify='silentfix')
