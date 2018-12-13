# bettermoments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1419754.svg)](https://doi.org/10.5281/zenodo.1419754)

![TW Hydrae]{docs/TWHya.png}

Measuring precise line-of-sight velocities is essential when looking for small scale deviations indicative of, for example, embedded planets. `bettermoments` helps you measure such velocities.

## Usage

_Work in progress:_ To start, clone the repository then run

```bash
pip install .
```

to install this locally. Then, in the directory of the image cube simply run

```bash
bettermoments path/to/cube.fits
```

which will create lovely line peak and line centroid maps (with uncertainties!). These will be saved in the same directory as the original cube. For more information on the available functions, use:

```bash
bettermoments -h
```

## Currently Working On

* Correct header information in the saved files.
* Analytical fits, e.g. Gaussians or Gauss-Hermite expansions.
* Documentation.

## Attribution

If you make use of this package in your research, please cite [Teague & Foreman-Mackey (2018)](https://arxiv.org/abs/1809.10295),

```
@ARTICLE{2018RNAAS...2c.173T,
       author = {{Teague}, Richard and {Foreman-Mackey}, Daniel},
        title = "{A Robust Method to Measure Centroids of Spectral Lines}",
      journal = {Research Notes of the American Astronomical Society},
         year = 2018,
        month = Sep,
       volume = {2},
          eid = {173},
        pages = {173},
          doi = {10.3847/2515-5172/aae265},
       adsurl = {https://ui.adsabs.harvard.edu/#abs/2018RNAAS...2c.173T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
