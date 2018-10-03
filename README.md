# bettermoments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1419754.svg)](https://doi.org/10.5281/zenodo.1419754)

Measuring precise line-of-sight velocities is essential when looking for small scale deviations indicative of, for example, embedded planets. `bettermoments` helps you measure such velocities.

## Approach

Rather than fitting the full line profile with an analytical expression, we fit a quadratic to the the peak pixel and its two adjacent pixels. For a single component, this has been shown om [Vakili & Hogg (2016)](https://arxiv.org/abs/1610.05873) to provide comparable precision to fitting the full spectrum with the true underlying profile. Thus, in the extremely likely scenario where the underlying profile is not known, this method will out perform more common methods, such as the intensity weighted average. This approach, including the correction to conserve flux, is also extensively discussed in Appendix C of [Courteau (1997)](https://arxiv.org/pdf/astro-ph/9709201.pdf).

We note that there are alternative implemenations of this method available, such as [NEMO](https://github.com/teuben/nemo).

## Usage

To start, install the `bettermoments` package by executing

```bash
pip install bettermoments
```

or if you have cloned the repository then you can change into the directory and run

```bash
pip install --user .
```

Then you can execute this module using

```python
import bettermoments as bm

x_max, x_max_sigma, y_max, y_max_sigma = bm.quadratic(data, uncertainties)
```

See the docstring for the `bettermoments.quadratic` function for more
information on the available options.

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
