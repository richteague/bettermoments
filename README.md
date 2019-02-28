# bettermoments

<p align='center'>
  <img src="https://github.com/richteague/bettermoments/blob/master/docs/TWHya.png" width="435" height="435"><br/>
  Measuring precise line-of-sight velocities from Doppler shifted lines is essential<br/>
  when looking for small scale deviations indicative of, for example, embedded planets.<br/><br/>
  <font size="7"><b>bettermoments</b> helps you measure such velocities.</font>
</p>

## Usage

To start, clone the repository then in your shell run

```bash
$ pip install .
```

to install this locally. **This currently only works with Python 3:** Then in any shell simply run

```console
$ bettermoments path/to/cube.fits
```

which will create lovely line peak and line centroid maps (with uncertainties!). These will be saved in the same directory as the original cube. For more information on the available functions, use:

```bash
$ bettermoments -h
```

or refer to the [FAQ](https://github.com/richteague/bettermoments/blob/master/docs/FAQ.md) for more detail.

## Currently Working On

* Analytical fits, e.g. Gaussians or Gauss-Hermite expansions.
* Documentation.

## Attribution

If you make use of this package in your research, particularly the `quadratic` method, please cite [Teague & Foreman-Mackey (2018)](https://arxiv.org/abs/1809.10295),

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
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/dee9bf4b8b6b4d35bab87fded4e6636f)](https://app.codacy.com/app/richteague/bettermoments?utm_source=github.com&utm_medium=referral&utm_content=richteague/bettermoments&utm_campaign=Badge_Grade_Dashboard)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1419754.svg)](https://doi.org/10.5281/zenodo.1419754)
