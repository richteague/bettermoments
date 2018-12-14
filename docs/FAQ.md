# Frequently Asked Questions

## What methods are available?

`bettermoments` has multiple methods for collapsing your datacube. Thus far the available options are:

#### Intensity

* **zeroth** - Simply integrated along the spectrum to return the total integrated intensity. More commonly known as the zeroth moment map.

Note that both **quadratic** and **maximum** also return the line peak if you want a radial brightness temperature profile.

#### Velocity / Line Center

* **quadratic** - As described in [Teague & Foremanmackey (2018)](http://iopscience.iop.org/article/10.3847/2515-5172/aae265/meta), this method fits a quadratic curve to the pixel of peak intensity and its two neighbouring pixels. This results in a line center and line peak.

* **maximum** - Similar to the **quadratic** method, but only uses the pixel of peak intensity. This limits precision of the line center to the velocity resolution of the data.

* **first** - The intensity weighted average velocity, or more commonly known as the first moment. This is very susceptable to noise in the spectrum so often requires clipping or masking the data to get nice results.

#### Velocity Dispersion

_Coming Soon_

#### Analytical Fits

_Coming Soon_

For more information on any of these functions, use the help: `bettermoments.collapse_cube.collapse_<method>?`.

## Which method is most appropriate for my data?

It depends on what you want to show. If you're worried that your line profile is complex, chances are that describing it with a single statistic is not the best thing to do and fitting a more appropriate analytical profile would be your best bet.

## How does the quadratic method work?

Rather than fitting the full line profile with an analytical expression, we fit a quadratic to the the peak pixel and its two adjacent pixels. For a single component, this has been shown om [Vakili & Hogg (2016)](https://arxiv.org/abs/1610.05873) to provide comparable precision to fitting the full spectrum with the true underlying profile. Thus, in the extremely likely scenario where the underlying profile is not known, this method will out perform more common methods, such as the intensity weighted average. This approach, including the correction to conserve flux, is also extensively discussed in Appendix C of [Courteau (1997)](https://arxiv.org/pdf/astro-ph/9709201.pdf).

We note that there are alternative implemenations of this method available, such as [NEMO](https://github.com/teuben/nemo).
