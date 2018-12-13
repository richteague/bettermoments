# Frequently Asked Questions

## Which method is most appropriate for my data?

It depends.

## How does the quadratic method work?

Rather than fitting the full line profile with an analytical expression, we fit a quadratic to the the peak pixel and its two adjacent pixels. For a single component, this has been shown om [Vakili & Hogg (2016)](https://arxiv.org/abs/1610.05873) to provide comparable precision to fitting the full spectrum with the true underlying profile. Thus, in the extremely likely scenario where the underlying profile is not known, this method will out perform more common methods, such as the intensity weighted average. This approach, including the correction to conserve flux, is also extensively discussed in Appendix C of [Courteau (1997)](https://arxiv.org/pdf/astro-ph/9709201.pdf).

We note that there are alternative implemenations of this method available, such as [NEMO](https://github.com/teuben/nemo).
