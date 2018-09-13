# bettermoments

Measuring precise line-of-sight velocities is essential when looking for small scale deviations indicative of, for example, embedded planets. `bettermoments` helps you measure such velocities.

## Approach

Rather than fitting the full line profile with an analytical expression, we fit a quadratic to the the peak pixel and its two adjacent pixels. This has been shown om [Vakili & Hogg (2016)](https://arxiv.org/abs/1610.05873) to provide comparable precision to fitting the full spectrum with the true underlying profile. Thus, in the extremely likely scenario where the underlying profile is not known, this method will out perform more common methods, such as the intensity weighted average.
