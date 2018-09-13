# bettermoments

**A robust method for inferring line-of-sight velocities from Doppler shifted spectra.**

Multiple methods have been used to infer the line-of-sight velocity from a Doppler shifted spectra, the most common being the intensity weighted average velocity (first moment maps), the velocity of the peak intensity (ninth moment / peak maps), or the line centre of an analytical profile fit to the spectrum. To get nice maps from these one often needs to include sigma-clipping or masking the data.

With `bettermoments` we provide an alternative which does not require such clipping and combines the sub-channel precision of the first moment map with the model independence of the ninth moment map. It has been shown in [Vakili & Hogg (2016)](https://arxiv.org/abs/1610.05873) that this approach can provide comparable precision to fitting the full spectrum with the true underlying profile without having to assume such a profile.

## Example

Below compares the resulting velocity maps of HD 135344B, a well studied protoplanetary disk ([van der Marel et al. 2016](www.google.com)), using the intensity weighted average velocity, left, the velocity of the peak pixel, centre, and our new, quadradtic approach, right.

![alt text](notebooks/moment_comparison.pdf "Comparison of moments.")
