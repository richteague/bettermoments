.. module:: bettermoments.collapse_cube
.. collapse_cube

API
===

Here we describe all the functions used to collapse the spectral cube which are
typically called by the command line interface. However, importing these into
your workflow may be useful.

In general, for the generated moment maps, ``MX``, where ``X`` is an integer
denotes a statistical moment. For the non-traditional methods, ``v0``, ``dV``
and ``Fnu`` represent the line center, width and peak, respectively.


Moment Maps
-----------

Implementation of traditional moment-map methods. See the `CASA documentation
<https://casa.nrao.edu/docs/CasaRef/image.moments.html>`_ for more information.

.. autofunction:: bettermoments.collapse_cube.collapse_zeroth

.. autofunction:: bettermoments.collapse_cube.collapse_first

.. autofunction:: bettermoments.collapse_cube.collapse_second

.. autofunction:: bettermoments.collapse_cube.collapse_eighth

.. autofunction:: bettermoments.collapse_cube.collapse_ninth

.. autofunction:: bettermoments.collapse_cube.collapse_maximum


Non-Traditional Methods
-----------------------

.. autofunction:: bettermoments.collapse_cube.collapse_quadratic

.. autofunction:: bettermoments.collapse_cube.collapse_width


Higher Order Gaussian Fits
--------------------------

Coming soon.
