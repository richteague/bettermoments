.. command_line

Command Line Interface
======================

This is the preferred way to interact with ``bettermoments``. If the install
has gone successfully, you should be able to run ``bettermoments`` from any
directory using,

.. code-block:: bash

    bettermoments path/to/cube.fits

Which, by default, will use the :func:`bettermoments.collapse_cube.collapse_quadratic`
function to calculate line center and line peak maps. This will automatically
extract the data array and spectral axis from the cube and provide them to the
appropriate functions.

.. warning::

    The command line interface will automatically overwrite any files with the
    same name. Make sure that you move or rename old files which you want to
    keep or use the ``--nooverwrite`` flag.

Different Methods
-----------------

To change the method applied to collapse the cube, use the :code:`-method [name]`
flag, where the names of the functions are found in the `API`.
For example, to calculate the zeroth moment map,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth

which will produce a ``*M0.fits`` file with the uncertainties in ``*dM0.fits``.

Masking
-------

When making a moment map it is often useful, sometimes necessary, to mask the
data to reduce the noise in the resulting image. There are a couple of different
options in ``bettermoments`` to do this.

Channel Selection
^^^^^^^^^^^^^^^^^

The most straight forward is a simple channel selection using the ``-firstchannel``
and ``-lastchannel`` arguments. By default these span the entire cube range.
For example,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth -firstchannel 5 -lastchannel 10

would create a zeroth moment map using the channels 5 to 10 inclusively.

Threshold Clipping
^^^^^^^^^^^^^^^^^^

One of the most common approaches is to apply a 'sigma clip', essentially
masking any pixels below some user-specified threshold, usual in untis of the
background RMS. In ``bettermoments`` this is applied with the ``-clip`` argument.

For example,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth -clip 2

would calculate a zeroth moment map out of all the pixels which have an absolute
value of greater than or equal to ``2 * RMS``. The background RMS is automatically
calculated using the central 50% of the pixels in the first and last 5 channels.
The number of channels used for this estimation can be changed with the
``-noisechannels`` argument. Rather than calculating the RMS automatically, you
can specify their own value with the ``-rms`` argument. Note that internally
the RMS is assumed to be homogeneous, both spatially and spectrally.

If you want include asymmetric bounds you can include two ``-clip`` values. For
example,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth -clip -3 2

would mask out all pixel values between ``-3 * RMS`` and ``2 * RMS``.

User-Defined Masks
^^^^^^^^^^^^^^^^^^

Sometimes you may want to include a user-defined mask, such at the CLEAN mask
used when imaging interferometric data. As long as the mask has the same shape
as the data in the image cube you can include this with,

.. code-block:: bash

    bettermoments path/to/cube.fits -mask path/to/mask.fits



Help
----

For help with the exact command line options, use

.. code-block:: bash

    bettermoment --help
