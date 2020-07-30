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
flag, where the names of the functions are found in the `API`
For example, to calculate the zeroth moment map,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth

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

One of the most common approaches is to apply a sigma clip.

Help
----

For help with the exact command line options, use

.. code-block:: bash

    bettermoment --help
