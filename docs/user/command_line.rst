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
    keep or use the ``-overwrite False`` flag.

Different Methods
-----------------

To change the method applied to collapse the cube, use the :code:`-method [name]`
flag, where the names of the functions are found in the `API`, but omitting the
``collapse_`` preface.

For example, to calculate the zeroth moment map,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth

Help
----

For help with the exact command line options, use

.. code-block:: bash

    bettermoment --help
