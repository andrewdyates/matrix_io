matrix_io
=========

Load and save matrices into Numpy from a variety of standard formats.

Use this module to load and save matrices in workflows to handle things like:
  * Column IDs
  * Row IDs
  * Import and export of matrices with missing values
  * Best practice ways of saving binary formats (pickled, protocol=2)
  * Differences in numpy versions
  * Import from text that just works

_Functions_:

 * `load(fname)`: load a matrix from file
 * `save(fname, M)`: save a matrix to file, optionally include row and column IDs for text formats
