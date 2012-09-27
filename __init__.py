#!/usr/bin/python
"""General Load / Save object for matrices.
Prevents a clutter of "special case" formatting code every time a matrix needs to be loaded.
"""
import numpy as np
import cPickle as pickle

class MIO(object):

  def __init__(self):
    self.row_ids = None
    self.col_ids = None
  
  def load(self, fname, delimit_c=None, header_c="#"):
    ext = fname.rpartition[2].lower()
    if ext == "npy":
      return np.load(fname)
    elif ext == "pkl":
      return pickle.load(open(fname))
    
    # Load from text files.
    elif ext == "tab":
      delimit_c = "\t"
    elif ext == "csv":
      delimit_c = ","
    elif ext == "txt":
      if delimit_c is None:
        raise Exception, "Manually set delimiter for .txt files."
    else:
      raise Exception, "Unrecognized file extension '%s' for file %s." % (ext, fname)

    # If the presence of column and row id are unspecified, check for them by reading the first few lines.
    fp = line_iter(open(fname), comment_c=header_c)
    first_line = fp.next()
    # If the first two delimited entries of the first line are not numbers, assume that the first line is a header.
    row = first_line.split(delimit_c)
    if not is_numeric(row[0]) and not is_numeric(row[1]):
      # Directly set column IDs from this row.
      self.col_ids = row
    else:
      self.col_ids = None
    # Examine the next line. If the next non-empty first column value is not a number, assume that the first column contains row IDs
    for line in fp:
      col1 = line.partition(delimit_c)[0]
      if not col1:
        continue
      else:
        has_row_ids = is_numeric(col1)
        
    # Trash fp and read file into matrix. Handle column and row IDs in fp iterator.
    del fp
    fp = line_iter(open(fname), comment_c=header_c)
    if self.col_ids is not None:
      fp.next()
    if has_row_ids:
      self.row_ids = []
      fp = named_row_iter(fp, varlist=self.row_ids, delimit_c=delimit_c)

    # Handle numpy1.5 error regarding missing 'read' and 'readline' functions
    if numpy.version.version < 1.6:
      fp = FakeFile(fp)

    M = np.genfromtxt(fp, usemask=True, delimiter=delimit_c, comments=header_c)
    return M
      
    
def is_numeric(x):
  try:
    float(x)
  except (ValueError, TypeError):
    return False
  else:
    return True
    
def line_iter(fp, comment_c="#"):
  """Iterate over lines, skipping over comments and blank lines.
  Strip \r if it exists.
  """
  for line in fp:
    if line[0] in (comment_c, '\n'):
      continue
    yield line.strip('\r\n')+"\n"

def named_row_iter(fp, varlist, delimit_c="\t"):
  """Yield next row of floats w/o ID in first column; append ID to varlist.

  Args:
    fp: [*str] of readable file-pointer-like iterator, yields delimited line
    varlist: [str] of row IDs to append to
  """
  for line in fp:
    name,c,row = line.partition(delimit_c)
    varlist.append(name)
    yield row

class FakeFile(object):
  def __init__(self, line_generator):
    self.s = line_generator
  def __iter__(self):
    return self
  def next(self):
    return self.s.next()
  def read(self):
    return "".join([q for q in self.s])
  def readline(self):
    return self.s.next()

    
    
