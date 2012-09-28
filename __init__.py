#!/usr/bin/python
"""General Load / Save object for matrices.
Handles row IDs columns and header column ID rows.
Prevents a clutter of "special case" formatting code every time a matrix needs to be loaded.
"""
import numpy as np
import os
import cPickle as pickle

  
def load(fname, delimit_c="\t", header_c="#"):
  """Load matrix based on file extension. Automatically extract row and column IDs if they exist.

  Args:
    fname: str of filename to load. Use .ext to determine load type.
    delimit_c: chr of row delimiter for text ftypes
    header_c: chr of header first-line delimiter for text ftypes
  Returns:
    {str: obj} of loaded objects:
      {
        'M': np.array numeric matrix loaded from `fname` 
        'row_ids': [str] of row IDs, if they were detected. Else, None.
        'col_ids': [str] of col IDs, if they were detected. Else, None.
      }
  """
  row_ids, col_ids = None, None
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
    pass
  else:
    print "WARNING: Unrecognized file extension '%s' for file %s." % (ext, fname)

  # If the presence of column and row id are unspecified, check for them by reading the first few lines.
  fp = line_iter(open(fname), comment_c=header_c)
  first_line = fp.next()
  # If the first two delimited entries of the first line are not numbers, assume that the first line is a header.
  row = first_line.split(delimit_c)
  if not is_numeric(row[0]) and not is_numeric(row[1]):
    # Directly set column IDs from this row.
    col_ids = row
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
  if col_ids is not None:
    fp.next()
  if has_row_ids:
    row_ids = []
    fp = named_row_iter(fp, varlist=row_ids, delimit_c=delimit_c)

  # Handle numpy1.5 error regarding missing 'read' and 'readline' functions
  if numpy.version.version < 1.6:
    fp = FakeFile(fp)

  M = np.genfromtxt(fp, usemask=True, delimiter=delimit_c, comments=header_c)
  return {
    'M': M,
    'row_ids': row_ids,
    'col_ids': col_ids
    }


def save(fname, M, row_ids=None, col_ids=None, ftype="pkl", delimit_c=None, fmt="%.6f"):
  """Save matrix. Return filename of matrix saved.
  Optionally include row_ids or col_ids if target ftype is text-based.

  Args:
    fname: str of path where to save matrix
    M: np.array of matrix to save
    row_ids: [str] of row IDs or None
    col_ids: [str] of row IDs or None
    fmt: str of numeric pattern if using ftype type is txt
  """
  # Default return value; modify if row_ids or col_ids are saved.
  basename,c,ext = os.path.basename(fname).rpartition('.')
  ext = ext.lower()
  if ext == "pkl":
    ftype = "pkl"
  elif ext == "npy":
    ftype = "npy"
  elif ext == "tab":
    ftype = "txt"
    delimit_c = "\t"
  elif ext == "csv":
    ftype = "txt"
    delimit_c = ","
  else:
    print "WARNING: Unknown file extension '%s' for file name %s. Using type %s..." % (ext, fname, ftype)


  if ftype == "txt" and (row_ids is not None or col_ids is not None):
    print "WARNING: row or column IDs cannot be saved for non-text ftype '%s'. Ignoring ID list." % (ext)
  else:
    if row_ids is not None:
      assert np.size(M,0) == len(row_ids)
    if col_ids is not None:
      assert np.size(M,1) == len(col_ids)
      
  # Write matrix
  if ftype == "pkl":
    pickle.dump(M, open(fname,"w"), protocol=2)
  elif ftype == "npy":
    np.save(fname, M)
  elif ftype == "txt":
    fp = open(fname, "w")
    if col_ids is not None:
      fp.write(delimit_c.join(col_ids)); fp.write("\n")
    for i, row in enumerate(M):
      if row_ids is not None:
        fp.write(row_ids[i] + delimit_c)
      fp.write(row_to_txt(row, fmt)); fp.write("\n")
      

def save_ids(fname, ids):
  fp = open(fname, "w")
  for s in ids:
    fp.write(s+"\n")
  return fname
      
def row_to_txt(row, fmt='%.6f'):
  s = []
  for i in range(len(row)):
    if row.mask[i]:
      s.append("")
    else:
      s.append(fmt%row[i])
  return "\t".join(s)

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

    
    
