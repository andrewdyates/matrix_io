#!/usr/bin/python
"""General Load / Save object for matrices.
Handles row IDs columns and header column ID rows.
Prevents a clutter of "special case" formatting code every time a matrix needs to be loaded.
"""
import numpy as np
import os
import cPickle as pickle

def idx_type(idx):
  """Return if idx is abstract type (str, num, list)"""
  if isinstance(idx, basestring):
    return 'str'
  try:
    float(idx) 
  except TypeError:
    pass
  else:
    return 'num'
  return 'list'
  

class NamedMatrix(object):
  """Named matrix wrapper around load / save module functions.

  Initialize with `load()` parameters or provide `load()` return dict.
  Can index by .row(), .col(), and e() using either label str, int, or list of either.
  """
  def __init__(self, D=None, **kwds):
    self.M = None
    if D is None:
      assert 'fp' in kwds
      fp = kwds.pop('fp',None)
      D = load(fp, **kwds)
    for k, v in D.items():
      setattr(self, k, v)
    assert self.M is not None
    if self.row_ids:
      self.load_row_ids(self.row_ids)
    if self.col_ids:
      self.load_col_ids(self.col_ids)
    self.m = np.size(self.M,0)
    self.n = np.size(self.M,1)
      
  def load_row_ids(self, row_ids):
    assert len(row_ids) == np.size(self.M, 0)
    self.row_ids = np.array(row_ids)
    self.row_idx = dict([(s,i) for i,s in enumerate(self.row_ids)])
    
  def load_col_ids(self, col_ids):
    assert len(col_ids) == np.size(self.M, 1)
    self.col_ids = np.array(col_ids)
    self.col_idx = dict([(s,i) for i,s in enumerate(self.col_ids)])

  def row(self, idx, mutate=False):
    # should this also return row ids?
    idx = self._idx_to_int(idx, "row")
    Q = self.M[idx,:]
    if mutate:
      self.M = Q
      if self.row_ids:
        row_ids = np.array(self.row_ids)[idx]
        self.load_row_ids(row_ids)
        self.m = np.size(self.M,0)
    return Q
  
  def col(self, idx, mutate=False):
    # should this also return col ids?
    idx = self._idx_to_int(idx, "col")
    Q = self.M[:,idx]
    if mutate:
      self.M = Q
      if self.col_ids:
        col_ids = np.array(self.col_ids)[idx]
        self.load_col_ids(col_ids)
        self.n = np.size(self.M,1)
    return Q

  def e(self, idx_row, idx_col):
    idx_row = self._idx_to_int(idx_row, "row")
    idx_col = self._idx_to_int(idx_col, "col")
    return self.M[idx_row,idx_col]

  def _idx_to_int(self, idx, q):
    assert q in ("row", "col")
    if q == "row":
      idx_map = self.row_idx
    elif q == "col":
      idx_map = self.col_idx
    dtype = idx_type(idx)
    if dtype == "list":
      if idx_type(idx[0]) == 'str':
        if idx_map:
          # Map strings to int indices
          idx = [idx_map[s] for s in idx]
        else:
          raise ValueError, "self.%s_idx not defined. Use an integer %s index." % (q, q)
    elif dtype == "str":
      if idx_map:
        idx = idx_map[idx]
      else:
        raise ValueError, "self.%s_idx not defined. Use an integer %s index." % (q, q)
    elif dtype == "int":
      pass
    return idx

  def save(self, fp, **kwds):
    save(self.M, fp, **kwds)


def load(fp, ftype=None, delimit_c=None, header_c="#", check_row_ids=True, check_col_ids=True, dtype=np.float):
  """Load matrix based on file extension. Automatically extract row and column IDs if they exist.

  First cell, if both in the Row_IDs and Col_IDs, assign to Row_IDs

  Args:
    fp: file-like object or string
      The file to read. It must support seek() and read() methods.
    ftype: str in FTYPES ("npy", "pkl", "txt") of file format. Inferred from filename ext.
    delimit_c: chr of row delimiter for text ftypes. Inferred from filename ext.
    header_c: chr of header first-line delimiter for text ftypes
  Returns:
    {str: obj} of loaded objects:
      {
        'M': np.array numeric matrix loaded from `fname` 
        'row_ids': [str] of row IDs, if they were detected. Else, None.
        'col_ids': [str] of col IDs, if they were detected. Else, None.
      }
  """
  FTYPES = ("npy", "pkl", "txt")
  assert fp
  row_ids, col_ids = None, None

  # fp is a filename string. Get type; open it.
  if isinstance(fp, basestring):
    ext = fp.rpartition(".")[2]
    if ext == "tab":
      if not delimit_c:
        delimit_c = "\t"
      ftype = "txt"
    elif ext == "csv":
      if not delimit_c:
        delimit_c = ","
      ftype = "txt"
    elif ext == "txt":
      assert delimit_c, "Delimiter must be specified for .txt extension type."
      ftype = "txt"
    elif ext == "npy":
      ftype = "npy"
    elif ext == "pkl":
      ftype = "pkl"
    else:
      print "WARNING: Unrecognized file extension %s. Using dtype %s." % (ext, ftype)
    fp_raw = open(fp)
  else:
    fp_raw = fp

  assert ftype and ftype in FTYPES, "ftype must be in [%s]. Currently, ftype is `%s`. If iterator passed rather than filename, specify `ftype` in function parameters." % (", ".join(FTYPES), ftype)
  if ftype == "npy":
    return {"M": np.load(fp_raw), 'ftype': 'npy'}
  elif ftype == "pkl":
    return {"M": pickle.load(fp_raw), 'ftype': 'pkl'}
  elif ftype == "txt":
    headers = []
    fp = line_iter(fp_raw, headers, comment_c=header_c)
    
  if check_col_ids: 
    # If the presence of column and row id are unspecified, check for them by reading the first few lines.
    first_line = fp.next()
    # If the first two delimited entries of the first line are not numbers, assume that the first line is a header.
    row = first_line.split(delimit_c)
    if not is_numeric(row[0]) and len(row) > 1 and not is_numeric(row[1]):
      # Directly set column IDs from this row.
      row[-1] = row[-1].rstrip('\n\r')
      col_ids = row  # if col_ids is not set, then it is None
      
  if check_row_ids:
    # Examine the next line. If the next non-empty first column value is not a number, assume that the first column contains row IDs
    for line in fp:
      col1 = line.partition(delimit_c)[0]
      if not col1:
        continue
      else:
        has_row_ids = not is_numeric(col1)
  else:
    has_row_ids = False
        
  # Rewind fp and read file into matrix. Handle column and row IDs in fp iterator.
  fp_raw.seek(0)
  fp = line_iter(fp_raw, comment_c=header_c)
  if col_ids is not None:
    fp.next()
  if has_row_ids:
    row_ids = []
    fp = named_row_iter(fp, varlist=row_ids, delimit_c=delimit_c)

  # Handle numpy1.5 error regarding missing 'read' and 'readline' functions
  if np.version.version < 1.6:
    fp = FakeFile(fp)

  M = np.genfromtxt(fp, usemask=True, delimiter=delimit_c, comments=header_c, dtype=dtype)

  # If column IDs has an extra entry, assume that the first entry is a filler label.
  if has_row_ids and col_ids is not None:
    if len(col_ids) == M.shape[1]+1:
      col_ids = col_ids[1:]

  return {
    'M': M,
    'row_ids': row_ids,
    'col_ids': col_ids,
    'ftype': ftype,
    'headers': headers,
    'dtype': dtype
    }


def save(M, fp, ftype="pkl", row_ids=None, col_ids=None, headers=None, delimit_c="\t", fmt="%.6f", comment_c="#", fill_upper_left=True):
  """Save matrix. Return filename of matrix saved.
  Optionally include row_ids or col_ids if target ftype is text-based.

  Args:
    M: np.array of matrix to save
    fp: file-like object or string.
      The file to read. It must support seek() and read() methods.
    ftype: str of saving file type; inferred from fp filename
    row_ids: [str] of row IDs or None
    col_ids: [str] of row IDs or None
    fmt: str of numeric pattern if using ftype type is txt
  Returns:
    str of ftype file format in which the matrix was saved.
  """
  assert isinstance(fp, basestring) or hasattr(fp, 'write'), "Parameter `fp` must either be a writable file object or a string of a valid file path. (Did you call save(M, fp...) with the parameters in the right order?)"
  FTYPES = ("pkl", "npy", "txt")
  if isinstance(fill_upper_left, basestring) and fill_upper_left.lower() in ('f', 'false', 'none'):
    fill_upper_left = False
  if isinstance(fp, basestring):
    basename,c,ext = os.path.basename(fp).rpartition('.')
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
    elif ext == "txt":
      assert delimit_c, "Delimiter must be specified for .txt extension type."
      ftype = "txt"
    else:
      assert ftype, "Cannot infer ftype from file extension %s." % (ext)
      print "WARNING: Unknown file extension '%s' for file name %s. Using type %s..." % (ext, fp, ftype)
    fp = open(fp, "w")

  assert ftype in FTYPES, "ftype must be in %s" % ", ".join(FTYPES)
  if ftype != "txt" and (row_ids is not None or col_ids is not None or headers is not None):
    print "WARNING: row or column IDs or headers cannot be saved for non-text ftype '%s'. Ignoring ID list." % (ext)
  else:
    if row_ids is not None:
      assert np.size(M,0) == len(row_ids)
    if col_ids is not None:
      assert np.size(M,1) == len(col_ids)
      
  # Write matrix
  if ftype == "pkl":
    pickle.dump(M, fp, protocol=2)
  elif ftype == "npy":
    np.save(fp, M)
  elif ftype == "txt":
    # Write headers
    if headers is not None:
      for line in headers:
        if line[0] != "#":
          fp.write(comment_c)
        fp.write(line)
        if line[-1] != "\n":
          fp.write("\n")
    # Write column header
    if col_ids is not None:
      row = col_ids
      # top left cell of first row ID in col ID row.
      if row_ids is not None:
        if fill_upper_left:
          row = ["COL_ID"] + row
      fp.write(delimit_c.join(row)); fp.write("\n")
    # Write row IDs
    for i, row in enumerate(M):
      if row_ids is not None:
        fp.write(row_ids[i] + delimit_c)
      fp.write(row_to_txt(row, fmt)); fp.write("\n")
  else:
    raise Exception, "Unknown file type. Cannot save matrix."
  return ftype
      

def save_ids(fname, ids):
  fp = open(fname, "w")
  for s in ids:
    fp.write(s+"\n")
  return fname
      
def row_to_txt(row, fmt='%.6f'):
  s = []
  for i in range(len(row)):
    if hasattr(row, 'mask') and row.mask[i]:
      s.append("")
    else:
      try:
        s.append(fmt%row[i])
      except TypeError:
        # Handle non-numeric types directly
        s.append(str(row[i]))
  return "\t".join(s)

def is_numeric(x):
  try:
    float(x)
  except (ValueError, TypeError):
    return False
  else:
    return True
    
def line_iter(fp, headers=None, comment_c="#"):
  """Iterate over lines, skipping over comments and blank lines.
  Strip \r if it exists.

  Append headers if headers array provided.
  """
  for line in fp:
    if line[0] in ('\n'):
      continue
    if line[0] == comment_c:
      if headers is not None:
        headers.append(line)
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

    
    
