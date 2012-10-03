#!/usr/bin/python
"""General Load / Save object for matrices.
Handles row IDs columns and header column ID rows.
Prevents a clutter of "special case" formatting code every time a matrix needs to be loaded.
"""
import numpy as np
import os
import cPickle as pickle

  
def load(fp, ftype=None, delimit_c=None, header_c="#"):
  """Load matrix based on file extension. Automatically extract row and column IDs if they exist.

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
  if type(fp) == str:
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

  assert ftype and ftype in FTYPES, "ftype must be in %s. If iterator passed rather than filename, specify `ftype` in function parameters." % ", ".join(FTYPES)
  if ftype == "npy":
    return {"M": np.load(fp_raw)}
  elif ftype == "pkl":
    return {"M": pickle.load(fp_raw)}
  elif ftype == "txt":
    headers = []
    fp = line_iter(fp_raw, headers, comment_c=header_c)

  # If the presence of column and row id are unspecified, check for them by reading the first few lines.
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
      has_row_ids = not is_numeric(col1)
        
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

  M = np.genfromtxt(fp, usemask=True, delimiter=delimit_c, comments=header_c)
  return {
    'M': M,
    'row_ids': row_ids,
    'col_ids': col_ids,
    'ftype': ftype,
    'headers': headers
    }


def save(M, fp, ftype="pkl", row_ids=None, col_ids=None, delimit_c=None, fmt="%.6f"):
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
  FTYPES = ("pkl", "npy", "txt")
  if type(fp) == str:
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
  if ftype == "txt" and (row_ids is not None or col_ids is not None):
    print "WARNING: row or column IDs cannot be saved for non-text ftype '%s'. Ignoring ID list." % (ext)
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
    # Write column header
    if col_ids is not None:
      fp.write(delimit_c.join(col_ids)); fp.write("\n")
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

    
    
