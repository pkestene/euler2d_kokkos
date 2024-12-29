#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple plot of a vtk image data file (.vti).
To use when paraview is anavailable.
"""

import sys
import os

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt

# check if an input file name was given (VTI file)
if len(sys.argv)<2:
    sys.exit('Usage: %s file.vti' % sys.argv[0])

# check that input file exists
if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: file %s was not found!' % sys.argv[1])

# open vti file
print 'Reading data {}'.format(sys.argv[1])
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(sys.argv[1])
reader.Update()

# retrieve image dimensions
im = reader.GetOutput()
rows, cols, _ = im.GetDimensions()

# retrieve data 'rho' and shape it to a 2d array
rho = vtk_to_numpy ( im.GetCellData().GetArray(0) )
rho = rho.reshape(rows-1, cols-1)

# now open display
plt.imshow(rho)
plt.show()
