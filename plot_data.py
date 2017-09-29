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

# test si on a donne un argument (nom du fichier vti a lire)
if len(sys.argv)<2:
    sys.exit('Usage: %s fichier.vti' % sys.argv[0])

# verifie que le fichier existe bien
if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: file %s was not found!' % sys.argv[1])

# ouvre le fichier vti
print 'Reading data {}'.format(sys.argv[1])
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(sys.argv[1])
reader.Update()

# recupere l'image et ses dimensions
im = reader.GetOutput()
rows, cols, _ = im.GetDimensions()

# recupere le tableau 'rho' et le remet en forme 2D
rho = vtk_to_numpy ( im.GetCellData().GetArray(0) )
rho = rho.reshape(rows-1, cols-1)

# affichage graphique
plt.imshow(rho)
plt.show()
