#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import configparser
import subprocess

def sedov_plot(ini_filename):

    config = configparser.ConfigParser()
    config.read(ini_filename)

    nbins=config.get('blast','num_radial_bins', fallback=1000)
    tEnd=config.get('run','tEnd', fallback=0.1)

    omega=0
    gamma=1.4
    output_filename='sedov_blast_analytical_2d.dat'
    Eblast = 0.311357
    dim=2

    print('sedov args : nbins={} Eblast={} dim={} omega={} gamma={} tEnd={} output_filename={}'.format(nbins,Eblast,dim,omega,gamma,tEnd,output_filename))


    # compute analytical solution
    subprocess.run(["./sedov_fortran/sedov3_qp", str(nbins), str(Eblast), str(dim), str(omega), str(gamma), str(tEnd), str(output_filename)])

    # load analytical solution
    sedov_ana=np.loadtxt(output_filename, skiprows=2)
    sedov_ana_r = sedov_ana[:,1]
    sedov_ana_density = sedov_ana[:,2]

    # load numerical solution
    sedov_num_r = np.load('sedov_blast_radial_distances.npy')
    sedov_num_density = np.load('sedov_blast_density_profile.npy')

    fig = plt.figure()
    plt.plot(sedov_ana_r, sedov_ana_density, 'go-', label='analytical')
    plt.plot(sedov_num_r, sedov_num_density, 'xb-', label='numerical')
    plt.legend()
    plt.title('Sedov blast at tEnd={} with gamma={}'.format(tEnd,gamma))
    plt.show()

###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Display sedov density plots.')
    parser.add_argument('--ini', type=str, default='test_sedov_blast_2d.ini', help='kanop ini parameter file')
    args = parser.parse_args()

    sedov_plot(args.ini)
