# Comparing numerical and analytical solution of the Sedov blast run

1. run kanop to get the radial profile of density:
```shell
../euler2d ./test_sedov_blast_2d.ini
```
this will create 2 output files:
- `sedov_blast_radial_distances.npy` : radial distance data
- `sedov_blast_density_profile.npy` : density data

2. run python script `sedov_plot.py`:
```shell
./sedov_plot.py --ini ./test_sedov_blast_2d.ini
```
this will compute the analytical solution (using fortran code from F.X. Timmes) and plot the results.
