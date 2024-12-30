# SPDX-FileCopyrightText: 2024 euler2d_kokkos authors
#
# SPDX-License-Identifier: Unlicense

add_subdirectory(sedov_fortran)

configure_file(test_sedov_blast_2d.ini test_sedov_blast_2d.ini COPYONLY)
configure_file(sedov_plot.py sedov_plot.py COPYONLY)
