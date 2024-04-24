add_executable(sedov3_qp
  sedov3_qp.f90)

# TODO: refactor sedov3_qp.f90 to avoid all the warnings
remove_definitions(-Wall)
