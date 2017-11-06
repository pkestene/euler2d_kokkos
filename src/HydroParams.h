/**
 * \file HydroParams.h
 * \brief Hydrodynamics solver parameters.
 *
 * \date April, 16 2016
 * \author P. Kestener
 */
#ifndef HYDRO_PARAMS_H_
#define HYDRO_PARAMS_H_

#include "kokkos_shared.h"
#include "real_type.h"
#include "config/ConfigMap.h"

#include <stdbool.h>

#include "hydro_common.h"

//! hydro field indexes
enum ComponentIndex {
  ID=0,  /*!< ID Density field index */
  IP=1,  /*!< IP Pressure/Energy field index */
  IU=2,  /*!< X velocity / momentum index */
  IV=3  /*!< Y velocity / momentum index */ 
};

//! face index
enum FaceIdType {
  FACE_XMIN=0,
  FACE_XMAX=1,
  FACE_YMIN=2,
  FACE_YMAX=3
};

//! Riemann solver type for hydro fluxes
enum RiemannSolverType {
  RIEMANN_APPROX, /*!< quasi-exact Riemann solver (hydro-only) */ 
  RIEMANN_HLL,    /*!< HLL hydro and MHD Riemann solver */
  RIEMANN_HLLC    /*!< HLLC hydro-only Riemann solver */ 
};

//! type of boundary condition (note that BC_COPY is only used in the
//! MPI version for inside boundary)
enum BoundaryConditionType {
  BC_UNDEFINED, 
  BC_DIRICHLET,   /*!< reflecting border condition */
  BC_NEUMANN,     /*!< absorbing border condition */
  BC_PERIODIC,    /*!< periodic border condition */
  BC_COPY         /*!< only used in MPI parallelized version */
};

//! enum component index
enum ComponentIndex2D {
  IX = 0,
  IY = 1
};

//! direction used in directional splitting scheme
enum Direction {
  XDIR=1, 
  YDIR=2
};

//! location of the outside boundary
enum BoundaryLocation {
  XMIN = 0, 
  XMAX = 1, 
  YMIN = 2, 
  YMAX = 3
};

//! implementation version
enum ImplementationVersion {
  IMPL_VERSION_0,
  IMPL_VERSION_1
};

//! problem type
enum ProblemType {
  PROBLEM_IMPLODE,
  PROBLEM_BLAST
};

// variable names in the order as in component index
extern const char * varNames[4];

struct HydroSettings {

  // hydro (numerical scheme) parameters
  real_t gamma0;      /*!< specific heat capacity ratio (adiabatic index)*/
  real_t gamma6;
  real_t cfl;         /*!< Courant-Friedrich-Lewy parameter.*/
  real_t slope_type;  /*!< type of slope computation (2 for second order scheme).*/
  int    iorder;      /*!< */
  real_t smallr;      /*!< small density cut-off*/
  real_t smallc;      /*!< small speed of sound cut-off*/
  real_t smallp;      /*!< small pressure cut-off*/
  real_t smallpp;     /*!< smallp times smallr*/

  KOKKOS_INLINE_FUNCTION
  HydroSettings() : gamma0(1.4), gamma6(1.0), cfl(1.0), slope_type(2.0),
		    iorder(1),
		    smallr(1e-8), smallc(1e-8), smallp(1e-6), smallpp(1e-6) {}
  
}; // struct HydroSettings

/*
 * Hydro Parameters (declaration)
 */
struct HydroParams {
  
  // run parameters
  int    nStepmax;   /*!< maximun number of time steps. */
  real_t tEnd;       /*!< end of simulation time. */
  int    nOutput;    /*!< number of time steps between 2 consecutive outputs. */
  bool   enableOutput; /*!< enable output file write. */
  
  // geometry parameters
  int nx;     /*!< logical size along X (without ghost cells).*/
  int ny;     /*!< logical size along Y (without ghost cells).*/
  int ghostWidth;  
  int imin;   /*!< index minimum at X border*/
  int imax;   /*!< index maximum at X border*/
  int jmin;   /*!< index minimum at Y border*/
  int jmax;   /*!< index maximum at Y border*/
  
  int isize;  /*!< total size (in cell unit) along X direction with ghosts.*/
  int jsize;  /*!< total size (in cell unit) along Y direction with ghosts.*/
  
  real_t xmin;
  real_t xmax;
  real_t ymin;
  real_t ymax;
  real_t dx;       /*!< x resolution */
  real_t dy;       /*!< y resolution */
  
  int boundary_type_xmin;
  int boundary_type_xmax;
  int boundary_type_ymin;
  int boundary_type_ymax;
  
  // IO parameters
  bool ioVTK;    /*!< enable VTK  output file format (using VTI).*/
  bool ioHDF5;  /*!< enable HDF5 output file format.*/
  
  // hydro settings (will be passed to device functions)
  HydroSettings settings;
  
  int niter_riemann;  /*!< number of iteration usd in quasi-exact riemann solver*/
  int riemannSolverType;
  
  int problemType;
  
  // blast problem parameters
  real_t blast_radius;
  real_t blast_center_x;
  real_t blast_center_y;
  real_t blast_density_in;
  real_t blast_density_out;
  real_t blast_pressure_in;
  real_t blast_pressure_out;
  
  // other parameters
  int implementationVersion=0; /*!< triggers which implementation to use (currently 3 versions)*/

  HydroParams() :
    nStepmax(0), tEnd(0.0), nOutput(0), enableOutput(true),
    nx(0), ny(0), ghostWidth(2),
    imin(0), imax(0), jmin(0), jmax(0),
    isize(0), jsize(0),
    xmin(0.0), xmax(1.0), ymin(0.0), ymax(1.0),
    dx(0.0), dy(0.0),
    boundary_type_xmin(1),
    boundary_type_xmax(1),
    boundary_type_ymin(1),
    boundary_type_ymax(1),
    ioVTK(true), ioHDF5(false),
    settings(),
    niter_riemann(10), riemannSolverType(), problemType(),
    blast_radius(10.0),
    blast_center_x(0.0), blast_center_y(0.0),
    blast_density_in(1.0), blast_density_out(1.2),
    blast_pressure_in(10.0), blast_pressure_out(0.1),
    implementationVersion(0) {}

  void setup(ConfigMap& map);
  void init();
  void print();
  
}; // struct HydroParams


#endif // HYDRO_PARAMS_H_
