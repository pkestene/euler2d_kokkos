/**
 *
 */
#ifndef HYDRO_RUN_H_
#define HYDRO_RUN_H_

#include "Timer.h"
#include "HydroParams.h"
#include "kokkos_shared.h"

/**
 * Main hydrodynamics data structure.
 */
class HydroRun
{

public:

  HydroRun(HydroParams& params, ConfigMap& configMap);
  virtual ~HydroRun();
  
  // hydroParams
  HydroParams& params;
  ConfigMap&   configMap;
  
  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
  DataArray     Q;     /*!< hydrodynamics primitive    variables array  */

  /* implementation 0 */
  DataArray Fluxes_x; /*!< implementation 0 */
  DataArray Fluxes_y; /*!< implementation 0 */
  
  /* implementation 1 only */
  DataArray Slopes_x; /*!< implementation 1 only */
  DataArray Slopes_y; /*!< implementation 1 only */

  //riemann_solver_t riemann_solver_fn; /*!< riemann solver function pointer */

  Timer boundaries_timer, godunov_timer;
  
  // methods
  real_t compute_dt(int useU);
  
  void godunov_unsplit(int nStep, real_t dt);
  
  void godunov_unsplit_cpu(DataArray data_in, 
			   DataArray data_out, 
			   real_t dt, 
			   int nStep);
  
  void convertToPrimitives(DataArray Udata);
  
  void computeTrace(DataArray Udata, real_t dt);
  
  void computeFluxesAndUpdate(DataArray Udata, 
			      real_t dt);
  
  void make_boundaries(DataArray Udata);

  // host routines (initialization)
  void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);

  // host routines (save data to file, device data are copied into host
  // inside this routine)
  void saveVTK(DataArray Udata, int iStep, std::string name);
  
  int isize, jsize, ijsize;
  
}; // class HydroRun

#endif // HYDRO_RUN_H_
