#include "HydroParams.h"

#include <cstdlib> // for exit
#include <cstdio>  // for fprintf
#include <cstring> // for strcmp
#include <iostream>

#include "config/inih/ini.h" // our INI file reader

const char * varNames[4] = { "rho", "E", "mx", "my" };

/*
 * Hydro Parameters (read parameter file)
 */
void HydroParams::setup(ConfigMap &configMap)
{

  /* initialize RUN parameters */
  nStepmax = configMap.getInteger("run","nstepmax",1000);
  tEnd     = configMap.getFloat  ("run","tend",0.0);
  nOutput  = configMap.getInteger("run","noutput",100);
  if (nOutput == -1)
    enableOutput = false;
  
  /* initialize MESH parameters */
  nx = configMap.getInteger("mesh","nx", 2);
  ny = configMap.getInteger("mesh","ny", 2);

  xmin = configMap.getFloat("mesh", "xmin", 0.0);
  ymin = configMap.getFloat("mesh", "ymin", 0.0);

  xmax = configMap.getFloat("mesh", "xmax", 1.0);
  ymax = configMap.getFloat("mesh", "ymax", 1.0);

  boundary_type_xmin  = static_cast<int>(configMap.getInteger("mesh","boundary_type_xmin", BC_DIRICHLET));
  boundary_type_xmax  = static_cast<int>(configMap.getInteger("mesh","boundary_type_xmax", BC_DIRICHLET));
  boundary_type_ymin  = static_cast<int>(configMap.getInteger("mesh","boundary_type_ymin", BC_DIRICHLET));
  boundary_type_ymax  = static_cast<int>(configMap.getInteger("mesh","boundary_type_ymax", BC_DIRICHLET));

  settings.gamma0         = configMap.getFloat("hydro","gamma0", 1.4);
  settings.cfl            = configMap.getFloat("hydro", "cfl", 0.5);
  settings.iorder         = configMap.getInteger("hydro","iorder", 2);
  settings.slope_type     = configMap.getFloat("hydro","slope_type",1.0);
  settings.smallc         = configMap.getFloat("hydro","smallc", 1e-10);
  settings.smallr         = configMap.getFloat("hydro","smallr", 1e-10);

  niter_riemann  = configMap.getInteger("hydro","niter_riemann", 10);
  std::string riemannSolverStr = std::string(configMap.getString("hydro","riemann", "approx"));
  if ( !riemannSolverStr.compare("approx") ) {
    riemannSolverType = RIEMANN_APPROX;
  } else if ( !riemannSolverStr.compare("hll") ) {
    riemannSolverType = RIEMANN_HLL;
  } else if ( !riemannSolverStr.compare("hllc") ) {
    riemannSolverType = RIEMANN_HLLC;
  } else {
    std::cout << "Riemann Solver specified in parameter file is invalid\n";
    std::cout << "Use the default one : approx\n";
    riemannSolverType = RIEMANN_APPROX;
  }
    
  std::string problemStr = std::string(configMap.getString("hydro","problem", "unknown"));
  if ( !problemStr.compare("implode") ) {
    problemType = PROBLEM_IMPLODE;
  } else if ( !problemStr.compare("blast") ) {
    problemType = PROBLEM_BLAST;
  } else {
    std::cout << "Problem is invalid\n";
    std::cout << "Use the default one : implode\n";
    problemType = PROBLEM_IMPLODE;
  }

  blast_radius   = configMap.getFloat("blast","radius", (xmin+xmax)/2.0/10);
  blast_center_x = configMap.getFloat("blast","center_x", (xmin+xmax)/2);
  blast_center_y = configMap.getFloat("blast","center_y", (ymin+ymax)/2);
  blast_density_in  = configMap.getFloat("blast","density_in", 1.0);
  blast_density_out = configMap.getFloat("blast","density_out", 1.2);
  blast_pressure_in  = configMap.getFloat("blast","pressure_in", 10.0);
  blast_pressure_out = configMap.getFloat("blast","pressure_out", 0.1);

  implementationVersion  = configMap.getFloat("OTHER","implementationVersion", 0);
  if (implementationVersion != 0 and
      implementationVersion != 1) {
    std::cout << "Implementation version is invalid (must be 0 or 1)\n";
    std::cout << "Use the default : 0\n";
    implementationVersion = 0;
  }

  init();

} // HydroParams::setup

// =======================================================
// =======================================================
void HydroParams::init()
{

  // set other parameters
  imax = nx - 1 + 2*ghostWidth;
  jmax = ny - 1 + 2*ghostWidth;
  
  isize = imax - imin + 1;
  jsize = jmax - jmin + 1;
  
  dx = (xmax - xmin) / nx;
  dy = (ymax - ymin) / ny;
  
  settings.smallp  = settings.smallc*settings.smallc/
    settings.gamma0;
  settings.smallpp = settings.smallr*settings.smallp;
  settings.gamma6  = (settings.gamma0 + ONE_F)/(TWO_F * settings.gamma0);
  
  // check that given parameters are valid
  if ( (implementationVersion != 0) && 
       (implementationVersion != 1) && 
       (implementationVersion != 2) ) {
    fprintf(stderr, "The implementation version parameter should 0,1 or 2 !!!");
    fprintf(stderr, "Check your parameter file, section OTHER");
    exit(EXIT_FAILURE);
  } else {
    fprintf(stdout, "Using Euler implementation version %d\n", implementationVersion);
  }

} // HydroParams::init


// =======================================================
// =======================================================
void HydroParams::print()
{
  
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  printf( "nx         : %d\n", nx);
  printf( "ny         : %d\n", ny);
  printf( "dx         : %f\n", dx);
  printf( "dy         : %f\n", dy);
  printf( "imin       : %d\n", imin);
  printf( "imax       : %d\n", imax);
  printf( "jmin       : %d\n", jmin);      
  printf( "jmax       : %d\n", jmax);      
  printf( "nStepmax   : %d\n", nStepmax);
  printf( "tEnd       : %f\n", tEnd);
  printf( "nOutput    : %d\n", nOutput);
  printf( "gamma0     : %f\n", settings.gamma0);
  printf( "cfl        : %f\n", settings.cfl);
  printf( "smallr     : %12.10f\n", settings.smallr);
  printf( "smallc     : %12.10f\n", settings.smallc);
  //printf( "niter_riemann : %d\n", niter_riemann);
  printf( "iorder     : %d\n", settings.iorder);
  printf( "slope_type : %f\n", settings.slope_type);
  printf( "riemann    : %d\n", riemannSolverType);
  printf( "problem    : %d\n", problemType);
  printf( "implementation version : %d\n",implementationVersion);
  printf( "##########################\n");
  
} // HydroParams::print
