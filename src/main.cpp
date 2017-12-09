/**
 * Hydro2d solver for teaching purpose.
 *
 * \date April, 16 2016
 * \author P. Kestener
 */

#include <cstdlib>
#include <cstdio>

#include "kokkos_shared.h"

#include "real_type.h"   // choose between single and double precision
#include "HydroParams.h" // read parameter file
#include "HydroBaseFunctor.h"
#include "HydroRun.h"    // memory allocation for hydro arrays
#include "Timer.h"  // for timer

#ifdef KOKKOS_ENABLE_CUDA
#include "CudaTimer.h"
#endif

int main(int argc, char *argv[])
{

  /*
   * Initialize kokkos (host + device)
   * 
   * If CUDA is enabled, Kokkos will try to use the default GPU, 
   * i.e. GPU #0 if you have multiple GPUs.
   */
  Kokkos::initialize(argc, argv);

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";
  }
  real_t t=0, dt=0;
  int    nStep=0;

  Timer total_timer, io_timer, dt_timer;
  
  if (argc != 2) {
    fprintf(stderr, "Error: wrong number of argument; input filename must be the only parameter on the command line\n");
    exit(EXIT_FAILURE);
  }

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);
  
  // print parameters on screen
  params.print();

  // initialize workspace memory (U, U2, ...)
  HydroRun *hydro = new HydroRun(params, configMap);
  dt = hydro->compute_dt( nStep%2 );

  // initialize boundaries
  hydro->make_boundaries(hydro->U);
  hydro->make_boundaries(hydro->U2);

  // start computation
  std::cout << "Start computation....\n";
  total_timer.start();

  // Hydrodynamics solver loop
  while (t < params.tEnd && nStep < params.nStepmax) {

    if (nStep % 10 == 0) {
      std::cout << "time step=" << nStep << std::endl;
    }

    // output
    if (params.enableOutput) {
      if (nStep % params.nOutput == 0) {
	std::cout << "Output results at time t=" << t << " step " << nStep
		  << " dt=" << dt << std::endl;
	io_timer.start();
	if (nStep % 2 == 0)
	  hydro->saveVTK(hydro->U, nStep, "U");
	else
	  hydro->saveVTK(hydro->U2, nStep, "U");
	io_timer.stop();
      } // end output
    } // end enable output
    
    // compute new dt
    dt_timer.start();
    dt = hydro->compute_dt( nStep%2 );
    dt_timer.stop();
    
    // perform one step integration
    hydro->godunov_unsplit(nStep, dt);

    // increase time
    nStep++;
    t+=dt;

  } // end solver loop

  // end of computation
  total_timer.stop();

  // print monitoring information
  {
    int isize = params.isize;
    int jsize = params.jsize;
    
    real_t t_tot   = total_timer.elapsed();
    real_t t_comp  = hydro->godunov_timer.elapsed();
    real_t t_dt    = dt_timer.elapsed();
    real_t t_bound = hydro->boundaries_timer.elapsed();
    real_t t_io    = io_timer.elapsed();
    printf("total       time : %5.3f secondes\n",t_tot);
    printf("godunov     time : %5.3f secondes %5.2f%%\n",t_comp,100*t_comp/t_tot);
    printf("compute dt  time : %5.3f secondes %5.2f%%\n",t_dt,100*t_dt/t_tot);
    printf("boundaries  time : %5.3f secondes %5.2f%%\n",t_bound,100*t_bound/t_tot);
    printf("io          time : %5.3f secondes %5.2f%%\n",t_io,100*t_io/t_tot);
    printf("Perf             : %10.2f number of Mcell-updates/s\n",nStep*isize*jsize/t_tot*1e-6);
  }
  
  delete hydro;

  Kokkos::finalize();
  
  return EXIT_SUCCESS;

} // end main
