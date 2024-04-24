/**
 * Hydro2d solver for teaching purpose.
 *
 * \date April, 16 2016
 * \author P. Kestener
 */

#include <cstdio>
#include <cstdlib>

#include "kokkos_shared.h"

#include "HydroBaseFunctor.h"
#include "ComputeRadialProfileFunctor.h"
#include "HydroParams.h" // read parameter file
#include "HydroRun.h"    // memory allocation for hydro arrays
#include "real_type.h"   // choose between single and double precision
#include "Timer.h"       // measure time

int
main(int argc, char * argv[])
{
  using DefaultDevice =
    Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
  using device = DefaultDevice;
  // using device = Kokkos::OpenMP::device_type;
  // using device = Kokkos::CudaSpace::device_type;

  using real_t = euler2d::real_t;

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
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";
  }
  real_t t = 0, dt = 0;
  int    nStep = 0;

  Timer total_timer, io_timer, dt_timer;

  if (argc != 2)
  {
    fprintf(stderr,
            "Error: wrong number of argument; input filename must be "
            "the only parameter on the command line\n");
    exit(EXIT_FAILURE);
  }

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string(argv[1]);
  ConfigMap   configMap(input_file);

  // test: create a HydroParams object
  euler2d::HydroParams params = euler2d::HydroParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  // initialize workspace memory (U, U2, ...)
  euler2d::HydroRun<device> * hydro = new euler2d::HydroRun<device>(params, configMap);
  dt = hydro->compute_dt(nStep % 2);

  // initialize boundaries
  hydro->make_boundaries(hydro->U);
  hydro->make_boundaries(hydro->U2);

  Kokkos::Profiling::pushRegion("main_loop");

  // start computation
  std::cout << "Start computation....\n";
  total_timer.start();

  // Hydrodynamics solver loop
  while (t < params.tEnd && nStep < params.nStepmax)
  {

    if (nStep % 10 == 0)
    {
      std::cout << "time step=" << nStep << std::endl;
    }

    // output
    if (params.enableOutput)
    {
      Kokkos::Profiling::pushRegion("output");
      if (params.nOutput > 0 and nStep % params.nOutput == 0)
      {
        std::cout << "Output results at time t=" << t << " step " << nStep << " dt=" << dt
                  << std::endl;
        io_timer.start();
        if (nStep % 2 == 0)
          hydro->saveData(hydro->U, nStep, "U");
        else
          hydro->saveData(hydro->U2, nStep, "U");
        io_timer.stop();
      } // end output
      Kokkos::Profiling::popRegion();
    } // end enable output

    // compute new dt
    dt_timer.start();
    dt = hydro->compute_dt(nStep % 2);

    // correct dt if necessary
    if (t + dt > params.tEnd)
    {
      dt = params.tEnd - t;
    }

    dt_timer.stop();

    // perform one step integration
    hydro->godunov_unsplit(nStep, dt);

    // increase time
    nStep++;
    t += dt;

  } // end solver loop

  // save last time step
  if (params.enableOutput)
  {
    if (params.nOutput > 0)
    {
      std::cout << "Output results at time t=" << t << " step " << nStep << " dt=" << dt
                << std::endl;
      io_timer.start();
      if (nStep % 2 == 0)
        hydro->saveData(hydro->U, nStep, "U");
      else
        hydro->saveData(hydro->U2, nStep, "U");
      io_timer.stop();
    } // end output
  } // end enable output

  // end of computation
  total_timer.stop();
  Kokkos::Profiling::popRegion();

  // write XDMF wrapper
#ifdef USE_HDF5
  printf("Last time step is : %d\n", nStep);

  if (params.nOutput > 0 and params.ioHDF5)
    hydro->write_xdmf_time_series(nStep);
#endif

  // post-processing for Sedov blast
  if (params.problemType == euler2d::PROBLEM_BLAST and params.blast_total_energy_inside > 0)
  {
    euler2d::ComputeRadialProfileFunctor<device>::apply(params, hydro->U);
  }

  // print monitoring information
  {
    int isize = params.isize;
    int jsize = params.jsize;

    real_t t_tot = total_timer.elapsed();
    real_t t_comp = hydro->godunov_timer.elapsed();
    real_t t_dt = dt_timer.elapsed();
    real_t t_bound = hydro->boundaries_timer.elapsed();
    real_t t_io = io_timer.elapsed();
    printf("total       time : %5.3f secondes\n", t_tot);
    printf("godunov     time : %5.3f secondes %5.2f%%\n", t_comp, 100 * t_comp / t_tot);
    printf("compute dt  time : %5.3f secondes %5.2f%%\n", t_dt, 100 * t_dt / t_tot);
    printf("boundaries  time : %5.3f secondes %5.2f%%\n", t_bound, 100 * t_bound / t_tot);
    printf("io          time : %5.3f secondes %5.2f%%\n", t_io, 100 * t_io / t_tot);
    printf("Perf             : %10.2f number of Mcell-updates/s\n",
           1.0 * nStep * isize * jsize / t_tot * 1e-6);
  }

  delete hydro;

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // end main
