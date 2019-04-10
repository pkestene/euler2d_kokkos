#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "HydroRun.h"
#include "HydroParams.h"
#include "Timer.h"

// the actual computational functors called in HydroRun
#include "HydroRunFunctors.h"

// Kokkos
#include "kokkos_shared.h"

static bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}


// =======================================================
// =======================================================
/**
 *
 */
HydroRun::HydroRun(HydroParams& params, ConfigMap& configMap) :
  params(params),
  configMap(configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(),
  Slopes_x(), Slopes_y()
{

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ijsize = params.ijsize;

  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", ijsize);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",ijsize);
  Q     = DataArray("Q", ijsize);

  if (params.implementationVersion == 0) {

    Fluxes_x = DataArray("Fluxes_x", ijsize);
    Fluxes_y = DataArray("Fluxes_y", ijsize);
    
  } else if (params.implementationVersion == 1) {

    Slopes_x = DataArray("Slope_x", ijsize);
    Slopes_y = DataArray("Slope_y", ijsize);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", ijsize);
    Fluxes_y = Fluxes_x;
    
  } 
  
  // default riemann solver
  // riemann_solver_fn = &HydroRun::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &HydroRun::riemann_hllc;
  
  /*
   * initialize hydro array at t=0
   */
  if ( params.problemType == PROBLEM_IMPLODE) {

    init_implode(U);

  } else if (params.problemType == PROBLEM_BLAST) {

    init_blast(U);

  } else {

    std::cout << "Problem : " << params.problemType
	      << " is not recognized / implemented in initHydroRun."
	      << std::endl;
    std::cout <<  "Use default - implode" << std::endl;
    init_implode(U);

  }

  // copy U into U2
  Kokkos::deep_copy(U2,U);

} // HydroRun::HydroRun


// =======================================================
// =======================================================
/**
 *
 */
HydroRun::~HydroRun()
{

} // HydroRun::~HydroRun

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \param[in] useU integer, if 0 use data in U else use U2
 *
 * \return dt time step
 */
real_t HydroRun::compute_dt(int useU)
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  // which array is the current one ?
  if (useU == 0)
    Udata = U;
  else
    Udata = U2;

  // call device functor
  ComputeDtFunctor::apply(params, Udata, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // HydroRun::compute_dt

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void HydroRun::godunov_unsplit(int nStep, real_t dt)
{
  
  if ( nStep % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt, nStep);
  } else {
    godunov_unsplit_cpu(U2, U , dt, nStep);
  }
  
} // HydroRun::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
void HydroRun::godunov_unsplit_cpu(DataArray data_in, 
				   DataArray data_out, 
				   real_t dt, 
				   int nStep)
{

  real_t dtdx;
  real_t dtdy;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

  // fill ghost cell in data_in
  boundaries_timer.start();
  make_boundaries(data_in);
  boundaries_timer.stop();
    
  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  godunov_timer.start();

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  if (params.implementationVersion == 0) {
    
    // compute fluxes
    ComputeAndStoreFluxesFunctor::apply(params, Q,
					Fluxes_x, Fluxes_y,
					dtdx, dtdy);

    // actual update
    UpdateFunctor::apply(params, data_out,
			 Fluxes_x, Fluxes_y);
    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor::apply(params, Q, Slopes_x, Slopes_y);

    // now trace along X axis
    ComputeTraceAndFluxes_Functor<XDIR>::apply(params, Q,
					       Slopes_x, Slopes_y,
					       Fluxes_x,
					       dtdx, dtdy);
    
    // and update along X axis
    UpdateDirFunctor<XDIR>::apply(params, data_out, Fluxes_x);
    
    // now trace along Y axis
    ComputeTraceAndFluxes_Functor<YDIR>::apply(params, Q,
					       Slopes_x, Slopes_y,
					       Fluxes_y,
					       dtdx, dtdy);
    
    // and update along Y axis
    UpdateDirFunctor<YDIR>::apply(params, data_out, Fluxes_y);
    
  } // end params.implementationVersion == 1
  
  godunov_timer.stop();
  
} // HydroRun::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void HydroRun::convertToPrimitives(DataArray Udata)
{
  // call device functor
  ConvertToPrimitivesFunctor::apply(params, Udata, Q);
  
} // HydroRun::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void HydroRun::make_boundaries(DataArray Udata)
{
  const int ghostWidth=params.ghostWidth;
  const int isize = params.isize;
  const int jsize = params.jsize;

  // call device functors
  MakeBoundariesFunctor<FACE_XMIN>::apply(params, Udata);
  MakeBoundariesFunctor<FACE_XMAX>::apply(params, Udata);
  MakeBoundariesFunctor<FACE_YMIN>::apply(params, Udata);
  MakeBoundariesFunctor<FACE_YMAX>::apply(params, Udata);
  
} // HydroRun::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void HydroRun::init_implode(DataArray Udata)
{
  
  InitImplodeFunctor::apply(params, Udata);
  
} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
void HydroRun::init_blast(DataArray Udata)
{
  
  InitBlastFunctor::apply(params, Udata);

} // HydroRun::init_blast

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void HydroRun::saveVTK(DataArray Udata,
		       int iStep,
		       std::string name)
{

  const int isize  = params.isize;
  const int nx = params.nx;
  const int ny = params.ny;
  const int imin = params.imin;
  const int imax = params.imax;
  const int jmin = params.jmin;
  const int jmax = params.jmax;
  const int ghostWidth = params.ghostWidth;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;
  
  // concatenate file prefix + file number + suffix
  std::string filename     = outputDir + "/" + outputPrefix+"_"+stepNum.str() + ".vti";
  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header
  outFile << "<?xml version=\"1.0\"?>\n";
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 0  << "\" "
	  << "Origin=\""
	  << params.xmin << " " << params.ymin << " " << 0.0 << "\" "
	  << "Spacing=\""
	  << params.dx << " " << params.dy << " " << 0.0 << "\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 0  << " "
	  << "\">\n";
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";
  outFile << "    <CellData>\n";

  // write data array (ascii), remove ghost cells
  for ( iVar=0; iVar<NBVAR; iVar++) {
    outFile << "    <DataArray type=\"";
    if (useDouble)
      outFile << "Float64";
    else
      outFile << "Float32";
    outFile << "\" Name=\"" << varNames[iVar] << "\" format=\"ascii\" >\n";

    for (j=0; j<params.jsize; ++j) {
      for (i=0; i<params.isize; ++i) {      

        int index = coord2index(i,j,params.isize, params.jsize);

        if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
            i>=imin+ghostWidth and i<=imax-ghostWidth) {
          outFile << Uhost(index, iVar) << " ";
        }
      }
    }
    outFile << "\n    </DataArray>\n";
  } // end for iVar

  outFile << "    </CellData>\n";

  // write footer
  outFile << "  </Piece>\n";
  outFile << "  </ImageData>\n";
  outFile << "</VTKFile>\n";
  
  outFile.close();

} // HydroRun::saveVTK

