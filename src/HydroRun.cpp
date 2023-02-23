#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "HydroRun.h"
#include "HydroParams.h"
#include "Timer.h"
#include "hdf5.h"
// the actual computational functors called in HydroRun
#include "HydroRunFunctors.h"

// Kokkos
#include "kokkos_shared.h"

namespace euler2d {

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

  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", isize, jsize);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",isize, jsize);
  Q     = DataArray("Q", isize, jsize);

  if (params.implementationVersion == 0) {

    Fluxes_x = DataArray("Fluxes_x", isize, jsize);
    Fluxes_y = DataArray("Fluxes_y", isize, jsize);

  } else if (params.implementationVersion == 1) {

    Slopes_x = DataArray("Slope_x", isize, jsize);
    Slopes_y = DataArray("Slope_y", isize, jsize);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", isize, jsize);
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

  Kokkos::Profiling::pushRegion("compute_dt");
  // call device functor
  ComputeDtFunctor::apply(params, Udata, invDt);

  dt = params.settings.cfl/invDt;
  Kokkos::Profiling::popRegion();

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
  Kokkos::Profiling::pushRegion("make_boundaries");
  boundaries_timer.start();
  make_boundaries(data_in);
  boundaries_timer.stop();
  Kokkos::Profiling::popRegion();

  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);

  // start main computation
  godunov_timer.start();

  // convert conservative variable into primitives ones for the entire domain
  Kokkos::Profiling::pushRegion("compute_primitives");
  convertToPrimitives(data_in);
  Kokkos::Profiling::popRegion();

  if (params.implementationVersion == 0) {

    Kokkos::Profiling::pushRegion("hydro_impl0");
    // compute fluxes
    ComputeAndStoreFluxesFunctor::apply(params, Q,
					Fluxes_x, Fluxes_y,
					dtdx, dtdy);

    // actual update
    UpdateFunctor::apply(params, data_out,
			 Fluxes_x, Fluxes_y);
    Kokkos::Profiling::popRegion();

  } else if (params.implementationVersion == 1) {

    Kokkos::Profiling::pushRegion("hydro_impl1");

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
    Kokkos::Profiling::popRegion();

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

void HydroRun::saveData(DataArray Udata,
		       int iStep,
		       std::string name)
{
  if (params.ioHDF5) 
    saveHDF5(Udata, iStep, name);
  else
    saveVTK(Udata, iStep, name);
}

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

  const int ijsize = params.isize*params.jsize;
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

    for (int index=0; index<ijsize; ++index) {
      //index2coord(index,i,j,isize,jsize);

      // enforce the use of left layout (Ok for CUDA)
      // but for OpenMP, we will need to transpose
      j = index / isize;
      i = index - j*isize;

      if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	  i>=imin+ghostWidth and i<=imax-ghostWidth) {
    	outFile << Uhost(i, j, iVar) << " ";
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

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////
// Write as HDF5 format.
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void HydroRun::saveHDF5(DataArray Udata,
		       int iStep,
		       std::string name)
{

  const int ijsize = params.isize*params.jsize;
  const int isize  = params.isize;
  const int nx = params.nx;
  const int ny = params.ny;
  const int xysize = nx*ny;
  const int imin = params.imin;
  const int imax = params.imax;
  const int jmin = params.jmin;
  const int jmax = params.jmax;
  const int ghostWidth = params.ghostWidth;
  int i,j,id=0;

  hid_t file, dataset;       /* file and dataset handles */
  hid_t datatype, dataspace; /* handles */
  hsize_t dimsf[2];          /* dataset dimensions */
  H5T_order_t order;         /* little endian or big endian */
  real_t* E_host = new real_t[xysize];
  real_t* mx_host = new real_t[xysize];
  real_t* my_host = new real_t[xysize];
  real_t* rho_host = new real_t[xysize];

  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);

  for (int index=0; index<ijsize; ++index) {
    j = index / isize;
    i = index - j*isize;

    if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	  i>=imin+ghostWidth and i<=imax-ghostWidth) {
      rho_host[id] = Uhost(i,j,0);
      E_host[id] = Uhost(i,j,1);
      mx_host[id] = Uhost(i,j,2);
      my_host[id] = Uhost(i,j,3);
      id+=1;
    }
  }

  // local variables
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  // check scalar data type

  if (sizeof(real_t) == sizeof(double)) {
    datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
  } else {
    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
  }

  if (isBigEndian()) {
    order = H5T_ORDER_BE;
  } else {
    order = H5T_ORDER_LE;
  }

  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;

  // concatenate file prefix + file number + suffix
  std::string filename     = outputDir + "/" + outputPrefix+"_"+stepNum.str() + ".h5";

  /*
   * Create a new file using H5F_ACC_TRUNC access,
   * default file creation properties, and default file
   * access properties.
   */
  file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  /*
   * Describe the size of the array and create the data space for fixed
   * size dataset.
   */
  dimsf[0] = ny;
  dimsf[1] = nx;
  dataspace = H5Screate_simple(2, dimsf, NULL);

  /*
   * Define datatype for the data in the file.
   * We will store little endian numbers.
   */
  H5Tset_order(datatype, order);

  /*
   * Create a new dataset within the file using defined dataspace and
   * datatype and default dataset creation properties.
   */
  dataset = H5Dcreate2(file, "/rho", datatype, dataspace, 0, 0, H5P_DEFAULT);
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho_host);

  dataset = H5Dcreate2(file, "/E", datatype, dataspace, 0, 0, H5P_DEFAULT);
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, E_host);

  dataset = H5Dcreate2(file, "/mx", datatype, dataspace, 0, 0, H5P_DEFAULT);
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mx_host);

  dataset = H5Dcreate2(file, "/my", datatype, dataspace, 0, 0, H5P_DEFAULT);
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, my_host);

  delete []E_host;
  delete []mx_host;
  delete []my_host;
  delete []rho_host;
  /*
   * Close/release resources.
   */
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Dclose(dataset);
  H5Fclose(file);

} // HydroRun::saveHDF5

/* 
 * write xdmf file to provide metadata of H5 file
 * can be opened by ParaView
 * point to data file : xdmf2d.h5
 */
void HydroRun::write_xdmf_xml()
{
    FILE *xmf = 0;
    const int nx = params.nx;
    const int ny = params.ny;
    const char* type;
    int precision;

    if (sizeof(real_t) == sizeof(double)) {
      type = "Double";
      precision = 8;
    } else {
      type = "Float";
      precision = 4;
    }
    /*
     * Open the file and write the XML description of the mesh..
     */
    xmf = fopen("xdmf2d.xmf", "w");
    fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf(xmf, " <Domain>\n");
    fprintf(xmf, "   <Grid Name=\"mesh1\" GridType=\"Uniform\">\n");
    fprintf(xmf, "     <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"%d %d 1\"/>\n", ny+1, nx+1);
    fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(xmf, "       <DataItem Name=\"origin\" Dimensions=\"3\" NumberType=\"%s\" Precision=\"%d\" Format=\"XML\">\n", type, precision);
    fprintf(xmf, "        0.0 %f %f\n", params.xmin, params.ymin);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "       <DataItem Name=\"spacing\" Dimensions=\"3\" NumberType=\"%s\" Precision=\"%d\" Format=\"XML\">\n", type, precision);
    fprintf(xmf, "        0.0 %f %f\n", params.dx, params.dy);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Geometry>\n");
    fprintf(xmf, "     <Attribute Name=\"E\" AttributeType=\"Scalar\" Center=\"Cell\">\n");
    fprintf(xmf, "       <DataItem Dimensions=\"%d %d\" NumberType=\"%s\" Precision=\"%d\" Format=\"HDF\">\n", ny, nx, type, precision);
    fprintf(xmf, "        xdmf2d.h5:/E\n");
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Attribute>\n");
    fprintf(xmf, "     <Attribute Name=\"mx\" AttributeType=\"Scalar\" Center=\"Cell\">\n");
    fprintf(xmf, "       <DataItem Dimensions=\"%d %d\" NumberType=\"%s\" Precision=\"%d\" Format=\"HDF\">\n", ny, nx, type, precision);
    fprintf(xmf, "        xdmf2d.h5:/mx\n");
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Attribute>\n");
    fprintf(xmf, "     <Attribute Name=\"my\" AttributeType=\"Scalar\" Center=\"Cell\">\n");
    fprintf(xmf, "       <DataItem Dimensions=\"%d %d\" NumberType=\"%s\" Precision=\"%d\" Format=\"HDF\">\n", ny, nx, type, precision);
    fprintf(xmf, "        xdmf2d.h5:/my\n");
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Attribute>\n");
    fprintf(xmf, "     <Attribute Name=\"rho\" AttributeType=\"Scalar\" Center=\"Cell\">\n");
    fprintf(xmf, "       <DataItem Dimensions=\"%d %d\" NumberType=\"%s\" Precision=\"%d\" Format=\"HDF\">\n", ny, nx, type, precision);
    fprintf(xmf, "        xdmf2d.h5:/rho\n");
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Attribute>\n");
    fprintf(xmf, "   </Grid>\n");
    fprintf(xmf, " </Domain>\n");
    fprintf(xmf, "</Xdmf>\n");
    fclose(xmf);
}

} // namespace euler2d
