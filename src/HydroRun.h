/**
 *
 */
#ifndef HYDRO_RUN_H_
#define HYDRO_RUN_H_

#include "Timer.h"
#include "HydroParams.h"
#include "kokkos_shared.h"
// the actual computational functors called in HydroRun
#include "HydroRunFunctors.h"

#include <string>
#include <cstdio>
#include <cstdbool>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

#ifdef USE_HDF5
#  include <hdf5.h>
#endif

namespace euler2d
{

inline bool
isBigEndian()
{
  const int i = 1;
  return ((*(char *)&i) == 0);
}

// =======================================================
// =======================================================
/**
 * Main hydrodynamics data structure.
 */
template <typename device_t>
class HydroRun
{

public:
  using DataArray_t = DataArray<device_t>;
  using DataArrayHost_t = DataArrayHost<device_t>;

  HydroRun(HydroParams & params, ConfigMap & configMap);
  virtual ~HydroRun() = default;

  // hydroParams
  HydroParams & params;
  ConfigMap &   configMap;

  DataArray_t     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost_t Uhost; /*!< U mirror on host memory space */
  DataArray_t     U2;    /*!< hydrodynamics conservative variables arrays */
  DataArray_t     Q;     /*!< hydrodynamics primitive    variables array  */

  /* implementation 0 */
  DataArray_t Fluxes_x; /*!< implementation 0 */
  DataArray_t Fluxes_y; /*!< implementation 0 */

  /* implementation 1 only */
  DataArray_t Slopes_x; /*!< implementation 1 only */
  DataArray_t Slopes_y; /*!< implementation 1 only */

  // riemann_solver_t riemann_solver_fn; /*!< riemann solver function pointer */

  Timer boundaries_timer, godunov_timer;

  // methods
  real_t
  compute_dt(int useU);

  void
  godunov_unsplit(int nStep, real_t dt);

  void
  godunov_unsplit_impl(DataArray_t data_in, DataArray_t data_out, real_t dt, int nStep);

  void
  convertToPrimitives(DataArray_t Udata);

  void
  computeTrace(DataArray_t Udata, real_t dt);

  void
  computeFluxesAndUpdate(DataArray_t Udata, real_t dt);

  void
  make_boundaries(DataArray_t Udata);

  // host routines (initialization)
  void
  init_implode(DataArray_t Udata);

  void
  init_blast(DataArray_t Udata);

  void
  init_four_quadrant(DataArray_t Udata);

  void
  init_discontinuity(DataArray_t Udata);

  void
  init_shocked_bubble(DataArray_t Udata);

  // host routines (save data to file, device data are copied into host
  // inside this routine)
  void
  saveData(DataArray_t Udata, int iStep, std::string name);

  void
  saveVTK(DataArray_t Udata, int iStep, std::string name);

#ifdef USE_HDF5
  void
  saveHDF5(DataArray_t Udata, int iStep, std::string name);

  /**
   *  Write a wrapper file using the Xmdf file format (XML) to allow
   * Paraview/Visit to read these h5 files as a time series.
   */
  void
  write_xdmf_time_series(int last_time_step);
#endif
}; // class HydroRun


// =======================================================
// =======================================================
/**
 *
 */
template <typename device_t>
HydroRun<device_t>::HydroRun(HydroParams & params, ConfigMap & configMap)
  : params(params)
  , configMap(configMap)
  , U(Kokkos::view_alloc(Kokkos::WithoutInitializing, "U"), params.isize, params.jsize)
  , Uhost(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, U))
  , U2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "U2"), params.isize, params.jsize)
  , Q(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Q"), params.isize, params.jsize)
  , Fluxes_x()
  , Fluxes_y()
  , Slopes_x()
  , Slopes_y()
{

  const int isize = params.isize;
  const int jsize = params.jsize;

  /*
   * memory allocation (use sizes with ghosts included)
   */
  if (params.implementationVersion == 0)
  {
    Fluxes_x = DataArray_t("Fluxes_x", isize, jsize);
    Fluxes_y = DataArray_t("Fluxes_y", isize, jsize);
  }
  else if (params.implementationVersion == 1)
  {
    Slopes_x = DataArray_t("Slope_x", isize, jsize);
    Slopes_y = DataArray_t("Slope_y", isize, jsize);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray_t("Fluxes_x", isize, jsize);
    Fluxes_y = Fluxes_x;
  }

  // default riemann solver
  // riemann_solver_fn = &HydroRun<device_t>::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &HydroRun<device_t>::riemann_hllc;

  /*
   * initialize hydro array at t=0
   */
  if (params.problemType == PROBLEM_IMPLODE)
  {
    init_implode(U);
  }
  else if (params.problemType == PROBLEM_BLAST)
  {
    init_blast(U);
  }
  else if (params.problemType == PROBLEM_FOUR_QUADRANT)
  {
    init_four_quadrant(U);
  }
  else if (params.problemType == PROBLEM_DISCONTINUITY)
  {
    init_discontinuity(U);
  }
  else if (params.problemType == PROBLEM_SHOCKED_BUBBLE)
  {
    init_shocked_bubble(U);
  }
  else
  {
    std::cout << "Problem : " << params.problemType
              << " is not recognized / implemented in initHydroRun." << std::endl;
    std::cout << "Use default - implode" << std::endl;
    init_implode(U);
  }

  // copy U into U2
  Kokkos::deep_copy(U2, U);

} // HydroRun<device_t>::HydroRun

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \param[in] useU integer, if 0 use data in U else use U2
 *
 * \return dt time step
 */
template <typename device_t>
real_t
HydroRun<device_t>::compute_dt(int useU)
{

  real_t      dt;
  real_t      invDt = ZERO_F;
  DataArray_t Udata;

  // which array is the current one ?
  if (useU == 0)
    Udata = U;
  else
    Udata = U2;

  Kokkos::Profiling::pushRegion("compute_dt");
  // call device functor
  ComputeDtFunctor<device_t>::apply(params, Udata, invDt);

  dt = params.settings.cfl / invDt;
  Kokkos::Profiling::popRegion();

  return dt;

} // HydroRun<device_t>::compute_dt

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template <typename device_t>
void
HydroRun<device_t>::godunov_unsplit(int nStep, real_t dt)
{

  if (nStep % 2 == 0)
  {
    godunov_unsplit_impl(U, U2, dt, nStep);
  }
  else
  {
    godunov_unsplit_impl(U2, U, dt, nStep);
  }

} // HydroRun<device_t>::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
template <typename device_t>
void
HydroRun<device_t>::godunov_unsplit_impl(DataArray_t data_in,
                                         DataArray_t data_out,
                                         real_t      dt,
                                         int         nStep)
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

  if (params.implementationVersion == 0)
  {

    Kokkos::Profiling::pushRegion("hydro_impl0");
    // compute fluxes
    ComputeAndStoreFluxesFunctor<device_t>::apply(params, Q, Fluxes_x, Fluxes_y, dtdx, dtdy);

    // actual update
    UpdateFunctor<device_t>::apply(params, data_out, Fluxes_x, Fluxes_y);
    Kokkos::Profiling::popRegion();
  }
  else if (params.implementationVersion == 1)
  {

    Kokkos::Profiling::pushRegion("hydro_impl1");

    // call device functor to compute slopes
    ComputeSlopesFunctor<device_t>::apply(params, Q, Slopes_x, Slopes_y);

    // now trace along X axis
    ComputeTraceAndFluxes_Functor<device_t, XDIR>::apply(
      params, Q, Slopes_x, Slopes_y, Fluxes_x, dtdx, dtdy);

    // and update along X axis
    UpdateDirFunctor<device_t, XDIR>::apply(params, data_out, Fluxes_x);

    // now trace along Y axis
    ComputeTraceAndFluxes_Functor<device_t, YDIR>::apply(
      params, Q, Slopes_x, Slopes_y, Fluxes_y, dtdx, dtdy);

    // and update along Y axis
    UpdateDirFunctor<device_t, YDIR>::apply(params, data_out, Fluxes_y);
    Kokkos::Profiling::popRegion();
  }
  else if (params.implementationVersion == 2)
  {
    Kokkos::Profiling::pushRegion("hydro_impl2");
    // compute fluxes and update
    ComputeFluxesAndUpdateFunctor<device_t>::apply(params, Q, data_out, dtdx, dtdy);
  }

  godunov_timer.stop();

} // HydroRun<device_t>::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
template <typename device_t>
void
HydroRun<device_t>::convertToPrimitives(DataArray_t Udata)
{
  // call device functor
  ConvertToPrimitivesFunctor<device_t>::apply(params, Udata, Q);

} // HydroRun<device_t>::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template <typename device_t>
void
HydroRun<device_t>::make_boundaries(DataArray_t Udata)
{

  // call device functors
  MakeBoundariesFunctor<device_t, FACE_XMIN>::apply(params, Udata);
  MakeBoundariesFunctor<device_t, FACE_XMAX>::apply(params, Udata);
  MakeBoundariesFunctor<device_t, FACE_YMIN>::apply(params, Udata);
  MakeBoundariesFunctor<device_t, FACE_YMAX>::apply(params, Udata);

} // HydroRun<device_t>::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
template <typename device_t>
void
HydroRun<device_t>::init_implode(DataArray_t Udata)
{

  InitImplodeFunctor<device_t>::apply(params, Udata);

} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template <typename device_t>
void
HydroRun<device_t>::init_blast(DataArray_t Udata)
{

  InitBlastFunctor<device_t>::apply(params, Udata);

} // HydroRun<device_t>::init_blast

// =======================================================
// =======================================================
/**
 * Hydrodynamical discontinuity Test.
 *
 */
template <typename device_t>
void
HydroRun<device_t>::init_discontinuity(DataArray_t Udata)
{

  InitDiscontinuityFunctor<device_t>::apply(params, Udata);

} // HydroRun<device_t>::init_discontinuity

// =======================================================
// =======================================================
/**
 * Shocked bubble.
 *
 */
template <typename device_t>
void
HydroRun<device_t>::init_shocked_bubble(DataArray_t Udata)
{

  InitShockedBubbleFunctor<device_t>::apply(params, Udata);

} // HydroRun<device_t>::init_shocked_bubble

// =======================================================
// =======================================================
/**
 * Hydrodynamical four quadrant Test.
 *
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * Here only problem #3 is implemented. See https://github.com/pkestene/euler_kokkos for complete
 * list of all 19 initial conditions.
 */
template <typename device_t>
void
HydroRun<device_t>::init_four_quadrant(DataArray_t Udata)
{

  InitFourQuadrantFunctor<device_t>::apply(params, Udata);

} // HydroRun<device_t>::init_four_quadrant

// =======================================================
// =======================================================
template <typename device_t>
void
HydroRun<device_t>::saveData(DataArray_t Udata, int iStep, std::string name)
{
#ifdef USE_HDF5
  if (params.ioHDF5)
    saveHDF5(Udata, iStep, name);
#endif

  if (params.ioVTK)
    saveVTK(Udata, iStep, name);
} // HydroRun<device_t>::saveData

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
template <typename device_t>
void
HydroRun<device_t>::saveVTK(DataArray_t Udata, int iStep, std::string name)
{

  const int ijsize = params.isize * params.jsize;
  const int isize = params.isize;
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
  int         i, j, iVar;
  std::string outputDir = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double))
  {
    useDouble = true;
  }

  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;

  // concatenate file prefix + file number + suffix
  std::string filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vti";

  // open file
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);

  // write header
  outFile << "<?xml version=\"1.0\"?>\n";
  if (isBigEndian())
  {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  }
  else
  {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\"" << 0 << " " << nx << " " << 0 << " " << ny << " " << 0
          << " " << 0 << "\" "
          << "Origin=\"" << params.xmin << " " << params.ymin << " " << 0.0 << "\" "
          << "Spacing=\"" << params.dx << " " << params.dy << " " << 0.0 << "\">\n";
  outFile << "  <Piece Extent=\"" << 0 << " " << nx << " " << 0 << " " << ny << " " << 0 << " " << 0
          << " "
          << "\">\n";

  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";
  outFile << "    <CellData>\n";

  // write data array (ascii), remove ghost cells
  for (iVar = 0; iVar < NBVAR; iVar++)
  {
    outFile << "    <DataArray type=\"";
    if (useDouble)
      outFile << "Float64";
    else
      outFile << "Float32";
    outFile << "\" Name=\"" << varNames[iVar] << "\" format=\"ascii\" >\n";

    for (int index = 0; index < ijsize; ++index)
    {
      // index2coord(index,i,j,isize,jsize);

      // enforce the use of left layout (Ok for CUDA)
      // but for OpenMP, we will need to transpose
      j = index / isize;
      i = index - j * isize;

      if (j >= jmin + ghostWidth and j <= jmax - ghostWidth and i >= imin + ghostWidth and
          i <= imax - ghostWidth)
      {
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

} // HydroRun<device_t>::saveVTK

#ifdef USE_HDF5
// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////
// Write as HDF5 format.
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
template <typename device_t>
void
HydroRun<device_t>::saveHDF5(DataArray_t Udata, int iStep, std::string name)
{

  const int ijsize = params.isize * params.jsize;
  const int isize = params.isize;
  const int nx = params.nx;
  const int ny = params.ny;
  const int xysize = nx * ny;
  const int imin = params.imin;
  const int imax = params.imax;
  const int jmin = params.jmin;
  const int jmax = params.jmax;
  const int ghostWidth = params.ghostWidth;

  hid_t       file, dataset;       /* file and dataset handles */
  hid_t       datatype, dataspace; /* handles */
  hsize_t     dimsf[2];            /* dataset dimensions */
  H5T_order_t order;               /* little endian or big endian */

  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);

  // local variables
  std::string outputDir = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  // check scalar data type

  if (sizeof(real_t) == sizeof(double))
  {
    datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
  }
  else
  {
    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
  }

  if (isBigEndian())
  {
    order = H5T_ORDER_BE;
  }
  else
  {
    order = H5T_ORDER_LE;
  }

  real_t * data_host = new real_t[xysize];

  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;

  // concatenate file prefix + file number + suffix
  std::string filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".h5";

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

  // now we write data
  for (int ivar = 0; ivar < NBVAR; ++ivar)
  {

    // copy Uhost into data_host by removing ghost border
    int id = 0;
    for (int index = 0; index < ijsize; ++index)
    {
      int j = index / isize;
      int i = index - j * isize;

      if (j >= jmin + ghostWidth and j <= jmax - ghostWidth and i >= imin + ghostWidth and
          i <= imax - ghostWidth)
      {
        data_host[id] = Uhost(i, j, ivar);
        id += 1;
      }
    } // end for index

    /*
     * Create a new dataset within the file using defined dataspace and
     * datatype and default dataset creation properties.
     */
    std::string varName = "/" + std::string(varNames[ivar]);
    dataset = H5Dcreate2(file, varName.c_str(), datatype, dataspace, 0, 0, H5P_DEFAULT);
    H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_host);
  } // end for ivar

  delete[] data_host;

  /*
   * Close/release resources.
   */
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Dclose(dataset);
  H5Fclose(file);

} // HydroRun<device_t>::saveHDF5

// =======================================================
// =======================================================
/*
 * write xdmf file to provide metadata of H5 file
 * can be opened by ParaView
 * point to data file : xdmf2d.h5
 */
template <typename device_t>
void
HydroRun<device_t>::write_xdmf_time_series(int last_time_step)
{
  FILE *      xdmf = 0;
  const int & nx = params.nx;
  const int & ny = params.ny;

  // get data type as a string for Xdmf
  std::string dataTypeName = sizeof(real_t) == sizeof(double) ? "Double" : "Float";

  /*
   * Open the file and write the XML description of the mesh..
   */
  std::string outputDir = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
  std::string xdmfFilename = outputPrefix + ".xmf";

  xdmf = fopen(xdmfFilename.c_str(), "w");

  fprintf(xdmf, "<?xml version=\"1.0\" ?>\n");
  fprintf(xdmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
  fprintf(xdmf, "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n");
  fprintf(xdmf, "  <Domain>\n");
  fprintf(xdmf,
          "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n");

  // for each time step write a <grid> </grid> item
  const int nbOutput = (params.nStepmax + params.nOutput - 1) / params.nOutput + 1;
  for (int iOut = 0; iOut < nbOutput; ++iOut)
  {
    int iStep = iOut * params.nOutput;

    if (iOut == (nbOutput - 1))
    {
      iStep = last_time_step;
    }

    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << iStep;

    // take care that the following filename must be exactly the same as in routine outputHdf5 !!!
    std::string baseName = outputPrefix + "_" + outNum.str();
    std::string hdf5Filename = outputPrefix + "_" + outNum.str() + ".h5";
    std::string hdf5FilenameFull = outputDir + "/" + outputPrefix + "_" + outNum.str() + ".h5";

    fprintf(xdmf, "      <Grid Name=\"%s\" GridType=\"Uniform\">\n", baseName.c_str());
    fprintf(xdmf, "      <Time Value=\"%d\" />\n", iStep);

    // topology = CoRectMesh
    fprintf(xdmf,
            "        <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\"%d %d\"/>\n",
            ny,
            nx);

    // geometry
    fprintf(xdmf, "        <Geometry Type=\"ORIGIN_DXDY\">\n");
    fprintf(xdmf, "          <DataStructure\n");
    fprintf(xdmf, "            Name=\"Origin\"\n");
    fprintf(xdmf, "            DataType=\"%s\"\n", dataTypeName.c_str());
    fprintf(xdmf, "            Dimensions=\"2\"\n");
    fprintf(xdmf, "            Format=\"XML\">\n");
    fprintf(xdmf, "            0 0\n");
    fprintf(xdmf, "          </DataStructure>\n");
    fprintf(xdmf, "          <DataStructure\n");
    fprintf(xdmf, "            Name=\"Spacing\"\n");
    fprintf(xdmf, "            DataType=\"%s\"\n", dataTypeName.c_str());
    fprintf(xdmf, "            Dimensions=\"2\"\n");
    fprintf(xdmf, "            Format=\"XML\">\n");
    fprintf(xdmf, "            1 1\n");
    fprintf(xdmf, "          </DataStructure>\n");
    fprintf(xdmf, "        </Geometry>\n");


    // save all scalar field
    for (int iVar = 0; iVar < NBVAR; iVar++)
    {
      fprintf(xdmf, "      <Attribute Center=\"Node\" Name=\"%s\">\n", varNames[iVar]);
      fprintf(xdmf, "        <DataStructure\n");
      fprintf(xdmf, "           DataType=\"%s\"\n", dataTypeName.c_str());
      fprintf(xdmf, "           Dimensions=\"%d %d\"\n", ny, nx);
      fprintf(xdmf, "           Format=\"HDF\">\n");
      fprintf(xdmf, "           %s:/%s\n", hdf5Filename.c_str(), varNames[iVar]);
      fprintf(xdmf, "        </DataStructure>\n");
      fprintf(xdmf, "      </Attribute>\n");
    }

    // finalize grid file for the current time step
    fprintf(xdmf, "    </Grid>\n");

  } // end for iStep

  fprintf(xdmf, "    </Grid>\n");
  fprintf(xdmf, "  </Domain>\n");
  fprintf(xdmf, "</Xdmf>\n");
  fclose(xdmf);

} // HydroRun<device_t>::write_xdmf_xml
#endif // USE_HDF5

} // namespace euler2d

#endif // HYDRO_RUN_H_
