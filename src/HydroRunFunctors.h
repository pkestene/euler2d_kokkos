#ifndef HYDRO_RUN_FUNCTORS_H_
#define HYDRO_RUN_FUNCTORS_H_

#include "HydroBaseFunctor.h"
#include "kokkos_shared.h"

namespace euler2d
{

/*************************************************/
/*************************************************/
/*************************************************/
template<typename device_t>
class ComputeDtFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  ComputeDtFunctor(HydroParams params, DataArray_t Udata)
    : HydroBaseFunctor(params)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata, real_t & invDt)
  {
    ComputeDtFunctor    computeDtFunctor(params, Udata);
    Kokkos::Max<real_t> reducer(invDt);
    Kokkos::parallel_reduce(
      "Computedt",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      computeDtFunctor,
      reducer);
  }

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j, real_t & invDt) const
  {
    const int    isize = params.isize;
    const int    jsize = params.jsize;
    const int    ghostWidth = params.ghostWidth;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c = 0.0;
      real_t     vx, vy;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, ID);
      uLoc[IP] = Udata(i, j, IP);
      uLoc[IU] = Udata(i, j, IU);
      uLoc[IV] = Udata(i, j, IV);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      vx = c + fabs(qLoc[IU]);
      vy = c + fabs(qLoc[IV]);

      invDt = fmax(invDt, vx / dx + vy / dy);
    }

  } // operator ()

  DataArray_t Udata;

}; // ComputeDtFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template<typename device_t>
class ConvertToPrimitivesFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  ConvertToPrimitivesFunctor(HydroParams params, DataArray_t Udata, DataArray_t Qdata)
    : HydroBaseFunctor(params)
    , Udata(Udata)
    , Qdata(Qdata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata, DataArray_t Qdata)
  {
    ConvertToPrimitivesFunctor convertToPrimitivesFunctor(params, Udata, Qdata);
    Kokkos::parallel_for(
      "ConvertToPrimitives",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      convertToPrimitivesFunctor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    // const int ghostWidth = params.ghostWidth;

    if (j >= 0 && j < jsize && i >= 0 && i < isize)
    {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t     c;

      // get local conservative variable
      uLoc[ID] = Udata(i, j, ID);
      uLoc[IP] = Udata(i, j, IP);
      uLoc[IU] = Udata(i, j, IU);
      uLoc[IV] = Udata(i, j, IV);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);

      // copy q state in q global
      Qdata(i, j, ID) = qLoc[ID];
      Qdata(i, j, IP) = qLoc[IP];
      Qdata(i, j, IU) = qLoc[IU];
      Qdata(i, j, IV) = qLoc[IV];
    }
  }

  DataArray_t Udata;
  DataArray_t Qdata;

}; // ConvertToPrimitivesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/* NOT USED CURRENTLY */
// class ComputeFluxesAndUpdateFunctor : public HydroBaseFunctor {

// public:

//   ComputeFluxesAndUpdateFunctor(HydroParams params,
// 				DataArray_t Udata,
// 				DataArray_t Qm_x,
// 				DataArray_t Qm_y,
// 				DataArray_t Qp_x,
// 				DataArray_t Qp_y,
// 				real_t dtdx,
// 				real_t dtdy) :
//     HydroBaseFunctor(params), Udata(Udata),
//     Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
//     dtdx(dtdx), dtdy(dtdy) {};

//   // static method which does it all: create and execute functor
//   static void apply(HydroParams params,
// 		    DataArray_t Udata,
// 		    DataArray_t Qm_x,
// 		    DataArray_t Qm_y,
// 		    DataArray_t Qp_x,
// 		    DataArray_t Qp_y,
// 		    real_t dtdx,
// 		    real_t dtdy)
//   {
//     const int ijsize = params.isize*params.jsize;
//     ComputeFluxesAndUpdateFunctor computeFluxesAndUpdateFunctor(params, Udata,
// 								Qm_x, Qm_y,
// 								Qp_x, Qp_y,
// 								dtdx, dtdy);
//     Kokkos::parallel_for("ComputeFluxesAndUpdate", Kokkos::RangePolicy<exec_space>(0,ijsize),
//     computeFluxesAndUpdateFunctor);
//   }

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const int& index_) const
//   {
//     const int isize = params.isize;
//     const int jsize = params.jsize;
//     const int ghostWidth = params.ghostWidth;

//     int i,j;
//     index2coord(index_,i,j,isize,jsize);

//     if(j >= ghostWidth && j <= jsize - ghostWidth &&
//        i >= ghostWidth && i <= isize - ghostWidth) {

//       HydroState qleft, qright;
//       HydroState flux_x, flux_y;
//       HydroState qgdnv;

//       //
//       // Solve Riemann problem at X-interfaces and compute
//       // X-fluxes
//       //
//       qleft[ID]   = Qm_x(i-1,j , ID);
//       qleft[IP]   = Qm_x(i-1,j , IP);
//       qleft[IU]   = Qm_x(i-1,j , IU);
//       qleft[IV]   = Qm_x(i-1,j , IV);

//       qright[ID]  = Qp_x(i  ,j , ID);
//       qright[IP]  = Qp_x(i  ,j , IP);
//       qright[IU]  = Qp_x(i  ,j , IU);
//       qright[IV]  = Qp_x(i  ,j , IV);

//       // compute hydro flux_x
//       riemann_hllc(qleft,qright,qgdnv,flux_x);

//       //
//       // Solve Riemann problem at Y-interfaces and compute Y-fluxes
//       //
//       qleft[ID]   = Qm_y(i  ,j-1, ID);
//       qleft[IP]   = Qm_y(i  ,j-1, IP);
//       qleft[IU]   = Qm_y(i  ,j-1, IV); // watchout IU, IV permutation
//       qleft[IV]   = Qm_y(i  ,j-1, IU); // watchout IU, IV permutation

//       qright[ID]  = Qp_y(i  ,j , ID);
//       qright[IP]  = Qp_y(i  ,j , IP);
//       qright[IU]  = Qp_y(i  ,j , IV); // watchout IU, IV permutation
//       qright[IV]  = Qp_y(i  ,j , IU); // watchout IU, IV permutation

//       // compute hydro flux_y
//       riemann_hllc(qleft,qright,qgdnv,flux_y);

//       //
//       // update hydro array
//       //
//       Udata(i-1,j  , ID) += - flux_x[ID]*dtdx;
//       Udata(i-1,j  , IP) += - flux_x[IP]*dtdx;
//       Udata(i-1,j  , IU) += - flux_x[IU]*dtdx;
//       Udata(i-1,j  , IV) += - flux_x[IV]*dtdx;

//       Udata(i  ,j  , ID) +=   flux_x[ID]*dtdx;
//       Udata(i  ,j  , IP) +=   flux_x[IP]*dtdx;
//       Udata(i  ,j  , IU) +=   flux_x[IU]*dtdx;
//       Udata(i  ,j  , IV) +=   flux_x[IV]*dtdx;

//       Udata(i  ,j-1, ID) += - flux_y[ID]*dtdy;
//       Udata(i  ,j-1, IP) += - flux_y[IP]*dtdy;
//       Udata(i  ,j-1, IU) += - flux_y[IV]*dtdy; // watchout IU and IV swapped
//       Udata(i  ,j-1, IV) += - flux_y[IU]*dtdy; // watchout IU and IV swapped

//       Udata(i  ,j  , ID) +=   flux_y[ID]*dtdy;
//       Udata(i  ,j  , IP) +=   flux_y[IP]*dtdy;
//       Udata(i  ,j  , IU) +=   flux_y[IV]*dtdy; // watchout IU and IV swapped
//       Udata(i  ,j  , IV) +=   flux_y[IU]*dtdy; // watchout IU and IV swapped

//     }

//   }

//   DataArray_t Udata;
//   DataArray_t Qm_x, Qm_y, Qp_x, Qp_y;
//   real_t dtdx, dtdy;

// }; // ComputeFluxesAndUpdateFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/* NOT USED CURRENTLY */
// class ComputeTraceFunctor : public HydroBaseFunctor {

// public:

//   ComputeTraceFunctor(HydroParams params,
// 		      DataArray_t Qdata,
// 		      DataArray_t Qm_x,
// 		      DataArray_t Qm_y,
// 		      DataArray_t Qp_x,
// 		      DataArray_t Qp_y,
// 		      real_t dtdx,
// 		      real_t dtdy) :
//     HydroBaseFunctor(params),
//     Qdata(Qdata),
//     Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
//     dtdx(dtdx), dtdy(dtdy) {};

//   // static method which does it all: create and execute functor
//   static void apply(HydroParams params,
// 		    DataArray_t Qdata,
// 		    DataArray_t Qm_x,
// 		    DataArray_t Qm_y,
// 		    DataArray_t Qp_x,
// 		    DataArray_t Qp_y,
// 		    real_t dtdx,
// 		    real_t dtdy)
//   {

//     const int ijsize = params.isize*params.jsize;
//     ComputeTraceFunctor computeTraceFunctor(params, Qdata,
// 					    Qm_x, Qm_y,
// 					    Qp_x, Qp_y,
// 					    dtdx, dtdy);
//     Kokkos::parallel_for("ComputeTrace",Kokkos::RangePolicy<exec_space>(0,ijsize), computeTraceFunctor);

//   }


//   KOKKOS_INLINE_FUNCTION
//   void operator()(const int& index) const
//   {
//     const int isize = params.isize;
//     const int jsize = params.jsize;
//     const int ghostWidth = params.ghostWidth;

//     int i,j;
//     index2coord(index,i,j,isize,jsize);

//     if(j >= 1 && j <= jsize - ghostWidth &&
//        i >= 1 && i <= isize - ghostWidth) {

//       HydroState qLoc   ; // local primitive variables
//       HydroState qPlusX ;
//       HydroState qMinusX;
//       HydroState qPlusY ;
//       HydroState qMinusY;

//       HydroState dqX;
//       HydroState dqY;

//       HydroState qmX;
//       HydroState qmY;
//       HydroState qpX;
//       HydroState qpY;

//       // get primitive variables state vector
//       {
// 	qLoc   [ID] = Qdata(i  ,j  , ID);
// 	qPlusX [ID] = Qdata(i+1,j  , ID);
// 	qMinusX[ID] = Qdata(i-1,j  , ID);
// 	qPlusY [ID] = Qdata(i  ,j+1, ID);
// 	qMinusY[ID] = Qdata(i  ,j-1, ID);

// 	qLoc   [IP] = Qdata(i  ,j  , IP);
// 	qPlusX [IP] = Qdata(i+1,j  , IP);
// 	qMinusX[IP] = Qdata(i-1,j  , IP);
// 	qPlusY [IP] = Qdata(i  ,j+1, IP);
// 	qMinusY[IP] = Qdata(i  ,j-1, IP);

// 	qLoc   [IU] = Qdata(i  ,j  , IU);
// 	qPlusX [IU] = Qdata(i+1,j  , IU);
// 	qMinusX[IU] = Qdata(i-1,j  , IU);
// 	qPlusY [IU] = Qdata(i  ,j+1, IU);
// 	qMinusY[IU] = Qdata(i  ,j-1, IU);

// 	qLoc   [IV] = Qdata(i  ,j  , IV);
// 	qPlusX [IV] = Qdata(i+1,j  , IV);
// 	qMinusX[IV] = Qdata(i-1,j  , IV);
// 	qPlusY [IV] = Qdata(i  ,j+1, IV);
// 	qMinusY[IV] = Qdata(i  ,j-1, IV);

//       } //

//       // get hydro slopes dq
//       slope_unsplit_hydro_2d(qLoc,
// 			     qPlusX, qMinusX,
// 			     qPlusY, qMinusY,
// 			     dqX, dqY);

//       // compute qm, qp
//       trace_unsplit_hydro_2d(qLoc,
// 			     dqX, dqY,
// 			     dtdx, dtdy,
// 			     qmX, qmY,
// 			     qpX, qpY);

//       // store qm, qp : only what is really needed
//       Qm_x(i  ,j  , ID) = qmX[ID];
//       Qp_x(i  ,j  , ID) = qpX[ID];
//       Qm_y(i  ,j  , ID) = qmY[ID];
//       Qp_y(i  ,j  , ID) = qpY[ID];

//       Qm_x(i  ,j  , IP) = qmX[IP];
//       Qp_x(i  ,j  , IP) = qpX[IP];
//       Qm_y(i  ,j  , IP) = qmY[IP];
//       Qp_y(i  ,j  , IP) = qpY[IP];

//       Qm_x(i  ,j  , IU) = qmX[IU];
//       Qp_x(i  ,j  , IU) = qpX[IU];
//       Qm_y(i  ,j  , IU) = qmY[IU];
//       Qp_y(i  ,j  , IU) = qpY[IU];

//       Qm_x(i  ,j  , IV) = qmX[IV];
//       Qp_x(i  ,j  , IV) = qpX[IV];
//       Qm_y(i  ,j  , IV) = qmY[IV];
//       Qp_y(i  ,j  , IV) = qpY[IV];

//     }
//   }

//   DataArray_t Qdata;
//   DataArray_t Qm_x, Qm_y, Qp_x, Qp_y;
//   real_t dtdx, dtdy;

// }; // ComputeTraceFunctor


/*************************************************/
/*************************************************/
/*************************************************/
template<typename device_t>
class ComputeAndStoreFluxesFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  ComputeAndStoreFluxesFunctor(HydroParams params,
                               DataArray_t   Qdata,
                               DataArray_t   FluxData_x,
                               DataArray_t   FluxData_y,
                               real_t      dtdx,
                               real_t      dtdy)
    : HydroBaseFunctor(params)
    , Qdata(Qdata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray_t   Qdata,
        DataArray_t   FluxData_x,
        DataArray_t   FluxData_y,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeAndStoreFluxesFunctor functor(params, Qdata, FluxData_x, FluxData_y, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeAndStoreFluxes",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth && j <= jsize - ghostWidth && i >= ghostWidth && i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;

      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // get primitive variables state vector
      qLoc[ID] = Qdata(i, j, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j, ID);
      qNeighbors_2[ID] = Qdata(i, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i, j - 1, ID);

      qLoc[IP] = Qdata(i, j, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j, IP);
      qNeighbors_2[IP] = Qdata(i, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i, j - 1, IP);

      qLoc[IU] = Qdata(i, j, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j, IU);
      qNeighbors_2[IU] = Qdata(i, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i, j - 1, IU);

      qLoc[IV] = Qdata(i, j, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j, IV);
      qNeighbors_2[IV] = Qdata(i, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i, j - 1, IV);

      slope_unsplit_hydro_2d(
        qLoc, qNeighbors_0, qNeighbors_1, qNeighbors_2, qNeighbors_3, dqX, dqY);

      // slopes at left neighbor along X
      qLocNeighbor[ID] = Qdata(i - 1, j, ID);
      qNeighbors_0[ID] = Qdata(i, j, ID);
      qNeighbors_1[ID] = Qdata(i - 2, j, ID);
      qNeighbors_2[ID] = Qdata(i - 1, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i - 1, j - 1, ID);

      qLocNeighbor[IP] = Qdata(i - 1, j, IP);
      qNeighbors_0[IP] = Qdata(i, j, IP);
      qNeighbors_1[IP] = Qdata(i - 2, j, IP);
      qNeighbors_2[IP] = Qdata(i - 1, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i - 1, j - 1, IP);

      qLocNeighbor[IU] = Qdata(i - 1, j, IU);
      qNeighbors_0[IU] = Qdata(i, j, IU);
      qNeighbors_1[IU] = Qdata(i - 2, j, IU);
      qNeighbors_2[IU] = Qdata(i - 1, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i - 1, j - 1, IU);

      qLocNeighbor[IV] = Qdata(i - 1, j, IV);
      qNeighbors_0[IV] = Qdata(i, j, IV);
      qNeighbors_1[IV] = Qdata(i - 2, j, IV);
      qNeighbors_2[IV] = Qdata(i - 1, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i - 1, j - 1, IV);

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along X
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_XMAX, qleft);

      // Solve Riemann problem at X-interfaces and compute X-fluxes
      // riemann_2d(qleft,qright,&qgdnv,&flux_x);
      riemann_hllc(qleft, qright, qgdnv, flux_x);

      //
      // store fluxes X
      //
      FluxData_x(i, j, ID) = flux_x[ID] * dtdx;
      FluxData_x(i, j, IP) = flux_x[IP] * dtdx;
      FluxData_x(i, j, IU) = flux_x[IU] * dtdx;
      FluxData_x(i, j, IV) = flux_x[IV] * dtdx;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      qLocNeighbor[ID] = Qdata(i, j - 1, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j - 1, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j - 1, ID);
      qNeighbors_2[ID] = Qdata(i, j, ID);
      qNeighbors_3[ID] = Qdata(i, j - 2, ID);

      qLocNeighbor[IP] = Qdata(i, j - 1, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j - 1, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j - 1, IP);
      qNeighbors_2[IP] = Qdata(i, j, IP);
      qNeighbors_3[IP] = Qdata(i, j - 2, IP);

      qLocNeighbor[IU] = Qdata(i, j - 1, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j - 1, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j - 1, IU);
      qNeighbors_2[IU] = Qdata(i, j, IU);
      qNeighbors_3[IU] = Qdata(i, j - 2, IU);

      qLocNeighbor[IV] = Qdata(i, j - 1, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j - 1, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j - 1, IV);
      qNeighbors_2[IV] = Qdata(i, j, IV);
      qNeighbors_3[IV] = Qdata(i, j - 2, IV);

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_YMAX, qleft);

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qright[IU]), &(qright[IV]));
      // riemann_2d(qleft,qright,qgdnv,flux_y);
      riemann_hllc(qleft, qright, qgdnv, flux_y);

      //
      // store fluxes Y
      //
      FluxData_y(i, j, ID) = flux_y[ID] * dtdy;
      FluxData_y(i, j, IP) = flux_y[IP] * dtdy;
      FluxData_y(i, j, IU) = flux_y[IV] * dtdy; //
      FluxData_y(i, j, IV) = flux_y[IU] * dtdy; //

    } // end if

  } // end operator ()

  DataArray_t Qdata;
  DataArray_t FluxData_x;
  DataArray_t FluxData_y;
  real_t    dtdx, dtdy;

}; // ComputeAndStoreFluxesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template<typename device_t>
class UpdateFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  UpdateFunctor(HydroParams params, DataArray_t Udata, DataArray_t FluxData_x, DataArray_t FluxData_y)
    : HydroBaseFunctor(params)
    , Udata(Udata)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata, DataArray_t FluxData_x, DataArray_t FluxData_y)
  {
    UpdateFunctor functor(params, Udata, FluxData_x, FluxData_y);
    Kokkos::parallel_for(
      "UpdateFunctor",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      Udata(i, j, ID) += FluxData_x(i, j, ID);
      Udata(i, j, IP) += FluxData_x(i, j, IP);
      Udata(i, j, IU) += FluxData_x(i, j, IU);
      Udata(i, j, IV) += FluxData_x(i, j, IV);

      Udata(i, j, ID) -= FluxData_x(i + 1, j, ID);
      Udata(i, j, IP) -= FluxData_x(i + 1, j, IP);
      Udata(i, j, IU) -= FluxData_x(i + 1, j, IU);
      Udata(i, j, IV) -= FluxData_x(i + 1, j, IV);

      Udata(i, j, ID) += FluxData_y(i, j, ID);
      Udata(i, j, IP) += FluxData_y(i, j, IP);
      Udata(i, j, IU) += FluxData_y(i, j, IU);
      Udata(i, j, IV) += FluxData_y(i, j, IV);

      Udata(i, j, ID) -= FluxData_y(i, j + 1, ID);
      Udata(i, j, IP) -= FluxData_y(i, j + 1, IP);
      Udata(i, j, IU) -= FluxData_y(i, j + 1, IU);
      Udata(i, j, IV) -= FluxData_y(i, j + 1, IV);

    } // end if

  } // end operator ()

  DataArray_t Udata;
  DataArray_t FluxData_x;
  DataArray_t FluxData_y;

}; // UpdateFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <typename device_t>
class ComputeFluxesAndUpdateFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  ComputeFluxesAndUpdateFunctor(HydroParams params,
                                DataArray_t Qdata,
                                DataArray_t Udata,
                                real_t      dtdx,
                                real_t      dtdy)
    : HydroBaseFunctor(params)
    , Qdata(Qdata)
    , Udata(Udata)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Qdata, DataArray_t Udata, real_t dtdx, real_t dtdy)
  {
    ComputeFluxesAndUpdateFunctor functor(params, Qdata, Udata, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeFluxesAndUpdate",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth && j <= jsize - ghostWidth && i >= ghostWidth && i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;

      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // get primitive variables state vector
      qLoc[ID] = Qdata(i, j, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j, ID);
      qNeighbors_2[ID] = Qdata(i, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i, j - 1, ID);

      qLoc[IP] = Qdata(i, j, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j, IP);
      qNeighbors_2[IP] = Qdata(i, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i, j - 1, IP);

      qLoc[IU] = Qdata(i, j, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j, IU);
      qNeighbors_2[IU] = Qdata(i, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i, j - 1, IU);

      qLoc[IV] = Qdata(i, j, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j, IV);
      qNeighbors_2[IV] = Qdata(i, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i, j - 1, IV);

      slope_unsplit_hydro_2d(
        qLoc, qNeighbors_0, qNeighbors_1, qNeighbors_2, qNeighbors_3, dqX, dqY);

      // slopes at left neighbor along X
      qLocNeighbor[ID] = Qdata(i - 1, j, ID);
      qNeighbors_0[ID] = Qdata(i, j, ID);
      qNeighbors_1[ID] = Qdata(i - 2, j, ID);
      qNeighbors_2[ID] = Qdata(i - 1, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i - 1, j - 1, ID);

      qLocNeighbor[IP] = Qdata(i - 1, j, IP);
      qNeighbors_0[IP] = Qdata(i, j, IP);
      qNeighbors_1[IP] = Qdata(i - 2, j, IP);
      qNeighbors_2[IP] = Qdata(i - 1, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i - 1, j - 1, IP);

      qLocNeighbor[IU] = Qdata(i - 1, j, IU);
      qNeighbors_0[IU] = Qdata(i, j, IU);
      qNeighbors_1[IU] = Qdata(i - 2, j, IU);
      qNeighbors_2[IU] = Qdata(i - 1, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i - 1, j - 1, IU);

      qLocNeighbor[IV] = Qdata(i - 1, j, IV);
      qNeighbors_0[IV] = Qdata(i, j, IV);
      qNeighbors_1[IV] = Qdata(i - 2, j, IV);
      qNeighbors_2[IV] = Qdata(i - 1, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i - 1, j - 1, IV);

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along X
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_XMAX, qleft);

      // Solve Riemann problem at X-interfaces and compute X-fluxes
      // riemann_2d(qleft,qright,&qgdnv,&flux_x);
      riemann_hllc(qleft, qright, qgdnv, flux_x);

      //
      // Update with fluxes along X
      //
      if (j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, ID), flux_x[ID] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IP), flux_x[IP] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IU), flux_x[IU] * dtdx);
        Kokkos::atomic_add(&Udata(i, j, IV), flux_x[IV] * dtdx);
      }
      if (j < jsize - ghostWidth and i > ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i - 1, j, ID), flux_x[ID] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IP), flux_x[IP] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IU), flux_x[IU] * dtdx);
        Kokkos::atomic_sub(&Udata(i - 1, j, IV), flux_x[IV] * dtdx);
      }

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      qLocNeighbor[ID] = Qdata(i, j - 1, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j - 1, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j - 1, ID);
      qNeighbors_2[ID] = Qdata(i, j, ID);
      qNeighbors_3[ID] = Qdata(i, j - 2, ID);

      qLocNeighbor[IP] = Qdata(i, j - 1, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j - 1, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j - 1, IP);
      qNeighbors_2[IP] = Qdata(i, j, IP);
      qNeighbors_3[IP] = Qdata(i, j - 2, IP);

      qLocNeighbor[IU] = Qdata(i, j - 1, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j - 1, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j - 1, IU);
      qNeighbors_2[IU] = Qdata(i, j, IU);
      qNeighbors_3[IU] = Qdata(i, j - 2, IU);

      qLocNeighbor[IV] = Qdata(i, j - 1, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j - 1, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j - 1, IV);
      qNeighbors_2[IV] = Qdata(i, j, IV);
      qNeighbors_3[IV] = Qdata(i, j - 2, IV);

      slope_unsplit_hydro_2d(qLocNeighbor,
                             qNeighbors_0,
                             qNeighbors_1,
                             qNeighbors_2,
                             qNeighbors_3,
                             dqX_neighbor,
                             dqY_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //

      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(
        qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_YMAX, qleft);

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]), &(qleft[IV]));
      swapValues(&(qright[IU]), &(qright[IV]));
      // riemann_2d(qleft,qright,qgdnv,flux_y);
      riemann_hllc(qleft, qright, qgdnv, flux_y);
      swapValues(&(flux_y[IU]), &(flux_y[IV]));

      //
      // store fluxes Y
      //
      if (j < jsize - ghostWidth and i < isize - ghostWidth)
      {
        Kokkos::atomic_add(&Udata(i, j, ID), flux_y[ID] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IP), flux_y[IP] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IU), flux_y[IU] * dtdy);
        Kokkos::atomic_add(&Udata(i, j, IV), flux_y[IV] * dtdy);
      }
      if (j > ghostWidth and i < jsize - ghostWidth)
      {
        Kokkos::atomic_sub(&Udata(i, j - 1, ID), flux_y[ID] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IP), flux_y[IP] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IU), flux_y[IU] * dtdy);
        Kokkos::atomic_sub(&Udata(i, j - 1, IV), flux_y[IV] * dtdy);
      }

    } // end if

  } // end operator ()

  DataArray_t Qdata;
  DataArray_t Udata;
  real_t      dtdx, dtdy;

}; // ComputeFluxesAndUpdateFunctor


/*************************************************/
/*************************************************/
/*************************************************/
template <typename device_t, Direction dir>
class UpdateDirFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  UpdateDirFunctor(HydroParams params, DataArray_t Udata, DataArray_t FluxData)
    : HydroBaseFunctor(params)
    , Udata(Udata)
    , FluxData(FluxData){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata, DataArray_t FluxData)
  {
    UpdateDirFunctor<device_t, dir> functor(params, Udata, FluxData);
    Kokkos::parallel_for(
      "UpdateDir",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      if (dir == XDIR)
      {

        Udata(i, j, ID) += FluxData(i, j, ID);
        Udata(i, j, IP) += FluxData(i, j, IP);
        Udata(i, j, IU) += FluxData(i, j, IU);
        Udata(i, j, IV) += FluxData(i, j, IV);

        Udata(i, j, ID) -= FluxData(i + 1, j, ID);
        Udata(i, j, IP) -= FluxData(i + 1, j, IP);
        Udata(i, j, IU) -= FluxData(i + 1, j, IU);
        Udata(i, j, IV) -= FluxData(i + 1, j, IV);
      }
      else if (dir == YDIR)
      {

        Udata(i, j, ID) += FluxData(i, j, ID);
        Udata(i, j, IP) += FluxData(i, j, IP);
        Udata(i, j, IU) += FluxData(i, j, IU);
        Udata(i, j, IV) += FluxData(i, j, IV);

        Udata(i, j, ID) -= FluxData(i, j + 1, ID);
        Udata(i, j, IP) -= FluxData(i, j + 1, IP);
        Udata(i, j, IU) -= FluxData(i, j + 1, IU);
        Udata(i, j, IV) -= FluxData(i, j + 1, IV);
      }

    } // end if

  } // end operator ()

  DataArray_t Udata;
  DataArray_t FluxData;

}; // UpdateDirFunctor


/*************************************************/
/*************************************************/
/*************************************************/
template<typename device_t>
class ComputeSlopesFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  ComputeSlopesFunctor(HydroParams params, DataArray_t Qdata, DataArray_t Slopes_x, DataArray_t Slopes_y)
    : HydroBaseFunctor(params)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Qdata, DataArray_t Slopes_x, DataArray_t Slopes_y)
  {
    ComputeSlopesFunctor functor(params, Qdata, Slopes_x, Slopes_y);
    Kokkos::parallel_for(
      "ComputeSlopes",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth - 1 && j <= jsize - ghostWidth && i >= ghostWidth - 1 &&
        i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;

      // Local slopes and neighbor slopes
      HydroState dqX{};
      HydroState dqY{};

      // get primitive variables state vector
      qLoc[ID] = Qdata(i, j, ID);
      qNeighbors_0[ID] = Qdata(i + 1, j, ID);
      qNeighbors_1[ID] = Qdata(i - 1, j, ID);
      qNeighbors_2[ID] = Qdata(i, j + 1, ID);
      qNeighbors_3[ID] = Qdata(i, j - 1, ID);

      qLoc[IP] = Qdata(i, j, IP);
      qNeighbors_0[IP] = Qdata(i + 1, j, IP);
      qNeighbors_1[IP] = Qdata(i - 1, j, IP);
      qNeighbors_2[IP] = Qdata(i, j + 1, IP);
      qNeighbors_3[IP] = Qdata(i, j - 1, IP);

      qLoc[IU] = Qdata(i, j, IU);
      qNeighbors_0[IU] = Qdata(i + 1, j, IU);
      qNeighbors_1[IU] = Qdata(i - 1, j, IU);
      qNeighbors_2[IU] = Qdata(i, j + 1, IU);
      qNeighbors_3[IU] = Qdata(i, j - 1, IU);

      qLoc[IV] = Qdata(i, j, IV);
      qNeighbors_0[IV] = Qdata(i + 1, j, IV);
      qNeighbors_1[IV] = Qdata(i - 1, j, IV);
      qNeighbors_2[IV] = Qdata(i, j + 1, IV);
      qNeighbors_3[IV] = Qdata(i, j - 1, IV);

      slope_unsplit_hydro_2d(
        qLoc, qNeighbors_0, qNeighbors_1, qNeighbors_2, qNeighbors_3, dqX, dqY);

      // copy back slopes in global arrays
      Slopes_x(i, j, ID) = dqX[ID];
      Slopes_y(i, j, ID) = dqY[ID];

      Slopes_x(i, j, IP) = dqX[IP];
      Slopes_y(i, j, IP) = dqY[IP];

      Slopes_x(i, j, IU) = dqX[IU];
      Slopes_y(i, j, IU) = dqY[IU];

      Slopes_x(i, j, IV) = dqX[IV];
      Slopes_y(i, j, IV) = dqY[IV];

    } // end if

  } // end operator ()

  DataArray_t Qdata;
  DataArray_t Slopes_x, Slopes_y;

}; // ComputeSlopesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template<typename device_t, Direction dir>
class ComputeTraceAndFluxes_Functor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  ComputeTraceAndFluxes_Functor(HydroParams params,
                                DataArray_t   Qdata,
                                DataArray_t   Slopes_x,
                                DataArray_t   Slopes_y,
                                DataArray_t   Fluxes,
                                real_t      dtdx,
                                real_t      dtdy)
    : HydroBaseFunctor(params)
    , Qdata(Qdata)
    , Slopes_x(Slopes_x)
    , Slopes_y(Slopes_y)
    , Fluxes(Fluxes)
    , dtdx(dtdx)
    , dtdy(dtdy){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params,
        DataArray_t   Qdata,
        DataArray_t   Slopes_x,
        DataArray_t   Slopes_y,
        DataArray_t   Fluxes,
        real_t      dtdx,
        real_t      dtdy)
  {
    ComputeTraceAndFluxes_Functor<device_t, dir> functor(
      params, Qdata, Slopes_x, Slopes_y, Fluxes, dtdx, dtdy);
    Kokkos::parallel_for(
      "ComputeTraceAndFluxes",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    if (j >= ghostWidth && j <= jsize - ghostWidth && i >= ghostWidth && i <= isize - ghostWidth)
    {

      // local primitive variables
      HydroState qLoc; // local primitive variables

      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;

      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux;

      //
      // compute reconstructed states at left interface along X
      //
      qLoc[ID] = Qdata(i, j, ID);
      dqX[ID] = Slopes_x(i, j, ID);
      dqY[ID] = Slopes_y(i, j, ID);

      qLoc[IP] = Qdata(i, j, IP);
      dqX[IP] = Slopes_x(i, j, IP);
      dqY[IP] = Slopes_y(i, j, IP);

      qLoc[IU] = Qdata(i, j, IU);
      dqX[IU] = Slopes_x(i, j, IU);
      dqY[IU] = Slopes_y(i, j, IU);

      qLoc[IV] = Qdata(i, j, IV);
      dqX[IV] = Slopes_x(i, j, IV);
      dqY[IV] = Slopes_y(i, j, IV);

      if (dir == XDIR)
      {

        // left interface : right state
        trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_XMIN, qright);

        qLocNeighbor[ID] = Qdata(i - 1, j, ID);
        dqX_neighbor[ID] = Slopes_x(i - 1, j, ID);
        dqY_neighbor[ID] = Slopes_y(i - 1, j, ID);

        qLocNeighbor[IP] = Qdata(i - 1, j, IP);
        dqX_neighbor[IP] = Slopes_x(i - 1, j, IP);
        dqY_neighbor[IP] = Slopes_y(i - 1, j, IP);

        qLocNeighbor[IU] = Qdata(i - 1, j, IU);
        dqX_neighbor[IU] = Slopes_x(i - 1, j, IU);
        dqY_neighbor[IU] = Slopes_y(i - 1, j, IU);

        qLocNeighbor[IV] = Qdata(i - 1, j, IV);
        dqX_neighbor[IV] = Slopes_x(i - 1, j, IV);
        dqY_neighbor[IV] = Slopes_y(i - 1, j, IV);

        // left interface : left state
        trace_unsplit_2d_along_dir(
          qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_XMAX, qleft);

        // Solve Riemann problem at X-interfaces and compute X-fluxes
        riemann_hllc(qleft, qright, qgdnv, flux);

        //
        // store fluxes
        //
        Fluxes(i, j, ID) = flux[ID] * dtdx;
        Fluxes(i, j, IP) = flux[IP] * dtdx;
        Fluxes(i, j, IU) = flux[IU] * dtdx;
        Fluxes(i, j, IV) = flux[IV] * dtdx;
      }
      else if (dir == YDIR)
      {

        // left interface : right state
        trace_unsplit_2d_along_dir(qLoc, dqX, dqY, dtdx, dtdy, FACE_YMIN, qright);

        qLocNeighbor[ID] = Qdata(i, j - 1, ID);
        dqX_neighbor[ID] = Slopes_x(i, j - 1, ID);
        dqY_neighbor[ID] = Slopes_y(i, j - 1, ID);

        qLocNeighbor[IP] = Qdata(i, j - 1, IP);
        dqX_neighbor[IP] = Slopes_x(i, j - 1, IP);
        dqY_neighbor[IP] = Slopes_y(i, j - 1, IP);

        qLocNeighbor[IU] = Qdata(i, j - 1, IU);
        dqX_neighbor[IU] = Slopes_x(i, j - 1, IU);
        dqY_neighbor[IU] = Slopes_y(i, j - 1, IU);

        qLocNeighbor[IV] = Qdata(i, j - 1, IV);
        dqX_neighbor[IV] = Slopes_x(i, j - 1, IV);
        dqY_neighbor[IV] = Slopes_y(i, j - 1, IV);

        // left interface : left state
        trace_unsplit_2d_along_dir(
          qLocNeighbor, dqX_neighbor, dqY_neighbor, dtdx, dtdy, FACE_YMAX, qleft);

        // Solve Riemann problem at Y-interfaces and compute Y-fluxes
        swapValues(&(qleft[IU]), &(qleft[IV]));
        swapValues(&(qright[IU]), &(qright[IV]));
        riemann_hllc(qleft, qright, qgdnv, flux);

        //
        // update hydro array
        //
        Fluxes(i, j, ID) = flux[ID] * dtdy;
        Fluxes(i, j, IP) = flux[IP] * dtdy;
        Fluxes(i, j, IU) = flux[IV] * dtdy; // IU/IV swapped
        Fluxes(i, j, IV) = flux[IU] * dtdy; // IU/IV swapped
      }

    } // end if

  } // end operator ()

  DataArray_t Qdata;
  DataArray_t Slopes_x, Slopes_y;
  DataArray_t Fluxes;
  real_t    dtdx, dtdy;

}; // ComputeTraceAndFluxes_Functor

/*************************************************/
/*************************************************/
/*************************************************/
template<typename device_t>
class InitImplodeFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  InitImplodeFunctor(HydroParams params, DataArray_t Udata)
    : HydroBaseFunctor(params)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata)
  {
    InitImplodeFunctor functor(params, Udata);
    Kokkos::parallel_for(
      "InitImplode",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    real_t x = xmin + dx / 2 + (i - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j - ghostWidth) * dy;

    real_t tmp = x + y * y;
    if (tmp > 0.5 && tmp < 1.5)
    {
      Udata(i, j, ID) = 1.0;
      Udata(i, j, IP) = 1.0 / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;
    }
    else
    {
      Udata(i, j, ID) = 0.125;
      Udata(i, j, IP) = 0.14 / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;
    }

  } // end operator ()

  DataArray_t Udata;

}; // InitImplodeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Initialize blast test case.
 *
 * Two types of initialization:
 *
 * - if parameter total_energy_inside is positive, we initialize in total energy
 * - if parameter total_energy_inside is negative or null, we initialize using pressure in /
 * pressure out
 *
 * If you want to do the well know sedov blast test, you need to initialize using total energy, not
 * pressure.
 */
template <typename device_t>
class InitBlastFunctor : public HydroBaseFunctor
{

public:
  struct TagRegularInit
  {};
  struct TagCorrectTotalEnergy
  {};

  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  InitBlastFunctor(HydroParams params, DataArray_t Udata)
    : HydroBaseFunctor(params)
    , Udata(Udata) {};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata)
  {
    real_t              volume_inside = ZERO_F;
    Kokkos::Sum<real_t> reducer(volume_inside);

    InitBlastFunctor functor(params, Udata);
    Kokkos::parallel_reduce("InitBlast - regular",
                            Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>, TagRegularInit>(
                              { 0, 0 }, { params.isize, params.jsize }),
                            functor,
                            reducer);

    if (params.blast_total_energy_inside > 0)
    {
      functor.m_volume_inside = volume_inside;
      Kokkos::parallel_for(
        "InitBlast - correct total energy",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>, TagCorrectTotalEnergy>(
          { 0, 0 }, { params.isize, params.jsize }),
        functor);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRegularInit const &, const int & i, const int & j, real_t & volume) const
  {

    const int ghostWidth = params.ghostWidth;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius = params.blast_radius;
    const real_t radius2 = blast_radius * blast_radius;
    const real_t blast_center_x = params.blast_center_x;
    const real_t blast_center_y = params.blast_center_y;
    const real_t blast_density_in = params.blast_density_in;
    const real_t blast_density_out = params.blast_density_out;
    const real_t blast_pressure_in = params.blast_pressure_in;
    const real_t blast_pressure_out = params.blast_pressure_out;

    real_t x = xmin + dx / 2 + (i - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j - ghostWidth) * dy;

    real_t d2 =
      (x - blast_center_x) * (x - blast_center_x) + (y - blast_center_y) * (y - blast_center_y);

    if (d2 < radius2)
    {
      Udata(i, j, ID) = blast_density_in;
      Udata(i, j, IP) = blast_pressure_in / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;

      volume += dx * dy;
    }
    else
    {
      Udata(i, j, ID) = blast_density_out;
      Udata(i, j, IP) = blast_pressure_out / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;
    }

  } // end operator () - TagRegularInit

  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagCorrectTotalEnergy const &, const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius = params.blast_radius;
    const real_t radius2 = blast_radius * blast_radius;
    const real_t blast_center_x = params.blast_center_x;
    const real_t blast_center_y = params.blast_center_y;
    const real_t blast_total_energy_inside = params.blast_total_energy_inside;

    real_t x = xmin + dx / 2 + (i - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j - ghostWidth) * dy;

    real_t d2 =
      (x - blast_center_x) * (x - blast_center_x) + (y - blast_center_y) * (y - blast_center_y);

    if (d2 < radius2)
    {
      Udata(i, j, IP) = blast_total_energy_inside / m_volume_inside;
    }

  } // end operator () - TagCorrectTotalEnergy

  DataArray_t Udata;
  real_t      m_volume_inside = 0.0;

}; // InitBlastFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <typename device_t>
class InitFourQuadrantFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  InitFourQuadrantFunctor(HydroParams params, DataArray_t Udata)
    : HydroBaseFunctor(params)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata)
  {
    InitFourQuadrantFunctor functor(params, Udata);
    Kokkos::parallel_for(
      "InitFourQuadrant",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_FUNCTION void
  primToCons(Kokkos::Array<real_t, 4> & U, real_t gamma0) const
  {

    real_t rho = U[ID];
    real_t p = U[IP];
    real_t u = U[IU];
    real_t v = U[IV];

    U[IU] *= rho; // rho*u
    U[IV] *= rho; // rho*v
    U[IP] = p / (gamma0 - 1.0) + rho * (u * u + v * v) * 0.5;

  } // primToCons


  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    // four quadrant problem parameters
    const real_t xt = 0.8;
    const real_t yt = 0.8;

    const real_t gamma0 = params.settings.gamma0;

    // clang-format off
    // primitive variables
    Kokkos::Array<real_t, 4> U0{ 1.5   , 1.5  , 0.0  , 0.0   };
    Kokkos::Array<real_t, 4> U1{ 0.5323, 0.3  , 1.206, 0.0   };
    Kokkos::Array<real_t, 4> U2{ 0.138 , 0.029, 1.206, 1.206 };
    Kokkos::Array<real_t, 4> U3{ 0.5323, 0.3  , 0.0  , 1.206 };
    // clang-format on

    primToCons(U0, gamma0);
    primToCons(U1, gamma0);
    primToCons(U2, gamma0);
    primToCons(U3, gamma0);

    real_t x = xmin + dx / 2 + (i - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j - ghostWidth) * dy;

    if (x < xt)
    {
      if (y < yt)
      {
        // region 2
        Udata(i, j, ID) = U2[ID];
        Udata(i, j, IP) = U2[IP];
        Udata(i, j, IU) = U2[IU];
        Udata(i, j, IV) = U2[IV];
      }
      else
      {
        // region 1
        Udata(i, j, ID) = U1[ID];
        Udata(i, j, IP) = U1[IP];
        Udata(i, j, IU) = U1[IU];
        Udata(i, j, IV) = U1[IV];
      }
    }
    else
    {
      if (y < yt)
      {
        // region 3
        Udata(i, j, ID) = U3[ID];
        Udata(i, j, IP) = U3[IP];
        Udata(i, j, IU) = U3[IU];
        Udata(i, j, IV) = U3[IV];
      }
      else
      {
        // region 0
        Udata(i, j, ID) = U0[ID];
        Udata(i, j, IP) = U0[IP];
        Udata(i, j, IU) = U0[IU];
        Udata(i, j, IV) = U0[IV];
      }
    }

  } // end operator ()

  DataArray_t Udata;

}; // InitFourQuadrantFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <typename device_t>
class InitDiscontinuityFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  InitDiscontinuityFunctor(HydroParams params, DataArray_t Udata)
    : HydroBaseFunctor(params)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata)
  {
    InitDiscontinuityFunctor functor(params, Udata);
    Kokkos::parallel_for(
      "InitDiscontinuity",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int ghostWidth = params.ghostWidth;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t gamma0 = params.settings.gamma0;

    real_t x = xmin + dx / 2 + (i - ghostWidth) * dx;
    real_t y = ymin + dy / 2 + (j - ghostWidth) * dy;

    if (x + y < 1)
    {
      Udata(i, j, ID) = 1.0 + x * x;
      Udata(i, j, IP) = 1.0 / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;
    }
    else
    {
      Udata(i, j, ID) = 0.25;
      Udata(i, j, IP) = 1.0 / (gamma0 - 1.0);
      Udata(i, j, IU) = 0.0;
      Udata(i, j, IV) = 0.0;
    }

  } // end operator ()

  DataArray_t Udata;

}; // InitDiscontinuityFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <typename device_t, FaceIdType faceId>
class MakeBoundariesFunctor : public HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;
  using exec_space = typename device_t::execution_space;

  MakeBoundariesFunctor(HydroParams params, DataArray_t Udata)
    : HydroBaseFunctor(params)
    , Udata(Udata){};


  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, DataArray_t Udata)
  {
    int nbIter = params.ghostWidth * std::max(params.isize, params.jsize);

    MakeBoundariesFunctor<device_t, faceId> functor(params, Udata);
    Kokkos::parallel_for("MakeBoundaries", Kokkos::RangePolicy<exec_space>(0, nbIter), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index) const
  {
    const int nx = params.nx;
    const int ny = params.ny;

    // const int isize = params.isize;
    // const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    const int imin = params.imin;
    const int imax = params.imax;

    const int jmin = params.jmin;
    const int jmax = params.jmax;

    int i, j;

    int i0, j0;

    if (faceId == FACE_XMIN)
    {

      // boundary xmin
      int boundary_type = params.boundary_type_xmin;

      j = index / ghostWidth;
      i = index - j * ghostWidth;

      if (j >= jmin && j <= jmax && i >= 0 && i < ghostWidth)
      {

        for (int iVar = 0; iVar < NBVAR; iVar++)
        {

          real_t sign = 1.0;

          if (boundary_type == BC_DIRICHLET)
          {
            i0 = 2 * ghostWidth - 1 - i;
            if (iVar == IU)
              sign = -1.0;
          }
          else if (boundary_type == BC_NEUMANN)
          {
            i0 = ghostWidth;
          }
          else
          { // periodic
            i0 = nx + i;
          }

          Udata(i, j, iVar) = Udata(i0, j, iVar) * sign;
        }
      }
    }

    if (faceId == FACE_XMAX)
    {

      // boundary xmax
      int boundary_type = params.boundary_type_xmax;

      j = index / ghostWidth;
      i = index - j * ghostWidth;
      i += (nx + ghostWidth);

      if (j >= jmin && j <= jmax && i >= nx + ghostWidth && i <= nx + 2 * ghostWidth - 1)
      {

        for (int iVar = 0; iVar < NBVAR; iVar++)
        {

          real_t sign = 1.0;

          if (boundary_type == BC_DIRICHLET)
          {
            i0 = 2 * nx + 2 * ghostWidth - 1 - i;
            if (iVar == IU)
              sign = -1.0;
          }
          else if (boundary_type == BC_NEUMANN)
          {
            i0 = nx + ghostWidth - 1;
          }
          else
          { // periodic
            i0 = i - nx;
          }

          Udata(i, j, iVar) = Udata(i0, j, iVar) * sign;
        }
      }
    }

    if (faceId == FACE_YMIN)
    {

      // boundary ymin
      int boundary_type = params.boundary_type_ymin;

      i = index / ghostWidth;
      j = index - i * ghostWidth;

      if (i >= imin && i <= imax && j >= 0 && j < ghostWidth)
      {

        for (int iVar = 0; iVar < NBVAR; iVar++)
        {

          real_t sign = 1.0;

          if (boundary_type == BC_DIRICHLET)
          {
            j0 = 2 * ghostWidth - 1 - j;
            if (iVar == IV)
              sign = -1.0;
          }
          else if (boundary_type == BC_NEUMANN)
          {
            j0 = ghostWidth;
          }
          else
          { // periodic
            j0 = ny + j;
          }

          Udata(i, j, iVar) = Udata(i, j0, iVar) * sign;
        }
      }
    }

    if (faceId == FACE_YMAX)
    {

      // boundary ymax
      int boundary_type = params.boundary_type_ymax;

      i = index / ghostWidth;
      j = index - i * ghostWidth;
      j += (ny + ghostWidth);

      if (i >= imin && i <= imax && j >= ny + ghostWidth && j <= ny + 2 * ghostWidth - 1)
      {

        for (int iVar = 0; iVar < NBVAR; iVar++)
        {

          real_t sign = 1.0;

          if (boundary_type == BC_DIRICHLET)
          {
            j0 = 2 * ny + 2 * ghostWidth - 1 - j;
            if (iVar == IV)
              sign = -1.0;
          }
          else if (boundary_type == BC_NEUMANN)
          {
            j0 = ny + ghostWidth - 1;
          }
          else
          { // periodic
            j0 = j - ny;
          }

          Udata(i, j, iVar) = Udata(i, j0, iVar) * sign;
        }
      }
    }

  } // end operator ()

  DataArray_t Udata;

}; // MakeBoundariesFunctor

} // namespace euler2d

#endif // HYDRO_RUN_FUNCTORS_H_
