#ifndef HYDRO_RUN_FUNCTORS_H_
#define HYDRO_RUN_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__


#include "HydroBaseFunctor.h"

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor : public HydroBaseFunctor {

public:
  
  ComputeDtFunctor(HydroParams params,
		   DataArray Udata) :
    HydroBaseFunctor(params),
    Udata(Udata)  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata,
		    real_t& invDt) {
    const int ijsize = params.isize*params.jsize;
    ComputeDtFunctor computeDtFunctor(params, Udata);
    Kokkos::parallel_reduce(ijsize, computeDtFunctor, invDt);
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
  } // init

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &index, real_t &invDt) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c=0.0;
      real_t vx, vy;
      
      // get local conservative variable
      uLoc[ID] = Udata(i,j,ID);
      uLoc[IP] = Udata(i,j,IP);
      uLoc[IU] = Udata(i,j,IU);
      uLoc[IV] = Udata(i,j,IV);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      vx = c+FABS(qLoc[IU]);
      vy = c+FABS(qLoc[IV]);

      invDt = FMAX(invDt, vx/dx + vy/dy);
      
    }
	    
  } // operator ()


  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  
  DataArray Udata;
  
}; // ComputeDtFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor : public HydroBaseFunctor {

public:

  ConvertToPrimitivesFunctor(HydroParams params,
			     DataArray Udata,
			     DataArray Qdata) :
    HydroBaseFunctor(params), Udata(Udata), Qdata(Qdata)  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata,
		    DataArray Qdata)
  {
    const int ijsize = params.isize*params.jsize;
    ConvertToPrimitivesFunctor convertToPrimitivesFunctor(params, Udata, Qdata);
    Kokkos::parallel_for(ijsize, convertToPrimitivesFunctor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    //const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= 0 && j < jsize  &&
       i >= 0 && i < isize ) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc[ID] = Udata(i,j,ID);
      uLoc[IP] = Udata(i,j,IP);
      uLoc[IU] = Udata(i,j,IU);
      uLoc[IV] = Udata(i,j,IV);
      
      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);

      // copy q state in q global
      Qdata(i,j,ID) = qLoc[ID];
      Qdata(i,j,IP) = qLoc[IP];
      Qdata(i,j,IU) = qLoc[IU];
      Qdata(i,j,IV) = qLoc[IV];
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qdata;
    
}; // ConvertToPrimitivesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/* NOT USED CURRENTLY */
class ComputeFluxesAndUpdateFunctor : public HydroBaseFunctor {

public:

  ComputeFluxesAndUpdateFunctor(HydroParams params,
				DataArray Udata,
				DataArray Qm_x,
				DataArray Qm_y,
				DataArray Qp_x,
				DataArray Qp_y,
				real_t dtdx,
				real_t dtdy) :
    HydroBaseFunctor(params), Udata(Udata),
    Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
    dtdx(dtdx), dtdy(dtdy) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata,
		    DataArray Qm_x,
		    DataArray Qm_y,
		    DataArray Qp_x,
		    DataArray Qp_y,
		    real_t dtdx,
		    real_t dtdy)
  {
    const int ijsize = params.isize*params.jsize;
    ComputeFluxesAndUpdateFunctor computeFluxesAndUpdateFunctor(params, Udata,
								Qm_x, Qm_y,
								Qp_x, Qp_y,
								dtdx, dtdy);
    Kokkos::parallel_for(ijsize, computeFluxesAndUpdateFunctor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index_) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index_,i,j,isize,jsize);
    
    if(j >= ghostWidth && j <= jsize - ghostWidth &&
       i >= ghostWidth && i <= isize - ghostWidth) {
      
      HydroState qleft, qright;
      HydroState flux_x, flux_y;
      HydroState qgdnv;

      //
      // Solve Riemann problem at X-interfaces and compute
      // X-fluxes
      //
      qleft[ID]   = Qm_x(i-1,j , ID);
      qleft[IP]   = Qm_x(i-1,j , IP);
      qleft[IU]   = Qm_x(i-1,j , IU);
      qleft[IV]   = Qm_x(i-1,j , IV);
      
      qright[ID]  = Qp_x(i  ,j , ID);
      qright[IP]  = Qp_x(i  ,j , IP);
      qright[IU]  = Qp_x(i  ,j , IU);
      qright[IV]  = Qp_x(i  ,j , IV);
      
      // compute hydro flux_x
      riemann_hllc(qleft,qright,qgdnv,flux_x);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      qleft[ID]   = Qm_y(i  ,j-1, ID);
      qleft[IP]   = Qm_y(i  ,j-1, IP);
      qleft[IU]   = Qm_y(i  ,j-1, IV); // watchout IU, IV permutation
      qleft[IV]   = Qm_y(i  ,j-1, IU); // watchout IU, IV permutation

      qright[ID]  = Qp_y(i  ,j , ID);
      qright[IP]  = Qp_y(i  ,j , IP);
      qright[IU]  = Qp_y(i  ,j , IV); // watchout IU, IV permutation
      qright[IV]  = Qp_y(i  ,j , IU); // watchout IU, IV permutation
      
      // compute hydro flux_y
      riemann_hllc(qleft,qright,qgdnv,flux_y);
            
      //
      // update hydro array
      //
      Udata(i-1,j  , ID) += - flux_x[ID]*dtdx;
      Udata(i-1,j  , IP) += - flux_x[IP]*dtdx;
      Udata(i-1,j  , IU) += - flux_x[IU]*dtdx;
      Udata(i-1,j  , IV) += - flux_x[IV]*dtdx;

      Udata(i  ,j  , ID) +=   flux_x[ID]*dtdx;
      Udata(i  ,j  , IP) +=   flux_x[IP]*dtdx;
      Udata(i  ,j  , IU) +=   flux_x[IU]*dtdx;
      Udata(i  ,j  , IV) +=   flux_x[IV]*dtdx;

      Udata(i  ,j-1, ID) += - flux_y[ID]*dtdy;
      Udata(i  ,j-1, IP) += - flux_y[IP]*dtdy;
      Udata(i  ,j-1, IU) += - flux_y[IV]*dtdy; // watchout IU and IV swapped
      Udata(i  ,j-1, IV) += - flux_y[IU]*dtdy; // watchout IU and IV swapped

      Udata(i  ,j  , ID) +=   flux_y[ID]*dtdy;
      Udata(i  ,j  , IP) +=   flux_y[IP]*dtdy;
      Udata(i  ,j  , IU) +=   flux_y[IV]*dtdy; // watchout IU and IV swapped
      Udata(i  ,j  , IV) +=   flux_y[IU]*dtdy; // watchout IU and IV swapped
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qm_x, Qm_y, Qp_x, Qp_y;
  real_t dtdx, dtdy;
  
}; // ComputeFluxesAndUpdateFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/* NOT USED CURRENTLY */
class ComputeTraceFunctor : public HydroBaseFunctor {

public:

  ComputeTraceFunctor(HydroParams params,
		      DataArray Udata,
		      DataArray Qdata,
		      DataArray Qm_x,
		      DataArray Qm_y,
		      DataArray Qp_x,
		      DataArray Qp_y,
		      real_t dtdx,
		      real_t dtdy) :
    HydroBaseFunctor(params),
    Udata(Udata), Qdata(Qdata),
    Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
    dtdx(dtdx), dtdy(dtdy) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata,
		    DataArray Qdata,
		    DataArray Qm_x,
		    DataArray Qm_y,
		    DataArray Qp_x,
		    DataArray Qp_y,
		    real_t dtdx,
		    real_t dtdy)
  {

    const int ijsize = params.isize*params.jsize;
    ComputeTraceFunctor computeTraceFunctor(params, Udata, Qdata,
					    Qm_x, Qm_y,
					    Qp_x, Qp_y,
					    dtdx, dtdy);
    Kokkos::parallel_for(ijsize, computeTraceFunctor);
    
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= 1 && j <= jsize - ghostWidth &&
       i >= 1 && i <= isize - ghostWidth) {

      HydroState qLoc   ; // local primitive variables
      HydroState qPlusX ;
      HydroState qMinusX;
      HydroState qPlusY ;
      HydroState qMinusY;

      HydroState dqX;
      HydroState dqY;

      HydroState qmX;
      HydroState qmY;
      HydroState qpX;
      HydroState qpY;
      
      // get primitive variables state vector
      {
	qLoc   [ID] = Qdata(i  ,j  , ID);
	qPlusX [ID] = Qdata(i+1,j  , ID);
	qMinusX[ID] = Qdata(i-1,j  , ID);
	qPlusY [ID] = Qdata(i  ,j+1, ID);
	qMinusY[ID] = Qdata(i  ,j-1, ID);

	qLoc   [IP] = Qdata(i  ,j  , IP);
	qPlusX [IP] = Qdata(i+1,j  , IP);
	qMinusX[IP] = Qdata(i-1,j  , IP);
	qPlusY [IP] = Qdata(i  ,j+1, IP);
	qMinusY[IP] = Qdata(i  ,j-1, IP);

	qLoc   [IU] = Qdata(i  ,j  , IU);
	qPlusX [IU] = Qdata(i+1,j  , IU);
	qMinusX[IU] = Qdata(i-1,j  , IU);
	qPlusY [IU] = Qdata(i  ,j+1, IU);
	qMinusY[IU] = Qdata(i  ,j-1, IU);

	qLoc   [IV] = Qdata(i  ,j  , IV);
	qPlusX [IV] = Qdata(i+1,j  , IV);
	qMinusX[IV] = Qdata(i-1,j  , IV);
	qPlusY [IV] = Qdata(i  ,j+1, IV);
	qMinusY[IV] = Qdata(i  ,j-1, IV);

      } // 
      
      // get hydro slopes dq
      slope_unsplit_hydro_2d(qLoc, 
			     qPlusX, qMinusX, 
			     qPlusY, qMinusY, 
			     dqX, dqY);
      
      // compute qm, qp
      trace_unsplit_hydro_2d(qLoc, 
			     dqX, dqY,
			     dtdx, dtdy, 
			     qmX, qmY,
			     qpX, qpY);

      // store qm, qp : only what is really needed
      Qm_x(i  ,j  , ID) = qmX[ID];
      Qp_x(i  ,j  , ID) = qpX[ID];
      Qm_y(i  ,j  , ID) = qmY[ID];
      Qp_y(i  ,j  , ID) = qpY[ID];
      
      Qm_x(i  ,j  , IP) = qmX[IP];
      Qp_x(i  ,j  , ID) = qpX[IP];
      Qm_y(i  ,j  , ID) = qmY[IP];
      Qp_y(i  ,j  , ID) = qpY[IP];
      
      Qm_x(i  ,j  , IU) = qmX[IU];
      Qp_x(i  ,j  , IU) = qpX[IU];
      Qm_y(i  ,j  , IU) = qmY[IU];
      Qp_y(i  ,j  , IU) = qpY[IU];
      
      Qm_x(i  ,j  , IV) = qmX[IV];
      Qp_x(i  ,j  , IV) = qpX[IV];
      Qm_y(i  ,j  , IV) = qmY[IV];
      Qp_y(i  ,j  , IV) = qpY[IV];
      
    }
  }

  DataArray Udata, Qdata;
  DataArray Qm_x, Qm_y, Qp_x, Qp_y;
  real_t dtdx, dtdy;
  
}; // ComputeTraceFunctor


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAndStoreFluxesFunctor : public HydroBaseFunctor {

public:

  ComputeAndStoreFluxesFunctor(HydroParams params,
			       DataArray Qdata,
			       DataArray FluxData_x,
			       DataArray FluxData_y,		       
			       real_t dtdx,
			       real_t dtdy) :
    HydroBaseFunctor(params),
    Qdata(Qdata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y), 
    dtdx(dtdx),
    dtdy(dtdy) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Qdata,
		    DataArray FluxData_x,
		    DataArray FluxData_y,		       
		    real_t dtdx,
		    real_t dtdy)
  {
    const int ijsize = params.isize*params.jsize;
    ComputeAndStoreFluxesFunctor functor(params, Qdata,
					 FluxData_x, FluxData_y,
					 dtdx, dtdy);
    Kokkos::parallel_for(ijsize, functor);
    
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {
      
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
      qLoc[ID]         = Qdata(i  ,j  , ID);
      qNeighbors_0[ID] = Qdata(i+1,j  , ID);
      qNeighbors_1[ID] = Qdata(i-1,j  , ID);
      qNeighbors_2[ID] = Qdata(i  ,j+1, ID);
      qNeighbors_3[ID] = Qdata(i  ,j-1, ID);
      
      qLoc[IP]         = Qdata(i  ,j  , IP);
      qNeighbors_0[IP] = Qdata(i+1,j  , IP);
      qNeighbors_1[IP] = Qdata(i-1,j  , IP);
      qNeighbors_2[IP] = Qdata(i  ,j+1, IP);
      qNeighbors_3[IP] = Qdata(i  ,j-1, IP);
      
      qLoc[IU]         = Qdata(i  ,j  , IU);
      qNeighbors_0[IU] = Qdata(i+1,j  , IU);
      qNeighbors_1[IU] = Qdata(i-1,j  , IU);
      qNeighbors_2[IU] = Qdata(i  ,j+1, IU);
      qNeighbors_3[IU] = Qdata(i  ,j-1, IU);
      
      qLoc[IV]         = Qdata(i  ,j  , IV);
      qNeighbors_0[IV] = Qdata(i+1,j  , IV);
      qNeighbors_1[IV] = Qdata(i-1,j  , IV);
      qNeighbors_2[IV] = Qdata(i  ,j+1, IV);
      qNeighbors_3[IV] = Qdata(i  ,j-1, IV);
      
      slope_unsplit_hydro_2d(qLoc, 
			     qNeighbors_0, qNeighbors_1, 
			     qNeighbors_2, qNeighbors_3,
			     dqX, dqY);
	
      // slopes at left neighbor along X      
      qLocNeighbor[ID] = Qdata(i-1,j  , ID);
      qNeighbors_0[ID] = Qdata(i  ,j  , ID);
      qNeighbors_1[ID] = Qdata(i-2,j  , ID);
      qNeighbors_2[ID] = Qdata(i-1,j+1, ID);
      qNeighbors_3[ID] = Qdata(i-1,j-1, ID);
      
      qLocNeighbor[IP] = Qdata(i-1,j  , IP);
      qNeighbors_0[IP] = Qdata(i  ,j  , IP);
      qNeighbors_1[IP] = Qdata(i-2,j  , IP);
      qNeighbors_2[IP] = Qdata(i-1,j+1, IP);
      qNeighbors_3[IP] = Qdata(i-1,j-1, IP);
      
      qLocNeighbor[IU] = Qdata(i-1,j  , IU);
      qNeighbors_0[IU] = Qdata(i  ,j  , IU);
      qNeighbors_1[IU] = Qdata(i-2,j  , IU);
      qNeighbors_2[IU] = Qdata(i-1,j+1, IU);
      qNeighbors_3[IU] = Qdata(i-1,j-1, IU);
      
      qLocNeighbor[IV] = Qdata(i-1,j  , IV);
      qNeighbors_0[IV] = Qdata(i  ,j  , IV);
      qNeighbors_1[IV] = Qdata(i-2,j  , IV);
      qNeighbors_2[IV] = Qdata(i-1,j+1, IV);
      qNeighbors_3[IV] = Qdata(i-1,j-1, IV);
      
      slope_unsplit_hydro_2d(qLocNeighbor, 
			     qNeighbors_0, qNeighbors_1, 
			     qNeighbors_2, qNeighbors_3,
			     dqX_neighbor, dqY_neighbor);
      
      //
      // compute reconstructed states at left interface along X
      //
      
      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc,
				 dqX, dqY,
				 dtdx, dtdy, FACE_XMIN, qright);
      
      // left interface : left state
      trace_unsplit_2d_along_dir(qLocNeighbor,
				 dqX_neighbor,dqY_neighbor,
				 dtdx, dtdy, FACE_XMAX, qleft);
      
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //riemann_2d(qleft,qright,&qgdnv,&flux_x);
      riemann_hllc(qleft,qright,qgdnv,flux_x);
	
      //
      // store fluxes X
      //
      FluxData_x(i  ,j  , ID) = flux_x[ID] * dtdx;
      FluxData_x(i  ,j  , IP) = flux_x[IP] * dtdx;
      FluxData_x(i  ,j  , IU) = flux_x[IU] * dtdx;
      FluxData_x(i  ,j  , IV) = flux_x[IV] * dtdx;
      
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      qLocNeighbor[ID] = Qdata(i  ,j-1, ID);
      qNeighbors_0[ID] = Qdata(i+1,j-1, ID);
      qNeighbors_1[ID] = Qdata(i-1,j-1, ID);
      qNeighbors_2[ID] = Qdata(i  ,j  , ID);
      qNeighbors_3[ID] = Qdata(i  ,j-2, ID);
      
      qLocNeighbor[IP] = Qdata(i  ,j-1, IP);
      qNeighbors_0[IP] = Qdata(i+1,j-1, IP);
      qNeighbors_1[IP] = Qdata(i-1,j-1, IP);
      qNeighbors_2[IP] = Qdata(i  ,j  , IP);
      qNeighbors_3[IP] = Qdata(i  ,j-2, IP);
      
      qLocNeighbor[IU] = Qdata(i  ,j-1, IU);
      qNeighbors_0[IU] = Qdata(i+1,j-1, IU);
      qNeighbors_1[IU] = Qdata(i-1,j-1, IU);
      qNeighbors_2[IU] = Qdata(i  ,j  , IU);
      qNeighbors_3[IU] = Qdata(i  ,j-2, IU);
      
      qLocNeighbor[IV] = Qdata(i  ,j-1, IV);
      qNeighbors_0[IV] = Qdata(i+1,j-1, IV);
      qNeighbors_1[IV] = Qdata(i-1,j-1, IV);
      qNeighbors_2[IV] = Qdata(i  ,j  , IV);
      qNeighbors_3[IV] = Qdata(i  ,j-2, IV);
	
      slope_unsplit_hydro_2d(qLocNeighbor, 
			     qNeighbors_0, qNeighbors_1, 
			     qNeighbors_2, qNeighbors_3,
			     dqX_neighbor, dqY_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //
	
      // left interface : right state
      trace_unsplit_2d_along_dir(qLoc,
				 dqX, dqY,
				 dtdx, dtdy, FACE_YMIN, qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(qLocNeighbor,
				 dqX_neighbor,dqY_neighbor,
				 dtdx, dtdy, FACE_YMAX, qleft);

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft[IU]) ,&(qleft[IV]) );
      swapValues(&(qright[IU]),&(qright[IV]));
      //riemann_2d(qleft,qright,qgdnv,flux_y);
      riemann_hllc(qleft,qright,qgdnv,flux_y);

      //
      // store fluxes Y
      //
      FluxData_y(i  ,j  , ID) = flux_y[ID] * dtdy;
      FluxData_y(i  ,j  , IP) = flux_y[IP] * dtdy;
      FluxData_y(i  ,j  , IU) = flux_y[IU] * dtdy;
      FluxData_y(i  ,j  , IV) = flux_y[IV] * dtdy;
          
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray FluxData_x;
  DataArray FluxData_y;
  real_t dtdx, dtdy;
  
}; // ComputeAndStoreFluxesFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor : public HydroBaseFunctor {

public:

  UpdateFunctor(HydroParams params,
		DataArray Udata,
		DataArray FluxData_x,
		DataArray FluxData_y) :
    HydroBaseFunctor(params),
    Udata(Udata), 
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata,
		    DataArray FluxData_x,
		    DataArray FluxData_y)
  {
    const int ijsize = params.isize*params.jsize;
    UpdateFunctor functor(params, Udata,
			  FluxData_x, FluxData_y);
    Kokkos::parallel_for(ijsize, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      Udata(i  ,j  , ID) +=  FluxData_x(i  ,j  , ID);
      Udata(i  ,j  , IP) +=  FluxData_x(i  ,j  , IP);
      Udata(i  ,j  , IU) +=  FluxData_x(i  ,j  , IU);
      Udata(i  ,j  , IV) +=  FluxData_x(i  ,j  , IV);

      Udata(i  ,j  , ID) -=  FluxData_x(i+1,j  , ID);
      Udata(i  ,j  , IP) -=  FluxData_x(i+1,j  , IP);
      Udata(i  ,j  , IU) -=  FluxData_x(i+1,j  , IU);
      Udata(i  ,j  , IV) -=  FluxData_x(i+1,j  , IV);
      
      Udata(i  ,j  , ID) +=  FluxData_y(i  ,j  , ID);
      Udata(i  ,j  , IP) +=  FluxData_y(i  ,j  , IP);
      Udata(i  ,j  , IU) +=  FluxData_y(i  ,j  , IV); //
      Udata(i  ,j  , IV) +=  FluxData_y(i  ,j  , IU); //
      
      Udata(i  ,j  , ID) -=  FluxData_y(i  ,j+1, ID);
      Udata(i  ,j  , IP) -=  FluxData_y(i  ,j+1, IP);
      Udata(i  ,j  , IU) -=  FluxData_y(i  ,j+1, IV); //
      Udata(i  ,j  , IV) -=  FluxData_y(i  ,j+1, IU); //

    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData_x;
  DataArray FluxData_y;
  
}; // UpdateFunctor


/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class UpdateDirFunctor : public HydroBaseFunctor {

public:

  UpdateDirFunctor(HydroParams params,
		   DataArray Udata,
		   DataArray FluxData) :
    HydroBaseFunctor(params),
    Udata(Udata), 
    FluxData(FluxData) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata,
		    DataArray FluxData)
  {
    const int ijsize = params.isize*params.jsize;
    UpdateDirFunctor<dir> functor(params, Udata, FluxData);
    Kokkos::parallel_for(ijsize, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      if (dir == XDIR) {

	Udata(i  ,j  , ID) +=  FluxData(i  ,j  , ID);
	Udata(i  ,j  , IP) +=  FluxData(i  ,j  , IP);
	Udata(i  ,j  , IU) +=  FluxData(i  ,j  , IU);
	Udata(i  ,j  , IV) +=  FluxData(i  ,j  , IV);

	Udata(i  ,j  , ID) -=  FluxData(i+1,j  , ID);
	Udata(i  ,j  , IP) -=  FluxData(i+1,j  , IP);
	Udata(i  ,j  , IU) -=  FluxData(i+1,j  , IU);
	Udata(i  ,j  , IV) -=  FluxData(i+1,j  , IV);

      } else if (dir == YDIR) {

	Udata(i  ,j  , ID) +=  FluxData(i  ,j  , ID);
	Udata(i  ,j  , IP) +=  FluxData(i  ,j  , IP);
	Udata(i  ,j  , IU) +=  FluxData(i  ,j  , IU);
	Udata(i  ,j  , IV) +=  FluxData(i  ,j  , IV);
	
	Udata(i  ,j  , ID) -=  FluxData(i  ,j+1, ID);
	Udata(i  ,j  , IP) -=  FluxData(i  ,j+1, IP);
	Udata(i  ,j  , IU) -=  FluxData(i  ,j+1, IU);
	Udata(i  ,j  , IV) -=  FluxData(i  ,j+1, IV);

      }
      
    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData;
  
}; // UpdateDirFunctor

    
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor : public HydroBaseFunctor {
  
public:
  
  ComputeSlopesFunctor(HydroParams params,
		       DataArray Qdata,
		       DataArray Slopes_x,
		       DataArray Slopes_y) :
    HydroBaseFunctor(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Qdata,
		    DataArray Slopes_x,
		    DataArray Slopes_y)
  {
    const int ijsize = params.isize*params.jsize;
    ComputeSlopesFunctor functor(params, Qdata, Slopes_x, Slopes_x);
    Kokkos::parallel_for(ijsize, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth-1 && j <= jsize-ghostWidth  &&
       i >= ghostWidth-1 && i <= isize-ghostWidth ) {

      	// local primitive variables
	HydroState qLoc; // local primitive variables

	// local primitive variables in neighborbood
	HydroState qNeighbors_0;
	HydroState qNeighbors_1;
	HydroState qNeighbors_2;
	HydroState qNeighbors_3;

	// Local slopes and neighbor slopes
	HydroState dqX;
	HydroState dqY;
      
	// get primitive variables state vector
	qLoc[ID]         = Qdata(i  ,j  , ID);
	qNeighbors_0[ID] = Qdata(i+1,j  , ID);
	qNeighbors_1[ID] = Qdata(i-1,j  , ID);
	qNeighbors_2[ID] = Qdata(i  ,j+1, ID);
	qNeighbors_3[ID] = Qdata(i  ,j-1, ID);

	qLoc[IP]         = Qdata(i  ,j  , IP);
	qNeighbors_0[IP] = Qdata(i+1,j  , IP);
	qNeighbors_1[IP] = Qdata(i-1,j  , IP);
	qNeighbors_2[IP] = Qdata(i  ,j+1, IP);
	qNeighbors_3[IP] = Qdata(i  ,j-1, IP);
	
	qLoc[IU]         = Qdata(i  ,j  , IU);
	qNeighbors_0[IU] = Qdata(i+1,j  , IU);
	qNeighbors_1[IU] = Qdata(i-1,j  , IU);
	qNeighbors_2[IU] = Qdata(i  ,j+1, IU);
	qNeighbors_3[IU] = Qdata(i  ,j-1, IU);
	
	qLoc[IV]         = Qdata(i  ,j  , IV);
	qNeighbors_0[IV] = Qdata(i+1,j  , IV);
	qNeighbors_1[IV] = Qdata(i-1,j  , IV);
	qNeighbors_2[IV] = Qdata(i  ,j+1, IV);
	qNeighbors_3[IV] = Qdata(i  ,j-1, IV);
	
	slope_unsplit_hydro_2d(qLoc, 
			       qNeighbors_0, qNeighbors_1, 
			       qNeighbors_2, qNeighbors_3,
			       dqX, dqY);
	
	// copy back slopes in global arrays
	Slopes_x(i  ,j, ID) = dqX[ID];
	Slopes_y(i  ,j, ID) = dqY[ID];
	
	Slopes_x(i  ,j, IP) = dqX[IP];
	Slopes_y(i  ,j, IP) = dqY[IP];
	
	Slopes_x(i  ,j, IU) = dqX[IU];
	Slopes_y(i  ,j, IU) = dqY[IU];
	
	Slopes_x(i  ,j, IV) = dqX[IV];
	Slopes_y(i  ,j, IV) = dqY[IV];
      
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray Slopes_x, Slopes_y;
  
}; // ComputeSlopesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor : public HydroBaseFunctor {
  
public:
  
  ComputeTraceAndFluxes_Functor(HydroParams params,
				DataArray Qdata,
				DataArray Slopes_x,
				DataArray Slopes_y,
				DataArray Fluxes,
				real_t    dtdx,
				real_t    dtdy) :
    HydroBaseFunctor(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y),
    Fluxes(Fluxes),
    dtdx(dtdx), dtdy(dtdy) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Qdata,
		    DataArray Slopes_x,
		    DataArray Slopes_y,
		    DataArray Fluxes,
		    real_t    dtdx,
		    real_t    dtdy)
  {
    const int ijsize = params.isize*params.jsize;
    ComputeTraceAndFluxes_Functor<dir> functor(params, Qdata, Slopes_x, Slopes_x, Fluxes,
					       dtdx, dtdy);
    Kokkos::parallel_for(ijsize, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {

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
	qLoc[ID] = Qdata   (i  ,j, ID);
	dqX[ID]  = Slopes_x(i  ,j, ID);
	dqY[ID]  = Slopes_y(i  ,j, ID);
	
	qLoc[IP] = Qdata   (i  ,j, IP);
	dqX[IP]  = Slopes_x(i  ,j, IP);
	dqY[IP]  = Slopes_y(i  ,j, IP);
	
	qLoc[IU] = Qdata   (i  ,j, IU);
	dqX[IU]  = Slopes_x(i  ,j, IU);
	dqY[IU]  = Slopes_y(i  ,j, IU);
	
	qLoc[IV] = Qdata   (i  ,j, IV);
	dqX[IV]  = Slopes_x(i  ,j, IV);
	dqY[IV]  = Slopes_y(i  ,j, IV);

	if (dir == XDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(qLoc,
				     dqX, dqY,
				     dtdx, dtdy, FACE_XMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (i-1,j  , ID);
	  dqX_neighbor[ID] = Slopes_x(i-1,j  , ID);
	  dqY_neighbor[ID] = Slopes_y(i-1,j  , ID);
	  
	  qLocNeighbor[IP] = Qdata   (i-1,j  , IP);
	  dqX_neighbor[IP] = Slopes_x(i-1,j  , IP);
	  dqY_neighbor[IP] = Slopes_y(i-1,j  , IP);
	  
	  qLocNeighbor[IU] = Qdata   (i-1,j  , IU);
	  dqX_neighbor[IU] = Slopes_x(i-1,j  , IU);
	  dqY_neighbor[IU] = Slopes_y(i-1,j  , IU);
	  
	  qLocNeighbor[IV] = Qdata   (i-1,j  , IV);
	  dqX_neighbor[IV] = Slopes_x(i-1,j  , IV);
	  dqY_neighbor[IV] = Slopes_y(i-1,j  , IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,
				     dtdx, dtdy, FACE_XMAX, qleft);
	  
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann_hllc(qleft,qright,qgdnv,flux);

	  //
	  // store fluxes
	  //	
	  Fluxes(i  ,j , ID) =  flux[ID]*dtdx;
	  Fluxes(i  ,j , IP) =  flux[IP]*dtdx;
	  Fluxes(i  ,j , IU) =  flux[IU]*dtdx;
	  Fluxes(i  ,j , IV) =  flux[IV]*dtdx;

	} else if (dir == YDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(qLoc,
				     dqX, dqY,
				     dtdx, dtdy, FACE_YMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (i  ,j-1, ID);
	  dqX_neighbor[ID] = Slopes_x(i  ,j-1, ID);
	  dqY_neighbor[ID] = Slopes_y(i  ,j-1, ID);
	  
	  qLocNeighbor[IP] = Qdata   (i  ,j-1, IP);
	  dqX_neighbor[IP] = Slopes_x(i  ,j-1, IP);
	  dqY_neighbor[IP] = Slopes_y(i  ,j-1, IP);
	  
	  qLocNeighbor[IU] = Qdata   (i  ,j-1, IU);
	  dqX_neighbor[IU] = Slopes_x(i  ,j-1, IU);
	  dqY_neighbor[IU] = Slopes_y(i  ,j-1, IU);
	  
	  qLocNeighbor[IV] = Qdata   (i  ,j-1, IV);
	  dqX_neighbor[IV] = Slopes_x(i  ,j-1, IV);
	  dqY_neighbor[IV] = Slopes_y(i  ,j-1, IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,
				     dtdx, dtdy, FACE_YMAX, qleft);
	  
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  swapValues(&(qleft[IU]) ,&(qleft[IV]) );
	  swapValues(&(qright[IU]),&(qright[IV]));
	  riemann_hllc(qleft,qright,qgdnv,flux);
	  
	  //
	  // update hydro array
	  //	  
	  Fluxes(i  ,j  , ID) =  flux[ID]*dtdy;
	  Fluxes(i  ,j  , IP) =  flux[IP]*dtdy;
	  Fluxes(i  ,j  , IU) =  flux[IV]*dtdy; // IU/IV swapped
	  Fluxes(i  ,j  , IV) =  flux[IU]*dtdy; // IU/IV swapped

	}
	      
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray Slopes_x, Slopes_y;
  DataArray Fluxes;
  real_t dtdx, dtdy;
  
}; // ComputeTraceAndFluxes_Functor
    
/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor : public HydroBaseFunctor {

public:
  InitImplodeFunctor(HydroParams params,
		     DataArray Udata) :
    HydroBaseFunctor(params), Udata(Udata)  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata)
  {
    const int ijsize = params.isize*params.jsize;
    InitImplodeFunctor functor(params, Udata);
    Kokkos::parallel_for(ijsize, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    
    real_t tmp = x+y*y;
    if (tmp > 0.5 && tmp < 1.5) {
      Udata(i  ,j  , ID) = 1.0;
      Udata(i  ,j  , IP) = 1.0/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
    } else {
      Udata(i  ,j  , ID) = 0.125;
      Udata(i  ,j  , IP) = 0.14/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
    }
    
  } // end operator ()

  DataArray Udata;

}; // InitImplodeFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor : public HydroBaseFunctor {

public:
  InitBlastFunctor(HydroParams params,
		   DataArray Udata) :
    HydroBaseFunctor(params), Udata(Udata)  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata)
  {
    const int ijsize = params.isize*params.jsize;
    InitBlastFunctor functor(params, Udata);
    Kokkos::parallel_for(ijsize, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius      = params.blast_radius;
    const real_t radius2           = blast_radius*blast_radius;
    const real_t blast_center_x    = params.blast_center_x;
    const real_t blast_center_y    = params.blast_center_y;
    const real_t blast_density_in  = params.blast_density_in;
    const real_t blast_density_out = params.blast_density_out;
    const real_t blast_pressure_in = params.blast_pressure_in;
    const real_t blast_pressure_out= params.blast_pressure_out;
  

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;

    real_t d2 = 
      (x-blast_center_x)*(x-blast_center_x)+
      (y-blast_center_y)*(y-blast_center_y);    
    
    if (d2 < radius2) {
      Udata(i  ,j  , ID) = blast_density_in;
      Udata(i  ,j  , IP) = blast_pressure_in/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
    } else {
      Udata(i  ,j  , ID) = blast_density_out;
      Udata(i  ,j  , IP) = blast_pressure_out/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
    }
    
  } // end operator ()
  
  DataArray Udata;
  
}; // InitBlastFunctor
  

/*************************************************/
/*************************************************/
/*************************************************/
template <FaceIdType faceId>
class MakeBoundariesFunctor : public HydroBaseFunctor {
  
public:
  
  MakeBoundariesFunctor(HydroParams params,
			DataArray Udata) :
    HydroBaseFunctor(params), Udata(Udata)  {};
  

  // static method which does it all: create and execute functor
  static void apply(HydroParams params,
		    DataArray Udata)
  {
    int nbIter = params.ghostWidth*std::max(params.isize,
					    params.jsize);
    
    MakeBoundariesFunctor<faceId> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int nx = params.nx;
    const int ny = params.ny;
    
    //const int isize = params.isize;
    //const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    const int imin = params.imin;
    const int imax = params.imax;
    
    const int jmin = params.jmin;
    const int jmax = params.jmax;
    
    int i,j;

    int i0, j0;
    int iVar;

    if (faceId == FACE_XMIN) {
      
      // boundary xmin
      int boundary_type = params.boundary_type_xmin;

      j = index / ghostWidth;
      i = index - j*ghostWidth;
      
      if(j >= jmin && j <= jmax    &&
	 i >= 0    && i <ghostWidth) {
	
	real_t sign=1.0;
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*ghostWidth-1-i;
	    if (iVar==IU) sign=-ONE_F;
	  } else if( boundary_type == BC_NEUMANN ) {
	    i0=ghostWidth;
	  } else { // periodic
	    i0=nx+i;
	  }
	  
	  Udata(i  ,j  , iVar) = Udata(i0  ,j  , iVar)*sign;
	  
	}
	
      }
    }

    if (faceId == FACE_XMAX) {
      
      // boundary xmax
      int boundary_type = params.boundary_type_xmax;

      j = index / ghostWidth;
      i = index - j*ghostWidth;
      i += (nx+ghostWidth);

      if(j >= jmin          && j <= jmax             &&
	 i >= nx+ghostWidth && i <= nx+2*ghostWidth-1) {
	
	real_t sign=1.0;
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*nx+2*ghostWidth-1-i;
	    if (iVar==IU) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    i0=nx+ghostWidth-1;
	  } else { // periodic
	    i0=i-nx;
	  }
	  
	  Udata(i  ,j  , iVar) = Udata(i0 ,j  , iVar)*sign;
	  
	}
      }
    }
    
    if (faceId == FACE_YMIN) {
      
      // boundary ymin
      int boundary_type = params.boundary_type_ymin;

      i = index / ghostWidth;
      j = index - i*ghostWidth;

      if(i >= imin && i <= imax    &&
	 j >= 0    && j <ghostWidth) {
	
	real_t sign=1.0;
	
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ghostWidth-1-j;
	    if (iVar==IV) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ghostWidth;
	  } else { // periodic
	    j0=ny+j;
	  }
	  
	  Udata(i  ,j  , iVar) = Udata(i  ,j0 , iVar)*sign;
	}
      }
    }

    if (faceId == FACE_YMAX) {

      // boundary ymax
      int boundary_type = params.boundary_type_ymax;

      i = index / ghostWidth;
      j = index - i*ghostWidth;
      j += (ny+ghostWidth);
      if(i >= imin          && i <= imax              &&
	 j >= ny+ghostWidth && j <= ny+2*ghostWidth-1) {
	
	real_t sign=1.0;
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ny+2*ghostWidth-1-j;
	    if (iVar==IV) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ny+ghostWidth-1;
	  } else { // periodic
	    j0=j-ny;
	  }
	  
	  Udata(i  ,j  , iVar) = Udata(i  ,j0  , iVar)*sign;
	  
	}

      }
    }
    
  } // end operator ()

  DataArray Udata;
  
}; // MakeBoundariesFunctor
  
#endif // HYDRO_RUN_FUNCTORS_H_

