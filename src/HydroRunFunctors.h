#ifndef HYDRO_RUN_FUNCTORS_H_
#define HYDRO_RUN_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__


#include "HydroBaseFunctor.h"

/**
 * small utility to wrap selection between team and range policy
 */
template<class Functor>
static void launch_functor(HydroParams params, Functor functor)
{
  
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  
  // loop over i used for "vectorization"
  // loop over j used for team/thread parallelism
  
  Kokkos::parallel_for(
    team_policy_t(params.nbTeams, 
                  Kokkos::AUTO, /* team size chosen by kokkos */
                  team_policy_t::vector_length_max()),
    functor);  
  
} // end launch_functor

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
      uLoc[ID] = Udata(index,ID);
      uLoc[IP] = Udata(index,IP);
      uLoc[IU] = Udata(index,IU);
      uLoc[IV] = Udata(index,IV);

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

    // create functor
    ConvertToPrimitivesFunctor functor(params, Udata, Qdata);

    launch_functor(params,functor);

  } // end apply

  /** 
   * entry point for team policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& thread) const
  {

    // the teams league must distribute last dimension (here j) into
    // nbTeams chuncks
    // so compute chunck size per team (rounded up)
    int chunck_size_y = (params.jsize+params.nbTeams-1)/params.nbTeams;
    
    int chunk_size_per_team = chunck_size_y;

    // team id
    int teamId = thread.league_rank();
    
    // compute j start
    int jStart = teamId * chunck_size_y;

    auto U_ID = Kokkos::subview(Udata, Kokkos::ALL(), 0);
    auto U_IP = Kokkos::subview(Udata, Kokkos::ALL(), 1);
    auto U_IU = Kokkos::subview(Udata, Kokkos::ALL(), 2);
    auto U_IV = Kokkos::subview(Udata, Kokkos::ALL(), 3);
    
    auto Q_ID = Kokkos::subview(Qdata, Kokkos::ALL(), 0);
    auto Q_IP = Kokkos::subview(Qdata, Kokkos::ALL(), 1);
    auto Q_IU = Kokkos::subview(Qdata, Kokkos::ALL(), 2);
    auto Q_IV = Kokkos::subview(Qdata, Kokkos::ALL(), 3);
    
    // spread work among the thread in the team
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, chunk_size_per_team), 
      [=](const int &index) {

        // index goes from 0 to chunck_size_ter_team
        // re-compute j from offset + index 
        int j =  jStart + index;

        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(thread, params.isize),
          [=](const int &i) {
            
            //do_compute(INDEX(i,j));

	    const int isize = params.isize;
	    const int jsize = params.jsize;
	    //const int ghostWidth = params.ghostWidth;
	    
	    int ij = coord2index(i,j,isize,jsize);
	    
	    if(j >= 0 && j < jsize  &&
	       i >= 0 && i < isize ) {
	      
	      HydroState uLoc; // conservative    variables in current cell
	      HydroState qLoc; // primitive    variables in current cell
	      real_t c;
	      
	      // get local conservative variable
	      /*uLoc[ID] = Udata(ij,ID);
	      uLoc[IP] = Udata(ij,IP);
	      uLoc[IU] = Udata(ij,IU);
	      uLoc[IV] = Udata(ij,IV);*/
	      uLoc[ID] = U_ID(ij);
	      uLoc[IP] = U_IP(ij);
	      uLoc[IU] = U_IU(ij);
	      uLoc[IV] = U_IV(ij);
	      
	      // get primitive variables in current cell
	      computePrimitives(uLoc, &c, qLoc);
	      
	      // copy q state in q global
	      /*Qdata(ij,ID) = qLoc[ID];
	      Qdata(ij,IP) = qLoc[IP];
	      Qdata(ij,IU) = qLoc[IU];
	      Qdata(ij,IV) = qLoc[IV];*/
	      Q_ID(ij) = qLoc[ID];
	      Q_IP(ij) = qLoc[IP];
	      Q_IU(ij) = qLoc[IU];
	      Q_IV(ij) = qLoc[IV];
	      
	    }

          }); // end vector range
      });
    
  } // end team policy functor

  DataArray Udata;
  DataArray Qdata;
    
}; // ConvertToPrimitivesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/* NOT USED CURRENTLY */
// class ComputeFluxesAndUpdateFunctor : public HydroBaseFunctor {

// public:

//   ComputeFluxesAndUpdateFunctor(HydroParams params,
// 				DataArray Udata,
// 				DataArray Qm_x,
// 				DataArray Qm_y,
// 				DataArray Qp_x,
// 				DataArray Qp_y,
// 				real_t dtdx,
// 				real_t dtdy) :
//     HydroBaseFunctor(params), Udata(Udata),
//     Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
//     dtdx(dtdx), dtdy(dtdy) {};

//   // static method which does it all: create and execute functor
//   static void apply(HydroParams params,
// 		    DataArray Udata,
// 		    DataArray Qm_x,
// 		    DataArray Qm_y,
// 		    DataArray Qp_x,
// 		    DataArray Qp_y,
// 		    real_t dtdx,
// 		    real_t dtdy)
//   {
//     ComputeFluxesAndUpdateFunctor functor(params, Udata,
//                                           Qm_x, Qm_y,
//                                           Qp_x, Qp_y,
//                                           dtdx, dtdy);

//     launch_functor(params,functor);
    
//   } // end apply

//   /** 
//    * entry point for team policy
//    */
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const thread_t& thread) const
//   {

//     // the teams league must distribute last dimension (here j) into
//     // nbTeams chuncks
//     // so compute chunck size per team (rounded up)
//     int chunck_size_y = (params.jsize+params.nbTeams-1)/params.nbTeams;
    
//     int chunk_size_per_team = chunck_size_y;

//     // team id
//     int teamId = thread.league_rank();
    
//     // compute j start
//     int jStart = teamId * chunck_size_y; 

//     // spread work among the thread in the team
//     Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(thread, chunk_size_per_team), 
//       [=](const int &index) {

//         // index goes from 0 to chunck_size_ter_team
//         // re-compute j from offset + index 
//         int j =  jStart + index;

//         Kokkos::parallel_for(
//           Kokkos::ThreadVectorRange(thread, params.isize),
//           [=](const int &i) {
            
//             do_compute(i,j);

//           }); // end vector range
//       });
    
//   } // end team policy functor

//   /** 
//    * entry point for range policy
//    */
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const int& index_) const
//   {
//     const int isize = params.isize;
//     const int jsize = params.jsize;

//     int i,j;
//     index2coord(index_,i,j,isize,jsize);
    
//     do_compute(i,j);

//   } // end range policy functor

//   /**
//    * Actual computation.
//    */
//   KOKKOS_INLINE_FUNCTION
//   void do_compute(const int& i, const int& j) const
//   {

//     const int isize = params.isize;
//     const int jsize = params.jsize;
//     const int ghostWidth = params.ghostWidth;

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
    
//   } // end do_compute
  
//   DataArray Udata;
//   DataArray Qm_x, Qm_y, Qp_x, Qp_y;
//   real_t dtdx, dtdy;
  
// }; // ComputeFluxesAndUpdateFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/* NOT USED CURRENTLY */
// class ComputeTraceFunctor : public HydroBaseFunctor {

// public:

//   ComputeTraceFunctor(HydroParams params,
// 		      DataArray Udata,
// 		      DataArray Qdata,
// 		      DataArray Qm_x,
// 		      DataArray Qm_y,
// 		      DataArray Qp_x,
// 		      DataArray Qp_y,
// 		      real_t dtdx,
// 		      real_t dtdy) :
//     HydroBaseFunctor(params),
//     Udata(Udata), Qdata(Qdata),
//     Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
//     dtdx(dtdx), dtdy(dtdy) {};
  
//   // static method which does it all: create and execute functor
//   static void apply(HydroParams params,
// 		    DataArray Udata,
// 		    DataArray Qdata,
// 		    DataArray Qm_x,
// 		    DataArray Qm_y,
// 		    DataArray Qp_x,
// 		    DataArray Qp_y,
// 		    real_t dtdx,
// 		    real_t dtdy)
//   {

//     const int ijsize = params.isize*params.jsize;
//     ComputeTraceFunctor computeTraceFunctor(params, Udata, Qdata,
// 					    Qm_x, Qm_y,
// 					    Qp_x, Qp_y,
// 					    dtdx, dtdy);
//     Kokkos::parallel_for(ijsize, computeTraceFunctor);
    
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
//       Qp_x(i  ,j  , ID) = qpX[IP];
//       Qm_y(i  ,j  , ID) = qmY[IP];
//       Qp_y(i  ,j  , ID) = qpY[IP];
      
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

//   DataArray Udata, Qdata;
//   DataArray Qm_x, Qm_y, Qp_x, Qp_y;
//   real_t dtdx, dtdy;
  
// }; // ComputeTraceFunctor


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
    ComputeAndStoreFluxesFunctor functor(params, Qdata,
					 FluxData_x, FluxData_y,
					 dtdx, dtdy);

    launch_functor(params,functor);

  } // end apply

  /** 
   * entry point for team policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& thread) const
  {

    // the teams league must distribute last dimension (here j) into
    // nbTeams chuncks
    // so compute chunck size per team (rounded up)
    int chunck_size_y = (params.jsize+params.nbTeams-1)/params.nbTeams;
    
    int chunk_size_per_team = chunck_size_y;

    // team id
    int teamId = thread.league_rank();
    
    // compute j start
    int jStart = teamId * chunck_size_y; 

    auto Q_ID = Kokkos::subview(Qdata, Kokkos::ALL(), 0);
    auto Q_IP = Kokkos::subview(Qdata, Kokkos::ALL(), 1);
    auto Q_IU = Kokkos::subview(Qdata, Kokkos::ALL(), 2);
    auto Q_IV = Kokkos::subview(Qdata, Kokkos::ALL(), 3);

    auto Flux_x_ID = Kokkos::subview(FluxData_x, Kokkos::ALL(), 0);
    auto Flux_x_IP = Kokkos::subview(FluxData_x, Kokkos::ALL(), 1);
    auto Flux_x_IU = Kokkos::subview(FluxData_x, Kokkos::ALL(), 2);
    auto Flux_x_IV = Kokkos::subview(FluxData_x, Kokkos::ALL(), 3);

    auto Flux_y_ID = Kokkos::subview(FluxData_y, Kokkos::ALL(), 0);
    auto Flux_y_IP = Kokkos::subview(FluxData_y, Kokkos::ALL(), 1);
    auto Flux_y_IU = Kokkos::subview(FluxData_y, Kokkos::ALL(), 2);
    auto Flux_y_IV = Kokkos::subview(FluxData_y, Kokkos::ALL(), 3);

    // spread work among the thread in the team
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, chunk_size_per_team), 
      [=](const int &index) {

        // index goes from 0 to chunck_size_ter_team
        // re-compute j from offset + index 
        int j =  jStart + index;

        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(thread, params.isize),
          [=](const int &i) {

            //do_compute(INDEX(i,j));
            const int isize = params.isize;
            const int jsize = params.jsize;
            const int ghostWidth = params.ghostWidth;    

            const int di = 1;
            const int dj = isize;

	    int ij = coord2index(i,j,isize,jsize);
            
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
              qLoc[ID]         = Q_ID(ij);
              qNeighbors_0[ID] = Q_ID(ij+di);
              qNeighbors_1[ID] = Q_ID(ij-di);
              qNeighbors_2[ID] = Q_ID(ij+dj);
              qNeighbors_3[ID] = Q_ID(ij-dj);
              
              qLoc[IP]         = Q_IP(ij);
              qNeighbors_0[IP] = Q_IP(ij+di);
              qNeighbors_1[IP] = Q_IP(ij-di);
              qNeighbors_2[IP] = Q_IP(ij+dj);
              qNeighbors_3[IP] = Q_IP(ij-dj);
              
              qLoc[IU]         = Q_IU(ij);
              qNeighbors_0[IU] = Q_IU(ij+di);
              qNeighbors_1[IU] = Q_IU(ij-di);
              qNeighbors_2[IU] = Q_IU(ij+dj);
              qNeighbors_3[IU] = Q_IU(ij-dj);
              
              qLoc[IV]         = Q_IV(ij);
              qNeighbors_0[IV] = Q_IV(ij+di);
              qNeighbors_1[IV] = Q_IV(ij-di);
              qNeighbors_2[IV] = Q_IV(ij+dj);
              qNeighbors_3[IV] = Q_IV(ij-dj);
              
              slope_unsplit_hydro_2d(qLoc, 
                                     qNeighbors_0, qNeighbors_1, 
                                     qNeighbors_2, qNeighbors_3,
                                     dqX, dqY);
              
              // slopes at left neighbor along X      
              qLocNeighbor[ID] = Q_ID(ij-  di);
              qNeighbors_0[ID] = Q_ID(ij);
              qNeighbors_1[ID] = Q_ID(ij-2*di);
              qNeighbors_2[ID] = Q_ID(ij-  di+dj);
              qNeighbors_3[ID] = Q_ID(ij-  di-dj);
              
              qLocNeighbor[IP] = Q_IP(ij-di);
              qNeighbors_0[IP] = Q_IP(ij);
              qNeighbors_1[IP] = Q_IP(ij-2*di);
              qNeighbors_2[IP] = Q_IP(ij-  di+dj);
              qNeighbors_3[IP] = Q_IP(ij-  di-dj);

              qLocNeighbor[IU] = Q_IU(ij-di);
              qNeighbors_0[IU] = Q_IU(ij);
              qNeighbors_1[IU] = Q_IU(ij-2*di);
              qNeighbors_2[IU] = Q_IU(ij-  di+dj);
              qNeighbors_3[IU] = Q_IU(ij-  di-dj);

              qLocNeighbor[IV] = Q_IV(ij-di);
              qNeighbors_0[IV] = Q_IV(ij);
              qNeighbors_1[IV] = Q_IV(ij-2*di);
              qNeighbors_2[IV] = Q_IV(ij-  di+dj);
              qNeighbors_3[IV] = Q_IV(ij-  di-dj);
              
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
              Flux_x_ID(ij) = flux_x[ID] * dtdx;
              Flux_x_IP(ij) = flux_x[IP] * dtdx;
              Flux_x_IU(ij) = flux_x[IU] * dtdx;
              Flux_x_IV(ij) = flux_x[IV] * dtdx;
              
              //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              // deal with left interface along Y !
              //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              
              // slopes at left neighbor along Y
              qLocNeighbor[ID] = Q_ID(ij   -  dj);
              qNeighbors_0[ID] = Q_ID(ij+di-  dj);
              qNeighbors_1[ID] = Q_ID(ij-di-  dj);
              qNeighbors_2[ID] = Q_ID(ij        );
              qNeighbors_3[ID] = Q_ID(ij   -2*dj);
              
              qLocNeighbor[IP] = Q_IP(ij   -  dj);
              qNeighbors_0[IP] = Q_IP(ij+di-  dj);
              qNeighbors_1[IP] = Q_IP(ij-di-  dj);
              qNeighbors_2[IP] = Q_IP(ij        );
              qNeighbors_3[IP] = Q_IP(ij   -2*dj);
              
              qLocNeighbor[IU] = Q_IU(ij   -  dj);
              qNeighbors_0[IU] = Q_IU(ij+di-  dj);
              qNeighbors_1[IU] = Q_IU(ij-di-  dj);
              qNeighbors_2[IU] = Q_IU(ij        );
              qNeighbors_3[IU] = Q_IU(ij   -2*dj);
              
              qLocNeighbor[IV] = Q_IV(ij   -  dj);
              qNeighbors_0[IV] = Q_IV(ij+di-  dj);
              qNeighbors_1[IV] = Q_IV(ij-di-  dj);
              qNeighbors_2[IV] = Q_IV(ij        );
              qNeighbors_3[IV] = Q_IV(ij   -2*dj);              
              
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
              Flux_y_ID(ij) = flux_y[ID] * dtdy;
              Flux_y_IP(ij) = flux_y[IP] * dtdy;
              Flux_y_IU(ij) = flux_y[IU] * dtdy;
              Flux_y_IV(ij) = flux_y[IV] * dtdy;
              
            } // end if
            
          }); // end vector range
      });
    
  } // end team policy functor
  
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
    UpdateFunctor functor(params, Udata,
			  FluxData_x, FluxData_y);

    launch_functor(params,functor);

  } // end apply

  /** 
   * entry point for team policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& thread) const
  {

    // the teams league must distribute last dimension (here j) into
    // nbTeams chuncks
    // so compute chunck size per team (rounded up)
    int chunck_size_y = (params.jsize+params.nbTeams-1)/params.nbTeams;
    
    int chunk_size_per_team = chunck_size_y;

    // team id
    int teamId = thread.league_rank();
    
    // compute j start
    int jStart = teamId * chunck_size_y; 

    // spread work among the thread in the team
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, chunk_size_per_team), 
      [=](const int &index) {

        // index goes from 0 to chunck_size_ter_team
        // re-compute j from offset + index 
        int j =  jStart + index;

        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(thread, params.isize),
          [=](const int &i) {
            
            do_compute(INDEX(i,j));

          }); // end vector range
      });
    
  } // end team policy functor

  /** 
   * entry point for range policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    do_compute(index);

  } // end range policy functor

  /**
   * Actual computation.
   */
  KOKKOS_INLINE_FUNCTION
  void do_compute(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;    
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      Udata(INDEX(i  ,j  ), ID) +=  FluxData_x(INDEX(i  ,j  ), ID);
      Udata(INDEX(i  ,j  ), IP) +=  FluxData_x(INDEX(i  ,j  ), IP);
      Udata(INDEX(i  ,j  ), IU) +=  FluxData_x(INDEX(i  ,j  ), IU);
      Udata(INDEX(i  ,j  ), IV) +=  FluxData_x(INDEX(i  ,j  ), IV);

      Udata(INDEX(i  ,j  ), ID) -=  FluxData_x(INDEX(i+1,j  ), ID);
      Udata(INDEX(i  ,j  ), IP) -=  FluxData_x(INDEX(i+1,j  ), IP);
      Udata(INDEX(i  ,j  ), IU) -=  FluxData_x(INDEX(i+1,j  ), IU);
      Udata(INDEX(i  ,j  ), IV) -=  FluxData_x(INDEX(i+1,j  ), IV);
      
      Udata(INDEX(i  ,j  ), ID) +=  FluxData_y(INDEX(i  ,j  ), ID);
      Udata(INDEX(i  ,j  ), IP) +=  FluxData_y(INDEX(i  ,j  ), IP);
      Udata(INDEX(i  ,j  ), IU) +=  FluxData_y(INDEX(i  ,j  ), IV); //
      Udata(INDEX(i  ,j  ), IV) +=  FluxData_y(INDEX(i  ,j  ), IU); //
      
      Udata(INDEX(i  ,j  ), ID) -=  FluxData_y(INDEX(i  ,j+1), ID);
      Udata(INDEX(i  ,j  ), IP) -=  FluxData_y(INDEX(i  ,j+1), IP);
      Udata(INDEX(i  ,j  ), IU) -=  FluxData_y(INDEX(i  ,j+1), IV); //
      Udata(INDEX(i  ,j  ), IV) -=  FluxData_y(INDEX(i  ,j+1), IU); //

    } // end if
    
  } // end do_compute
  
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
    UpdateDirFunctor<dir> functor(params, Udata, FluxData);
    launch_functor(params,functor);
  } // end apply

  /** 
   * entry point for team policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& thread) const
  {

    // the teams league must distribute last dimension (here j) into
    // nbTeams chuncks
    // so compute chunck size per team (rounded up)
    int chunck_size_y = (params.jsize+params.nbTeams-1)/params.nbTeams;
    
    int chunk_size_per_team = chunck_size_y;

    // team id
    int teamId = thread.league_rank();
    
    // compute j start
    int jStart = teamId * chunck_size_y; 

    // spread work among the thread in the team
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, chunk_size_per_team), 
      [=](const int &index) {

        // index goes from 0 to chunck_size_ter_team
        // re-compute j from offset + index 
        int j =  jStart + index;

        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(thread, params.isize),
          [=](const int &i) {
            
            do_compute(INDEX(i,j));

          }); // end vector range
      });
    
  } // end team policy functor

  /** 
   * entry point for range policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    
    do_compute(index);

  } // end range policy functor

  /**
   * Actual computation.
   */
  KOKKOS_INLINE_FUNCTION
  void do_compute(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;    
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      if (dir == XDIR) {

	Udata(INDEX(i  ,j  ), ID) +=  FluxData(INDEX(i  ,j  ), ID);
	Udata(INDEX(i  ,j  ), IP) +=  FluxData(INDEX(i  ,j  ), IP);
	Udata(INDEX(i  ,j  ), IU) +=  FluxData(INDEX(i  ,j  ), IU);
	Udata(INDEX(i  ,j  ), IV) +=  FluxData(INDEX(i  ,j  ), IV);

	Udata(INDEX(i  ,j  ), ID) -=  FluxData(INDEX(i+1,j  ), ID);
	Udata(INDEX(i  ,j  ), IP) -=  FluxData(INDEX(i+1,j  ), IP);
	Udata(INDEX(i  ,j  ), IU) -=  FluxData(INDEX(i+1,j  ), IU);
	Udata(INDEX(i  ,j  ), IV) -=  FluxData(INDEX(i+1,j  ), IV);

      } else if (dir == YDIR) {

	Udata(INDEX(i  ,j  ), ID) +=  FluxData(INDEX(i  ,j  ), ID);
	Udata(INDEX(i  ,j  ), IP) +=  FluxData(INDEX(i  ,j  ), IP);
	Udata(INDEX(i  ,j  ), IU) +=  FluxData(INDEX(i  ,j  ), IU);
	Udata(INDEX(i  ,j  ), IV) +=  FluxData(INDEX(i  ,j  ), IV);
	
	Udata(INDEX(i  ,j  ), ID) -=  FluxData(INDEX(i  ,j+1), ID);
	Udata(INDEX(i  ,j  ), IP) -=  FluxData(INDEX(i  ,j+1), IP);
	Udata(INDEX(i  ,j  ), IU) -=  FluxData(INDEX(i  ,j+1), IU);
	Udata(INDEX(i  ,j  ), IV) -=  FluxData(INDEX(i  ,j+1), IV);

      }
      
    } // end if
    
  } // end do_compute
  
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
    ComputeSlopesFunctor functor(params, Qdata, Slopes_x, Slopes_x);
    launch_functor(params,functor);
  } // end apply

  /** 
   * entry point for team policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& thread) const
  {

    // the teams league must distribute last dimension (here j) into
    // nbTeams chuncks
    // so compute chunck size per team (rounded up)
    int chunck_size_y = (params.jsize+params.nbTeams-1)/params.nbTeams;
    
    int chunk_size_per_team = chunck_size_y;

    // team id
    int teamId = thread.league_rank();
    
    // compute j start
    int jStart = teamId * chunck_size_y; 

    // spread work among the thread in the team
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, chunk_size_per_team), 
      [=](const int &index) {

        // index goes from 0 to chunck_size_ter_team
        // re-compute j from offset + index 
        int j =  jStart + index;

        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(thread, params.isize),
          [=](const int &i) {
            
            do_compute(INDEX(i,j));

          }); // end vector range
      });
    
  } // end team policy functor

  /** 
   * entry point for range policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    do_compute(index);

  } // end range policy functor

  /**
   * Actual computation.
   */
  KOKKOS_INLINE_FUNCTION
  void do_compute(const int& index) const
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
	qLoc[ID]         = Qdata(INDEX(i  ,j  ), ID);
	qNeighbors_0[ID] = Qdata(INDEX(i+1,j  ), ID);
	qNeighbors_1[ID] = Qdata(INDEX(i-1,j  ), ID);
	qNeighbors_2[ID] = Qdata(INDEX(i  ,j+1), ID);
	qNeighbors_3[ID] = Qdata(INDEX(i  ,j-1), ID);

	qLoc[IP]         = Qdata(INDEX(i  ,j  ), IP);
	qNeighbors_0[IP] = Qdata(INDEX(i+1,j  ), IP);
	qNeighbors_1[IP] = Qdata(INDEX(i-1,j  ), IP);
	qNeighbors_2[IP] = Qdata(INDEX(i  ,j+1), IP);
	qNeighbors_3[IP] = Qdata(INDEX(i  ,j-1), IP);
	
	qLoc[IU]         = Qdata(INDEX(i  ,j  ), IU);
	qNeighbors_0[IU] = Qdata(INDEX(i+1,j  ), IU);
	qNeighbors_1[IU] = Qdata(INDEX(i-1,j  ), IU);
	qNeighbors_2[IU] = Qdata(INDEX(i  ,j+1), IU);
	qNeighbors_3[IU] = Qdata(INDEX(i  ,j-1), IU);
	
	qLoc[IV]         = Qdata(INDEX(i  ,j  ), IV);
	qNeighbors_0[IV] = Qdata(INDEX(i+1,j  ), IV);
	qNeighbors_1[IV] = Qdata(INDEX(i-1,j  ), IV);
	qNeighbors_2[IV] = Qdata(INDEX(i  ,j+1), IV);
	qNeighbors_3[IV] = Qdata(INDEX(i  ,j-1), IV);
	
	slope_unsplit_hydro_2d(qLoc, 
			       qNeighbors_0, qNeighbors_1, 
			       qNeighbors_2, qNeighbors_3,
			       dqX, dqY);
	
	// copy back slopes in global arrays
	Slopes_x(INDEX(i  ,j), ID) = dqX[ID];
	Slopes_y(INDEX(i  ,j), ID) = dqY[ID];
	
	Slopes_x(INDEX(i  ,j), IP) = dqX[IP];
	Slopes_y(INDEX(i  ,j), IP) = dqY[IP];
	
	Slopes_x(INDEX(i  ,j), IU) = dqX[IU];
	Slopes_y(INDEX(i  ,j), IU) = dqY[IU];
	
	Slopes_x(INDEX(i  ,j), IV) = dqX[IV];
	Slopes_y(INDEX(i  ,j), IV) = dqY[IV];
      
    } // end if
    
  } // end do_compute
  
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
    ComputeTraceAndFluxes_Functor<dir> functor(params, Qdata, Slopes_x, Slopes_x, Fluxes,
					       dtdx, dtdy);
    launch_functor(params,functor);
  } // end apply

  /** 
   * entry point for team policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& thread) const
  {

    // the teams league must distribute last dimension (here j) into
    // nbTeams chuncks
    // so compute chunck size per team (rounded up)
    int chunck_size_y = (params.jsize+params.nbTeams-1)/params.nbTeams;
    
    int chunk_size_per_team = chunck_size_y;

    // team id
    int teamId = thread.league_rank();
    
    // compute j start
    int jStart = teamId * chunck_size_y; 

    // spread work among the thread in the team
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, chunk_size_per_team), 
      [=](const int &index) {

        // index goes from 0 to chunck_size_ter_team
        // re-compute j from offset + index 
        int j =  jStart + index;

        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(thread, params.isize),
          [=](const int &i) {
            
            do_compute(INDEX(i,j));

          }); // end vector range
      });
    
  } // end team policy functor

  /** 
   * entry point for range policy
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    do_compute(index);

  } // end range policy functor

  /**
   * Actual computation.
   */
  KOKKOS_INLINE_FUNCTION
  void do_compute(const int& index) const
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
	qLoc[ID] = Qdata   (INDEX(i  ,j), ID);
	dqX[ID]  = Slopes_x(INDEX(i  ,j), ID);
	dqY[ID]  = Slopes_y(INDEX(i  ,j), ID);
	
	qLoc[IP] = Qdata   (INDEX(i  ,j), IP);
	dqX[IP]  = Slopes_x(INDEX(i  ,j), IP);
	dqY[IP]  = Slopes_y(INDEX(i  ,j), IP);
	
	qLoc[IU] = Qdata   (INDEX(i  ,j), IU);
	dqX[IU]  = Slopes_x(INDEX(i  ,j), IU);
	dqY[IU]  = Slopes_y(INDEX(i  ,j), IU);
	
	qLoc[IV] = Qdata   (INDEX(i  ,j), IV);
	dqX[IV]  = Slopes_x(INDEX(i  ,j), IV);
	dqY[IV]  = Slopes_y(INDEX(i  ,j), IV);

	if (dir == XDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(qLoc,
				     dqX, dqY,
				     dtdx, dtdy, FACE_XMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (INDEX(i-1,j  ), ID);
	  dqX_neighbor[ID] = Slopes_x(INDEX(i-1,j  ), ID);
	  dqY_neighbor[ID] = Slopes_y(INDEX(i-1,j  ), ID);
	  
	  qLocNeighbor[IP] = Qdata   (INDEX(i-1,j  ), IP);
	  dqX_neighbor[IP] = Slopes_x(INDEX(i-1,j  ), IP);
	  dqY_neighbor[IP] = Slopes_y(INDEX(i-1,j  ), IP);
	  
	  qLocNeighbor[IU] = Qdata   (INDEX(i-1,j  ), IU);
	  dqX_neighbor[IU] = Slopes_x(INDEX(i-1,j  ), IU);
	  dqY_neighbor[IU] = Slopes_y(INDEX(i-1,j  ), IU);
	  
	  qLocNeighbor[IV] = Qdata   (INDEX(i-1,j  ), IV);
	  dqX_neighbor[IV] = Slopes_x(INDEX(i-1,j  ), IV);
	  dqY_neighbor[IV] = Slopes_y(INDEX(i-1,j  ), IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,
				     dtdx, dtdy, FACE_XMAX, qleft);
	  
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann_hllc(qleft,qright,qgdnv,flux);

	  //
	  // store fluxes
	  //	
	  Fluxes(INDEX(i  ,j ), ID) =  flux[ID]*dtdx;
	  Fluxes(INDEX(i  ,j ), IP) =  flux[IP]*dtdx;
	  Fluxes(INDEX(i  ,j ), IU) =  flux[IU]*dtdx;
	  Fluxes(INDEX(i  ,j ), IV) =  flux[IV]*dtdx;

	} else if (dir == YDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(qLoc,
				     dqX, dqY,
				     dtdx, dtdy, FACE_YMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (INDEX(i  ,j-1), ID);
	  dqX_neighbor[ID] = Slopes_x(INDEX(i  ,j-1), ID);
	  dqY_neighbor[ID] = Slopes_y(INDEX(i  ,j-1), ID);
	  
	  qLocNeighbor[IP] = Qdata   (INDEX(i  ,j-1), IP);
	  dqX_neighbor[IP] = Slopes_x(INDEX(i  ,j-1), IP);
	  dqY_neighbor[IP] = Slopes_y(INDEX(i  ,j-1), IP);
	  
	  qLocNeighbor[IU] = Qdata   (INDEX(i  ,j-1), IU);
	  dqX_neighbor[IU] = Slopes_x(INDEX(i  ,j-1), IU);
	  dqY_neighbor[IU] = Slopes_y(INDEX(i  ,j-1), IU);
	  
	  qLocNeighbor[IV] = Qdata   (INDEX(i  ,j-1), IV);
	  dqX_neighbor[IV] = Slopes_x(INDEX(i  ,j-1), IV);
	  dqY_neighbor[IV] = Slopes_y(INDEX(i  ,j-1), IV);
	  
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
	  Fluxes(INDEX(i  ,j  ), ID) =  flux[ID]*dtdy;
	  Fluxes(INDEX(i  ,j  ), IP) =  flux[IP]*dtdy;
	  Fluxes(INDEX(i  ,j  ), IU) =  flux[IV]*dtdy; // IU/IV swapped
	  Fluxes(INDEX(i  ,j  ), IV) =  flux[IU]*dtdy; // IU/IV swapped

	}
	      
    } // end if
    
  } // end do_compute
  
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
      Udata(INDEX(i  ,j  ), ID) = 1.0;
      Udata(INDEX(i  ,j  ), IP) = 1.0/(gamma0-1.0);
      Udata(INDEX(i  ,j  ), IU) = 0.0;
      Udata(INDEX(i  ,j  ), IV) = 0.0;
    } else {
      Udata(INDEX(i  ,j  ), ID) = 0.125;
      Udata(INDEX(i  ,j  ), IP) = 0.14/(gamma0-1.0);
      Udata(INDEX(i  ,j  ), IU) = 0.0;
      Udata(INDEX(i  ,j  ), IV) = 0.0;
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
      Udata(INDEX(i  ,j  ), ID) = blast_density_in;
      Udata(INDEX(i  ,j  ), IP) = blast_pressure_in/(gamma0-1.0);
      Udata(INDEX(i  ,j  ), IU) = 0.0;
      Udata(INDEX(i  ,j  ), IV) = 0.0;
    } else {
      Udata(INDEX(i  ,j  ), ID) = blast_density_out;
      Udata(INDEX(i  ,j  ), IP) = blast_pressure_out/(gamma0-1.0);
      Udata(INDEX(i  ,j  ), IU) = 0.0;
      Udata(INDEX(i  ,j  ), IV) = 0.0;
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

    if (faceId == FACE_XMIN) {
      
      // boundary xmin
      int boundary_type = params.boundary_type_xmin;

      j = index / ghostWidth;
      i = index - j*ghostWidth;
      
      if(j >= jmin && j <= jmax    &&
	 i >= 0    && i <ghostWidth) {
	
	for ( int iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  real_t sign=1.0;
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*ghostWidth-1-i;
	    if (iVar==IU) sign=-1.0;
	  } else if( boundary_type == BC_NEUMANN ) {
	    i0=ghostWidth;
	  } else { // periodic
	    i0=nx+i;
	  }
	  
	  Udata(INDEX(i  ,j  ), iVar) = Udata(INDEX(i0  ,j  ), iVar)*sign;
	  
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
	
	for ( int iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  real_t sign=1.0;
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*nx+2*ghostWidth-1-i;
	    if (iVar==IU) sign=-1.0;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    i0=nx+ghostWidth-1;
	  } else { // periodic
	    i0=i-nx;
	  }
	  
	  Udata(INDEX(i  ,j  ), iVar) = Udata(INDEX(i0 ,j  ), iVar)*sign;
	  
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
	
	for ( int iVar=0; iVar<NBVAR; iVar++ ) {

	  real_t sign=1.0;

	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ghostWidth-1-j;
	    if (iVar==IV) sign=-1.0;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ghostWidth;
	  } else { // periodic
	    j0=ny+j;
	  }
	  
	  Udata(INDEX(i  ,j  ), iVar) = Udata(INDEX(i  ,j0 ), iVar)*sign;
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
	
	for ( int iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  real_t sign=1.0;
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ny+2*ghostWidth-1-j;
	    if (iVar==IV) sign=-1.0;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ny+ghostWidth-1;
	  } else { // periodic
	    j0=j-ny;
	  }
	  
	  Udata(INDEX(i  ,j  ), iVar) = Udata(INDEX(i  ,j0  ), iVar)*sign;
	  
	}

      }
    }
    
  } // end operator ()

  DataArray Udata;
  
}; // MakeBoundariesFunctor
  
#endif // HYDRO_RUN_FUNCTORS_H_

