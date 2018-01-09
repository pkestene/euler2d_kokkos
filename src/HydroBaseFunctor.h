#ifndef HYDRO_BASE_FUNCTOR_H_
#define HYDRO_BASE_FUNCTOR_H_

#include "kokkos_shared.h"

#include "HydroParams.h"

/**
 * Base class to derive actual kokkos functor.
 * params is passed by copy.
 */
class HydroBaseFunctor
{

public:

  HydroBaseFunctor(HydroParams params) : params(params) {};
  virtual ~HydroBaseFunctor() {};

  HydroParams params;
  
  // utility routines used in various computational kernels

  KOKKOS_INLINE_FUNCTION
  void swapValues(real_t *a, real_t *b) const
  {
    
    real_t tmp = *a;
    
    *a = *b;
    *b = tmp;
    
  } // swapValues
  
  /**
   * Equation of state:
   * compute pressure p and speed of sound c, from density rho and
   * internal energy eint using the "calorically perfect gas" equation
   * of state : \f$ eint=\frac{p}{\rho (\gamma-1)} \f$
   * Recall that \f$ \gamma \f$ is equal to the ratio of specific heats
   *  \f$ \left[ c_p/c_v \right] \f$.
   * 
   * @param[in]  rho  density
   * @param[in]  eint internal energy
   * @param[out] p    pressure
   * @param[out] c    speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void eos(real_t rho,
	   real_t eint,
	   real_t* p,
	   real_t* c) const
  {
    const real_t gamma0 = params.settings.gamma0;
    const real_t smallp = params.settings.smallp;
    
    *p = FMAX((gamma0 - ONE_F) * rho * eint, rho * smallp);
    *c = SQRT(gamma0 * (*p) / rho);
    
  } // eos
  
  /**
   * Convert conservative variables (rho, rho*u, rho*v, e) to 
   * primitive variables (rho,u,v,p)
   * @param[in]  u  conservative variables array
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant NBVAR)
   * @param[out] c  local speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void computePrimitives(const HydroState& u,
			 real_t* c,
			 HydroState& q) const
  {
    const real_t gamma0 = params.settings.gamma0;
    const real_t smallr = params.settings.smallr;
    const real_t smallp = params.settings.smallp;
    
    real_t d, p, ux, uy;
    
    d = fmax(u[ID], smallr);
    ux = u[IU] / d;
    uy = u[IV] / d;
    
    real_t eken = HALF_F * (ux*ux + uy*uy);
    real_t e = u[IP] / d - eken;
    
    // compute pressure and speed of sound
    p = fmax((gamma0 - 1.0) * d * e, d * smallp);
    *c = sqrt(gamma0 * (p) / d);
    
    q[ID] = d;
    q[IP] = p;
    q[IU] = ux;
    q[IV] = uy;
    
  } // computePrimitive

  
  /**
   * Trace computations for unsplit Godunov scheme.
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] qNeighbors : state in the neighbor cells (2 neighbors
   * per dimension, in the following order x+, x-, y+, y-, z+, z-)
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[out] qm        : qm state (one per dimension)
   * \param[out] qp        : qp state (one per dimension)
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_2d(const HydroState& q, 
			const HydroState& qNeighbors_0,
			const HydroState& qNeighbors_1,
			const HydroState& qNeighbors_2,
			const HydroState& qNeighbors_3,
			real_t c, 
			real_t dtdx, 
			real_t dtdy, 
			HydroState& qm_x,
			HydroState& qm_y,
			HydroState& qp_x,
			HydroState& qp_y) const
  {
    
    const real_t gamma0 = params.settings.gamma0;
    const real_t smallr = params.settings.smallr;
    
    // first compute slopes
    HydroState dqX, dqY;
    dqX[ID] = 0.0;
    dqX[IP] = 0.0;
    dqX[IU] = 0.0;
    dqX[IV] = 0.0;
    dqY[ID] = 0.0;
    dqY[IP] = 0.0;
    dqY[IU] = 0.0;
    dqY[IV] = 0.0;
      
    slope_unsplit_hydro_2d(q, 
			   qNeighbors_0, qNeighbors_1, 
			   qNeighbors_2, qNeighbors_3,
			   dqX, dqY);
      
    // Cell centered values
    real_t r =  q[ID];
    real_t p =  q[IP];
    real_t u =  q[IU];
    real_t v =  q[IV];
      
    // TVD slopes in all directions
    real_t drx = dqX[ID];
    real_t dpx = dqX[IP];
    real_t dux = dqX[IU];
    real_t dvx = dqX[IV];
      
    real_t dry = dqY[ID];
    real_t dpy = dqY[IP];
    real_t duy = dqY[IU];
    real_t dvy = dqY[IV];
      
    // source terms (with transverse derivatives)
    real_t sr0 = -u*drx-v*dry - (dux+dvy)*r;
    real_t sp0 = -u*dpx-v*dpy - (dux+dvy)*gamma0*p;
    real_t su0 = -u*dux-v*duy - (dpx    )/r;
    real_t sv0 = -u*dvx-v*dvy - (dpy    )/r;
      
    // Right state at left interface
    qp_x[ID] = r - HALF_F*drx + sr0*dtdx*HALF_F;
    qp_x[IP] = p - HALF_F*dpx + sp0*dtdx*HALF_F;
    qp_x[IU] = u - HALF_F*dux + su0*dtdx*HALF_F;
    qp_x[IV] = v - HALF_F*dvx + sv0*dtdx*HALF_F;
    qp_x[ID] = fmax(smallr, qp_x[ID]);
      
    // Left state at right interface
    qm_x[ID] = r + HALF_F*drx + sr0*dtdx*HALF_F;
    qm_x[IP] = p + HALF_F*dpx + sp0*dtdx*HALF_F;
    qm_x[IU] = u + HALF_F*dux + su0*dtdx*HALF_F;
    qm_x[IV] = v + HALF_F*dvx + sv0*dtdx*HALF_F;
    qm_x[ID] = fmax(smallr, qm_x[ID]);
      
    // Top state at bottom interface
    qp_y[ID] = r - HALF_F*dry + sr0*dtdy*HALF_F;
    qp_y[IP] = p - HALF_F*dpy + sp0*dtdy*HALF_F;
    qp_y[IU] = u - HALF_F*duy + su0*dtdy*HALF_F;
    qp_y[IV] = v - HALF_F*dvy + sv0*dtdy*HALF_F;
    qp_y[ID] = fmax(smallr, qp_y[ID]);
      
    // Bottom state at top interface
    qm_y[ID] = r + HALF_F*dry + sr0*dtdy*HALF_F;
    qm_y[IP] = p + HALF_F*dpy + sp0*dtdy*HALF_F;
    qm_y[IU] = u + HALF_F*duy + su0*dtdy*HALF_F;
    qm_y[IV] = v + HALF_F*dvy + sv0*dtdy*HALF_F;
    qm_y[ID] = fmax(smallr, qm_y[ID]);
      
  } // trace_unsplit_2d


  /**
   * Trace computations for unsplit Godunov scheme.
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] dqX        : slope along X
   * \param[in] dqY        : slope along Y
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[in] dtdy       : dt over dy
   * \param[in] faceId     : which face will be reconstructed
   * \param[out] qface     : q reconstructed state at cell interface
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_2d_along_dir(const HydroState& q, 
				  const HydroState& dqX,
				  const HydroState& dqY,
				  real_t dtdx, 
				  real_t dtdy, 
				  int    faceId,
				  HydroState& qface) const
  {
  
    const real_t gamma0 = params.settings.gamma0;
    const real_t smallr = params.settings.smallr;

    // Cell centered values
    real_t r =  q[ID];
    real_t p =  q[IP];
    real_t u =  q[IU];
    real_t v =  q[IV];
  
    // TVD slopes in all directions
    real_t drx = dqX[ID];
    real_t dpx = dqX[IP];
    real_t dux = dqX[IU];
    real_t dvx = dqX[IV];
  
    real_t dry = dqY[ID];
    real_t dpy = dqY[IP];
    real_t duy = dqY[IU];
    real_t dvy = dqY[IV];
  
    // source terms (with transverse derivatives)
    real_t sr0 = -u*drx-v*dry - (dux+dvy)*r;
    real_t sp0 = -u*dpx-v*dpy - (dux+dvy)*gamma0*p;
    real_t su0 = -u*dux-v*duy - (dpx    )/r;
    real_t sv0 = -u*dvx-v*dvy - (dpy    )/r;
  
    if (faceId == FACE_XMIN) {
      // Right state at left interface
      qface[ID] = r - HALF_F*drx + sr0*dtdx*HALF_F;
      qface[IP] = p - HALF_F*dpx + sp0*dtdx*HALF_F;
      qface[IU] = u - HALF_F*dux + su0*dtdx*HALF_F;
      qface[IV] = v - HALF_F*dvx + sv0*dtdx*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_XMAX) {
      // Left state at right interface
      qface[ID] = r + HALF_F*drx + sr0*dtdx*HALF_F;
      qface[IP] = p + HALF_F*dpx + sp0*dtdx*HALF_F;
      qface[IU] = u + HALF_F*dux + su0*dtdx*HALF_F;
      qface[IV] = v + HALF_F*dvx + sv0*dtdx*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }
  
    if (faceId == FACE_YMIN) {
      // Top state at bottom interface
      qface[ID] = r - HALF_F*dry + sr0*dtdy*HALF_F;
      qface[IP] = p - HALF_F*dpy + sp0*dtdy*HALF_F;
      qface[IU] = u - HALF_F*duy + su0*dtdy*HALF_F;
      qface[IV] = v - HALF_F*dvy + sv0*dtdy*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_YMAX) {
      // Bottom state at top interface
      qface[ID] = r + HALF_F*dry + sr0*dtdy*HALF_F;
      qface[IP] = p + HALF_F*dpy + sp0*dtdy*HALF_F;
      qface[IU] = u + HALF_F*duy + su0*dtdy*HALF_F;
      qface[IV] = v + HALF_F*dvy + sv0*dtdy*HALF_F;
      qface[ID] = fmax(smallr, qface[ID]);
    }

  } // trace_unsplit_2d_along_dir


  /**
   * This another implementation of trace computations for 2D data; it
   * is used when unsplitVersion = 1
   *
   * Note that :
   * - hydro slopes computations are done outside this routine
   *
   * \param[in]  q  primitive variable state vector
   * \param[in]  dq primitive variable slopes
   * \param[in]  dtdx dt divided by dx
   * \param[in]  dtdy dt divided by dy
   * \param[out] qm
   * \param[out] qp
   *
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_hydro_2d(const HydroState& q,
			      const HydroState& dqX,
			      const HydroState& dqY,
			      real_t dtdx,
			      real_t dtdy,
			      HydroState& qm_x,
			      HydroState& qm_y,
			      HydroState& qp_x,
			      HydroState& qp_y) const
  {
  
    const real_t gamma0 = params.settings.gamma0;
    const real_t smallr = params.settings.smallr;
    const real_t smallp = params.settings.smallp;

    // Cell centered values
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];

    // Cell centered TVD slopes in X direction
    real_t drx = dqX[ID];  drx *= HALF_F;
    real_t dpx = dqX[IP];  dpx *= HALF_F;
    real_t dux = dqX[IU];  dux *= HALF_F;
    real_t dvx = dqX[IV];  dvx *= HALF_F;
  
    // Cell centered TVD slopes in Y direction
    real_t dry = dqY[ID];  dry *= HALF_F;
    real_t dpy = dqY[IP];  dpy *= HALF_F;
    real_t duy = dqY[IU];  duy *= HALF_F;
    real_t dvy = dqY[IV];  dvy *= HALF_F;

    // Source terms (including transverse derivatives)
    real_t sr0, su0, sv0, sp0;

    /*only true for cartesian grid */
    {
      sr0 = (-u*drx-dux*r)       *dtdx + (-v*dry-dvy*r)       *dtdy;
      su0 = (-u*dux-dpx/r)       *dtdx + (-v*duy      )       *dtdy;
      sv0 = (-u*dvx      )       *dtdx + (-v*dvy-dpy/r)       *dtdy;
      sp0 = (-u*dpx-dux*gamma0*p)*dtdx + (-v*dpy-dvy*gamma0*p)*dtdy;    
    } // end cartesian

    // Update in time the  primitive variables
    r = r + sr0;
    u = u + su0;
    v = v + sv0;
    p = p + sp0;

    // Face averaged right state at left interface
    qp_x[ID] = r - drx;
    qp_x[IU] = u - dux;
    qp_x[IV] = v - dvx;
    qp_x[IP] = p - dpx;
    qp_x[ID] = fmax(smallr,  qp_x[ID]);
    qp_x[IP] = fmax(smallp * qp_x[ID], qp_x[IP]);
  
    // Face averaged left state at right interface
    qm_x[ID] = r + drx;
    qm_x[IU] = u + dux;
    qm_x[IV] = v + dvx;
    qm_x[IP] = p + dpx;
    qm_x[ID] = fmax(smallr,  qm_x[ID]);
    qm_x[IP] = fmax(smallp * qm_x[ID], qm_x[IP]);

    // Face averaged top state at bottom interface
    qp_y[ID] = r - dry;
    qp_y[IU] = u - duy;
    qp_y[IV] = v - dvy;
    qp_y[IP] = p - dpy;
    qp_y[ID] = fmax(smallr,  qp_y[ID]);
    qp_y[IP] = fmax(smallp * qp_y[ID], qp_y[IP]);
  
    // Face averaged bottom state at top interface
    qm_y[ID] = r + dry;
    qm_y[IU] = u + duy;
    qm_y[IV] = v + dvy;
    qm_y[IP] = p + dpy;
    qm_y[ID] = fmax(smallr,  qm_y[ID]);
    qm_y[IP] = fmax(smallp * qm_y[ID], qm_y[IP]);
  
  } // trace_unsplit_hydro_2d


  /**
   * Compute primitive variables slopes (dqX,dqY) for one component from q and its neighbors.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
   * 
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q       : current primitive variable
   * \param[in]  qPlusX  : value in the next neighbor cell along XDIR
   * \param[in]  qMinusX : value in the previous neighbor cell along XDIR
   * \param[in]  qPlusY  : value in the next neighbor cell along YDIR
   * \param[in]  qMinusY : value in the previous neighbor cell along YDIR
   * \param[out] dqX     : reference to an array returning the X slopes
   * \param[out] dqY     : reference to an array returning the Y slopes
   *
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_2d_scalar(real_t q, 
				     real_t qPlusX,
				     real_t qMinusX,
				     real_t qPlusY,
				     real_t qMinusY,
				     real_t *dqX,
				     real_t *dqY) const
  {
    const real_t slope_type = params.settings.slope_type;

    real_t dlft, drgt, dcen, dsgn, slop, dlim;

    // slopes in first coordinate direction
    dlft = slope_type*(q      - qMinusX);
    drgt = slope_type*(qPlusX - q      );
    dcen = HALF_F * (qPlusX - qMinusX);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqX = dsgn * fmin( dlim, FABS(dcen) );
  
    // slopes in second coordinate direction
    dlft = slope_type*(q      - qMinusY);
    drgt = slope_type*(qPlusY - q      );
    dcen = HALF_F * (qPlusY - qMinusY);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqY = dsgn * fmin( dlim, FABS(dcen) );

  } // slope_unsplit_hydro_2d_scalar


  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
   * 
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q       : current primitive variable state
   * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
   * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
   * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
   * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
   * \param[out] dqX     : reference to an array returning the X slopes
   * \param[out] dqY     : reference to an array returning the Y slopes
   *
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_2d(const HydroState& q, 
			      const HydroState& qPlusX, 
			      const HydroState& qMinusX,
			      const HydroState& qPlusY,
			      const HydroState& qMinusY,
			      HydroState& dqX,
			      HydroState& dqY) const
  {
  
    const real_t slope_type = params.settings.slope_type;
    
    if (slope_type==0) {

      dqX[ID] = ZERO_F;
      dqX[IP] = ZERO_F;
      dqX[IU] = ZERO_F;
      dqX[IV] = ZERO_F;

      dqY[ID] = ZERO_F;
      dqY[IP] = ZERO_F;
      dqY[IU] = ZERO_F;
      dqY[IV] = ZERO_F;

      return;
    }

    if (slope_type==1 || slope_type==2) {  // minmod or average

      slope_unsplit_hydro_2d_scalar( q[ID], qPlusX[ID], qMinusX[ID], qPlusY[ID], qMinusY[ID], &(dqX[ID]), &(dqY[ID]));
      slope_unsplit_hydro_2d_scalar( q[IP], qPlusX[IP], qMinusX[IP], qPlusY[IP], qMinusY[IP], &(dqX[IP]), &(dqY[IP]));
      slope_unsplit_hydro_2d_scalar( q[IU], qPlusX[IU], qMinusX[IU], qPlusY[IU], qMinusY[IU], &(dqX[IU]), &(dqY[IU]));
      slope_unsplit_hydro_2d_scalar( q[IV], qPlusX[IV], qMinusX[IV], qPlusY[IV], qMinusY[IV], &(dqX[IV]), &(dqY[IV]));

    } // end slope_type == 1 or 2
  
  } // slope_unsplit_hydro_2d

  /**
   * Compute cell fluxes from the Godunov state
   * \param[in]  qgdnv input Godunov state
   * \param[out] flux  output flux vector
   */
  KOKKOS_INLINE_FUNCTION
  void cmpflx(const HydroState& qgdnv, 
	      HydroState& flux) const
  {
    const real_t gamma0 = params.settings.gamma0;

    // Compute fluxes
    // Mass density
    flux[ID] = qgdnv[ID] * qgdnv[IU];
  
    // Normal momentum
    flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP];
  
    // Transverse momentum
    flux[IV] = flux[ID] * qgdnv[IV];

    // Total energy
    real_t entho = ONE_F / (gamma0 - ONE_F);
    real_t ekin;
    ekin = HALF_F * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV]);
  
    real_t etot = qgdnv[IP] * entho + ekin;
    flux[IP] = qgdnv[IU] * (etot + qgdnv[IP]);

  } // cmpflx
  
  /** 
   * Riemann solver, equivalent to riemann_approx in RAMSES (see file
   * godunov_utils.f90 in RAMSES).
   * 
   * @param[in] qleft  : input left state
   * @param[in] qright : input right state
   * @param[out] qgdnv : output Godunov state
   * @param[out] flux  : output flux
   */
  KOKKOS_INLINE_FUNCTION
  void riemann_approx(const HydroState& qleft,
		      const HydroState& qright,
		      HydroState& qgdnv, 
		      HydroState& flux) const
  {
    const real_t gamma0  = params.settings.gamma0;
    const real_t gamma6  = params.settings.gamma6;
    const real_t smallr  = params.settings.smallr;
    const real_t smallc  = params.settings.smallc;
    const real_t smallp  = params.settings.smallp;
    const real_t smallpp = params.settings.smallpp;

    // Pressure, density and velocity
    real_t rl = fmax(qleft [ID], smallr);
    real_t ul =      qleft [IU];
    real_t pl = fmax(qleft [IP], rl*smallp);
    real_t rr = fmax(qright[ID], smallr);
    real_t ur =      qright[IU];
    real_t pr = fmax(qright[IP], rr*smallp);
  
    // Lagrangian sound speed
    real_t cl = gamma0*pl*rl;
    real_t cr = gamma0*pr*rr;
  
    // First guess
    real_t wl = SQRT(cl);
    real_t wr = SQRT(cr);
    real_t pstar = fmax(((wr*pl+wl*pr)+wl*wr*(ul-ur))/(wl+wr), (real_t) ZERO_F);
    real_t pold = pstar;
    real_t conv = ONE_F;
  
    // Newton-Raphson iterations to find pstar at the required accuracy
    for(int iter = 0; (iter < 10 /*niter_riemann*/) && (conv > 1e-6); ++iter)
      {
	real_t wwl = SQRT(cl*(ONE_F+gamma6*(pold-pl)/pl));
	real_t wwr = SQRT(cr*(ONE_F+gamma6*(pold-pr)/pr));
	real_t ql = 2.0f*wwl*wwl*wwl/(wwl*wwl+cl);
	real_t qr = 2.0f*wwr*wwr*wwr/(wwr*wwr+cr);
	real_t usl = ul-(pold-pl)/wwl;
	real_t usr = ur+(pold-pr)/wwr;
	real_t delp = fmax(qr*ql/(qr+ql)*(usl-usr),-pold);
      
	pold = pold+delp;
	conv = FABS(delp/(pold+smallpp));	 // Convergence indicator
      }
  
    // Star region pressure
    // for a two-shock Riemann problem
    pstar = pold;
    wl = SQRT(cl*(ONE_F+gamma6*(pstar-pl)/pl));
    wr = SQRT(cr*(ONE_F+gamma6*(pstar-pr)/pr));
  
    // Star region velocity
    // for a two shock Riemann problem
    real_t ustar = HALF_F * (ul + (pl-pstar)/wl + ur - (pr-pstar)/wr);
  
    // Left going or right going contact wave
    real_t sgnm = COPYSIGN(ONE_F, ustar);
  
    // Left or right unperturbed state
    real_t ro, uo, po, wo;
    if(sgnm > ZERO_F)
      {
	ro = rl;
	uo = ul;
	po = pl;
	wo = wl;
      }
    else
      {
	ro = rr;
	uo = ur;
	po = pr;
	wo = wr;
      }
    real_t co = fmax(smallc, SQRT(FABS(gamma0*po/ro)));
  
    // Star region density (Shock, fmax prevents vacuum formation in star region)
    real_t rstar = fmax((real_t) (ro/(ONE_F+ro*(po-pstar)/(wo*wo))), (real_t) (smallr));
    // Star region sound speed
    real_t cstar = fmax(smallc, SQRT(FABS(gamma0*pstar/rstar)));
  
    // Compute rarefaction head and tail speed
    real_t spout  = co    - sgnm*uo;
    real_t spin   = cstar - sgnm*ustar;
    // Compute shock speed
    real_t ushock = wo/ro - sgnm*uo;
  
    if(pstar >= po)
      {
	spin  = ushock;
	spout = ushock;
      }
  
    // Sample the solution at x/t=0
    real_t scr = fmax(spout-spin, smallc+FABS(spout+spin));
    real_t frac = HALF_F * (ONE_F + (spout + spin)/scr);

    if (frac != frac) /* Not a Number */
      frac = 0.0;
    else
      frac = frac >= 1.0 ? 1.0 : frac <= 0.0 ? 0.0 : frac;
  
    qgdnv[ID] = frac*rstar + (ONE_F-frac)*ro;
    qgdnv[IU] = frac*ustar + (ONE_F-frac)*uo;
    qgdnv[IP] = frac*pstar + (ONE_F-frac)*po;
  
    if(spout < ZERO_F)
      {
	qgdnv[ID] = ro;
	qgdnv[IU] = uo;
	qgdnv[IP] = po;
      }
  
    if(spin > ZERO_F)
      {
	qgdnv[ID] = rstar;
	qgdnv[IU] = ustar;
	qgdnv[IP] = pstar;
      }
  
    // transverse velocity
    if(sgnm > ZERO_F)
      {
	qgdnv[IV] = qleft[IV];
      }
    else
      {
	qgdnv[IV] = qright[IV];
      }
  
    cmpflx(qgdnv, flux);
  
  } // riemann_approx

    
  /** 
   * Riemann solver HLLC
   *
   * @param[in] qleft : input left state
   * @param[in] qright : input right state
   * @param[out] qgdnv : output Godunov state
   * @param[out] flux  : output flux
   */
  KOKKOS_INLINE_FUNCTION
  void riemann_hllc(const HydroState& qleft,
		    const HydroState& qright,
		    HydroState& qgdnv,
		    HydroState& flux) const
  {

    const real_t gamma0 = params.settings.gamma0;
    const real_t smallr = params.settings.smallr;
    const real_t smallp = params.settings.smallp;
    const real_t smallc = params.settings.smallc;

    const real_t entho = ONE_F / (gamma0 - ONE_F);
  
    // Left variables
    real_t rl = fmax(qleft[ID], smallr);
    real_t pl = fmax(qleft[IP], rl*smallp);
    real_t ul =      qleft[IU];
    
    real_t ecinl = HALF_F*rl*ul*ul;
    ecinl += HALF_F*rl*qleft[IV]*qleft[IV];

    real_t etotl = pl*entho+ecinl;
    real_t ptotl = pl;

    // Right variables
    real_t rr = fmax(qright[ID], smallr);
    real_t pr = fmax(qright[IP], rr*smallp);
    real_t ur =      qright[IU];

    real_t ecinr = HALF_F*rr*ur*ur;
    ecinr += HALF_F*rr*qright[IV]*qright[IV];
  
    real_t etotr = pr*entho+ecinr;
    real_t ptotr = pr;
    
    // Find the largest eigenvalues in the normal direction to the interface
    real_t cfastl = SQRT(fmax(gamma0*pl/rl,smallc*smallc));
    real_t cfastr = SQRT(fmax(gamma0*pr/rr,smallc*smallc));

    // Compute HLL wave speed
    real_t SL = fmin(ul,ur) - fmax(cfastl,cfastr);
    real_t SR = fmax(ul,ur) + fmax(cfastl,cfastr);

    // Compute lagrangian sound speed
    real_t rcl = rl*(ul-SL);
    real_t rcr = rr*(SR-ur);
    
    // Compute acoustic star state
    real_t ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
    real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

    // Left star region variables
    real_t rstarl    = rl*(SL-ul)/(SL-ustar);
    real_t etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar);
    
    // Right star region variables
    real_t rstarr    = rr*(SR-ur)/(SR-ustar);
    real_t etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar);
    
    // Sample the solution at x/t=0
    real_t ro, uo, ptoto, etoto;
    if (SL > ZERO_F) {
      ro=rl;
      uo=ul;
      ptoto=ptotl;
      etoto=etotl;
    } else if (ustar > ZERO_F) {
      ro=rstarl;
      uo=ustar;
      ptoto=ptotstar;
      etoto=etotstarl;
    } else if (SR > ZERO_F) {
      ro=rstarr;
      uo=ustar;
      ptoto=ptotstar;
      etoto=etotstarr;
    } else {
      ro=rr;
      uo=ur;
      ptoto=ptotr;
      etoto=etotr;
    }
      
    // Compute the Godunov flux
    flux[ID] = ro*uo;
    flux[IU] = ro*uo*uo+ptoto;
    flux[IP] = (etoto+ptoto)*uo;
    if (flux[ID] > ZERO_F) {
      flux[IV] = flux[ID]*qleft[IV];
    } else {
      flux[IV] = flux[ID]*qright[IV];
    }
  
  } // riemann_hllc

}; // class HydroBaseFunctor

#endif // HYDRO_BASE_FUNCTOR_H_
