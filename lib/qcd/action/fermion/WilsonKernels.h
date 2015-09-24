#ifndef  GRID_QCD_DHOP_H
#define  GRID_QCD_DHOP_H

namespace Grid {

  namespace QCD {

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Helper routines that implement Wilson stencil for a single site.
    // Common to both the WilsonFermion and WilsonFermion5D
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class Impl> class WilsonKernels : public FermionOperator<Impl> { 
    public:

     INHERIT_IMPL_TYPES(Impl);
     typedef FermionOperator<Impl> Base;
     
    public:
     void DiracOptDhopSite(CartesianStencil &st,DoubledGaugeField &U,
			   std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
			   int sF,int sU,const FermionField &in, FermionField &out);
      
     void DiracOptDhopSiteDag(CartesianStencil &st,DoubledGaugeField &U,
			      std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
			      int sF,int sU,const FermionField &in,FermionField &out);

     void DiracOptDhopDir(CartesianStencil &st,DoubledGaugeField &U,
			  std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
			  int sF,int sU,const FermionField &in, FermionField &out,int dirdisp,int gamma);
#define HANDOPT
#ifdef HANDOPT
     void DiracOptHandDhopSite(CartesianStencil &st,DoubledGaugeField &U,
			       std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
			       int sF,int sU,const FermionField &in, FermionField &out);

     void DiracOptHandDhopSiteDag(CartesianStencil &st,DoubledGaugeField &U,
				  std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
				  int sF,int sU,const FermionField &in, FermionField &out);
#else

     void DiracOptHandDhopSite(CartesianStencil &st,DoubledGaugeField &U,
			       std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
			       int sF,int sU,const FermionField &in, FermionField &out)
     {
       DiracOptDhopSite(st,U,buf,sF,sU,in,out); // will template override for Wilson Nc=3
     }

     void DiracOptHandDhopSiteDag(CartesianStencil &st,DoubledGaugeField &U,
				  std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
				  int sF,int sU,const FermionField &in, FermionField &out)
     {
       DiracOptDhopSiteDag(st,U,buf,sF,sU,in,out); // will template override for Wilson Nc=3
     }
#endif

     WilsonKernels(const ImplParams &p= ImplParams()) : Base(p) {};

    };

  }
}
#endif
