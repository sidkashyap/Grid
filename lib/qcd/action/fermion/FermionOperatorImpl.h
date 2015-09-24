#ifndef  GRID_QCD_FERMION_OPERATOR_IMPL_H
#define  GRID_QCD_FERMION_OPERATOR_IMPL_H

namespace Grid {

  namespace QCD {


    //////////////////////////////////////////////
    // Template parameter class constructs to package
    // externally control Fermion implementations
    // in orthogonal directions
    //
    // Ultimately need Impl to always define types where XXX is opaque
    //
    //    typedef typename XXX               Simd;
    //    typedef typename XXX     GaugeLinkField;	
    //    typedef typename XXX         GaugeField;
    //    typedef typename XXX      GaugeActField;
    //    typedef typename XXX       FermionField;
    //    typedef typename XXX  DoubledGaugeField;
    //    typedef typename XXX         SiteSpinor;
    //    typedef typename XXX     SiteHalfSpinor;	
    //    typedef typename XXX         Compressor;	
    //
    // and Methods:
    //    void ImportGauge(GridBase *GaugeGrid,DoubledGaugeField &Uds,const GaugeField &Umu)
    //    void DoubleStore(GridBase *GaugeGrid,DoubledGaugeField &Uds,const GaugeField &Umu)
    //    void multLink(SiteHalfSpinor &phi,const SiteDoubledGaugeField &U,const SiteHalfSpinor &chi,int mu,StencilEntry *SE,CartesianStencil &St)
    //    void InsertForce4D(GaugeField &mat,const FermionField &Btilde,const FermionField &A,int mu)
    //    void InsertForce5D(GaugeField &mat,const FermionField &Btilde,const FermionField &A,int mu)
    //
    //
    // To acquire the typedefs from "Base" (either a base class or template param) use:
    //
    // INHERIT_GIMPL_TYPES(Base)
    // INHERIT_FIMPL_TYPES(Base)
    // INHERIT_IMPL_TYPES(Base)
    //
    // The Fermion operators will do the following:
    //
    // struct MyOpParams { 
    //   RealD mass;
    // };
    //
    //
    // template<class Impl>
    // class MyOp : pubic<Impl> { 
    // public:
    //
    //    INHERIT_ALL_IMPL_TYPES(Impl);
    //
    //    MyOp(MyOpParams Myparm, ImplParams &ImplParam) :  Impl(ImplParam)
    //    {
    //
    //    };
    //    
    //  }
    //////////////////////////////////////////////

    
    ////////////////////////////////////////////////////////////////////////
    // Implementation dependent gauge types
    ////////////////////////////////////////////////////////////////////////

#define INHERIT_IMPL_TYPES(Base) \
    INHERIT_GIMPL_TYPES(Base)\
    INHERIT_FIMPL_TYPES(Base)

#define INHERIT_GIMPL_TYPES(GImpl) \
    typedef typename GImpl::Simd                           Simd;\
    typedef typename GImpl::GaugeLinkField       GaugeLinkField;\
    typedef typename GImpl::GaugeField               GaugeField;	
    
    // Composition with smeared link, bc's etc.. probably need multiple inheritance
    // Variable precision "S" and variable Nc
    template<class S,int Nrepresentation=Nc>
    class ImplGauge { 
    public:
    
      typedef S Simd;
    
      template<typename vtype> using iImplGaugeLink          = iScalar<iScalar<iMatrix<vtype, Nrepresentation> > >;
      template<typename vtype> using iImplGaugeField         = iVector<iScalar<iMatrix<vtype, Nrepresentation> >, Nd  >;
    
      typedef iImplGaugeLink    <Simd>           SiteGaugeLink;
      typedef iImplGaugeField   <Simd>           SiteGaugeField;
    
      typedef Lattice<SiteGaugeLink>                GaugeLinkField; // bit ugly naming; polarised gauge field, lorentz... all ugly
      typedef Lattice<SiteGaugeField>                   GaugeField;

    };

    ////////////////////////////////////////////////////////////////////////
    // Implementation dependent fermion types
    ////////////////////////////////////////////////////////////////////////

#define INHERIT_FIMPL_TYPES(Impl)\
    typedef typename Impl::FermionField           FermionField;		\
    typedef typename Impl::DoubledGaugeField DoubledGaugeField;		\
    typedef typename Impl::SiteSpinor               SiteSpinor;		\
    typedef typename Impl::SiteHalfSpinor       SiteHalfSpinor;		\
    typedef typename Impl::Compressor               Compressor;		\
    typedef typename Impl::ImplParams ImplParams;

    ///////
    // Single flavour four spinors with colour index
    ///////
    template<class S,int Nrepresentation=Nc>
    class WilsonImpl :  public ImplGauge<S,Nrepresentation> { 
    public:

      typedef ImplGauge<S,Nrepresentation> Gimpl;


      INHERIT_GIMPL_TYPES(Gimpl);

      template<typename vtype> using iImplSpinor             = iScalar<iVector<iVector<vtype, Nrepresentation>, Ns> >;
      template<typename vtype> using iImplHalfSpinor         = iScalar<iVector<iVector<vtype, Nrepresentation>, Nhs> >;
      template<typename vtype> using iImplDoubledGaugeField  = iVector<iScalar<iMatrix<vtype, Nrepresentation> >, Nds >;
    
      typedef iImplSpinor    <Simd>           SiteSpinor;
      typedef iImplHalfSpinor<Simd>           SiteHalfSpinor;
      typedef iImplDoubledGaugeField<Simd>    SiteDoubledGaugeField;

      typedef Lattice<SiteSpinor>                 FermionField;
      typedef Lattice<SiteDoubledGaugeField> DoubledGaugeField;

      typedef WilsonCompressor<SiteHalfSpinor,SiteSpinor> Compressor;
      typedef WilsonImplParams ImplParams;
      ImplParams Params;
      WilsonImpl(const ImplParams &p= ImplParams()) : Params(p) {}; 

      inline void multLink(SiteHalfSpinor &phi,const SiteDoubledGaugeField &U,const SiteHalfSpinor &chi,int mu,StencilEntry *SE,CartesianStencil &St){
        mult(&phi(),&U(mu),&chi());
      }

      inline void DoubleStore(GridBase *GaugeGrid,DoubledGaugeField &Uds,const GaugeField &Umu)
      {
        conformable(Uds._grid,GaugeGrid);
        conformable(Umu._grid,GaugeGrid);
        GaugeLinkField U(GaugeGrid);
        for(int mu=0;mu<Nd;mu++){
  	  U = PeekIndex<LorentzIndex>(Umu,mu);
	  PokeIndex<LorentzIndex>(Uds,U,mu);
	  U = adj(Cshift(U,mu,-1));
	  PokeIndex<LorentzIndex>(Uds,U,mu+4);
	}
      }
      
      inline void InsertForce4D(GaugeField &mat, FermionField &Btilde, FermionField &A,int mu){
	GaugeLinkField link(mat._grid);
	link = TraceIndex<SpinIndex>(outerProduct(Btilde,A)); 
	PokeIndex<LorentzIndex>(mat,link,mu);
      }   

      inline void InsertForce5D(GaugeField &mat, FermionField &Btilde, FermionField &Atilde,int mu){

	int Ls=Btilde._grid->_fdimensions[0];

	GaugeLinkField tmp(mat._grid);
	tmp = zero;
PARALLEL_FOR_LOOP
	for(int sss=0;sss<tmp._grid->oSites();sss++){
	  int sU=sss;
	  for(int s=0;s<Ls;s++){
	    int sF = s+Ls*sU;
	    tmp[sU] = tmp[sU]+ traceIndex<SpinIndex>(outerProduct(Btilde[sF],Atilde[sF])); // ordering here
	  }
	}
	PokeIndex<LorentzIndex>(mat,tmp,mu);
	
      }

    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // Flavour doubled spinors; is Gparity the only? what about C*?
    ////////////////////////////////////////////////////////////////////////////////////////

    template<class S,int Nrepresentation>
    class GparityWilsonImpl : public ImplGauge<S,Nrepresentation> { 
    public:

      typedef ImplGauge<S,Nrepresentation> Gimpl;

      INHERIT_GIMPL_TYPES(Gimpl);

      template<typename vtype> using iImplSpinor             = iVector<iVector<iVector<vtype, Nrepresentation>, Ns>, Ngp >;
      template<typename vtype> using iImplHalfSpinor         = iVector<iVector<iVector<vtype, Nrepresentation>, Nhs>, Ngp >;
      template<typename vtype> using iImplDoubledGaugeField  = iVector<iVector<iScalar<iMatrix<vtype, Nrepresentation> >, Nds >, Ngp >;
    
      typedef iImplSpinor    <Simd>           SiteSpinor;
      typedef iImplHalfSpinor<Simd>           SiteHalfSpinor;
      typedef iImplDoubledGaugeField<Simd>    SiteDoubledGaugeField;

      typedef Lattice<SiteSpinor>                 FermionField;
      typedef Lattice<SiteDoubledGaugeField> DoubledGaugeField;

      typedef WilsonCompressor<SiteHalfSpinor,SiteSpinor> Compressor;

      typedef GparityWilsonImplParams ImplParams;
      ImplParams Params;
      GparityWilsonImpl(const ImplParams &p= ImplParams()) : Params(p) {}; 
      
      // provide the multiply by link that is differentiated between Gparity (with flavour index) and non-Gparity
      inline void multLink(SiteHalfSpinor &phi,const SiteDoubledGaugeField &U,const SiteHalfSpinor &chi,int mu,StencilEntry *SE,CartesianStencil &St){

	typedef SiteHalfSpinor vobj;
	typedef typename SiteHalfSpinor::scalar_object sobj;

	vobj vtmp;
	sobj stmp;
	
	GridBase *grid = St._grid;
      
	const int Nsimd = grid->Nsimd();
	
	int direction    = St._directions[mu];
	int distance     = St._distances[mu];
	int ptype        = St._permute_type[mu]; 
	int sl           = St._grid->_simd_layout[direction];

	// Fixme X.Y.Z.T hardcode in stencil
	int mmu          = mu % Nd;

	// assert our assumptions
	assert((distance==1)||(distance==-1)); // nearest neighbour stencil hard code
	assert((sl==1)||(sl==2));
	
	std::vector<int> icoor;
      
	if ( SE->_around_the_world && Params.twists[mmu] ) {

	  if ( sl == 2 ) {

	    std::vector<sobj> vals(Nsimd);

	    extract(chi,vals);
	    for(int s=0;s<Nsimd;s++){

	      grid->iCoorFromIindex(icoor,s);
	      
	      assert((icoor[direction]==0)||(icoor[direction]==1));
	      
	      int permute_lane;
	      if ( distance == 1) {
		permute_lane = icoor[direction]?1:0;
	      } else {
		permute_lane = icoor[direction]?0:1;
	      }
	      
	      if ( permute_lane ) { 
		stmp(0) = vals[s](1);
		stmp(1) = vals[s](0);
		vals[s] = stmp;
	      }
	    }
	    merge(vtmp,vals);

	  } else { 
	    vtmp(0) = chi(1);
	    vtmp(1) = chi(0);
	  }
	  mult(&phi(0),&U(0)(mu),&vtmp(0));
	  mult(&phi(1),&U(1)(mu),&vtmp(1));
	  
	} else { 
	  mult(&phi(0),&U(0)(mu),&chi(0));
	  mult(&phi(1),&U(1)(mu),&chi(1));
	}
	
      }

      inline void DoubleStore(GridBase *GaugeGrid,DoubledGaugeField &Uds,const GaugeField &Umu)
      {
	
	conformable(Uds._grid,GaugeGrid);
	conformable(Umu._grid,GaugeGrid);
	
	GaugeLinkField Utmp(GaugeGrid);
	GaugeLinkField U(GaugeGrid);
	GaugeLinkField Uconj(GaugeGrid);
	
	Lattice<iScalar<vInteger> > coor(GaugeGrid);

	
	for(int mu=0;mu<Nd;mu++){
	  
	  LatticeCoordinate(coor,mu);
	  
	  U     = PeekIndex<LorentzIndex>(Umu,mu);
	  Uconj = conjugate(U);

	  // This phase could come from a simple bc 1,1,-1,1 ..
	  int neglink = GaugeGrid->GlobalDimensions()[mu]-1;
	  if ( Params.twists[mu] ) { 
	    Uconj = where(coor==neglink,-Uconj,Uconj);
	  }

	  
PARALLEL_FOR_LOOP
	  for(auto ss=U.begin();ss<U.end();ss++){
	    Uds[ss](0)(mu) = U[ss]();
	    Uds[ss](1)(mu) = Uconj[ss]();
	  }
	  
	  U     = adj(Cshift(U    ,mu,-1));      // correct except for spanning the boundary
	  Uconj = adj(Cshift(Uconj,mu,-1));
	  
	  Utmp = U;
	  if ( Params.twists[mu] ) { 
	    Utmp = where(coor==0,Uconj,Utmp);
	  }
	  
PARALLEL_FOR_LOOP
	  for(auto ss=U.begin();ss<U.end();ss++){
	    Uds[ss](0)(mu+4) = Utmp[ss]();
	  }
	  
	  Utmp = Uconj;
	  if ( Params.twists[mu] ) { 
	    Utmp = where(coor==0,U,Utmp);
	  }
	  
PARALLEL_FOR_LOOP
	  for(auto ss=U.begin();ss<U.end();ss++){
	    Uds[ss](1)(mu+4) = Utmp[ss]();
	  }
	  
	}
      }

      inline void InsertForce4D(GaugeField &mat, FermionField &Btilde, FermionField &A,int mu){
	assert(0);
	// Fixme
	return;
      }
      inline void InsertForce5D(GaugeField &mat, FermionField &Btilde, FermionField &A,int mu){
	assert(0);
	// Fixme
	return;
      }
    };

    typedef WilsonImpl<vComplex ,Nc> WilsonImplR; // Real.. whichever prec
    typedef WilsonImpl<vComplexF,Nc> WilsonImplF; // Float
    typedef WilsonImpl<vComplexD,Nc> WilsonImplD; // Double

    typedef GparityWilsonImpl<vComplex ,Nc> GparityWilsonImplR; // Real.. whichever prec
    typedef GparityWilsonImpl<vComplexF,Nc> GparityWilsonImplF; // Float
    typedef GparityWilsonImpl<vComplexD,Nc> GparityWilsonImplD; // Double

  }
}
#endif
