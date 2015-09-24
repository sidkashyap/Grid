#include <Grid.h>
namespace Grid {
namespace QCD {

template<class Impl> 
void WilsonKernels<Impl>::DiracOptDhopSite(CartesianStencil &st,DoubledGaugeField &U,
						  std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
						  int sF,int sU,const FermionField &in, FermionField &out)
{
  SiteHalfSpinor  tmp;    
  SiteHalfSpinor  chi;    
  SiteHalfSpinor Uchi;
  SiteSpinor result;
  StencilEntry *SE;
  int ptype;

  // Xp
  SE=st.GetEntry(ptype,Xp,sF);
  if ( SE->_is_local && SE->_permute ) {
    spProjXp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjXp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Xp,SE,st);
  spReconXp(result,Uchi);
    
  // Yp
  SE=st.GetEntry(ptype,Yp,sF);
  if ( SE->_is_local && SE->_permute ) {
    spProjYp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjYp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Yp,SE,st);
  accumReconYp(result,Uchi);

  // Zp
  SE=st.GetEntry(ptype,Zp,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjZp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjZp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Zp,SE,st);
  accumReconZp(result,Uchi);

  // Tp
  SE=st.GetEntry(ptype,Tp,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjTp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjTp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Tp,SE,st);
  accumReconTp(result,Uchi);

  // Xm
  SE=st.GetEntry(ptype,Xm,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjXm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjXm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Xm,SE,st);
  accumReconXm(result,Uchi);
  
  // Ym
  SE=st.GetEntry(ptype,Ym,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjYm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjYm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Ym,SE,st);
  accumReconYm(result,Uchi);
  
  // Zm
  SE=st.GetEntry(ptype,Zm,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjZm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjZm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Zm,SE,st);
  accumReconZm(result,Uchi);

  // Tm
  SE=st.GetEntry(ptype,Tm,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjTm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjTm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Tm,SE,st);
  accumReconTm(result,Uchi);

  vstream(out._odata[sF],result*(-0.5));
};

template<class Impl> 
void WilsonKernels<Impl>::DiracOptDhopSiteDag(CartesianStencil &st,DoubledGaugeField &U,
					      std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
					      int sF,int sU,const FermionField &in, FermionField &out)
{
  SiteHalfSpinor  tmp;    
  SiteHalfSpinor  chi;    
  SiteSpinor result;
  SiteHalfSpinor Uchi;
  StencilEntry *SE;
  int ptype;

  // Xp
  SE=st.GetEntry(ptype,Xm,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjXp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjXp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Xm,SE,st);
  spReconXp(result,Uchi);

  // Yp
  SE=st.GetEntry(ptype,Ym,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjYp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjYp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Ym,SE,st);
  accumReconYp(result,Uchi);
  
  // Zp
  SE=st.GetEntry(ptype,Zm,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjZp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjZp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Zm,SE,st);
  accumReconZp(result,Uchi);
  
  // Tp
  SE=st.GetEntry(ptype,Tm,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjTp(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjTp(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Tm,SE,st);
  accumReconTp(result,Uchi);
  
  // Xm
  SE=st.GetEntry(ptype,Xp,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjXm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjXm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Xp,SE,st);
  accumReconXm(result,Uchi);

  // Ym
  SE=st.GetEntry(ptype,Yp,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjYm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjYm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Yp,SE,st);
  accumReconYm(result,Uchi);

  // Zm
  SE=st.GetEntry(ptype,Zp,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjZm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjZm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Zp,SE,st);
  accumReconZm(result,Uchi);
    
  // Tm
  SE=st.GetEntry(ptype,Tp,sF);
  if (  SE->_is_local && SE->_permute ) {
    spProjTm(tmp,in._odata[SE->_offset]);
    permute(chi,tmp,ptype);
  } else if ( SE->_is_local ) {
    spProjTm(chi,in._odata[SE->_offset]);
  } else { 
    chi=buf[SE->_offset];
  }
  Impl::multLink(Uchi,U._odata[sU],chi,Tp,SE,st);
  accumReconTm(result,Uchi);
  
  vstream(out._odata[sF],result*(-0.5));
}

template<class Impl> 
void WilsonKernels<Impl>::DiracOptDhopDir(CartesianStencil &st,DoubledGaugeField &U,
					  std::vector<SiteHalfSpinor,alignedAllocator<SiteHalfSpinor> >  &buf,
					  int sF,int sU,const FermionField &in, FermionField &out,int dir,int gamma)
{
  SiteHalfSpinor  tmp;    
  SiteHalfSpinor  chi;    
  SiteSpinor   result;
  SiteHalfSpinor Uchi;
  StencilEntry *SE;
  int ptype;

  SE=st.GetEntry(ptype,dir,sF);

  // Xp
  if(gamma==Xp){
    if (  SE->_is_local && SE->_permute ) {
      spProjXp(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjXp(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconXp(result,Uchi);
  }

  // Yp
  if ( gamma==Yp ){
    if (  SE->_is_local && SE->_permute ) {
      spProjYp(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjYp(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconYp(result,Uchi);
  }
  
  // Zp
  if ( gamma ==Zp ){
    if (  SE->_is_local && SE->_permute ) {
      spProjZp(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjZp(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconZp(result,Uchi);
  }
  
  // Tp
  if ( gamma ==Tp ){
    if (  SE->_is_local && SE->_permute ) {
      spProjTp(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjTp(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconTp(result,Uchi);
  }

  // Xm
  if ( gamma==Xm ){
    if (  SE->_is_local && SE->_permute ) {
      spProjXm(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjXm(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconXm(result,Uchi);
  }

  // Ym
  if ( gamma == Ym ){
    if (  SE->_is_local && SE->_permute ) {
      spProjYm(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjYm(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconYm(result,Uchi);
  }

  // Zm
  if ( gamma == Zm ){
    if (  SE->_is_local && SE->_permute ) {
      spProjZm(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjZm(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconZm(result,Uchi);
  }
  
  // Tm
  if ( gamma==Tm ) {
    if (  SE->_is_local && SE->_permute ) {
      spProjTm(tmp,in._odata[SE->_offset]);
      permute(chi,tmp,ptype);
    } else if ( SE->_is_local ) {
      spProjTm(chi,in._odata[SE->_offset]);
    } else { 
      chi=buf[SE->_offset];
    }
    Impl::multLink(Uchi,U._odata[sU],chi,dir,SE,st);
    spReconTm(result,Uchi);
  }

  vstream(out._odata[sF],result*(-0.5));
}

  FermOpTemplateInstantiate(WilsonKernels);

}}
