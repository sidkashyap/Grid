#ifndef GRID_QCD_LINALG_UTILS_H
#define GRID_QCD_LINALG_UTILS_H

namespace Grid{
namespace QCD{
////////////////////////////////////////////////////////////////////////
//This file brings additional linear combination assist that is helpful
//to QCD such as chiral projectors and spin matrices applied to one of the inputs.
//These routines support five-D chiral fermions and contain s-subslice indexing 
//on the 5d (rb4d) checkerboarded lattices
////////////////////////////////////////////////////////////////////////
template<class vobj> 
void axpby_ssp(Lattice<vobj> &z, RealD a,const Lattice<vobj> &x,RealD b,const Lattice<vobj> &y,int s,int sp)
{
  z.checkerboard = x.checkerboard;
  conformable(x,y);
  conformable(x,z);
  GridBase *grid=x._grid;
  int Ls = grid->_rdimensions[0];
PARALLEL_FOR_LOOP
  for(int ss=0;ss<grid->oSites();ss+=Ls){ // adds Ls
    vobj tmp = a*x._odata[ss+s]+b*y._odata[ss+sp];
    vstream(z._odata[ss+s],tmp);
  }
}

template<class vobj> 
void ag5xpby_ssp(Lattice<vobj> &z,RealD a,const Lattice<vobj> &x,RealD b,const Lattice<vobj> &y,int s,int sp)
{
  z.checkerboard = x.checkerboard;
  conformable(x,y);
  conformable(x,z);
  GridBase *grid=x._grid;
  int Ls = grid->_rdimensions[0];
  Gamma G5(Gamma::Gamma5);
PARALLEL_FOR_LOOP
  for(int ss=0;ss<grid->oSites();ss+=Ls){ // adds Ls
    vobj tmp;
    tmp = G5*x._odata[ss+s]*a;
    tmp = tmp + b*y._odata[ss+sp];
    vstream(z._odata[ss+s],tmp);
  }
}

template<class vobj> 
void axpbg5y_ssp(Lattice<vobj> &z,RealD a,const Lattice<vobj> &x,RealD b,const Lattice<vobj> &y,int s,int sp)
{
  z.checkerboard = x.checkerboard;
  conformable(x,y);
  conformable(x,z);
  GridBase *grid=x._grid;
  int Ls = grid->_rdimensions[0];
  Gamma G5(Gamma::Gamma5);
PARALLEL_FOR_LOOP
  for(int ss=0;ss<grid->oSites();ss+=Ls){ // adds Ls
    vobj tmp;
    tmp = G5*y._odata[ss+sp]*b;
    tmp = tmp + a*x._odata[ss+s];
    vstream(z._odata[ss+s],tmp);
  }
}

template<class vobj> 
void ag5xpbg5y_ssp(Lattice<vobj> &z,RealD a,const Lattice<vobj> &x,RealD b,const Lattice<vobj> &y,int s,int sp)
{
  z.checkerboard = x.checkerboard;
  conformable(x,y);
  conformable(x,z);
  GridBase *grid=x._grid;
  int Ls = grid->_rdimensions[0];
  Gamma G5(Gamma::Gamma5);
PARALLEL_FOR_LOOP
  for(int ss=0;ss<grid->oSites();ss+=Ls){ // adds Ls
    vobj tmp1;
    vobj tmp2;
    tmp1 = a*x._odata[ss+s]+b*y._odata[ss+sp];
    tmp2 = G5*tmp1;
    vstream(z._odata[ss+s],tmp2);
  }
}

template<class vobj> 
void axpby_ssp_pminus(Lattice<vobj> &z,RealD a,const Lattice<vobj> &x,RealD b,const Lattice<vobj> &y,int s,int sp)
{
  z.checkerboard = x.checkerboard;
  conformable(x,y);
  conformable(x,z);
  GridBase *grid=x._grid;
  int Ls = grid->_rdimensions[0];
PARALLEL_FOR_LOOP
  for(int ss=0;ss<grid->oSites();ss+=Ls){ // adds Ls
    vobj tmp;
    spProj5m(tmp,y._odata[ss+sp]);
    tmp = a*x._odata[ss+s]+b*tmp;
    vstream(z._odata[ss+s],tmp);
  }
}

template<class vobj> 
void axpby_ssp_pplus(Lattice<vobj> &z,RealD a,const Lattice<vobj> &x,RealD b,const Lattice<vobj> &y,int s,int sp)
{
  z.checkerboard = x.checkerboard;
  conformable(x,y);
  conformable(x,z);
  GridBase *grid=x._grid;
  int Ls = grid->_rdimensions[0];
PARALLEL_FOR_LOOP
  for(int ss=0;ss<grid->oSites();ss+=Ls){ // adds Ls
    vobj tmp;
    spProj5p(tmp,y._odata[ss+sp]);
    tmp = a*x._odata[ss+s]+b*tmp;
    vstream(z._odata[ss+s],tmp);
  }
}

template<class vobj> 
void G5R5(Lattice<vobj> &z,const Lattice<vobj> &x)
{
  GridBase *grid=x._grid;
  z.checkerboard = x.checkerboard;
  conformable(x,z);
  int Ls = grid->_rdimensions[0];
  Gamma G5(Gamma::Gamma5);
PARALLEL_FOR_LOOP
  for(int ss=0;ss<grid->oSites();ss+=Ls){ // adds Ls
    vobj tmp;
    for(int s=0;s<Ls;s++){
      int sp = Ls-1-s;
      tmp = G5*x._odata[ss+s];
      vstream(z._odata[ss+sp],tmp);
    }
  }
}

}}
#endif 
