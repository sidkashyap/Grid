#include <Grid.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;


template<class d>
struct scal {
  d internal;
};

  Gamma::GammaMatrix Gmu [] = {
    Gamma::GammaX,
    Gamma::GammaY,
    Gamma::GammaZ,
    Gamma::GammaT
  };

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);



  const int Ls=8;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);


  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);

  LatticeFermion    src(FGrid); random(RNG5,src);
  LatticeFermion result(FGrid); result=zero;
  LatticeGaugeField Umu(UGrid); random(RNG4,Umu);

  std::vector<LatticeColourMatrix> U(4,UGrid);

  for(int mu=0;mu<Nd;mu++){
    U[mu] = PeekIndex<LorentzIndex>(Umu,mu);
  }

  ConjugateResidual<LatticeFermion> MCR(1.0e-8,10000);
  
  RealD mass=0.5;
  RealD M5=1.8;
  DomainWallFermionR Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);

  MdagMLinearOperator<DomainWallFermionR,LatticeFermion> HermOp(Ddwf);
  MCR(HermOp,src,result);

  Gamma5R5HermitianLinearOperator<DomainWallFermionR,LatticeFermion> g5HermOp(Ddwf);
  MCR(g5HermOp,src,result);


  Grid_finalize();
}
