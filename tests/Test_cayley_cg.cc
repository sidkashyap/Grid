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

template<class What> 
void  TestCGinversions(What & Ddwf, 
		       GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		       GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		       RealD mass, RealD M5,
		       GridParallelRNG *RNG4,
		       GridParallelRNG *RNG5);
template<class What> 
void  TestCGschur(What & Ddwf, 
		  GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		  GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		  RealD mass, RealD M5,
		  GridParallelRNG *RNG4,
		  GridParallelRNG *RNG5);

template<class What> 
void  TestCGunprec(What & Ddwf, 
		   GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		   GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		   RealD mass, RealD M5,
		   GridParallelRNG *RNG4,
		   GridParallelRNG *RNG5);

template<class What> 
void  TestCGprec(What & Ddwf, 
		 GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		 GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		 RealD mass, RealD M5,
		 GridParallelRNG *RNG4,
		 GridParallelRNG *RNG5);

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  int threads = GridThread::GetThreads();
  std::cout<<GridLogMessage << "Grid is setup to use "<<threads<<" threads"<<std::endl;

  const int Ls=8;
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);


  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);

  LatticeGaugeField Umu(UGrid); random(RNG4,Umu);
  std::vector<LatticeColourMatrix> U(4,UGrid);

  RealD mass=0.1;
  RealD M5  =1.8;
  std::cout<<GridLogMessage <<"DomainWallFermion test"<<std::endl;
  DomainWallFermionR Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);
  TestCGinversions<DomainWallFermionR>(Ddwf,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,&RNG4,&RNG5);

  RealD b=1.5;// Scale factor b+c=2, b-c=1
  RealD c=0.5;
  std::cout<<GridLogMessage <<"MobiusFermion test"<<std::endl;
  MobiusFermionR Dmob(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);
  TestCGinversions<MobiusFermionR>(Dmob,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,&RNG4,&RNG5);

  std::cout<<GridLogMessage <<"MobiusZolotarevFermion test"<<std::endl;
  MobiusZolotarevFermionR Dzolo(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c,0.1,2.0);
  TestCGinversions<MobiusZolotarevFermionR>(Dzolo,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,&RNG4,&RNG5);

  std::cout<<GridLogMessage <<"ScaledShamirFermion test"<<std::endl;
  ScaledShamirFermionR Dsham(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,2.0);
  TestCGinversions<ScaledShamirFermionR>(Dsham,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,&RNG4,&RNG5);

  std::cout<<GridLogMessage <<"ShamirZolotarevFermion test"<<std::endl;
  ShamirZolotarevFermionR Dshamz(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,0.1,2.0);
  TestCGinversions<ShamirZolotarevFermionR>(Dshamz,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,&RNG4,&RNG5);

  std::cout<<GridLogMessage <<"OverlapWilsonCayleyTanhFermion test"<<std::endl;
  OverlapWilsonCayleyTanhFermionR Dov(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,1.0);
  TestCGinversions<OverlapWilsonCayleyTanhFermionR>(Dov,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,&RNG4,&RNG5);

  std::cout<<GridLogMessage <<"OverlapWilsonCayleyZolotarevFermion test"<<std::endl;
  OverlapWilsonCayleyZolotarevFermionR Dovz(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,0.1,2.0);
  TestCGinversions<OverlapWilsonCayleyZolotarevFermionR>(Dovz,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,&RNG4,&RNG5);

  Grid_finalize();
}
template<class What> 
void  TestCGinversions(What & Ddwf, 
		       GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		       GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		       RealD mass, RealD M5,
		       GridParallelRNG *RNG4,
		       GridParallelRNG *RNG5)
{
  std::cout<<GridLogMessage << "Testing unpreconditioned inverter"<<std::endl;
  TestCGunprec<What>(Ddwf,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,RNG4,RNG5);
  std::cout<<GridLogMessage << "Testing red black preconditioned inverter"<<std::endl;
  TestCGprec<What>(Ddwf,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,RNG4,RNG5);
  std::cout<<GridLogMessage << "Testing red black Schur inverter"<<std::endl;
  TestCGschur<What>(Ddwf,FGrid,FrbGrid,UGrid,UrbGrid,mass,M5,RNG4,RNG5);
}

template<class What> 
void  TestCGunprec(What & Ddwf, 
		   GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		   GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		   RealD mass, RealD M5,
		   GridParallelRNG *RNG4,
		   GridParallelRNG *RNG5)
{
  LatticeFermion src   (FGrid); random(*RNG5,src);
  LatticeFermion result(FGrid); result=zero;

  MdagMLinearOperator<What,LatticeFermion> HermOp(Ddwf);
  ConjugateGradient<LatticeFermion> CG(1.0e-8,10000);
  CG(HermOp,src,result);

}
template<class What> 
void  TestCGprec(What & Ddwf, 
		 GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		 GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		 RealD mass, RealD M5,
		 GridParallelRNG *RNG4,
		 GridParallelRNG *RNG5)
{
  LatticeFermion src   (FGrid); random(*RNG5,src);
  LatticeFermion    src_o(FrbGrid);
  LatticeFermion result_o(FrbGrid);
  pickCheckerboard(Odd,src_o,src);
  result_o=zero;

  SchurDiagMooeeOperator<What,LatticeFermion> HermOpEO(Ddwf);
  ConjugateGradient<LatticeFermion> CG(1.0e-8,10000);
  CG(HermOpEO,src_o,result_o);
}


template<class What> 
void  TestCGschur(What & Ddwf, 
		   GridCartesian         * FGrid,	       GridRedBlackCartesian * FrbGrid,
		   GridCartesian         * UGrid,	       GridRedBlackCartesian * UrbGrid,
		   RealD mass, RealD M5,
		   GridParallelRNG *RNG4,
		   GridParallelRNG *RNG5)
{
  LatticeFermion src   (FGrid); random(*RNG5,src);
  LatticeFermion result(FGrid); result=zero;

  ConjugateGradient<LatticeFermion> CG(1.0e-8,10000);
  SchurRedBlackDiagMooeeSolve<LatticeFermion> SchurSolver(CG);
  SchurSolver(Ddwf,src,result);
}
