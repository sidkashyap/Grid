#ifndef QCD_EVEN_ODD_SCHUR_DIFFERENTIABLE_H
#define QCD_EVEN_ODD_SCHUR_DIFFERENTIABLE_H

namespace Grid{
  namespace QCD{

    // Base even odd HMC on the normal Mee based schur decomposition.
    //
    //     M = (Mee Meo) =  (1             0 )   (Mee   0               )  (1 Mee^{-1} Meo)
    //         (Moe Moo)    (Moe Mee^-1    1 )   (0   Moo-Moe Mee^-1 Meo)  (0   1         )
    //
    // Determinant is det of middle factor
    // This assumes Mee is indept of U.
    //
    template<class Impl>
    class SchurDifferentiableOperator :  public SchurDiagMooeeOperator<FermionOperator<Impl>,typename Impl::FermionField> 
      {
      public:
      INHERIT_IMPL_TYPES(Impl);

 	typedef FermionOperator<Impl> Matrix;

	SchurDifferentiableOperator (Matrix &Mat) : SchurDiagMooeeOperator<Matrix,FermionField>(Mat) {};

	void MpcDeriv(GaugeField &Force,const FermionField &U,const FermionField &V) {
	
	  GridBase *fgrid   = this->_Mat.FermionGrid();
	  GridBase *fcbgrid = this->_Mat.FermionRedBlackGrid();
	  GridBase *ugrid   = this->_Mat.GaugeGrid();
	  GridBase *ucbgrid = this->_Mat.GaugeRedBlackGrid();

	  Real coeff = 1.0;

	  FermionField tmp1(fcbgrid);
	  FermionField tmp2(fcbgrid);

	  conformable(fcbgrid,U._grid);
	  conformable(fcbgrid,V._grid);

	  // Assert the checkerboard?? or code for either
	  assert(U.checkerboard==Odd);
	  assert(V.checkerboard==U.checkerboard);

	  GaugeField ForceO(ucbgrid);
	  GaugeField ForceE(ucbgrid);

	  //  X^dag Der_oe MeeInv Meo Y
	  // Use Mooee as nontrivial but gauge field indept
	  this->_Mat.Meooe   (V,tmp1);      // odd->even -- implicit -0.5 factor to be applied
	  this->_Mat.MooeeInv(tmp1,tmp2);   // even->even 
	  this->_Mat.MoeDeriv(ForceO,U,tmp2,DaggerNo);
	  
	  //  Accumulate X^dag M_oe MeeInv Der_eo Y
	  this->_Mat.MeooeDag   (U,tmp1);    // even->odd -- implicit -0.5 factor to be applied
	  this->_Mat.MooeeInvDag(tmp1,tmp2); // even->even 
	  this->_Mat.MeoDeriv(ForceE,tmp2,V,DaggerNo);
	  
	  assert(ForceE.checkerboard==Even);
	  assert(ForceO.checkerboard==Odd);

	  setCheckerboard(Force,ForceE); 
	  setCheckerboard(Force,ForceO);
	  Force=-Force;
	}


	void MpcDagDeriv(GaugeField &Force,const FermionField &U,const FermionField &V) {
	
	  GridBase *fgrid   = this->_Mat.FermionGrid();
	  GridBase *fcbgrid = this->_Mat.FermionRedBlackGrid();
	  GridBase *ugrid   = this->_Mat.GaugeGrid();
	  GridBase *ucbgrid = this->_Mat.GaugeRedBlackGrid();

	  Real coeff = 1.0;

	  FermionField tmp1(fcbgrid);
	  FermionField tmp2(fcbgrid);

	  conformable(fcbgrid,U._grid);
	  conformable(fcbgrid,V._grid);

	  // Assert the checkerboard?? or code for either
	  assert(V.checkerboard==Odd);
	  assert(V.checkerboard==V.checkerboard);

	  GaugeField ForceO(ucbgrid);
	  GaugeField ForceE(ucbgrid);

	  //  X^dag Der_oe MeeInv Meo Y
	  // Use Mooee as nontrivial but gauge field indept
	  this->_Mat.MeooeDag   (V,tmp1);      // odd->even -- implicit -0.5 factor to be applied
	  this->_Mat.MooeeInvDag(tmp1,tmp2);   // even->even 
	  this->_Mat.MoeDeriv(ForceO,U,tmp2,DaggerYes);
	  
	  //  Accumulate X^dag M_oe MeeInv Der_eo Y
	  this->_Mat.Meooe   (U,tmp1);    // even->odd -- implicit -0.5 factor to be applied
	  this->_Mat.MooeeInv(tmp1,tmp2); // even->even 
	  this->_Mat.MeoDeriv(ForceE,tmp2,V,DaggerYes);

	  assert(ForceE.checkerboard==Even);
	  assert(ForceO.checkerboard==Odd);

	  setCheckerboard(Force,ForceE); 
	  setCheckerboard(Force,ForceO);
	  Force=-Force;
	}

    };

  }
}
#endif
