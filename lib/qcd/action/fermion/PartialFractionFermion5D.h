#ifndef  GRID_QCD_PARTIAL_FRACTION_H
#define  GRID_QCD_PARTIAL_FRACTION_H

namespace Grid {

  namespace QCD {

    template<class Impl>
    class PartialFractionFermion5D : public WilsonFermion5D<Impl>
    {
    public:
     INHERIT_IMPL_TYPES(Impl);

      const int part_frac_chroma_convention=1;

      void   Meooe_internal(const FermionField &in, FermionField &out,int dag);
      void   Mooee_internal(const FermionField &in, FermionField &out,int dag);
      void   MooeeInv_internal(const FermionField &in, FermionField &out,int dag);
      void   M_internal(const FermionField &in, FermionField &out,int dag);

      // override multiply
      virtual RealD  M    (const FermionField &in, FermionField &out);
      virtual RealD  Mdag (const FermionField &in, FermionField &out);

      // half checkerboard operaions
      virtual void   Meooe       (const FermionField &in, FermionField &out);
      virtual void   MeooeDag    (const FermionField &in, FermionField &out);
      virtual void   Mooee       (const FermionField &in, FermionField &out);
      virtual void   MooeeDag    (const FermionField &in, FermionField &out);
      virtual void   MooeeInv    (const FermionField &in, FermionField &out);
      virtual void   MooeeInvDag (const FermionField &in, FermionField &out);

      // force terms; five routines; default to Dhop on diagonal
      virtual void MDeriv  (GaugeField &mat,const FermionField &U,const FermionField &V,int dag);
      virtual void MoeDeriv(GaugeField &mat,const FermionField &U,const FermionField &V,int dag);
      virtual void MeoDeriv(GaugeField &mat,const FermionField &U,const FermionField &V,int dag);

      virtual void   Instantiatable(void) =0; // ensure no make-eee

      // Efficient support for multigrid coarsening
      virtual void  Mdir (const FermionField &in, FermionField &out,int dir,int disp);

      // Constructors
      PartialFractionFermion5D(GaugeField &_Umu,
			       GridCartesian         &FiveDimGrid,
			       GridRedBlackCartesian &FiveDimRedBlackGrid,
			       GridCartesian         &FourDimGrid,
			       GridRedBlackCartesian &FourDimRedBlackGrid,
			       RealD _mass,RealD M5,const ImplParams &p= ImplParams());

    protected:

      virtual void SetCoefficientsTanh(Approx::zolotarev_data *zdata,RealD scale);
      virtual void SetCoefficientsZolotarev(RealD zolo_hi,Approx::zolotarev_data *zdata);

      // Part frac
      RealD mass;
      RealD dw_diag;
      RealD R;
      RealD amax;
      RealD scale;
      std::vector<double> p; 
      std::vector<double> q;

    };


  }
}

#endif
