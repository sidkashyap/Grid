//--------------------------------------------------------------------
/*! @file HMC.h
 * @brief Classes for Hybrid Monte Carlo update
 *
 * @author Guido Cossu
 * Time-stamp: <2015-07-30 16:58:26 neo>
 */
//--------------------------------------------------------------------
#ifndef HMC_INCLUDED
#define HMC_INCLUDED

#include <string>


namespace Grid{
  namespace QCD{
    

    struct HMCparameters{
      Integer Nsweeps; /* @brief Number of sweeps in this run */
      Integer TotalSweeps; /* @brief If provided, the total number of sweeps */
      Integer ThermalizationSteps;
      Integer StartingConfig;
      Integer SaveInterval; //Setting to 0 does not save configurations
      std::string Filename_prefix; // To save configurations and rng seed
      
      HMCparameters();
    };
    
    //    template <class GaugeField, class Integrator, class Smearer, class Boundary> 
    template <class GaugeField, class IntegratorType>
    class HybridMonteCarlo {
    private:

      const HMCparameters Params;
      
      GridSerialRNG   &sRNG;   // Fixme: need a RNG management strategy.
      GridParallelRNG &pRNG; // Fixme: need a RNG management strategy.

      IntegratorType &TheIntegrator;

      /////////////////////////////////////////////////////////
      // Metropolis step
      /////////////////////////////////////////////////////////
      bool metropolis_test(const RealD DeltaH){

	RealD rn_test;

	RealD prob = std::exp(-DeltaH);

	random(sRNG,rn_test);
      
	std::cout<<GridLogMessage<< "--------------------------------------------\n";
	std::cout<<GridLogMessage<< "dH = "<<DeltaH << "  Random = "<< rn_test <<"\n";
	std::cout<<GridLogMessage<< "Acc. Probability = " << ((prob<1.0)? prob: 1.0)<< "   ";
      
	if((prob >1.0) || (rn_test <= prob)){       // accepted
	  std::cout<<GridLogMessage <<"-- ACCEPTED\n";
	  return true;
	} else {                               // rejected
	  std::cout<<GridLogMessage <<"-- REJECTED\n";
	  return false;
	}

      }

      /////////////////////////////////////////////////////////
      // Evolution
      /////////////////////////////////////////////////////////
      RealD evolve_step(GaugeField& U){

	TheIntegrator.refresh(U,pRNG); // set U and initialize P and phi's 

	RealD H0 = TheIntegrator.S(U); // initial state action  

	std::cout<<GridLogMessage<<"Total H before = "<< H0 << "\n";

	TheIntegrator.integrate(U);
      
	RealD H1 = TheIntegrator.S(U); // updated state action            

	std::cout<<GridLogMessage<<"Total H after = "<< H1 << "\n";

	return (H1-H0);
      }
      
    public:

      /////////////////////////////////////////
      // Constructor
      /////////////////////////////////////////
     HybridMonteCarlo(HMCparameters Pms,  IntegratorType &_Int, GridSerialRNG &_sRNG, GridParallelRNG &_pRNG ) :
        Params(Pms), 
	TheIntegrator(_Int), 
	sRNG(_sRNG),
	pRNG(_pRNG)
      {
      }
      ~HybridMonteCarlo(){};


      void evolve(GaugeField& Uin){

	Real DeltaH;
	
	// Thermalizations
	for(int iter=1; iter <= Params.ThermalizationSteps; ++iter){
	  std::cout<<GridLogMessage << "-- # Thermalization step = "<< iter <<  "\n";
	
	  DeltaH = evolve_step(Uin);
	  std::cout<<GridLogMessage<< "dH = "<< DeltaH << "\n";
	}

	// Actual updates (evolve a copy Ucopy then copy back eventually)
	GaugeField Ucopy(Uin._grid);
	for(int iter=Params.StartingConfig; iter < Params.Nsweeps+Params.StartingConfig; ++iter){
	  std::cout<<GridLogMessage << "-- # Sweep = "<< iter <<  "\n";
	  
	  Ucopy = Uin;

	  DeltaH = evolve_step(Ucopy);
		
	  if(metropolis_test(DeltaH)) Uin = Ucopy;

	}
      }
    };
    
  }// QCD
}// Grid


#endif 
