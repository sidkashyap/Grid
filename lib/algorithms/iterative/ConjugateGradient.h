#ifndef GRID_CONJUGATE_GRADIENT_H
#define GRID_CONJUGATE_GRADIENT_H

namespace Grid {

    /////////////////////////////////////////////////////////////
    // Base classes for iterative processes based on operators
    // single input vec, single output vec.
    /////////////////////////////////////////////////////////////

  template<class Field> 
    class ConjugateGradient : public OperatorFunction<Field> {
public:                                                
    RealD   Tolerance;
    Integer MaxIterations;
    ConjugateGradient(RealD tol,Integer maxit) : Tolerance(tol), MaxIterations(maxit) { 
    };


    void operator() (LinearOperatorBase<Field> &Linop,const Field &src, Field &psi){

      psi.checkerboard = src.checkerboard;
      conformable(psi,src);

      RealD cp,c,a,d,b,ssq,qq,b_pred;
      
      Field   p(src);
      Field mmp(src);
      Field   r(src);
      
      //Initial residual computation & set up
      RealD guess = norm2(psi);
      
      Linop.HermOpAndNorm(psi,mmp,d,b);
      
      r= src-mmp;
      p= r;
      
      a  =norm2(p);
      cp =a;
      ssq=norm2(src);

      std::cout<<GridLogIterative <<std::setprecision(4)<< "ConjugateGradient: guess "<<guess<<std::endl;
      std::cout<<GridLogIterative <<std::setprecision(4)<< "ConjugateGradient:   src "<<ssq  <<std::endl;
      std::cout<<GridLogIterative <<std::setprecision(4)<< "ConjugateGradient:    mp "<<d    <<std::endl;
      std::cout<<GridLogIterative <<std::setprecision(4)<< "ConjugateGradient:   mmp "<<b    <<std::endl;
      std::cout<<GridLogIterative <<std::setprecision(4)<< "ConjugateGradient:  cp,r "<<cp   <<std::endl;
      std::cout<<GridLogIterative <<std::setprecision(4)<< "ConjugateGradient:     p "<<a    <<std::endl;

      RealD rsq =  Tolerance* Tolerance*ssq;
      
      //Check if guess is really REALLY good :)
      if ( cp <= rsq ) {
	return;
      }
      
      std::cout<<GridLogIterative << std::setprecision(4)<< "ConjugateGradient: k=0 residual "<<cp<<" rsq"<<rsq<<std::endl;
      
      int k;
      for (k=1;k<=MaxIterations;k++){
	
	c=cp;
	
	Linop.HermOpAndNorm(p,mmp,d,qq);

	RealD    qqck = norm2(mmp);
	ComplexD dck  = innerProduct(p,mmp);
      
	a      = c/d;
	b_pred = a*(a*qq-d)/c;

	cp = axpy_norm(r,-a,mmp,r);
	b = cp/c;
	
	// Fuse these loops ; should be really easy
	psi= a*p+psi;
	p  = p*b+r;
	  
	std::cout<<GridLogIterative<<"ConjugateGradient: Iteration " <<k<<" residual "<<cp<< " target"<< rsq<<std::endl;
	
	// Stopping condition
	if ( cp <= rsq ) { 
	  
	  Linop.HermOpAndNorm(psi,mmp,d,qq);
	  p=mmp-src;
	  
	  RealD mmpnorm = sqrt(norm2(mmp));
	  RealD psinorm = sqrt(norm2(psi));
	  RealD srcnorm = sqrt(norm2(src));
	  RealD resnorm = sqrt(norm2(p));
	  RealD true_residual = resnorm/srcnorm;

	  std::cout<<GridLogMessage<<"ConjugateGradient: Converged on iteration " <<k
		   <<" computed residual "<<sqrt(cp/ssq)
		   <<" true residual     "<<true_residual
		   <<" target "<<Tolerance<<std::endl;
	  return;
	}
      }
      std::cout<<GridLogMessage<<"ConjugateGradient did NOT converge"<<std::endl;
      assert(0);
    }
  };
}
#endif
