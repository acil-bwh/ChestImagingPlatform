/**
 *
 *  $Date: 2012-09-05 16:59:15 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 231 $
 *  $Author: jross $
 *
 */

#ifndef _cipNewtonOptimizer_cxx
#define _cipNewtonOptimizer_cxx


#include "cipNewtonOptimizer.h"
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>


template < unsigned int Dimension >
cipNewtonOptimizer< Dimension >
::cipNewtonOptimizer()
{
  this->InitialParams = new PointType( Dimension, Dimension );
  this->OptimalParams = new PointType( Dimension, Dimension );

  this->GradientDifferenceTolerance = 0.5;
  this->SufficientDecreaseFactor    = 0.0001;
  this->Rho                         = 0.9;
}


template < unsigned int Dimension >
cipNewtonOptimizer< Dimension >
::~cipNewtonOptimizer()
{
}


template < unsigned int Dimension >
void cipNewtonOptimizer< Dimension >::SetInitialParameters( PointType* params )
{
  *this->InitialParams = *params;
}


template < unsigned int Dimension >
void cipNewtonOptimizer< Dimension >::GetOptimalParameters( PointType* params )
{
  (*params) = (*this->OptimalParams);
}


//DEB
template < unsigned int Dimension >
void cipNewtonOptimizer< Dimension >::Update( bool verbose )
{
  PointType* params = new PointType;
  (*params) = (*this->InitialParams);

  //
  // Initialize OptimalParams. It will be updated during the
  // optimization process below
  //
  (*this->OptimalParams) = (*params);

  double a;

  VectorType* g    = new VectorType( Dimension ); // The gradient
  VectorType* p    = new VectorType( Dimension ); // Search direction
  MatrixType* hInv = new MatrixType( Dimension, Dimension ); // Hessian inverse

  vnl_vector< double >  d( Dimension );
  vnl_matrix< double >  h( Dimension, Dimension );
  vnl_matrix< double >  v( Dimension, Dimension );

  //
  // Get the gradient and hessian at the current param location. Also
  // initialize OptimalValue (it will be updated during the
  // optimization below)
  //
  this->OptimalValue = this->Metric.GetValueGradientAndHessian( params, g, &h );

  double gradMag = std::sqrt( dot_product(*g,*g) );
  double gradMagDiff = DBL_MAX;
  double gradMagLast = gradMag;
  while ( gradMagDiff > this->GradientDifferenceTolerance )
    {
    //
    // Check for positive definiteness of the Hessian. If not positive
    // definite, perform Hessian modification by flipping the sign of
    // negative eigenvalues and reconstruction
    //
    vnl_symmetric_eigensystem_compute( h, v, d );

    if ( d[0] < 0 || d[1] < 0 )
      {
      h = std::abs(d[0])*outer_product(v.get_column(0),v.get_column(0)) + 
        std::abs(d[1])*outer_product(v.get_column(1),v.get_column(1));
      }

    (*hInv) =  vnl_matrix_inverse<double>(h);    

    (*p) = -(*hInv)*(*g); // Newton step    
    
    //
    // Determine the step length
    //
    a = this->LineSearch( params, p );
    
    (*params) = (*params) + a*(*p);
    
    //
    // Note that 'OptimalValue' and 'OptimalParams' are set at
    // each iteration, but upon convergence they will be set to the
    // final ("optimal") values
    //
    this->OptimalValue = this->Metric.GetValueGradientAndHessian( params, g, &h );
    (*this->OptimalParams) = (*params);
    
    gradMag = std::sqrt( dot_product(*g,*g) );
    gradMagDiff = gradMagLast - gradMag;
    gradMagLast = gradMag;
    
    if ( verbose )
      {
      std::cout << "Dimension:\t" << Dimension << std::endl;
      std::cout << "this->GradientTolerance:\t" << this->GradientTolerance << std::endl;
      std::cout << "this->OptimalValue:\t" << this->OptimalValue << std::endl;
      std::cout << "Hessian:\t" << h[0][0] << "\t" << h[0][1] << "\t" << h[1][0] << "\t" << h[1][1] << "\t" << std::endl;
      std::cout << "Hessian Inv:\t" << (*hInv)[0][0] << "\t" << (*hInv)[0][1] << "\t" << (*hInv)[1][0] << "\t" << (*hInv)[1][1] << "\t" << std::endl;
      std::cout << "a:\t" << a << std::endl;
//      std::cout << "p:\t" << p[0] << "\t" << p[1] << std::endl;
      std::cout << "gradMag:\t" << gradMag << std::endl;
      }
    }

  delete params;
  delete g;
  delete p;
  delete hInv;
}


template < unsigned int Dimension >
void cipNewtonOptimizer< Dimension >::Update()
{
  PointType* params = new PointType;
  (*params) = (*this->InitialParams);

  //
  // Initialize OptimalParams. It will be updated during the
  // optimization process below
  //
  (*this->OptimalParams) = (*params);

  double a;

  VectorType* g    = new VectorType( Dimension ); // The gradient
  VectorType* p    = new VectorType( Dimension ); // Search direction
  MatrixType* hInv = new MatrixType( Dimension, Dimension ); // Hessian inverse

  vnl_vector< double >  d( Dimension );
  vnl_matrix< double >  h( Dimension, Dimension );
  vnl_matrix< double >  v( Dimension, Dimension );

  //
  // Get the gradient and hessian at the current param location. Also
  // initialize OptimalValue (it will be updated during the
  // optimization below)
  //
  this->OptimalValue = this->Metric.GetValueGradientAndHessian( params, g, &h );

  double gradMag = std::sqrt( dot_product(*g,*g) );
  double gradMagDiff = DBL_MAX;
  double gradMagLast = gradMag;
  while ( gradMagDiff > this->GradientDifferenceTolerance )
    {
    //
    // Check for positive definiteness of the Hessian. If not positive
    // definite, perform Hessian modification by flipping the sign of
    // negative eigenvalues and reconstruction
    //
    vnl_symmetric_eigensystem_compute( h, v, d );

    if ( d[0] < 0 || d[1] < 0 )
      {
      h = std::abs(d[0])*outer_product(v.get_column(0),v.get_column(0)) + 
        std::abs(d[1])*outer_product(v.get_column(1),v.get_column(1));
      }

    (*hInv) =  vnl_matrix_inverse<double>(h);    

    (*p) = -(*hInv)*(*g); // Newton step    
    
    //
    // Determine the step length
    //
    a = this->LineSearch( params, p );
    
    (*params) = (*params) + a*(*p);
    
    //
    // Note that 'OptimalValue' and 'OptimalParams' are set at
    // each iteration, but upon convergence they will be set to the
    // final ("optimal") values
    //
    this->OptimalValue = this->Metric.GetValueGradientAndHessian( params, g, &h );

    (*this->OptimalParams) = (*params);
    
    gradMag = std::sqrt( dot_product(*g,*g) );
    gradMagDiff = gradMagLast - gradMag;
    gradMagLast = gradMag;
    }

  delete params;
  delete g;
  delete p;
  delete hInv;
}


//
// This function determines a step length, 'aOpt', along a search
// direction, 'p', from initial position, 'a0', such that the
// sufficient decrease condition is satisfied. The algorithm used is
// backtracking.
//
template < unsigned int Dimension >
double cipNewtonOptimizer< Dimension >::LineSearch( PointType* x0, VectorType* p )
{
  double a = 1.0;

  VectorType* g = new VectorType( Dimension );

  //
  // Get the metric value, 'f0', and the gradient, 'g', at initial
  // point, 'x0'.
  //
  double f0 = this->Metric.GetValueAndGradient( x0, g ); 

  //
  // Compute the following product to save a few computations 
  //
  double gOrig[2];
    gOrig[0] = (*g)[0];
    gOrig[1] = (*g)[1];

  double g0p = dot_product(*g,*p);

  //
  // The first condition in the while loop below evaluates the sufficient
  // decrease condition. This is all that is needed for the
  // backtracking algorithm
  // 
  PointType* params = new PointType;
  (*params) = (*x0) + a*(*p);

  double value = this->Metric.GetValueAndGradient( params, g );
  
  while ( value > f0 + this->SufficientDecreaseFactor*a*g0p )
    {
    a *= this->Rho;   

    (*params) = (*x0) + a*(*p);

    value = this->Metric.GetValueAndGradient( params, g );
    }

  double aOpt = a;

  delete g;
  delete params;

  return aOpt;
}

#endif
