/**
 *
 *  $Date: 2012-09-05 16:59:15 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 231 $
 *  $Author: jross $
 *
 */

#ifndef _cipNelderMeadSimplexOptimizer_cxx
#define _cipNelderMeadSimplexOptimizer_cxx

#include "cipNelderMeadSimplexOptimizer.h"
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>

cipNelderMeadSimplexOptimizer::cipNelderMeadSimplexOptimizer( unsigned int dimension )
{
  this->Dimension                = dimension;
  this->OptimalParams            = new double[this->Dimension];
  this->InitialParams            = new double[this->Dimension];
  this->NumberOfIterations       = 100;

  //
  // The initial simplex edge length will be used for the initial
  // simplex construction. A value of 3 is sufficient to cover a 
  // broad range parameter space locations about the initial parameter
  // location
  //
  this->InitialSimplexEdgeLength = 3.0; 
}


cipNelderMeadSimplexOptimizer::~cipNelderMeadSimplexOptimizer()
{
}


void cipNelderMeadSimplexOptimizer::SetInitialParameters( double* params )
{  
  for ( unsigned int i=0; i<this->Dimension; i++ )
    {
    this->InitialParams[i] = params[i];
    }
}


void cipNelderMeadSimplexOptimizer::GetOptimalParameters( double* params )
{
  for ( unsigned int i=0; i<this->Dimension; i++ )
    {
    params[i] = this->OptimalParams[i];
    }
}


//
// This method initializes the simplex for a given set of initial
// parameters and initial simplex edge length
//
void cipNelderMeadSimplexOptimizer::InitializeSimplex()
{
  //
  // 'flipper' will help create the vertex locations by flipping the
  // signs of the vertex coordinates during construction
  //
  std::vector< double > flipper;
  flipper.assign( this->Dimension, 1.0 );

  //
  // Note that the number of simplex vertices is one more than the
  // dimension of the objective function
  //
  for ( unsigned int i=0; i<this->Dimension+1; i++ )
    {
    SIMPLEXVERTEX vertex;
   
    //
    // The last vertex will be at the initial parameter values
    // specified by the user
    //
    if ( i==this->Dimension )
      {
      flipper.assign( this->Dimension, 0.0 );
      }

    //
    // Construct the coordinates for the current vertex with the help
    // of 'flipper' and the initial simplex edge length. 
    //
    for ( unsigned int j=0; j<this->Dimension; j++ )
      {
      double point = this->InitialParams[j] + flipper[j]*this->InitialSimplexEdgeLength/2.0;

      vertex.coordinates.push_back( point );
      }
    
    this->SimplexVertices.push_back( vertex );

    //
    // Update flipper so that it will create a novel vertex location
    // at the next iteration
    //
    for ( unsigned int j=0; j<this->Dimension; j++ )
      {
      if ( flipper[j] == 1.0 )
        {
        flipper[j] = -1.0;
        break;
        }
      }
    }
}


//
// At any given time during the optimization routine, this method can
// be called to update the vertex rankings wrt to their objective
// function values. Additionally, the "best value", "worst value",
// "best index", "worst index", and "worst runner-up" quantitiest are
// updated. These are needed for proper control flow of the algorithm.
//
void cipNelderMeadSimplexOptimizer::UpdateRankings()
{
  std::list< double > valueList;

  for ( unsigned int i=0; i<this->Dimension+1; i++ )
    {    
    valueList.push_back( this->SimplexVertices[i].value );

    //
    // Fill all 'rank' entries with a dummy variable so that we can
    // keep track of the vertices that have yet to have a ranking
    // assigned to them
    //
    this->SimplexVertices[i].rank = this->Dimension+2;
    }

  valueList.sort();

  std::list< double >::iterator listIt;

  unsigned int ranking = 0;
  for ( listIt = valueList.begin(); listIt != valueList.end(); listIt++ )
    {
    for ( unsigned int i=0; i<this->Dimension+1; i++ )
      {
      if ( this->SimplexVertices[i].value == *listIt && this->SimplexVertices[i].rank == this->Dimension+2 )
        {     
        this->SimplexVertices[i].rank = ranking;

        if ( ranking == 0 )
          {
          this->BestValue = *listIt;
          this->BestIndex = i;
          }
        if ( ranking == this->Dimension )
          {
          this->WorstValue = *listIt;
          this->WorstIndex = i;
          }
        if ( ranking == this->Dimension-1 )
          {
          this->WorstRunnerUpValue = *listIt;
          }  

        ranking++;      
        }
      }
    }

  for ( unsigned int i=0; i<this->Dimension; i++ )
    {
    this->OptimalParams[i] = this->SimplexVertices[this->BestIndex].coordinates[i];
    }

  std::cout << "val:\t" << this->BestValue << std::endl;
}


//
// Calling 'Update' will start the optimization
//
void cipNelderMeadSimplexOptimizer::Update()
{
  // We'll need a couple of boolean variables to keep track of
  // attempted contractions. Declare them here
  bool insideContractionSuccessful;
  bool outsideContractionSuccessful;

  // Construct the initial simplex
  this->InitializeSimplex();

  // Evalute the metric at each of these points. In the process,
  // identify the "worst" point, the "best" point, and keep a list of
  // the metric values at each vertex that we can sort after all the
  // computations N
  for ( unsigned int i=0; i<this->Dimension+1; i++ )
    {       
      this->SimplexVertices[i].value = this->Metric->GetValue( &(this->SimplexVertices[i].coordinates) );
    }

  // Now determine the ranking of each of these vertices wrt to the
  // objective function value
  this->UpdateRankings();

  // Following is the Nelder-Mead simplex reflection
  // algorithm. Iterate for the number of iterations specified by the
  // user 
  for ( unsigned int it=0; it<this->NumberOfIterations; it++ )
    {
    // Compute the center of gravity of the n best vertices 
    std::vector< double > xBar;
    for ( unsigned int i=0; i<this->Dimension; i++ )
      {
      xBar.push_back( 0.0 );
      
      for ( unsigned int j=0; j<this->Dimension+1; j++ )
        {
        if ( j != this->WorstIndex )
          {        
          xBar[i] += this->SimplexVertices[j].coordinates[i];
          }
        }
      
      xBar[i] /= static_cast< double >( this->Dimension );
      }

    // Now compute 'xBarNeg1'
    std::vector< double > xBarNeg1;
    for ( unsigned int i=0; i<this->Dimension; i++ )
      {
      double temp = xBar[i]-(this->SimplexVertices[this->WorstIndex].coordinates[i]-xBar[i]);
      xBarNeg1.push_back( temp );
      }

    // Evaluate at 'xBarNeg1'
    double fNeg1 = this->Metric->GetValue( &xBarNeg1 );

    if ( this->BestValue <= fNeg1 && fNeg1 < this->WorstRunnerUpValue )
      {
      // 'fNeg1' is neither the best value nor the worst. Replace the
      // worst coordinates with the coordinates at 'xBarNeg1'
      for ( unsigned int i=0; i<this->Dimension; i++ )
        {
        this->SimplexVertices[this->WorstIndex].coordinates[i] = xBarNeg1[i];
        }
      this->SimplexVertices[this->WorstIndex].value = fNeg1;
      this->UpdateRankings();
      }
    else if ( fNeg1 < this->BestValue )
      {
      // 'fNeg1' is better than our current best, so try to go farther
      // in this direction. Compute 'xBarNeg1'
      std::vector< double > xBarNeg2;
      for ( unsigned int i=0; i<this->Dimension; i++ )
        {
        double temp = xBar[i]-2.0*(this->SimplexVertices[this->WorstIndex].coordinates[i]-xBar[i]);
        xBarNeg2.push_back( temp );
        }
      
      // Now evaluate at 'xBarNeg2'
      double fNeg2 = this->Metric->GetValue( &xBarNeg2 );
      
      if ( fNeg2 < fNeg1 )
        {
        // 'fNeg2' is even better than 'fNeg1', so replace our worst
        // coordinates with the coordinates at 'xBarNeg2'

        for ( unsigned int i=0; i<this->Dimension; i++ )
          {
          this->SimplexVertices[this->WorstIndex].coordinates[i] = xBarNeg2[i];
          }

        this->SimplexVertices[this->WorstIndex].value = fNeg2;
        this->UpdateRankings();
        }
      else
        { 
        // 'fNeg2' is not better than 'fNeg1', so just replace the worst
        // coordinates with 'xBarNeg1'
        for ( unsigned int i=0; i<this->Dimension; i++ )
          {
          this->SimplexVertices[this->WorstIndex].coordinates[i] = xBarNeg1[i];
          }      
        this->SimplexVertices[this->WorstIndex].value = fNeg1;
        this->UpdateRankings();
        }
      }
    else if ( fNeg1 >= this->WorstRunnerUpValue )
      {
      // The reflected point, 'xBarNeg1', is still the worst, so
      // contract
      outsideContractionSuccessful = false;
      insideContractionSuccessful  = false;

      if ( this->WorstRunnerUpValue <= fNeg1 && fNeg1 < this->WorstValue )
        {
        // Try "outside" contraction
        std::vector< double > xBarNegHalf;
        for ( unsigned int i=0; i<this->Dimension; i++ )
          {
          double temp = xBar[i]-0.5*(this->SimplexVertices[this->WorstIndex].coordinates[i]-xBar[i]);
          xBarNegHalf.push_back( temp );
          }
        
        // Now evaluate at 'xBarNegHalf'
        double fNegHalf = this->Metric->GetValue( &xBarNegHalf );
        
        if ( fNegHalf <= fNeg1 )
          {
          outsideContractionSuccessful = true;
                    
          // 'fNegHalf' is better than 'fNeg1', so replace the worst 
          // coordinates with 'xBarNegHalf'
          for ( unsigned int i=0; i<this->Dimension; i++ )
            {
            this->SimplexVertices[this->WorstIndex].coordinates[i] = xBarNegHalf[i];
            }      
          this->SimplexVertices[this->WorstIndex].value = fNegHalf;
          this->UpdateRankings();
          }
        }
      else
        {
        // Try "inside" contraction
        std::vector< double > xBarPosHalf;
        for ( unsigned int i=0; i<this->Dimension; i++ )
          {
          double temp = xBar[i]+0.5*(this->SimplexVertices[this->WorstIndex].coordinates[i]-xBar[i]);
          xBarPosHalf.push_back( temp );
          }
        
        // Now evaluate at 'xBarPosHalf'
        double fPosHalf = this->Metric->GetValue( &xBarPosHalf );
        
        if ( fPosHalf < this->WorstValue )
          {
          insideContractionSuccessful = true;
          
          for ( unsigned int i=0; i<this->Dimension; i++ )
            {
            this->SimplexVertices[this->WorstIndex].coordinates[i] = xBarPosHalf[i];
            }      
          this->SimplexVertices[this->WorstIndex].value = fPosHalf;
          this->UpdateRankings();
          }
        }

      if ( !insideContractionSuccessful && !outsideContractionSuccessful )
        {
        // Neither "inside" nor "outside" contraction was successful, so
        // shrink all points towards the current best
        for ( unsigned int j=0; j<this->Dimension+1; j++ )
          {        
          if ( j != this->BestIndex )
            {
            for ( unsigned int i=0; i<this->Dimension; i++ )
              {
              this->SimplexVertices[j].coordinates[i] = 0.5*(this->SimplexVertices[this->BestIndex].coordinates[i] 
                                                               + this->SimplexVertices[j].coordinates[i]);
              } 
            this->SimplexVertices[j].value = this->Metric->GetValue( &this->SimplexVertices[j].coordinates );            
            }  
          }
        this->UpdateRankings();
        }
      } 
    }
}


#endif
