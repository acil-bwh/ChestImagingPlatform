/**
 *
 *  $Date: 2012-09-06 16:05:10 -0400 (Thu, 06 Sep 2012) $
 *  $Revision: 250 $
 *  $Author: jross $
 *
 */

#include "cipLobeBoundaryShapeModel.h"
#include "vnl/vnl_math.h"

cipLobeBoundaryShapeModel::cipLobeBoundaryShapeModel()
{
  this->ImageOrigin  = new double[3];
  this->ImageSpacing = new double[3];
}


cipLobeBoundaryShapeModel::~cipLobeBoundaryShapeModel()
{
  delete[] this->ImageOrigin;
  delete[] this->ImageSpacing;

  this->MeanSurfacePoints.clear();
  this->WeightedSurfacePoints.clear();
  this->Eigenvalues.clear();
  this->Eigenvectors.clear();
  this->ModeWeights.clear();
}


void cipLobeBoundaryShapeModel::SetImageOrigin( double const* origin )
{
  this->ImageOrigin[0] = origin[0];
  this->ImageOrigin[1] = origin[1];
  this->ImageOrigin[2] = origin[2];
}


double const* cipLobeBoundaryShapeModel::GetImageOrigin() const
{
  return this->ImageOrigin;
}


void cipLobeBoundaryShapeModel::SetImageSpacing( double const* spacing )
{
  this->ImageSpacing[0] = spacing[0];
  this->ImageSpacing[1] = spacing[1];
  this->ImageSpacing[2] = spacing[2];
}


double const* cipLobeBoundaryShapeModel::GetImageSpacing() const
{
  return this->ImageSpacing;
}


void cipLobeBoundaryShapeModel::SetEigenvalueSum( double sum )
{
  this->EigenvalueSum = sum;
}


double cipLobeBoundaryShapeModel::GetEigenvalueSum() const
{
  return this->EigenvalueSum;
}


void cipLobeBoundaryShapeModel::SetMeanSurfacePoints( std::vector< double* > const* points )
{
  this->MeanSurfacePoints.clear();

  for ( unsigned int i=0; i<(*points).size(); i++ )
    {
    double* point = new double[3];
      point[0] = (*points)[i][0];
      point[1] = (*points)[i][1];
      point[2] = (*points)[i][2];

    this->MeanSurfacePoints.push_back( point );
    }
}


std::vector< double* > const* cipLobeBoundaryShapeModel::GetMeanSurfacePoints() const
{
  return &this->MeanSurfacePoints;
}


std::vector< double* > const* cipLobeBoundaryShapeModel::GetWeightedSurfacePoints()
{
  this->ComputeWeightedSurfacePoints();

  return &this->WeightedSurfacePoints;
}


void cipLobeBoundaryShapeModel::ComputeWeightedSurfacePoints()
{
  this->WeightedSurfacePoints.clear();
  
  for ( unsigned int i=0; i<this->MeanSurfacePoints.size(); i++ )
    {
    double* point = new double[3];
      point[0] = this->MeanSurfacePoints[i][0];
      point[1] = this->MeanSurfacePoints[i][1];
      point[2] = this->MeanSurfacePoints[i][2];

    for ( unsigned int m=0; m<this->NumberOfModes; m++ )
      {                  
      point[2] += this->ModeWeights[m]*vcl_sqrt( this->Eigenvalues[m] )*this->Eigenvectors[m][i];
      }

    this->WeightedSurfacePoints.push_back( point );
    }
}


void cipLobeBoundaryShapeModel::SetEigenvalues( std::vector< double > const* eigVals )
{
  this->Eigenvalues.clear();

  for ( unsigned int i=0; i<(*eigVals).size(); i++ )
    {
    this->Eigenvalues.push_back( (*eigVals)[i] );
    }
}


std::vector< double > const* cipLobeBoundaryShapeModel::GetEigenvalues() const
{
  return &this->Eigenvalues;
}


void cipLobeBoundaryShapeModel::SetEigenvectors( std::vector< std::vector< double > > const* eigenvectors )
{
  this->Eigenvectors.clear();

  for ( unsigned int i=0; i<(*eigenvectors).size(); i++ )
    {
    std::vector< double > tempVec;

    for ( unsigned int j=0; j<(*eigenvectors)[i].size(); j++ )
      {
      tempVec.push_back( (*eigenvectors)[i][j] );
      }

    this->Eigenvectors.push_back( tempVec );
    }
}


std::vector< std::vector< double > > const* cipLobeBoundaryShapeModel::GetEigenvectors() const
{
  return &this->Eigenvectors;
}


void cipLobeBoundaryShapeModel::SetModeWeights( std::vector< double > const* weights )
{
  this->ModeWeights.clear();

  for ( unsigned int i=0; i<(*weights).size(); i++ )
    {
    this->ModeWeights.push_back( (*weights)[i] );
    }
}


std::vector< double >* cipLobeBoundaryShapeModel::GetModeWeights()
{
  return &this->ModeWeights;
}


void cipLobeBoundaryShapeModel::SetNumberOfModes( unsigned int numberOfModes )
{
  this->NumberOfModes = numberOfModes;
}


unsigned int cipLobeBoundaryShapeModel::GetNumberOfModes() const
{
  return this->NumberOfModes;
}
