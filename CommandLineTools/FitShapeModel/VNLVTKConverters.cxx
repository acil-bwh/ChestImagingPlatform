#include "VNLVTKConverters.h"
#include <vtkMath.h>
#include <algorithm>
#include <stdexcept>
#include <sstream>

// -------------------------------------------------------------------------
// helper methods

void
vnlMatrixToVTKMatrix( const vnl_matrix<double>& vnlMatrix, vtkMatrix4x4* vtkMatrix )
{
  if (vnlMatrix.rows() != 4 || vnlMatrix.cols() != 4)
  {
    throw std::runtime_error("vnlMatrixToVTKMatrix: matrix size mismatches.");
  }

  for (unsigned int i = 0; i < 4; i++)
  {
    for (unsigned int j = 0; j < 4; j++)
    {
      vtkMatrix->SetElement( i, j, vnlMatrix( i, j ) );
    }
  }
}

void
vtkMatrixToVNLMatrix( vtkMatrix4x4* vtkMatrix, vnl_matrix<double>& vnlMatrix )
{
  if (vnlMatrix.rows() != 4 || vnlMatrix.cols() != 4)
  {
    throw std::runtime_error("vtkMatrixToVNLMatrix: matrix size mismatches.");
  }

  for (unsigned int i = 0; i < 4; i++)
  {
    for (unsigned int j = 0; j < 4; j++)
    {
      vnlMatrix( i, j ) = vtkMatrix->GetElement( i, j );
    }
  }
}

void
vnlVectorToVTKPoints( const vnl_vector< double >& vnlVector, vtkPoints* points )
{
  unsigned int numPoints = points->GetNumberOfPoints();
  if (3 * numPoints != vnlVector.size())
  {
    std::cout << numPoints << " != " << vnlVector.size() / 3 << std::endl;
    throw std::runtime_error("vnlVectorToVTKPoints: size mismatches.");
  }

  double p[3];
  for (unsigned int i = 0, j = 0; i < numPoints; i++)
  {
    p[0] = vnlVector[j++]; p[1] = vnlVector[j++]; p[2] = vnlVector[j++];
    points->SetPoint( i, p );
  }
}

void
vtkPointsToVNLVector( vtkPoints* points, vnl_vector< double >& vnlVector )
{
  unsigned int numPoints = points->GetNumberOfPoints();
  if (3 * numPoints != vnlVector.size())
  {
    throw std::runtime_error("vtkPointsToVNLVector: size mismatches.");
  }

  double p[3];
  for (unsigned int i = 0, j = 0; i < numPoints; i++)
  {
    points->GetPoint( i, p );
    vnlVector[j++] = p[0]; vnlVector[j++] = p[1]; vnlVector[j++] = p[2];
  }
}

double
computeMaximumDistanceBetweenTwoPointSets( vtkPoints* source, vtkPoints* target )
{
  double avgDist, maxDist;
  computeDistanceBetweenTwoPointSets( source, target, avgDist, maxDist );
  return maxDist;
}

double
computeAverageDistanceBetweenTwoPointSets( vtkPoints* source, vtkPoints* target )
{
  double avgDist, maxDist;
  computeDistanceBetweenTwoPointSets( source, target, avgDist, maxDist );
  return avgDist;
}

void
computeDistanceBetweenTwoPointSets( vtkPoints* source, vtkPoints* target, double& avgDist, double& maxDist )
{
  unsigned int numPoints = source->GetNumberOfPoints();
  if (numPoints != target->GetNumberOfPoints())
  {
    throw std::runtime_error("computeAverageDistanceBetweenTwoPointSets: size mismatches.");
  }

  double sumDist = 0; maxDist = 0;
  for (unsigned int i = 0; i < numPoints; i++)
  {
    double dist = sqrt( vtkMath::Distance2BetweenPoints( source->GetPoint( i ),
                                                         target->GetPoint( i ) ) );
    sumDist += dist;
    maxDist = std::max( maxDist, dist );
  }
  avgDist = sumDist / numPoints;
}
