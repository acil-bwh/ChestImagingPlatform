#ifndef _vnl_vtk_converters_h_
#define _vnl_vtk_converters_h_

#include <vtkPoints.h>
#include <vtkMatrix4x4.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>

// helper methods
void vnlMatrixToVTKMatrix( const vnl_matrix<double>& vnlMatrix, vtkMatrix4x4* vtkMatrix );
void vtkMatrixToVNLMatrix( vtkMatrix4x4* vtkMatrix, vnl_matrix<double>& vnlMatrix );
void vnlVectorToVTKPoints( const vnl_vector< double >& vnlVector, vtkPoints* points );
void vtkPointsToVNLVector( vtkPoints* points, vnl_vector< double >& vnlVector );
double computeMaximumDistanceBetweenTwoPointSets( vtkPoints* source, vtkPoints* target );
double computeAverageDistanceBetweenTwoPointSets( vtkPoints* source, vtkPoints* target );
void computeDistanceBetweenTwoPointSets( vtkPoints* source, vtkPoints* target, double& avgDist, double& maxDist );

// other utilities
std::string int2str( int i );

#endif
