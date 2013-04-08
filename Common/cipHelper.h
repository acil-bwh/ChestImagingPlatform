//
//  \file cipHelper
//  \ingroup common
//  \brief This class is intended to contain a collection of functions that are routinely 
//	used in other programs.
//
//  $Date$
//  $Revision$
//  $Author$
//

#ifndef __cipHelper_h
#define __cipHelper_h

#include "cipConventions.h"
#include "vtkSmartPointer.h"
#include "vtkMutableDirectedGraph.h"
#include "vtkMutableUndirectedGraph.h" 

namespace cip {
  //
  // Function that downsamples a label map. Takes in as input a value for the downsampling amount and 
  // a pointer to a LabelMapType, and returns a pointer to a downsampled LabelMapType.
  //
  cip::LabelMapType::Pointer DownsampleLabelMap( short samplingAmount, cip::LabelMapType::Pointer inputLabelMap );

  //
  // Function that upsamples a label map. Takes in as input a value for the upsampling
  // amount and a pointer to a LabelMapType, and returns a pointer to a upsampled LabelMapType.
  //
  cip::LabelMapType::Pointer UpsampleLabelMap( short samplingAmount, cip::LabelMapType::Pointer inputLabelMap );

  //
  // Function that downsamples a CT. Takes in as input a value for the downsampling amount and 
  // a pointer to a CTType, and returns a pointer to a downsampled CTType. 
  //
  cip::CTType::Pointer DownsampleCT( short samplingAmount, cip::CTType::Pointer inputCT );

  //
  // Function that upsamples a label CT. Takes in as input a value for the upsampling
  // amount and a pointer to a CTType, and returns a pointer to a upsampled CTType. 
  //
  cip::CTType::Pointer UpsampleCT( short samplingAmount, cip::CTType::Pointer inputCT );

  //
  // Get the magnitude of the indicated vector
  //
  double GetVectorMagnitude( double vector[3] );

  //
  // Get the angle between the two vectors. By default, the answer will be returned 
  // in radians, but it can also be returned in degrees by setting 'returnDegrees'
  // to 'true'.
  //
  double GetAngleBetweenVectors( double vec1[3], double vec2[3], bool returnDegrees = false );

  //
  // Render a vtk-style graph for visualization
  //
  void ViewGraph( vtkSmartPointer< vtkMutableDirectedGraph > graph );

  //
  // View a vtk-style graph as poly data. It's assumed that the graph nodes correspond
  // to 3D points.
  //
  void ViewGraphAsPolyData( vtkSmartPointer< vtkMutableUndirectedGraph > graph );
}  

#endif
