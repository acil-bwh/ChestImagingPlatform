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
#include "vtkPolyData.h"

namespace cip {
  /** Function that downsamples a label map. Takes in as input a value for the downsampling amount and 
   * a pointer to a LabelMapType, and returns a pointer to a downsampled LabelMapType. */
  cip::LabelMapType::Pointer DownsampleLabelMap(short samplingAmount, cip::LabelMapType::Pointer inputLabelMap);

  /** Function that upsamples a label map. Takes in as input a value for the upsampling
   *amount and a pointer to a LabelMapType, and returns a pointer to a upsampled LabelMapType. */
  cip::LabelMapType::Pointer UpsampleLabelMap(short samplingAmount, cip::LabelMapType::Pointer inputLabelMap);

  /** Function that downsamples a CT. Takes in as input a value for the downsampling amount and 
   * a pointer to a CTType, and returns a pointer to a downsampled CTType. */
  cip::CTType::Pointer DownsampleCT(short samplingAmount, cip::CTType::Pointer inputCT);

  /** Function that upsamples a label CT. Takes in as input a value for the upsampling
   * amount and a pointer to a CTType, and returns a pointer to a upsampled CTType. */
  cip::CTType::Pointer UpsampleCT(short samplingAmount, cip::CTType::Pointer inputCT);

  /** Get the magnitude of the indicated vector */
  double GetVectorMagnitude(double vector[3]);

  /** Get the angle between the two vectors. By default, the answer will be returned 
   * in radians, but it can also be returned in degrees by setting 'returnDegrees' to 'true'. */
  double GetAngleBetweenVectors(double vec1[3], double vec2[3], bool returnDegrees = false);

  /** Render a vtk-style graph for visualization */
  void ViewGraph(vtkSmartPointer< vtkMutableDirectedGraph > graph);

  /** View a vtk-style graph as poly data. It's assumed that the graph nodes correspond
   * to 3D points. */
  void ViewGraphAsPolyData(vtkSmartPointer< vtkMutableUndirectedGraph > graph);

  /** Morphologically dilate label map. Only the label map value corresponding to the specified chest 
   * region and chest type is dilated. The rectangular kernel size is specified by the 'kernelRadius'
   * parameters.*/
  void DilateLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type, 
		      unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ);

  /** Morphologically erode label map. Only the label map value corresponding to the specified chest 
   * region and chest type is eroded. The rectangular kernel size is specified by the 'kernelRadius'
   * parameters.*/
  void ErodeLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type, 
		     unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ);

  /** Morphologically close label map. Only the label map value corresponding to the specified chest 
   * region and chest type is closed. The rectangular kernel size is specified by the 'kernelRadius'
   * parameters.*/
  void CloseLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type, 
		     unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ);

  /** Morphologically open label map. Only the label map value corresponding to the specified chest 
   * region and chest type is opened. The rectangular kernel size is specified by the 'kernelRadius'
   * parameters.*/
  void OpenLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type, 
		    unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ);
  
  /** Get the bounding with respect to a specified chest region - chest type combination. The bounding
   * box is returned as an ITK image region. */
  cip::LabelMapType::RegionType GetLabelMapChestRegionChestTypeBoundingBoxRegion(cip::LabelMapType::Pointer labelMap, 
										 unsigned char cipRegion = (unsigned char)(UNDEFINEDREGION), 
										 unsigned char cipType = (unsigned char)(UNDEFINEDTYPE));

  /** Similar to GetLabelMapChestRegionChestTypeBoundingBoxRegion, but this function will return an ITK bounding box region
   * padded according to the specified x, y, and z radii. The region is determined with respect to the specifed chest-region
   * chest-type combination. */
  cip::LabelMapType::RegionType GetLabelMapChestRegionChestTypePaddedBoundingBoxRegion(cip::LabelMapType::Pointer labelMap, 
										       unsigned char region, unsigned char type,
										       unsigned int radiusX, unsigned int radiusY, unsigned int radiusZ);
  /** This function is used to verify that the specified particles have 'ChestRegion' and 'ChestType' arrays. 
      If the particles don't have these arrays, they are assigned with default entries UNDEFINEDREGION and 
      UNDEFINEDTYPE */
  void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > );
}  

#endif
