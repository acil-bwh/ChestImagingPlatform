//
//  \file cipHelper
//  \ingroup common
//  \brief This class is intended to contain a collection of functions that are routinely 
//	used in other programs.
//

#ifndef __cipHelper_h
#define __cipHelper_h

#include "cipChestConventions.h"
#include "vtkSmartPointer.h"
#include "vtkMutableDirectedGraph.h"
#include "vtkMutableUndirectedGraph.h" 
#include "vtkPolyData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipThinPlateSplineSurface.h"
#include "itkImageSeriesReader.h"

namespace cip {
  /**
   *  Define typedefs used throughout the cip
   */
  typedef itk::Image< unsigned short, 3 >          LabelMapType;
  typedef itk::Image< unsigned short, 2 >          LabelMapSliceType;
  typedef itk::Image< short, 3 >                   CTType;
  typedef itk::Image< short, 2 >                   CTSliceType;
  typedef itk::Image< float, 3 >                   DistanceMapType;
  typedef itk::Image< float, 2 >                   DistanceMapSliceType;
  typedef itk::ImageFileReader< LabelMapType >     LabelMapReaderType;
  typedef itk::ImageFileWriter< LabelMapType >     LabelMapWriterType;
  typedef itk::ImageFileReader< CTType >           CTReaderType;
  typedef itk::ImageFileWriter< CTType >           CTWriterType;
  typedef itk::ImageFileReader< DistanceMapType >  DistanceMapReaderType;
  typedef itk::ImageFileWriter< DistanceMapType >  DistanceMapWriterType;
  typedef itk::ImageSeriesReader< CTType >         CTSeriesReaderType;
  
  /** Function to read CT from Directory */
  cip::CTType::Pointer ReadCTFromDirectory( std::string ctDir );
  
  /** Function to read CT from file */
  cip::CTType::Pointer ReadCTFromFile( std::string fileName );

  /** Function to read CT from file */
  cip::LabelMapType::Pointer ReadLabelMapFromFile( std::string fileName );
  
  /** Function that downsamples a label map. Takes in as input a value for the downsampling amount and
   * a pointer to a LabelMapType, and returns a pointer to a downsampled LabelMapType. */
  cip::LabelMapType::Pointer DownsampleLabelMap(unsigned short samplingAmount, cip::LabelMapType::Pointer inputLabelMap);

  /** Function that upsamples a label map. Takes in as input a value for the upsampling
   *amount and a pointer to a LabelMapType, and returns a pointer to a upsampled LabelMapType. */
  cip::LabelMapType::Pointer UpsampleLabelMap(unsigned short samplingAmount, cip::LabelMapType::Pointer inputLabelMap);

  /** Function that downsamples a label map slice. Takes in as input a value for the downsampling amount and 
   * a pointer to a LabelMapSliceType, and returns a pointer to a downsampled LabelMapSliceType. */
  cip::LabelMapSliceType::Pointer 
    DownsampleLabelMapSlice(unsigned short samplingAmount, cip::LabelMapSliceType::Pointer inputLabelMap);

  /** Function that upsamples a label map slice. Takes in as input a value for the upsampling
   *  amount and a pointer to a LabelMapSliceType, and returns a pointer to a upsampled LabelMapSliceType. */
  cip::LabelMapSliceType::Pointer 
    UpsampleLabelMapSlice(unsigned short samplingAmount, cip::LabelMapSliceType::Pointer inputLabelMap);

  /** Templated fucntion to downsample an itkImage data type */
  template<typename ImageType, typename InterpolatorType, unsigned int D> typename ImageType::Pointer 
    DownsampleImage(unsigned short samplingAmount, typename ImageType::Pointer inputImage);

  /** Function that downsamples a CT. Takes in as input a value for the downsampling amount and 
   * a pointer to a CTType, and returns a pointer to a downsampled CTType. */
  cip::CTType::Pointer DownsampleCT(unsigned short samplingAmount, cip::CTType::Pointer inputCT);

  /** Templated fucntion to upsample an itkImage data type */
  template<typename ImageType,typename InterpolatorType,unsigned int D> typename ImageType::Pointer 
    UpsampleImage(unsigned short samplingAmount, typename ImageType::Pointer inputImage);

  /** Function that upsamples a label CT. Takes in as input a value for the upsampling
   * amount and a pointer to a CTType, and returns a pointer to a upsampled CTType. */
  cip::CTType::Pointer UpsampleCT(unsigned short samplingAmount, cip::CTType::Pointer inputCT);

  /** Function that upsamples a distance map. Takes in as input a value for the upsampling
   * amount and a pointer to a CTType, and returns a pointer to a upsampled CTType. */
  cip::DistanceMapType::Pointer 
    UpsampleDistanceMap(unsigned short samplingAmount, cip::DistanceMapType::Pointer inputDM);
  
  /** Get the magnitude of the indicated vector */
  double GetVectorMagnitude(const cip::VectorType& vector);

  /** Get the angle between the two vectors. By default, the answer will be returned 
   * in radians, but it can also be returned in degrees by setting 'returnDegrees' to 'true'. */
  double GetAngleBetweenVectors(const cip::VectorType& vec1, const cip::VectorType& vec2, bool returnDegrees = false);

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
  cip::LabelMapType::RegionType 
    GetLabelMapChestRegionChestTypeBoundingBoxRegion(cip::LabelMapType::Pointer labelMap, 
						     unsigned char cipRegion = (unsigned char)(cip::UNDEFINEDREGION), 
						     unsigned char cipType = (unsigned char)(cip::UNDEFINEDTYPE));

  /** Similar to GetLabelMapChestRegionChestTypeBoundingBoxRegion, but this function will return an ITK bounding box region
   * padded according to the specified x, y, and z radii. The region is determined with respect to the specifed chest-region
   * chest-type combination. */
  cip::LabelMapType::RegionType 
    GetLabelMapChestRegionChestTypePaddedBoundingBoxRegion(cip::LabelMapType::Pointer labelMap, 
							   unsigned char region, unsigned char type,
							   unsigned int radiusX, unsigned int radiusY, unsigned int radiusZ);

  /** This function is used to verify that the specified particles have the 'ChestRegionChestType' array. 
      If the particles don't have this array, they are assigned with the default entry corresponding to UNDEFINEDREGION 
      and UNDEFINEDTYPE */
  void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > );

  /** This function will transfer all the point data arrays from the first polydata to the second. In the case
      when an array has the same name in both the "from" polydata and the "to" polydata, nothing will be done. 
      Both data sets must have the same number of points. */
  void GraftPointDataArrays( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );
  
  /** Given a thin plate spline surface and a point, this function will find the minimum distance
   *  to the surface */
  double GetDistanceToThinPlateSplineSurface( const cipThinPlateSplineSurface&, cip::PointType );

  /**Transfers the contents of a VTK polydata's field data to point data and vice-versa. 
   * Generally, field data applies to a dataset as a whole and need not have a one-to-one 
   * correspondence with the points. However, this may be the case in some instances 
   * (esp. with the particles datasets). In those cases it may be helpful to have the data 
   * contained in field data arrays also stored in point data arrays (e.g. for rendering 
   * purposes). Field data will only be transferred provided that the number of tuples in 
   * the field data array is the same as the number of points. */
  void TransferFieldDataToFromPointData( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData >, bool, bool, bool, bool );

  /** Given a thin plate spline surface and some point in 3D space, this function will 
   *  compute the closest point on the surface and set it to tpsPoint. */
  void GetClosestPointOnThinPlateSplineSurface( const cipThinPlateSplineSurface& tps, cip::PointType point, cip::PointType tpsPoint );

  /** Transfer the field data from one polydata to another  */
  void TransferFieldData( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );

  /** Check to see if two polydata data sets have the exact same field data. If the field
   *  data differs in any way, return false. Otherwise, return true. */
  bool HaveSameFieldData( vtkSmartPointer< vtkPolyData > polyData1, vtkSmartPointer< vtkPolyData > polyData2 );
}  

#endif
