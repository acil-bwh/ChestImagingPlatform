/** \file
 *  \ingroup commandLineTools
 *  \details This program is used to register a lung atlas convex hull
 *  mesh to the bones (ribs) in CT image. It used the iterative
 *  closest point algorithm with an affine transform to perform the
 *  registration. The input CT image is thresholded at a specified
 *  level: all voxels (physical points) above the threshold are added
 *  to the target point set. We assume that the ribs will be the bony
 *  objects that the mesh points will attract to. The final transform
 *  is written to file for image resampling using other tools.
 *
 * USAGE:
 *
 *   RegisterLungAtlas.exe  [-i \<int\>] [-b \<short\>] -o \<string\>
 *                          -m \<string\> -c \<string\>
 *                          [--] [--version] [-h]
 *
 * Where:
 *
 *   -i \<int\>,  --iterations \<int\>
 *     Number of iterations
 *
 *   -b \<short\>,  --bone \<short\>
 *     Threshold value for bone. Any voxel having HUintensity greater than or
 *     equal to this value will be considered bone and will be addedto the
 *     fixed point set. (Default: 600 HU)
 *
 *   -o \<string\>,  --trans \<string\>
 *     (required)  Output transform file name
 *
 *   -m \<string\>,  --mesh \<string\>
 *     (required)  Convex hull mesh file name
 *
 *   -c \<string\>,  --ct \<string\>
 *     (required)  CT file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 */

#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkAffineTransform.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMatrix.h"
#include "itkTransformFileWriter.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkIterativeClosestPointTransform.h"
#include "vtkLandmarkTransform.h"
#include "vtkVertexGlyphFilter.h"
#include "vtkCellArray.h"
#include "vtkIdList.h"
#include "vtkMatrix4x4.h"
#include "RegisterLungAtlasCLP.h"
#include "vtkXMLPolyDataReader.h"
#include <vtksys/SystemTools.hxx>

namespace
{
  typedef itk::Image< short, 3 >                           ImageType;
  typedef itk::ImageFileReader< ImageType >                ReaderType;
  typedef itk::ImageRegionIteratorWithIndex< ImageType >   IteratorType;
  typedef itk::AffineTransform< double, 3 >                TransformType;
  typedef itk::Matrix< double, 3, 3 >                      MatrixType;
}

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //

  PARSE_ARGS;
  //
  // Read the CT image
  //
  std::cout << "Reading CT image." << std::endl;
  ReaderType::Pointer ctReader = ReaderType::New();
  ctReader->SetFileName( ctFileName.c_str() );
  try
    {
      ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught reading CT image:";
      std::cerr << excp << std::endl;

      return cip::NRRDREADFAILURE;
    }

  //
  // Now fill the fixed point set with those physical point locations
  // corresponding to bone.
  //
  ImageType::PointType itkPoint;

  vtkSmartPointer< vtkPoints > targetPoints = vtkSmartPointer< vtkPoints >::New();

  IteratorType it( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
      if ( it.Get() >= boneThreshold )
	{
	  ctReader->GetOutput()->TransformIndexToPhysicalPoint( it.GetIndex(), itkPoint );

	  double vtkPoint[3];
	  vtkPoint[0] = itkPoint[0];
	  vtkPoint[1] = itkPoint[1];
	  vtkPoint[2] = itkPoint[2];

	  targetPoints->InsertNextPoint( vtkPoint );
	}

      ++it;
    }

  vtkSmartPointer< vtkPolyData > target = vtkSmartPointer< vtkPolyData >::New();
  target->SetPoints( targetPoints );

  vtkSmartPointer< vtkCellArray > targetArray = vtkSmartPointer< vtkCellArray >::New();

  for( unsigned int i=0; i<target->GetNumberOfPoints(); i++ )
    {
      vtkSmartPointer< vtkIdList > idList = vtkSmartPointer< vtkIdList >::New();
      idList->InsertNextId( i );

      targetArray->InsertNextCell( idList );
    }

  target->SetVerts( targetArray );
  //target->Update();

  //
  // Now read the convex hull mesh
  //
  std::cout << "Reading convex hull mesh... " << std::endl;

  vtkSmartPointer< vtkIterativeClosestPointTransform > icp = vtkSmartPointer< vtkIterativeClosestPointTransform >::New();

  std::string extension = vtksys::SystemTools::LowerCase( vtksys::SystemTools::GetFilenameLastExtension(convexHullMeshFileName) );

  vtkSmartPointer< vtkPolyDataReader > meshReader = vtkSmartPointer< vtkPolyDataReader >::New();
  vtkSmartPointer< vtkXMLPolyDataReader > meshReaderxml = vtkSmartPointer< vtkXMLPolyDataReader >::New();

  if( extension == std::string(".vtk") )
    {
       std::cout << "Reading convex hull vtk file... " << std::endl;
      meshReader->SetFileName( convexHullMeshFileName.c_str() );
      meshReader->Update();
      icp->SetSource( meshReader->GetOutput() );
    }
  else if( extension == std::string(".vtp") )
    {
      meshReaderxml->SetFileName(convexHullMeshFileName.c_str() );
      meshReaderxml->Update();
      icp->SetSource( meshReaderxml->GetOutput() );
    }

  std::cout << "Registering..." << std::endl;

  icp->SetTarget( target );
  icp->SetStartByMatchingCentroids( true );
  icp->GetLandmarkTransform()->SetModeToAffine();
  icp->SetMaximumNumberOfIterations( numberOfIterations );
  icp->Modified();
  icp->Update();

  vtkSmartPointer< vtkMatrix4x4 > vMatrix = icp->GetMatrix();
  std::cout << "The resulting matrix is: " << *vMatrix << std::endl;

  MatrixType iMatrix;
  iMatrix(0,0) = vMatrix->GetElement(0,0);  iMatrix(0,1) = vMatrix->GetElement(0,1);  iMatrix(0,2) = vMatrix->GetElement(0,2);
  iMatrix(1,0) = vMatrix->GetElement(1,0);  iMatrix(1,1) = vMatrix->GetElement(1,1);  iMatrix(1,2) = vMatrix->GetElement(1,2);
  iMatrix(2,0) = vMatrix->GetElement(2,0);  iMatrix(2,1) = vMatrix->GetElement(2,1);  iMatrix(2,2) = vMatrix->GetElement(2,2);

  TransformType::TranslationType translation;
  translation[0] = vMatrix->GetElement(0,3);
  translation[1] = vMatrix->GetElement(1,3);
  translation[2] = vMatrix->GetElement(2,3);

  TransformType::Pointer transform = TransformType::New();
  transform->SetMatrix( iMatrix );
  transform->SetTranslation( translation );

  std::cout << "Writing transform to file..." << std::endl;
  itk::TransformFileWriter::Pointer transformWriter = itk::TransformFileWriter::New();
  transformWriter->SetInput( transform );
  transformWriter->SetFileName( outputTransformFileName );
  transformWriter->Update();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

