/** \file
 *  \ingroup commandLineTools 
 *  \details This program will generate a 3D model from an input 
 *  3D label map using the discrete marching cubes algorithm  
 *
 *  USAGE: 
 *
 *  GenerateModel  [-r \<float\>] [--origSp \<bool\>] [-l \<unsigned short\>]
 *                 [-s \<unsigned int\>] -o \<string\> -i \<string\> [--]
 *                 [--version] [-h]
 *
 *  Where: 
 *
 *   -r \<float\>,  --reduc \<float\>
 *     Target reduction fraction for decimation
 *
 *   --origSp \<bool\>
 *     Set to 1 to used standard origin and spacing. Set to 0 by default.
 *
 *   -l \<unsigned short\>,  --label \<unsigned short\>
 *     Foreground label in the label map to be used for generating the model
 *
 *   -s \<unsigned int\>,  --smooth \<unsigned int\>
 *     Number of smoothing iterations
 *
 *   -o \<string\>,  --out \<string\>
 *     (required)  Output model file name
 *
 *   -i \<string\>,  --in \<string\>
 *     (required)  Input mask file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 *  $Date: 2012-09-19 17:00:08 -0400 (Wed, 19 Sep 2012) $
 *  $Revision: 281 $
 *  $Author: jross $
 *
 */


#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkVTKImageExport.h"
#include "itkImageRegionIterator.h"
#include "vtkImageData.h"
#include "vtkPolyDataNormals.h"
#include "vtkDecimatePro.h"
#include "vtkDiscreteMarchingCubes.h"
#include "vtkWindowedSincPolyDataFilter.h"
#include "vtkPolyDataWriter.h"
#include "vtkPolyData.h"
#include "vtkImageImport.h"
#include "vtkSmartPointer.h"
#include "GenerateModelCLP.h"

namespace
{
   typedef itk::Image< unsigned short, 3 >        ImageType;
   typedef itk::ImageFileReader< ImageType >      ReaderType;
   typedef itk::VTKImageExport< ImageType >       ExportType;
   typedef itk::ImageRegionIterator< ImageType >  IteratorType;


   void ConnectPipelines( ExportType::Pointer exporter, vtkImageImport* importer )
   {
    importer->SetUpdateInformationCallback(exporter->GetUpdateInformationCallback());
    importer->SetPipelineModifiedCallback(exporter->GetPipelineModifiedCallback());
    importer->SetWholeExtentCallback(exporter->GetWholeExtentCallback());
    importer->SetSpacingCallback(exporter->GetSpacingCallback());
    importer->SetOriginCallback(exporter->GetOriginCallback());
    importer->SetScalarTypeCallback(exporter->GetScalarTypeCallback());
    importer->SetNumberOfComponentsCallback(exporter->GetNumberOfComponentsCallback());
    importer->SetPropagateUpdateExtentCallback(exporter->GetPropagateUpdateExtentCallback());
    importer->SetUpdateDataCallback(exporter->GetUpdateDataCallback());
    importer->SetDataExtentCallback(exporter->GetDataExtentCallback());
    importer->SetBufferPointerCallback(exporter->GetBufferPointerCallback());
    importer->SetCallbackUserData(exporter->GetCallbackUserData());
    }

}


int main( int argc, char *argv[] )
{
  PARSE_ARGS;
  
  unsigned int   smootherIterations           = (unsigned int) smootherIterationsTemp;
  unsigned short foregroundLabel              = (unsigned short) foregroundLabelTemp;

    

  std::cout << "Reading mask..." << std::endl;    
  ReaderType::Pointer maskReader = ReaderType::New();
    maskReader->SetFileName( maskFileName );
  try
    {
    maskReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while reading mask image:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }
  
  if ( setStandardOriginAndSpacing == true )
    {
    ImageType::SpacingType spacing;
      spacing[0] = 1;
      spacing[1] = 1;
      spacing[2] = 1;

    ImageType::PointType origin;
      origin[0] = 0.0;
      origin[1] = 0.0;
      origin[2] = 0.0;

    maskReader->GetOutput()->SetOrigin( origin );
    maskReader->GetOutput()->SetSpacing( spacing );
    }

  IteratorType it( maskReader->GetOutput(), maskReader->GetOutput()->GetBufferedRegion() );
  
  //
  // If the user has not specified a foreground label, find the first
  // non-zero value and use that as the foreground value
  //
  if ( foregroundLabel == -1 )
    {
    it.GoToBegin();
    while ( !it.IsAtEnd() )
      {
      if ( it.Get() != 0 )
        {
        foregroundLabel = it.Get();
        
        break;
        }

      ++it;
      }

    }

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() == foregroundLabel )
      {
      it.Set( 1 );
      }
    else
      {
      it.Set( 0 );
      }
    
    ++it;
    }
  
  ExportType::Pointer refExporter = ExportType::New();
    refExporter->SetInput( maskReader->GetOutput() );
    
  vtkSmartPointer< vtkImageImport > refImporter = vtkSmartPointer< vtkImageImport >::New();

  ConnectPipelines( refExporter, refImporter );

  //
  // Perform marching cubes on the reference binary image and then
  // decimate 
  //
  std::cout << "Running marching cubes..." << std::endl;
  vtkSmartPointer< vtkDiscreteMarchingCubes > cubes = vtkSmartPointer< vtkDiscreteMarchingCubes >::New();
    cubes->SetInput( refImporter->GetOutput() );
    cubes->SetValue( 0, 1 );
    cubes->ComputeNormalsOff();
    cubes->ComputeScalarsOff();
    cubes->ComputeGradientsOff();
    cubes->Update();

  std::cout << "Smoothing model..." << std::endl;
  vtkSmartPointer< vtkWindowedSincPolyDataFilter > smoother = vtkSmartPointer< vtkWindowedSincPolyDataFilter >::New();
    smoother->SetInput( cubes->GetOutput() );
    smoother->SetNumberOfIterations( smootherIterations );
    smoother->BoundarySmoothingOff();
    smoother->FeatureEdgeSmoothingOff();
    smoother->SetPassBand( 0.001 );
    smoother->NonManifoldSmoothingOn();
    smoother->NormalizeCoordinatesOn();
    smoother->Update();

  std::cout << "Decimating model..." << std::endl;
  vtkSmartPointer< vtkDecimatePro > decimator = vtkSmartPointer< vtkDecimatePro >::New();
    decimator->SetInput( smoother->GetOutput() );
    decimator->SetTargetReduction( decimatorTargetReduction );
    decimator->PreserveTopologyOn();
    decimator->BoundaryVertexDeletionOff();
    decimator->Update();

  vtkSmartPointer< vtkPolyDataNormals > normals = vtkSmartPointer< vtkPolyDataNormals >::New();
    normals->SetInput( decimator->GetOutput() );
    normals->SetFeatureAngle( 90 );
    normals->Update();

  std::cout << "Writing model..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > modelWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    modelWriter->SetFileName( outputModelFileName.c_str() );
    modelWriter->SetInput( normals->GetOutput() );
    modelWriter->Write();  
    modelWriter->Delete();  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

// void ConnectUCharPipelines( UCharExportType::Pointer exporter, vtkImageImport* importer )
// {
//   importer->SetUpdateInformationCallback(exporter->GetUpdateInformationCallback());
//   importer->SetPipelineModifiedCallback(exporter->GetPipelineModifiedCallback());
//   importer->SetWholeExtentCallback(exporter->GetWholeExtentCallback());
//   importer->SetSpacingCallback(exporter->GetSpacingCallback());
//   importer->SetOriginCallback(exporter->GetOriginCallback());
//   importer->SetScalarTypeCallback(exporter->GetScalarTypeCallback());
//   importer->SetNumberOfComponentsCallback(exporter->GetNumberOfComponentsCallback());
//   importer->SetPropagateUpdateExtentCallback(exporter->GetPropagateUpdateExtentCallback());
//   importer->SetUpdateDataCallback(exporter->GetUpdateDataCallback());
//   importer->SetDataExtentCallback(exporter->GetDataExtentCallback());
//   importer->SetBufferPointerCallback(exporter->GetBufferPointerCallback());
//   importer->SetCallbackUserData(exporter->GetCallbackUserData());
// }



