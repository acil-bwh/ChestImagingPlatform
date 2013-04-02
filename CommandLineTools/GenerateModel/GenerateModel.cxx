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

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
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

typedef itk::Image< unsigned short, 3 >        ImageType;
typedef itk::ImageFileReader< ImageType >      ReaderType;
typedef itk::VTKImageExport< ImageType >       ExportType;
typedef itk::ImageRegionIterator< ImageType >  IteratorType;


void ConnectPipelines( ExportType::Pointer, vtkImageImport* );


int main( int argc, char *argv[] )
{
  //
  // Define command line arguments
  //
  std::string    maskFileName                 = "NA";
  std::string    outputModelFileName          = "NA";
  unsigned int   smootherIterations           = 2;
  unsigned short foregroundLabel              = -1;
  bool           setStandardOriginAndSpacing  = 0;
  float          decimatorTargetReduction     = 0.9;

  //
  // Descriptions of command line arguments for user help
  //
  std::string programDesc = "This program generates a 3D model given an input \
label map mask using the discrete marching cubes algorithm";

  std::string maskFileNameDesc                 = "Input mask file name";
  std::string outputModelFileNameDesc          = "Output model file name";
  std::string smootherIterationsDesc           = "Number of smoothing iterations";
  std::string foregroundLabelDesc              = "Foreground label in the label map to be \
used for generating the model";
  std::string setStandardOriginAndSpacingDesc  = "Set to 1 to used standard origin and spacing. \
Set to 0 by default.";
  std::string decimatorTargetReductionDesc     = "Target reduction fraction for decimation";

  //
  // Parse the input arguments
  //
  try
    {
      TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 281 $" );

      TCLAP::ValueArg<std::string>    maskFileNameArg( "i", "in", maskFileNameDesc, true, maskFileName, "string", cl);
      TCLAP::ValueArg<std::string>    outputModelFileNameArg( "o", "out", outputModelFileNameDesc, true, outputModelFileName, "string", cl );
      TCLAP::ValueArg<unsigned int>   smootherIterationsArg( "s", "smooth", smootherIterationsDesc, false, smootherIterations, "unsigned int", cl );
      TCLAP::ValueArg<unsigned short> foregroundLabelArg( "l", "label", foregroundLabelDesc, false, foregroundLabel, "unsigned short", cl );
      TCLAP::ValueArg<bool>           setStandardOriginAndSpacingArg( "", "origSp", setStandardOriginAndSpacingDesc, false, setStandardOriginAndSpacing, "bool", cl );
      TCLAP::ValueArg<float>          decimatorTargetReductionArg( "r", "reduc", decimatorTargetReductionDesc, false, decimatorTargetReduction, "float", cl);

      cl.parse( argc, argv );

      maskFileName                = maskFileNameArg.getValue();
      outputModelFileName         = outputModelFileNameArg.getValue();
      smootherIterations          = smootherIterationsArg.getValue();
      foregroundLabel             = foregroundLabelArg.getValue();
      setStandardOriginAndSpacing = setStandardOriginAndSpacingArg.getValue();
      decimatorTargetReduction    = decimatorTargetReductionArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
      std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
      return cip::ARGUMENTPARSINGERROR;
    }

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
  
  if ( setStandardOriginAndSpacing == 1 )
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

#endif
