#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "GenerateStenciledLabelMapFromParticlesCLP.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipChestConventions.h"
#include "cipParticlesToStenciledLabelMapImageFilter.h"
#include "cipSphereStencil.h"
#include "cipCylinderStencil.h"


typedef itk::Image< unsigned short, 3 >                          ImageType;
typedef itk::ImageFileReader< ImageType >                        ReaderType;
typedef itk::ImageFileWriter< ImageType >                        WriterType;
typedef cipParticlesToStenciledLabelMapImageFilter< ImageType >  StenciledLabelMapType; 

int main( int argc, char *argv[] )
{    
  PARSE_ARGS;

  if ( !useCylinderStencil && !useSphereStencil )
    {
      std::cerr << "Must specify to use either cylinder or sphere stencil" << std::endl;
      return cip::EXITFAILURE;
    }
  
  // Read the particles
  std::cout << "Reading particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( particlesFileName.c_str() );
    particlesReader->Update();    

  // Read the input label map
  std::cout << "Reading label map..." << std::endl;
  ReaderType::Pointer labelMapReader = ReaderType::New();
    labelMapReader->SetFileName( inLabelMapFileName );
  try
    {
    labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;     
    
    return cip::LABELMAPREADFAILURE;
    }

  // Set up the stencils
  cipCylinderStencil* cylinderStencil = new cipCylinderStencil();
    cylinderStencil->SetRadius( radius );
    cylinderStencil->SetHeight( height );

  cipSphereStencil* sphereStencil = new cipSphereStencil();
    sphereStencil->SetRadius( radius );

  // Now create the stenciled label map
  std::cout << "Creating stenciled label map..." << std::endl;
  StenciledLabelMapType::Pointer particlesToLabelMap = StenciledLabelMapType::New();
    particlesToLabelMap->SetParticlesData( particlesReader->GetOutput() );
    particlesToLabelMap->SetInput( labelMapReader->GetOutput() );
    particlesToLabelMap->SetScaleStencilPatternByParticleScale( scaleStencil );
    particlesToLabelMap->SetScaleStencilPatternByParticleDNNRadius( dnnRadiusStencil );
    particlesToLabelMap->SetDNNRadiusName( dnnRadiusName.c_str() );

  if ( areAirways )
    {
    particlesToLabelMap->SetChestParticleType( cip::AIRWAY );
    }
  else if ( areVessels )
    {
    particlesToLabelMap->SetChestParticleType( cip::VESSEL );
    }
  else if ( areFissures )
    {
    particlesToLabelMap->SetChestParticleType( cip::FISSURE );
    }
  if ( useCylinderStencil )
    {
    particlesToLabelMap->SetStencil( cylinderStencil );
    }
  if ( useSphereStencil )
    {
    particlesToLabelMap->SetStencil( sphereStencil );
    }
    particlesToLabelMap->Update();

  //
  // Write the label map to file
  //
  std::cout << "Writing label map..." << std::endl;
  WriterType::Pointer labelMapWriter = WriterType::New();
    labelMapWriter->SetFileName( outLabelMapFileName );
    labelMapWriter->SetInput( particlesToLabelMap->GetOutput() );
    labelMapWriter->UseCompressionOn();
  try
    {
    labelMapWriter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPWRITEFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
