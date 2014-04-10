/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads a particles dataset and creates a
 *  stenciled label map corresponding to them. An input label map is
 *  used simply to get spacing, origin, and dimensions of the output
 *  label map. Particles can correspond to vessels, airways, or
 *  fissures. Currently, sphere and cylinder stencils are
 *  supported. The user has the option of scaling the stencil pattern
 *  using the particle scale. Scaling in this case means scaling the
 *  radius for both the sphere and cylinder stencils. The height of
 *  the cylinder stencil remains fixed.  
 * 
 *  USAGE: 
 *
 *  GenerateStenciledLabelMapFromParticles  {-v|-a|-f} {-s|-c} [--height
 *                                          \<double\>] [-r \<double\>]
 *                                          [--ctSigma \<double\>] -p
 *                                          \<string\> -o \<string\> 
 *                                          -l \<string\> [--]
 *                                          [--version] [-h] 
 *
 *  Where: 
 *
 *   -v,  --vessel
 *     (OR required)  Set this flag to indicate that in the input particles
 *     correspond to vessels
 *         -- OR --
 *   -a,  --airway
 *     (OR required)  Set this flag to indicate that in the input particles
 *     correspond to airways
 *         -- OR --
 *   -f,  --fissure
 *     (OR required)  Set this flag to indicate that in the input particles
 *     correspond to fissures
 *
 *   -s,  --sphere
 *     (OR required)  Set this flag to indicate that the sphere stencil
 *     should be used
 *         -- OR --
 *   -c,  --cylinder
 *     (OR required)  Set this flag to indicate that the cylinder stencil
 *     should be used
 *
 *   --scale
 *     Setting this flag will cause the stencil pattern to be scaled
 *     according toparticle scale. If set, any radius value specified using
 *     the -r flag will be ignored. Scaling will beperformed using
 *     predetermined equations relating particle scale and CT point spread
 *     function sigma (setusing the -ctSigma flag)
 *
 *   --height \<double\>
 *     Cylinder stencil height in mm. Default is 1mm. This should typically
 *     beset to the inter-particle distance.
 *
 *   -r \<double\>,  --radius \<double\>
 *     Stencil radius in mm
 *
 *   --ctSigma \<double\>
 *     The CT scanner point spread function sigma. 0.0 by default.
 *
 *   -p \<string\>,  --particles \<string\>
 *     (required)  Input particles file name
 *
 *   -o \<string\>,  --outLabelMap \<string\>
 *     (required)  Output label map file name
 *
 *   -l \<string\>,  --inLabelMap \<string\>
 *     (required)  Input label map file name. Used to retrievespacing, origin
 *     , and dimensions for creating output label map
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
 *  TODO: Extend to cylinder stencil. Accomodate stencil scaling by
 *  particle scale. Flesh out program description comments and
 *  comments throughout.
 *
 *  $Date: 2012-06-11 17:57:09 -0700 (Mon, 11 Jun 2012) $
 *  $Revision: 155 $
 *  $Author: jross $
 *
 */

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
#include "cipConventions.h"
#include "cipParticlesToStenciledLabelMapImageFilter.h"
#include "cipSphereStencil.h"
#include "cipCylinderStencil.h"


typedef itk::Image< unsigned short, 3 >                          ImageType;
typedef itk::ImageFileReader< ImageType >                        ReaderType;
typedef itk::ImageFileWriter< ImageType >                        WriterType;
typedef cipParticlesToStenciledLabelMapImageFilter< ImageType >  StenciledLabelMapType; 

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
    //TCLAP::ValueArg<double>      ctPointSpreadSigmaArg( "", "ctSigma", ctPointSpreadSigmaDescription, false, ctPointSpreadSigma, "double", cl );
    //TCLAP::ValueArg<double>      radiusArg( "r", "radius", radiusDescription, false, radius, "double", cl );
    //TCLAP::ValueArg<double>      heightArg( "", "height", heightDescription, false, height, "double", cl );
    //TCLAP::SwitchArg             areVesselsArg( "v", "vessel", areVesselsDescription, false );
    //TCLAP::SwitchArg             areFissuresArg( "f", "fissure", areFissuresDescription, false );
    //TCLAP::SwitchArg             areAirwaysArg( "a", "airway", areAirwaysDescription, false );
   // TCLAP::SwitchArg             useSphereStencilArg( "s", "sphere", useSphereStencilDescription, false );
    //TCLAP::SwitchArg             useCylinderStencilArg( "c", "cylinder", useCylinderStencilDescription, false );
  //  TCLAP::SwitchArg             scaleStencilArg( "", "scale", scaleStencilDescription, cl, false );
/*
    std::vector< TCLAP::Arg* > particlesTypeXorList;
      particlesTypeXorList.push_back( &areVesselsArg );
      particlesTypeXorList.push_back( &areAirwaysArg ); 
      particlesTypeXorList.push_back( &areFissuresArg );

    cl.xorAdd( particlesTypeXorList );
    cl.xorAdd( useSphereStencilArg, useCylinderStencilArg );
    cl.parse( argc, argv );

    if ( scaleStencilArg.isSet() )
      {
      scaleStencil = true;
      }

    if ( areVesselsArg.isSet() )
      {
      areVessels = true;
      }
    if ( areAirwaysArg.isSet() )
      {
      areAirways = true;
      }
    if ( areFissuresArg.isSet() )
      {
      areFissures = true;
      }

    if ( useSphereStencilArg.isSet() )
      {
      useSphereStencil = true;
      }
    if ( useCylinderStencilArg.isSet() )
      {
      useCylinderStencil = true;
      }

    height                 = heightArg.getValue();
    radius                 = radiusArg.getValue();
    ctPointSpreadSigma     = ctPointSpreadSigmaArg.getValue();
    inLabelMapFileName     = inLabelMapFileNameArg.getValue();
    outLabelMapFileName    = outLabelMapFileNameArg.getValue();
    particlesFileName      = particlesFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }
*/
    
    PARSE_ARGS;
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
