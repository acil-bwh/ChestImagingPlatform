/** \file
 *  \ingroup commandLineTools 
 *  \details This program...
 */


#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "RemoveChestTypeFromLabelMapUsingParticlesCLP.h"
// #include "itkImage.h"
// #include "itkImageFileReader.h"
// #include "itkImageFileWriter.h"
// #include "vtkSmartPointer.h"
// #include "vtkPolyData.h"
// #include "vtkPolyDataReader.h"
// #include "cipParticlesToLabelMapImageFilter.h"

// typedef itk::Image< unsigned short, 3 >                          LabelMapType;
// typedef itk::Image< short, 3 >                                   CTType;
// typedef itk::ImageFileReader< LabelMapType >                     LabelMapReaderType;
// typedef itk::ImageFileWriter< LabelMapType >                     LabelMapWriterType;
// typedef itk::ImageFileReader< CTType >                           CTReaderType;
// typedef itk::cipParticlesToLabelMapImageToImageFilter< CTType >  ParticlesToLabelMapType;


int main( int argc, char *argv[] )
{
    
    PARSE_ARGS;
//   //
//   // Begin by defining the arguments to be passed
//   //
//   std::string ctFileName          = "NA";
//   std::string inLabelMapFileName  = "NA";
//   std::string outLabelMapFileName = "NA";
//   std::string particlesFileName   = "NA";
//   unsigned short cipType          = cip::AIRWAY;

//   //
//   // Describe the program and inputs for user help
//   //
//   std::string programDescription = "This program ...";
//   std::string ctFileNameDescription = "Input CT file name";
//   std::string particlesFileNameDescription = "Particles file name";
//   std::string inLabelMapFileNameDescription = "Input label map file name";
//   std::string outLabelMapFileNameDescription = "Output label map file name";
//   std::string cipTypeDescription = "The ChestType to be removed. This should correspond to the type\
// of particles being read in. The passed value should be an unsigned short and should\
// conform to the conventions layed out in cipChestConventions.h";

//   //
//   // Parse the input arguments
//   //
//   try
//     {
//     TCLAP::CmdLine cl( programDescription, ' ', "$Revision: 93 $" );

//     TCLAP::ValueArg<std::string>    ctFileNameArg( "c", "ctFileName", ctFileNameDescription, true, ctFileNameDescription, "string", cl );
//     TCLAP::ValueArg<std::string>    particlesFileNameArg( "p", "particles", particlesFileNameDescription, true, particlesFileNameDescription, "string", cl );
//     TCLAP::ValueArg<std::string>    inLabelMapFileNameArg( "i", "inLabelMap", inLabelMapFileNameDescription, true, inLabelMapFileName, "string", cl );
//     TCLAP::ValueArg<std::string>    outLabelMapFileNameArg( "o", "outLabelMap", outLabelMapFileNameDescription, true, outLabelMapFileName, "string", cl );
//     TCLAP::ValueArg<unsigned short> cipTypeArg( "t", "cipType", cipTypeDescription, false, cipType, "unsigned short", cl );

//     cl.parse( argc, argv );

//     ctFileName          = ctFileNameArg.getValue();
//     inLabelMapFileName  = inLabelMapFileNameArg.getValue();
//     outLabelMapFileName = outLabelMapFileNameArg.getValue();
//     particlesFileName   = particlesFileNameArg.getValue();
//     cipType             = cipTypeArg.getValue();
//     }
//   catch ( TCLAP::ArgException excp )
//     {
//     std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
//     return cip::ARGUMENTPARSINGERROR;
//     }

//   //
//   // Read the CT data
//   //
//   std::cout << "Reading CT image..." << std::endl;
//   CTReaderType::Pointer ctReader = CTReaderType::New();
//     ctReader->SetFileName( ctFileName );
//   try
//     {
//     ctReader->Update();
//     }
//   catch ( itk::ExceptionObject &excp )
//     {
//     std::cerr << "Exception caught reading label map:";
//     std::cerr << excp << std::endl;

//     return cip::NRRDREADFAILURE;
//     }

//   //
//   // Read the label map
//   //
//   std::cout << "Reading label map..." << std::endl;
//   LabelMapReaderType::Pointer labelMapReader = LabelMapReaderType::New();
//     labelMapReader->SetFileName( inLabelMapFileName );
//   try
//     {
//     labelMapReader->Update();
//     }
//   catch ( itk::ExceptionObject &excp )
//     {
//     std::cerr << "Exception caught reading label map:";
//     std::cerr << excp << std::endl;

//     return cip::LABELMAPREADFAILURE;
//     }

//   //
//   // Read the particles
//   //
//   std::cout << "Reading particles data..." << std::endl;
//   vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
//     particlesReader->SetFileName( particlesFileName.c_str() );
//     particlesReader->Update();

//   //
//   // Generate the label map, corresponding to the type of interest,
//   // from the particles data and input CT image
//   //
//   std::cout << "Generating label map from CT and particles data..." << std::endl;
//   ParticlesToLabelMapType::Pointer particlesToLabelMap = ParticlesToLabelMapType::New();
//     particlesToLabelMap->SetInput( ctReader->GetOutput() );
//     particlesToLabelMap->SetParticlesData( particlesReader->GetOutput() );
//     particlesToLabelMap->SetChestParticleType( cipType );
//     particlesToLabelMap->Update();

  //
  // Write the label map
  //
//   std::cout << "Writing label map..." << std::endl;
//   LabelMapWriterType::Pointer labelMapWriter = LabelMapWriterType::New();
//     labelMapWriter->SetFileName( outLabelMapFileName );
//     labelMapWriter->UseCompressionOn();
//     labelMapWriter->SetInput();
//   try
//     {
//     labelMapWriter->Update();
//     }
//   catch ( itk::ExceptionObject &excp )
//     {
//     std::cerr << "Exception caught reading label map:";
//     std::cerr << excp << std::endl;

//     return ChestConventions::LABELMAPREADFAILURE;
//     }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
