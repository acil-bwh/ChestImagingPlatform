/** \file
 *  \ingroup sandbox 
 *  \details This simple program ...
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"

#include "itkImageRegionIteratorWithIndex.h"

typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType > IteratorType;

int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed
  std::string inFileName  = "NA";
  std::string outFileName = "NA";

  // Input argument descriptions for user help
  std::string programDesc = "This program reads manually labeled pectoralis label maps that \
have incorrectly labeled chest regions and relables them properly.";

  std::string inFileNameDesc = "Input label map file name.";
  std::string outFileNameDesc = "Output label map file name.";

  // Parse the input arguments
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision$" );

    TCLAP::ValueArg<std::string> inFileNameArg( "i", "", inFileNameDesc, true, inFileName, "string", cl );
    TCLAP::ValueArg<std::string> outFileNameArg( "o", "", outFileNameDesc, true, outFileName, "string", cl );

    cl.parse( argc, argv );

    inFileName = inFileNameArg.getValue();
    outFileName = outFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  cip::ChestConventions conventions;

  // Use the following labels for relabeling
  unsigned short rightMajorLabel = conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::RIGHT), (unsigned char)(cip::PECTORALISMAJOR) );
  unsigned short rightMinorLabel = conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::RIGHT), (unsigned char)(cip::PECTORALISMINOR) );
  unsigned short leftMajorLabel  = conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::LEFT), (unsigned char)(cip::PECTORALISMAJOR) );
  unsigned short leftMinorLabel  = conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::LEFT), (unsigned char)(cip::PECTORALISMINOR) );

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName( inFileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  // Use the following map to create a mapping between the old label value
  // and the new label value
  std::map< unsigned short, unsigned short > relabelMapper;

  // TODO: Identify the new labels associated with the input (incorrect)
  // labels, and store them in 'relabelMapper'. Use the class methods
  // for 'ChestConventions' in cip::Conventions to help with this

  // Now relabel the input label map
  IteratorType it( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );
  
  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
      if ( it.Get() != 0 )
	{
	  unsigned short newValue = relabelMapper[it.Get()];
	  it.Set(newValue);
	}

      ++it;
    }

  std::cout << "Writing label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetFileName( outFileName );
    writer->UseCompressionOn();
    writer->SetInput( reader->GetOutput() );
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught writing label map:";
      std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
