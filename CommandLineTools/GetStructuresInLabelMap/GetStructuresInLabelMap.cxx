/** \file
 *  \ingroup commandLineTools 
 *  \details This 
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"

typedef itk::ImageRegionIterator< cip::LabelMapType > IteratorType;

int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed
  std::string labelMapFileName = "NA";

  // Input argument descriptions for user help
  std::string programDesc = "This program reads in a label map image and writes \
to the command line a list of the chest-region chest-type pairs that are present";

  std::string labelMapFileNameDesc = "Input label map file name";

  // Parse the input arguments
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision $" );

    TCLAP::ValueArg<std::string> labelMapFileNameArg( "i", "input", labelMapFileNameDesc, true, labelMapFileName, "string", cl );

    cl.parse( argc, argv );

    labelMapFileName = labelMapFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  cip::ChestConventions conventions;

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName(labelMapFileName);
  try
    {
    reader->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught reading label mape:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }

  std::list< unsigned short > labelsList;

  IteratorType it(reader->GetOutput(), reader->GetOutput()->GetBufferedRegion());

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
      labelsList.push_back( it.Get() );
      }

    ++it;
    }

  labelsList.unique();
  labelsList.sort();
  labelsList.unique();

  std::list< unsigned short >::iterator listIt = labelsList.begin();

  while ( listIt != labelsList.end() )
    {
    std::cout << conventions.GetChestRegionNameFromValue(*listIt) << "\t";
    std::cout << conventions.GetChestTypeNameFromValue(*listIt) << std::endl;

    ++listIt;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
