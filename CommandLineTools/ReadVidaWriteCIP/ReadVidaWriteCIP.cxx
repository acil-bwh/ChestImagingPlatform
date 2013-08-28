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
#include "itkImageRegionIteratorWithIndex.h"

typedef itk::Image< unsigned char, 3 >                VidaLabelMapType;
typedef itk::ImageFileReader< VidaLabelMapType >      VidaReaderType;
typedef itk::ImageRegionIterator< VidaLabelMapType >  VidaIteratorType;
typedef itk::ImageRegionIterator< cip::LabelMapType > CIPIteratorType;

int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed
  std::string   inLabelMapFileName    = "NA";
  std::string   inRefLabelMapFileName = "NA";
  std::string   outLabelMapFileName   = "NA";
  unsigned char cipRegion             = cip::UNDEFINEDREGION;
  unsigned char cipType               = cip::UNDEFINEDTYPE;

  //
  // Input argument descriptions for user help
  //
  std::string programDesc = "This ...";

  std::string inLabelMapFileNameDesc  = "Input label map file name in Vida format";
  std::string inRefLabelMapFileNameDesc = "Input label map for transferring proper origin and \
spacing information to the converted labelmap";
  std::string outLabelMapFileNameDesc = "Output label map file name in CIP format";
  std::string cipRegionDesc = "The CIP chest region of the structure contained in the Vida label map";
  std::string cipTypeDesc = "The CIP chest type of the structure contained in the Vida label map";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision$" );

    TCLAP::ValueArg<std::string> inLabelMapFileNameArg ( "i", "in", inLabelMapFileNameDesc, true, inLabelMapFileName, "string", cl );
    TCLAP::ValueArg<std::string> inRefLabelMapFileNameArg ( "r", "ref", inRefLabelMapFileNameDesc, true, inRefLabelMapFileName, "string", cl );
    TCLAP::ValueArg<std::string> outLabelMapFileNameArg ( "o", "out", outLabelMapFileNameDesc, true, outLabelMapFileName, "string", cl );
    TCLAP::ValueArg<int> cipRegionArg ( "", "region", cipRegionDesc, true, cipRegion, "unsigned char", cl );
    TCLAP::ValueArg<int> cipTypeArg ( "", "type", cipTypeDesc, true, cipType, "unsigned char", cl );

    cl.parse( argc, argv );

    inLabelMapFileName     = inLabelMapFileNameArg.getValue();
    inRefLabelMapFileName  = inRefLabelMapFileNameArg.getValue();
    outLabelMapFileName    = outLabelMapFileNameArg.getValue();
    cipRegion              = cipRegionArg.getValue();
    cipType                = cipTypeArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  cip::ChestConventions conventions;
  
  unsigned short cipValue = conventions.GetValueFromChestRegionAndType( (unsigned char)cipRegion, (unsigned char)cipType );
  std::cout << cipValue << std::endl;

  std::cout << "Reading Vida label map..." << std::endl;
  VidaReaderType::Pointer labelMapReader = VidaReaderType::New();
    labelMapReader->SetFileName(inLabelMapFileName);
  try
    {
    labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Reading reference label map..." << std::endl;
  cip::LabelMapReaderType::Pointer refLabelMapReader = cip::LabelMapReaderType::New();
    refLabelMapReader->SetFileName(inRefLabelMapFileName);
  try
    {
    refLabelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading reference label map:";
    std::cerr << excp << std::endl;
    }
  
  cip::LabelMapType::Pointer cipLabelMap = cip::LabelMapType::New();
    cipLabelMap->SetRegions( refLabelMapReader->GetOutput()->GetBufferedRegion().GetSize() );
    cipLabelMap->Allocate();
    cipLabelMap->FillBuffer( 0 );
    cipLabelMap->SetSpacing( refLabelMapReader->GetOutput()->GetSpacing() );
    cipLabelMap->SetOrigin( refLabelMapReader->GetOutput()->GetOrigin() );

  // Now transfer the Vida info to the CIP label map. Note that we
  // need to flip in z
  cip::LabelMapType::IndexType index;
  cip::LabelMapType::SizeType size = cipLabelMap->GetBufferedRegion().GetSize();

  VidaIteratorType vIt( labelMapReader->GetOutput(), labelMapReader->GetOutput()->GetBufferedRegion() );

  vIt.GoToBegin();
  while ( !vIt.IsAtEnd() )
    {
    if ( vIt.Get() != 0 )
      {
      index[0] = vIt.GetIndex()[0];
      index[1] = vIt.GetIndex()[1];
      index[2] = size[2] - vIt.GetIndex()[2];

      cipLabelMap->SetPixel( index, cipValue );
      }

    ++vIt;
    }

  std::cout << "Writing CIP label map..." << std::endl;
  cip::LabelMapWriterType::Pointer labelMapWriter = cip::LabelMapWriterType::New();
    labelMapWriter->SetFileName( outLabelMapFileName );
    labelMapWriter->SetInput( cipLabelMap );
    labelMapWriter->UseCompressionOn();
  try
    {
    labelMapWriter->Update();
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
