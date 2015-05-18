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

#include "ReadVidaWriteCIPCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImageRegionIteratorWithIndex.h"

namespace
{
   typedef itk::Image< unsigned char, 3 >                VidaLabelMapType;
   typedef itk::ImageFileReader< VidaLabelMapType >      VidaReaderType;
   typedef itk::ImageRegionIterator< VidaLabelMapType >  VidaIteratorType;
   typedef itk::ImageRegionIterator< cip::LabelMapType > CIPIteratorType;
}

int main( int argc, char *argv[] )
{
  unsigned char cipRegion             = cip::UNDEFINEDREGION;
  unsigned char cipType               = cip::UNDEFINEDTYPE;


  //
  // Parse the input arguments
  //
    PARSE_ARGS;
    
    if (cipRegionArg > -1)
        cipRegion              = cipRegionArg;
    if (cipTypeArg > -1)
        cipType                = cipTypeArg;


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
