/** \file
 *  \ingroup commandLineTools 
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "FindPatchMatchCLP.h"

typedef itk::ImageRegionIteratorWithIndex< cip::CTType > IteratorType;

int main( int argc, char *argv[] )
{    
  PARSE_ARGS;  

  std::cout << "Reading CT..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Reading patch..." << std::endl;
  cip::CTReaderType::Pointer patchReader = cip::CTReaderType::New();
    patchReader->SetFileName( patchFileName );
  try
    {
    patchReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading patch:";
    std::cerr << excp << std::endl;
    }

  cip::CTType::IndexType patchStart;
  cip::CTType::SizeType patchSize;
    patchSize[0] = patchReader->GetOutput()->GetBufferedRegion().GetSize()[0];
    patchSize[1] = patchReader->GetOutput()->GetBufferedRegion().GetSize()[1];
    patchSize[2] = patchReader->GetOutput()->GetBufferedRegion().GetSize()[2];

  cip::CTType::RegionType patchRegion;
  patchRegion.SetSize( patchSize );
  
  bool foundPatch = false;

  IteratorType ctIt( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );

  std::cout << "Searching for patch matches..." << std::endl;
  ctIt.GoToBegin();
  while ( !ctIt.IsAtEnd() )
    {
      patchStart[0] = ctIt.GetIndex()[0];
      patchStart[1] = ctIt.GetIndex()[1];
      patchStart[2] = ctIt.GetIndex()[2];

      patchRegion.SetIndex( patchStart );

      IteratorType pIt( patchReader->GetOutput(), patchRegion );

      foundPatch = true;      
      pIt.GoToBegin();
      while ( !pIt.IsAtEnd() )
	{
	  if ( ctReader->GetOutput()->GetPixel( ctIt.GetIndex() ) != pIt.Get() )
	    {
	      foundPatch = false;
	      break;
	    }

	  ++pIt;
	}
      
      if ( foundPatch )
	{
	  std::cout << patchStart << std::endl;
	}

      ++ctIt;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
