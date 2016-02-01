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

  cip::CTType::IndexType localStart;
  cip::CTType::SizeType patchSize;
    patchSize[0] = patchReader->GetOutput()->GetBufferedRegion().GetSize()[0];
    patchSize[1] = patchReader->GetOutput()->GetBufferedRegion().GetSize()[1];
    patchSize[2] = patchReader->GetOutput()->GetBufferedRegion().GetSize()[2];

  cip::CTType::IndexType index1;
  cip::CTType::IndexType index2;
  int patchSum = 0;
  short patchMax = -2000;
  for ( unsigned int i=0; i<patchSize[0]; i++ )
    {
      for ( unsigned int j=0; j<patchSize[1]; j++ )
	{
	  index1[0] = i;
	  index1[1] = j;
	  index1[2] = 0;
	  short val = patchReader->GetOutput()->GetPixel( index1 );
	  patchSum += val;
	  if ( val > patchMax )
	    {
	      patchMax = val;
	    }
	}
    }

  cip::CTType::RegionType localRegion;
    localRegion.SetSize( patchSize );

  bool foundPatch = false;

  IteratorType ctIt( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );
  IteratorType pIt( patchReader->GetOutput(), patchReader->GetOutput()->GetBufferedRegion() );

  std::cout << "Searching for patch matches..." << std::endl;
  ctIt.GoToBegin();
  while ( !ctIt.IsAtEnd() )
    {
      localStart[0] = ctIt.GetIndex()[0];
      localStart[1] = ctIt.GetIndex()[1];
      localStart[2] = ctIt.GetIndex()[2];

      if ( localStart[0] + patchSize[0] <= ctReader->GetOutput()->GetBufferedRegion().GetSize()[0] &&
      	   localStart[1] + patchSize[1] <= ctReader->GetOutput()->GetBufferedRegion().GetSize()[1] &&
      	   localStart[2] + patchSize[2] <= ctReader->GetOutput()->GetBufferedRegion().GetSize()[2] )
      	{
      	  localRegion.SetIndex( localStart );
	  IteratorType localIt( ctReader->GetOutput(), localRegion );

	  int localSum = 0;
	  short localMax = -2000;
      	  foundPatch = true;      
      	  pIt.GoToBegin();
	  localIt.GoToBegin();
      	  while ( !pIt.IsAtEnd() )
	    {
	      localSum += localIt.Get();
	      if ( localIt.Get() > localMax )
		{
		  localMax = localIt.Get();
		}
	      // if ( localIt.Get() != pIt.Get() )
      	      // 	{
      	      // 	  foundPatch = false;	
      	      // 	  break;
      	      // 	}

      	      ++pIt;
	      ++localIt;
      	    }

      	  //if ( foundPatch )
      	  if ( localSum == patchSum && localMax == patchMax )
      	    {
      	      std::cout << localStart << std::endl;
      	    }
      	}

      ++ctIt;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
