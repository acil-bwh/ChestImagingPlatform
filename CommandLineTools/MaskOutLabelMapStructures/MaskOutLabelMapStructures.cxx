#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImageRegionIterator.h"
#include "MaskOutLabelMapStructuresCLP.h"

typedef itk::ImageFileReader< cip::LabelMapType >     LabelMapReader;
typedef itk::ImageFileWriter< cip::LabelMapType >     LabelMapWriter;
typedef itk::ImageRegionIterator< cip::LabelMapType > IteratorType;

struct PAIR
{
  unsigned char chestRegion;
  unsigned char chestType;
};

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

  std::vector< unsigned char > maskRegions;
  std::vector< unsigned char > maskTypes;
  std::vector< PAIR > maskRegionTypePairs;

  for ( unsigned int i=0; i<maskRegionNames.size(); i++ )
    {
      maskRegions.push_back( conventions.GetChestRegionValueFromName( maskRegionNames[i] ) );
    }
  for ( unsigned int i=0; i<maskTypeNames.size(); i++ )
    {
      maskTypes.push_back( conventions.GetChestTypeValueFromName( maskTypeNames[i] ) );
    }
  for ( unsigned int i=0; i<maskRegionTypePairNames.size(); i+=2 )
    {
      PAIR tmp;
      tmp.chestRegion = conventions.GetChestRegionValueFromName( maskRegionTypePairNames[i] );
      tmp.chestType   = conventions.GetChestTypeValueFromName( maskRegionTypePairNames[i+1] );

      maskRegionTypePairs.push_back( tmp );
    }

  std::cout << "Reading label mape..." << std::endl;
  LabelMapReader::Pointer reader = LabelMapReader::New();
    reader->SetFileName( inLabelMapFileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading base label map:";
    std::cerr << excp << std::endl;
    }

  // Now iterate over the input and mask out the structures indicated by the 
  // user.
  unsigned char cipRegion, cipType;
  bool masked;

  IteratorType it( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
      if ( it.Get() != 0 )
	{
	  masked = false;

	  cipRegion = conventions.GetChestRegionFromValue( it.Get() );
	  cipType  = conventions.GetChestTypeFromValue( it.Get() ); 

	  for ( unsigned int i=0; i<maskRegions.size(); i++ )
	    {
	      if ( cipRegion == maskRegions[i] )
		{
		  it.Set( 0 );
		  masked = true;
		  break;
		}
	    }
	  if ( !masked )
	    {
	      for ( unsigned int i=0; i<maskTypes.size(); i++ )
		{
		  if ( cipType == maskTypes[i] )
		    {
		      it.Set( 0 );
		      masked = true;
		      break;
		    }
		}
	    }
	  if ( !masked )
	    {
	      for ( unsigned int i=0; i<maskRegionTypePairs.size(); i++ )
		{
		  if ( cipType == maskRegionTypePairs[i].chestType && 
		       cipRegion == maskRegionTypePairs[i].chestRegion )
		    {
		      it.Set( 0 );
		      break;
		    }
		}
	    }
	}      

      ++it;
    }

  std::cout << "Writing masked label map..." << std::endl;
  LabelMapWriter::Pointer writer = LabelMapWriter::New();
    writer->SetInput( reader->GetOutput() );
    writer->SetFileName( outLabelMapFileName );
    writer->UseCompressionOn();
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

  return 0;
}

#endif
