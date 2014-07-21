/** \file
 *  \ingroup commandLineTools 
 *  \details This program ...
 *  
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "GenerateInteractiveLobeSegmentationFissureMaskCLP.h"
#include "cipChestConventions.h"
#include "itkResampleImageFilter.h"
#include "itkImage.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkIdentityTransform.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"


typedef itk::Image< unsigned short, 3 >                    LabelMapType;
typedef itk::ImageFileReader< LabelMapType >               ReaderType;
typedef itk::ImageFileWriter< LabelMapType >               WriterType;
typedef itk::ImageRegionIteratorWithIndex< LabelMapType >  LabelMapIteratorType;


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
/*  std::string   inLabelMapFileName   = "NA";
  std::string   outLabelMapFileName  = "NA";
  int           radius               = 5;
  
  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( "", ' ', "$Revision: 153 $" );

    //::ValueArg<std::string>  inLabelMapFileNameArg ( "i", "in", "", true, inLabelMapFileName, "string", cl );
    //TCLAP::ValueArg<std::string>  outLabelMapFileNameArg ( "o", "out", "", true, outLabelMapFileName, "string", cl );
   // TCLAP::ValueArg<int>          radiusArg ( "r", "radius", "", false, radius, "int", cl );

    cl.parse( argc, argv );

    inLabelMapFileName  = inLabelMapFileNameArg.getValue();
    outLabelMapFileName = outLabelMapFileNameArg.getValue();
    radius              = radiusArg.getValue();    
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }  
*/
  //
  // Instantiate ChestConventions for convenience
  //
  ChestConventions conventions;

  unsigned short fissureLabel = 
    conventions.GetValueFromChestRegionAndType( static_cast< unsigned char >( cip::UNDEFINEDREGION ), static_cast< unsigned char >( cip::FISSURE ) );

  //
  // Read the label map image
  //
  std::cout << "Reading label map..." << std::endl;
  ReaderType::Pointer reader = ReaderType::New(); 
    reader->SetFileName( inLabelMapFileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading image:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }

  //
  // Create the output label map to fill
  //
  LabelMapType::Pointer outLabelMap = LabelMapType::New();
    outLabelMap->SetRegions( reader->GetOutput()->GetBufferedRegion().GetSize() );
    outLabelMap->Allocate();
    outLabelMap->FillBuffer( 0 );
    outLabelMap->SetOrigin( reader->GetOutput()->GetOrigin() );
    outLabelMap->SetSpacing( reader->GetOutput()->GetSpacing() );

  //
  // Isolate the chest region / type of interest
  //
  std::cout << "Creating fissure mask..." << std::endl;
  LabelMapIteratorType it( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 && outLabelMap->GetPixel( it.GetIndex() ) == 0 )
      {
      unsigned char tmpRegion = conventions.GetChestRegionFromValue( it.Get() );
      unsigned char tmpType   = conventions.GetChestTypeFromValue( it.Get() );

      if ( tmpRegion > 0 && tmpType > 0 )
	{	  
	  for ( int x = it.GetIndex()[0]-radius; x <= it.GetIndex()[0]+radius; x++ )
	    {
	      for ( int y = it.GetIndex()[1]-radius; y <= it.GetIndex()[1]+radius; y++ )
		{
		  for ( int z = it.GetIndex()[2]-radius; z <= it.GetIndex()[2]+radius; z++ )
		    {
		      LabelMapType::IndexType index;
      		        index[0] = x;
			index[1] = y;
			index[2] = z;		       

		      if ( reader->GetOutput()->GetBufferedRegion().IsInside( index ) )
			{
			  if ( reader->GetOutput()->GetPixel( index ) != 0 )
			    {
			      //outLabelMap->SetPixel( index, fissureLabel );
			      outLabelMap->SetPixel( index, 1 );
			    }
			}
		    }
		}
	    }
	}
      }
    
    ++it;
    }

  LabelMapIteratorType rIt( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType oIt( outLabelMap, outLabelMap->GetBufferedRegion() );

  rIt.GoToBegin();
  oIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
      rIt.Set( oIt.Get() );

      ++rIt;
      ++oIt;
    }

  std::cout << "Writing label map..." << std::endl;
  WriterType::Pointer writer = WriterType::New();
    writer->SetInput( reader->GetOutput() );
    writer->SetFileName( outLabelMapFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing image:";
    std::cerr << excp << std::endl;
    
    return cip::LABELMAPWRITEFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
