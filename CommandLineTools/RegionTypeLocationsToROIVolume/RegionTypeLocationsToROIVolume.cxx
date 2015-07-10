/** \file
*  \ingroup commandLineTools
*/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "cipChestRegionChestTypeLocations.h"
#include "RegionTypeLocationsToROIVolumeCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading reference CT..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading reference CT image:";
    std::cerr << excp << std::endl;
    }

  cip::LabelMapType::SizeType size;
    size[0] = ctReader->GetOutput()->GetBufferedRegion().GetSize()[0];
    size[1] = ctReader->GetOutput()->GetBufferedRegion().GetSize()[1];
    size[2] = ctReader->GetOutput()->GetBufferedRegion().GetSize()[2];

  cip::LabelMapType::PointType origin;
    origin[0] = ctReader->GetOutput()->GetOrigin()[0];
    origin[1] = ctReader->GetOutput()->GetOrigin()[1];
    origin[2] = ctReader->GetOutput()->GetOrigin()[2];

  cip::LabelMapType::SpacingType spacing;
    spacing[0] = ctReader->GetOutput()->GetSpacing()[0];
    spacing[1] = ctReader->GetOutput()->GetSpacing()[1];
    spacing[2] = ctReader->GetOutput()->GetSpacing()[2];

  cip::LabelMapType::Pointer roiVolume = cip::LabelMapType::New();
    roiVolume->SetRegions( size );
    roiVolume->Allocate();
    roiVolume->FillBuffer( 0 );
    roiVolume->SetOrigin( origin );
    roiVolume->SetSpacing( spacing );

  bool isPointsFile;
  std::cout << "Reading locations file..." << std::endl;
  cipChestRegionChestTypeLocationsIO locationsIO;
    if ( strcmp(inputIndicesFile.c_str(), "NA") != 0 )
      {
      locationsIO.SetFileName( inputIndicesFile );
      isPointsFile = false;
      }
    else if ( strcmp(inputPointsFile.c_str(), "NA") != 0 )
      {
      locationsIO.SetFileName( inputPointsFile );
      isPointsFile = true;
      }
    else
      {
      std::cerr << "No locations file specified." << std::endl;
      return cip::EXITFAILURE;
      }
      locationsIO.Read();

  std::cout << "Creating ROIs..." << std::endl;
  cip::ChestConventions conventions;
  cip::LabelMapType::IndexType index;
  cip::LabelMapType::IndexType tmpIndex;
  cip::LabelMapType::PointType point;

  unsigned short inc = 1;
  unsigned short roiVal;
  for ( unsigned int i=0; i<locationsIO.GetOutput()->GetNumberOfTuples(); i++ )
    {
      unsigned char cipRegion = locationsIO.GetOutput()->GetChestRegionValue( i );
      unsigned char cipType   = locationsIO.GetOutput()->GetChestTypeValue( i );
 
      if ( seg )
	{
	  roiVal = inc;
	}
      else
	{
	  roiVal = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );
	}
      inc++;

      unsigned int* indexLocation = new unsigned int[3];
      cip::PointType pointLocation(3);
      if ( isPointsFile )
	{
	  locationsIO.GetOutput()->GetLocation( i, pointLocation );
	  point[0] = pointLocation[0];
	  point[1] = pointLocation[1];
	  point[2] = pointLocation[2];

   	  ctReader->GetOutput()->TransformPhysicalPointToIndex( point, index );
	}
      else
	{
	  locationsIO.GetOutput()->GetLocation( i, indexLocation );
	  index[0] = indexLocation[0];
	  index[1] = indexLocation[1];
	  index[2] = indexLocation[2];
	}

      for ( unsigned int x = index[0] - xRadius; x <= index[0] + xRadius; x++ )
	{
	  for ( unsigned int y = index[1] - yRadius; y <= index[1] + yRadius; y++ )
	    {
	      for ( unsigned int z = index[2] - zRadius; z <= index[2] + zRadius; z++ )
		{
		  tmpIndex[0] = x;
		  tmpIndex[1] = y;
		  tmpIndex[2] = z;

		  if ( roiVolume->GetBufferedRegion().IsInside( tmpIndex ) )
		    {
		      roiVolume->SetPixel( tmpIndex, roiVal ); 
		    }
		}
	    }
	}      
    }

  std::cout << "Writing ROI volume..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetInput( roiVolume );
    writer->SetFileName( outVolumeFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing ROI image:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
