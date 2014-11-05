/** \file
 *  \ingroup commandLineTools 
 *  \details 
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "LabelMapFromRegionAndTypePointsCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

  // Read the region and type points
  std::cout << "Reading region and type points file..." << std::endl;
  cipChestRegionChestTypeLocationsIO* pointsReader = new cipChestRegionChestTypeLocationsIO();
    pointsReader->SetFileName( regionAndTypePoints );
    pointsReader->Read();

  cip::LabelMapType::SpacingType spacing;
  cip::LabelMapType::SizeType    size;
  cip::LabelMapType::PointType   origin;

  if ( labelMapFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading label map file..." << std::endl;
      cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
        reader->SetFileName( labelMapFileName );
      try
	{
	reader->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught reading label map:";
        std::cerr << excp << std::endl;
	}

      spacing   = reader->GetOutput()->GetSpacing();
      size      = reader->GetOutput()->GetBufferedRegion().GetSize();
      origin[0] = reader->GetOutput()->GetOrigin()[0];
      origin[1] = reader->GetOutput()->GetOrigin()[1];
      origin[2] = reader->GetOutput()->GetOrigin()[2];
    }
  else if ( ctFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading CT file..." << std::endl;
      cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
        reader->SetFileName( ctFileName );
      try
	{
	reader->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught reading label map:";
	std::cerr << excp << std::endl;
	}

      spacing = reader->GetOutput()->GetSpacing();
      size    = reader->GetOutput()->GetBufferedRegion().GetSize();
      origin[0] = reader->GetOutput()->GetOrigin()[0];
      origin[1] = reader->GetOutput()->GetOrigin()[1];
      origin[2] = reader->GetOutput()->GetOrigin()[2];
    }
  else
    {
      std::cerr << "Must specify either a CT or label map file" << std::endl;
      return cip::INSUFFICIENTDATAFAILURE;
    }

  cip::LabelMapType::Pointer labelMap = cip::LabelMapType::New();
    labelMap->SetRegions( size );
    labelMap->Allocate();
    labelMap->FillBuffer( 0 );
    labelMap->SetSpacing( spacing );
    labelMap->SetOrigin( origin );

  for ( unsigned int i=0; i<pointsReader->GetOutput()->GetNumberOfTuples(); i++ )
    {
    cip::PointType point(3);
    cip::LabelMapType::IndexType index;

    pointsReader->GetOutput()->GetLocation( i, point );

    cip::LabelMapType::PointType imagePoint;
      imagePoint[0] = point[0];
      imagePoint[1] = point[1];
      imagePoint[2] = point[2];

    labelMap->TransformPhysicalPointToIndex( imagePoint, index );

    unsigned char cipRegion = pointsReader->GetOutput()->GetChestRegionValue( i );
    unsigned char cipType   = pointsReader->GetOutput()->GetChestTypeValue( i );

    unsigned short value = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );

    labelMap->SetPixel( index, value );
    }

  if ( outLabelMapFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Writing label map..." << std::endl;
      cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
        writer->SetFileName( outLabelMapFileName );
	writer->UseCompressionOn();
	writer->SetInput( labelMap );
      try
	{
	writer->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught writing label map:";
	std::cerr << excp << std::endl;
	}
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}



#endif
