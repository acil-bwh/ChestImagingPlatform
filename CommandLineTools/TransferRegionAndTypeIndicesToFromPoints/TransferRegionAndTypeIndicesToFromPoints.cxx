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
#include "TransferRegionAndTypeIndicesToFromPointsCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading reference CT..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFile );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading reference CT image:";
    std::cerr << excp << std::endl;
    }

  cip::CTType::IndexType ctIndex;
  cip::CTType::PointType ctPoint;
  unsigned char cipRegion, cipType;

  if ( strcmp(inputIndicesFile.c_str(), "NA") != 0 )
    {
      std::cout << "Reading indices file..." << std::endl;
      cipChestRegionChestTypeLocationsIO indicesIO;
        indicesIO.SetFileName( inputIndicesFile );
	indicesIO.Read();

      std::cout << "Transfering indices to points..." << std::endl;
      cipChestRegionChestTypeLocations points;
      for ( unsigned int i=0; i<indicesIO.GetOutput()->GetNumberOfTuples(); i++ )
      	{
	  unsigned int* indexLocation = new unsigned int[3];
	  double*       pointLocation = new double[3];

      	  indicesIO.GetOutput()->GetLocation( i, indexLocation );

	  cipRegion = indicesIO.GetOutput()->GetChestRegionValue( i );
	  cipType   = indicesIO.GetOutput()->GetChestTypeValue( i );

	  ctIndex[0] = indexLocation[0];
	  ctIndex[1] = indexLocation[1];
	  ctIndex[2] = indexLocation[2];
	  ctReader->GetOutput()->TransformIndexToPhysicalPoint( ctIndex, ctPoint );

	  pointLocation[0] = ctPoint[0];
	  pointLocation[1] = ctPoint[1];
	  pointLocation[2] = ctPoint[2];
	  points.SetChestRegionChestTypeLocation( cipRegion, cipType, pointLocation );
      	}

      std::cout << "Writing points file..." << std::endl;
      cipChestRegionChestTypeLocationsIO pointsIO;
        pointsIO.SetFileName( outputPointsFile );
      	pointsIO.SetInput( &points );
      	pointsIO.Write();
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
