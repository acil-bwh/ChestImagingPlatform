#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include <fstream>
#include "vtkPolyData.h"
#include "vtkFloatArray.h"
#include "vtkFieldData.h"
#include "vtkPolyDataReader.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipLabelMapToLungLobeLabelMapImageFilter.h"
#include "cipThinPlateSplineSurface.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "cipLobeSurfaceModelIO.h"
#include <time.h>
#include "SegmentLungLobesCLP.h"
#include "cipExceptionObject.h"

typedef cipLabelMapToLungLobeLabelMapImageFilter LungLobeSegmentationType;

void AppendFissurePoints( std::vector< double* >*, vtkPolyData* );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Instatiate ChestConventions for general convenience later
  cip::ChestConventions conventions;

  std::cout << "Reading lung label map..." << std::endl;
  cip::LabelMapReaderType::Pointer leftLungRightLungReader = cip::LabelMapReaderType::New();
    leftLungRightLungReader->SetFileName( inLabelMapFileName );
  try
    {
    leftLungRightLungReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while reading label map:";
    std::cerr << excp << std::endl;
      
    return cip::LABELMAPREADFAILURE;
    }
  
  // Define the vectors of physical points needed for creating the TPS
  std::vector< double* > loPoints;
  std::vector< double* > roPoints;
  std::vector< double* > rhPoints;

  LungLobeSegmentationType::Pointer lobeSegmenter = LungLobeSegmentationType::New();

  // If the user has specified a region-type points file name, read
  // the data
  if ( regionTypePointsFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading region-type points file..." << std::endl;
    cipChestRegionChestTypeLocationsIO regionTypesIO;
      regionTypesIO.SetFileName( regionTypePointsFileName );
      regionTypesIO.Read();

    // Loop through the points and identify those (if any) that
    // correspond to defined fissures
    unsigned char cipRegion;
    unsigned char cipType;

    for ( unsigned int i=0; i<regionTypesIO.GetOutput()->GetNumberOfTuples(); i++ )
      {
      cipRegion = regionTypesIO.GetOutput()->GetChestRegionValue(i);
      cipType   = regionTypesIO.GetOutput()->GetChestTypeValue(i);

      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)( cip::LEFTLUNG ) ) )
        {
        if ( cipType == (unsigned char)( cip::OBLIQUEFISSURE ) )
          {
          double* location = new double[3];

          regionTypesIO.GetOutput()->GetLocation( i, location );
          loPoints.push_back( location );
          }
        }
      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)( cip::RIGHTLUNG ) ) )
        {
        if ( cipType == (unsigned char)( cip::OBLIQUEFISSURE ) )
          {
          double* location = new double[3];

          regionTypesIO.GetOutput()->GetLocation( i, location );
          roPoints.push_back( location );
          }
        else if ( cipType == (unsigned char)( cip::HORIZONTALFISSURE ) )
          {
          double* location = new double[3];

          regionTypesIO.GetOutput()->GetLocation( i, location );
          rhPoints.push_back( location );
          }
        }
      }
    }

  // Read Fiducial points if they are available
  if( rightHorizontalFiducials.size() > 0 )
    {
      cip::LabelMapType::PointType lpsPoint;
      cip::LabelMapType::IndexType index;
      
      for( ::size_t i = 0; i < rightHorizontalFiducials.size(); ++i )
	{
	  // seeds come in ras, convert to lps
	  lpsPoint[0] = -rightHorizontalFiducials[i][0];
	  lpsPoint[1] = -rightHorizontalFiducials[i][1];
	  lpsPoint[2] = rightHorizontalFiducials[i][2];
	  
	  leftLungRightLungReader->GetOutput()->TransformPhysicalPointToIndex(lpsPoint, index);
	  double* location = new double[3];
  	    location[0] = index[0];
	    location[1] = index[1];
	    location[2] = index[2];

	  rhPoints.push_back(location);
	}
    }
  
  if( rightObliqueFiducials.size() > 0 )
    {
      cip::LabelMapType::PointType lpsPoint;
      cip::LabelMapType::IndexType index;
      
      for( ::size_t i = 0; i < rightObliqueFiducials.size(); ++i )
	{
	  // seeds come in ras, convert to lps
	  lpsPoint[0] = -rightObliqueFiducials[i][0];
	  lpsPoint[1] = -rightObliqueFiducials[i][1];
	  lpsPoint[2] = rightObliqueFiducials[i][2];
	  
	  leftLungRightLungReader->GetOutput()->TransformPhysicalPointToIndex(lpsPoint, index);
	  double* location = new double[3];
       	    location[0] = index[0];
	    location[1] = index[1];
	    location[2] = index[2];

	  roPoints.push_back(location);
	}
    }
  
  if( leftObliqueFiducials.size() > 0 )
    {
      cip::LabelMapType::PointType lpsPoint;
      cip::LabelMapType::IndexType index;
      
      for( ::size_t i = 0; i < leftObliqueFiducials.size(); ++i )
	{
	  // seeds come in ras, convert to lps
	  lpsPoint[0] = -leftObliqueFiducials[i][0];
	  lpsPoint[1] = -leftObliqueFiducials[i][1];
	  lpsPoint[2] = leftObliqueFiducials[i][2];
	  
	  leftLungRightLungReader->GetOutput()->TransformPhysicalPointToIndex(lpsPoint, index);
	  
	  double* location = new double[3];
	    location[0] = index[0];
	    location[1] = index[1];
	    location[2] = index[2];

	  loPoints.push_back(location);
	}
    }

  if ( rhParticlesFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading right horizontal particles..." << std::endl;
    vtkPolyDataReader* rhParticlesReader = vtkPolyDataReader::New();
      rhParticlesReader->SetFileName( rhParticlesFileName.c_str() );
      rhParticlesReader->Update();    

    std::cout << "Appending right horizontal fissure points..." << std::endl;
    AppendFissurePoints( &rhPoints, rhParticlesReader->GetOutput() );
    }
  if ( roParticlesFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading right oblique particles..." << std::endl;
    vtkPolyDataReader* roParticlesReader = vtkPolyDataReader::New();
      roParticlesReader->SetFileName( roParticlesFileName.c_str() );
      roParticlesReader->Update();    

    std::cout << "Appending right oblique fissure points..." << std::endl;
    AppendFissurePoints( &roPoints, roParticlesReader->GetOutput() );
    }
  if ( loParticlesFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading left oblique particles..." << std::endl;
    vtkPolyDataReader* loParticlesReader = vtkPolyDataReader::New();
      loParticlesReader->SetFileName( loParticlesFileName.c_str() );
      loParticlesReader->Update();    

    std::cout << "Appending left oblique fissure points..." << std::endl;
    AppendFissurePoints( &loPoints, loParticlesReader->GetOutput() );
    }
  if ( leftShapeModelFileName.compare( "NA" ) != 0 )
    { 
    std::cout << "Reading left lung shape model..." << std::endl;
    cip::LobeSurfaceModelIO* leftShapeModelIO = new cip::LobeSurfaceModelIO(); 
      leftShapeModelIO->SetFileName( leftShapeModelFileName );
    try
      {
      leftShapeModelIO->Read();
      }
    catch ( cip::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading shape model:" << std::endl;
      std::cerr << excp << std::endl; 
      }

    cipThinPlateSplineSurface* loTPS = new cipThinPlateSplineSurface();
      loTPS->SetSurfacePoints( leftShapeModelIO->GetOutput()->GetWeightedSurfacePoints() );

    lobeSegmenter->SetLeftObliqueThinPlateSplineSurface( loTPS );
    }
  if ( rightShapeModelFileName.compare( "NA" ) != 0 )
    { 
    std::cout << "Reading right lung shape model..." << std::endl;
    cip::LobeSurfaceModelIO* rightShapeModelIO = new cip::LobeSurfaceModelIO(); 
      rightShapeModelIO->SetFileName( rightShapeModelFileName );
    try
      {
      rightShapeModelIO->Read();
      }
    catch ( cip::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading shape model:" << std::endl;
      std::cerr << excp << std::endl; 
      }
    rightShapeModelIO->GetOutput()->SetRightLungSurfaceModel( true );

    cipThinPlateSplineSurface* roTPS = new cipThinPlateSplineSurface();
    roTPS->SetSurfacePoints( rightShapeModelIO->GetOutput()->GetRightObliqueWeightedSurfacePoints() );

    cipThinPlateSplineSurface* rhTPS = new cipThinPlateSplineSurface();
    rhTPS->SetSurfacePoints( rightShapeModelIO->GetOutput()->GetRightHorizontalWeightedSurfacePoints() );

    lobeSegmenter->SetRightObliqueThinPlateSplineSurface( roTPS );
    lobeSegmenter->SetRightHorizontalThinPlateSplineSurface( rhTPS );
    }
  
  if ( (rhPoints.size() > 0 && roPoints.size() == 0 && rightShapeModelFileName.compare( "NA" ) == 0) ||
       (rhPoints.size() == 0 && roPoints.size() > 0 && rightShapeModelFileName.compare( "NA" ) == 0) )
    {
      std::cerr << "Insufficient data to segment right lung lobes. Exiting." << std::endl;
      return cip::INSUFFICIENTDATAFAILURE;
    }

  std::cout << "Segmenting lobes..." << std::endl;
  lobeSegmenter->SetLeftObliqueFissurePoints( &loPoints );
  lobeSegmenter->SetRightObliqueFissurePoints( &roPoints );
  lobeSegmenter->SetRightHorizontalFissurePoints( &rhPoints );
  lobeSegmenter->SetInput( leftLungRightLungReader->GetOutput() );
  lobeSegmenter->SetThinPlateSplineSurfaceFromPointsLambda( lambda );
  lobeSegmenter->Update();
  
  std::cout << "Writing lung lobe label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetInput( lobeSegmenter->GetOutput() );
    writer->UseCompressionOn();
    writer->SetFileName( outLabelMapFileName );
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;
      
    return cip::LABELMAPWRITEFAILURE;
    }
    
  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

void AppendFissurePoints( std::vector< double* >* fissurePoints, vtkPolyData* particles )
{  
  unsigned int inc = 1; //static_cast< unsigned int >( vcl_ceil(
                        //particles->GetNumberOfPoints()/750.0 ) );

  bool addPoint;

  for ( unsigned int i=0; i<particles->GetNumberOfPoints(); i += inc )
    {
    addPoint = true;

    double* position = new double[3];
      position[0] = particles->GetPoint(i)[0];
      position[1] = particles->GetPoint(i)[1];
      position[2] = particles->GetPoint(i)[2];

    for ( unsigned int j=0; j<fissurePoints->size(); j++ )
      {
      if ( (*fissurePoints)[j][0] == position[0] && (*fissurePoints)[j][1] == position[1] )
        {
        addPoint = false;
        }
      }
    if ( addPoint )
      {
      fissurePoints->push_back( position );
      }
    }
}

#endif
