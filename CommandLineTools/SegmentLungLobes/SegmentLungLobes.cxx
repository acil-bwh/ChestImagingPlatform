/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads a label map image (where the labels
 *  correspond to the conventions laid out cipChestConventions.h) as
 *  well as particles defining the lobe boundaries and produces a
 *  label map with the lung lobes identified. The input is assumed to
 *  have the left and right lungs uniquely labeled. The user can pass
 *  particles for the left lung only (left oblique fissure particles),
 *  right lung only (both right oblique and right horizontal fissure
 *  particles) or both. Thin plate splines are used to define the
 *  interpolation boundaries between the lobes. It is assumed that the
 *  input particles datasets are "clean" in the sense that each
 *  particle corresponds to (or is very likely to correspond to) the
 *  fissure it represents.
 *
 *  $Date: 2012-05-09 07:05:14 -0700 (Wed, 09 May 2012) $
 *  $Revision: 128 $
 *  $Author: jross $
 *
 *  USAGE:
 *
 *  SegmentLungLobes.exe  [-l \<double\>] --lobes \<string\>
 *                        --rhParticles \<string\>
 *                        --roParticles \<string\>
 *                        --loParticles \<string\> --leftRight
 *                        \<string\> [--] [--version] [-h]
 *
 *  Where:
 *
 *   -l \<double\>,  --lambda \<double\>
 *     Thin plate spline smoothing parameter
 *
 *   --lobes \<string\>
 *     (required)  Lung lobe label map file name
 *
 *   --regionType <string>
 *     Region and type points file name
 *
 *   --rhParticles \<string\>
 *     (required)  Right horizontal particles file name
 *
 *   --roParticles \<string\>
 *     (required)  Right oblique particles file name
 *
 *   --loParticles \<string\>
 *     (required)  Left oblique particles file name
 *
 *   --leftRight \<string\>
 *     (required)  Left-lung-right-lung file name. The assumption forthis
 *     input label map is that the left and right lungs have been labeled
 *     correctly. Note that these are theONLY lung regions that are assumed
 *     to be labeled.
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 */


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
#include <time.h>
#include "SegmentLungLobesCLP.h"

typedef cipLabelMapToLungLobeLabelMapImageFilter LungLobeSegmentationType;

void AppendFissurePoints( std::vector< double* >*, vtkPolyData* );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Instatiate ChestConventions for general convenience later
  cip::ChestConventions conventions;
  
  // Define the vectors of physical points needed for creating the TPS
  std::vector< double* > loPoints;
  std::vector< double* > roPoints;
  std::vector< double* > rhPoints;
  
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

      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::LEFTLUNG ) ) )
        {
        if ( cipType == static_cast< unsigned char >( cip::OBLIQUEFISSURE ) )
          {
          double* location = new double[3];

          regionTypesIO.GetOutput()->GetLocation( i, location );
          loPoints.push_back( location );
          }
        }
      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::RIGHTLUNG ) ) )
        {
        if ( cipType == static_cast< unsigned char >( cip::OBLIQUEFISSURE ) )
          {
          double* location = new double[3];

          regionTypesIO.GetOutput()->GetLocation( i, location );
          roPoints.push_back( location );
          }
        else if ( cipType == static_cast< unsigned char >( cip::HORIZONTALFISSURE ) )
          {
          double* location = new double[3];

          regionTypesIO.GetOutput()->GetLocation( i, location );
          rhPoints.push_back( location );
          }
        }
      }
    }
  
  // Read in the left-lung-right-lung label map
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
      location[0]=index[0];
      location[1]=index[1];
      location[2]=index[2];
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
      location[0]=index[0];
      location[1]=index[1];
      location[2]=index[2];
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
      location[0]=index[0];
      location[1]=index[1];
      location[2]=index[2];
      loPoints.push_back(location);
  	}
  }


  // Read in the particles data
  if ( loParticlesFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading left oblique particles..." << std::endl;
    vtkPolyDataReader* loParticlesReader = vtkPolyDataReader::New();
      loParticlesReader->SetFileName( loParticlesFileName.c_str() );
      loParticlesReader->Update();    

    std::cout << "Appending left oblique fissure points..." << std::endl;
    AppendFissurePoints( &loPoints, loParticlesReader->GetOutput() );
    }

  if ( roParticlesFileName.compare( "NA" ) != 0 && rhParticlesFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading right oblique particles..." << std::endl;
    vtkPolyDataReader* roParticlesReader = vtkPolyDataReader::New();
      roParticlesReader->SetFileName( roParticlesFileName.c_str() );
      roParticlesReader->Update();    

    std::cout << "Appending right oblique fissure points..." << std::endl;
    AppendFissurePoints( &roPoints, roParticlesReader->GetOutput() );

    std::cout << "Reading right horizontal particles..." << std::endl;
    vtkPolyDataReader* rhParticlesReader = vtkPolyDataReader::New();
      rhParticlesReader->SetFileName( rhParticlesFileName.c_str() );
      rhParticlesReader->Update();    

    std::cout << "Appending right horizontal fissure points..." << std::endl;
    AppendFissurePoints( &rhPoints, rhParticlesReader->GetOutput() );
    }

  // At this stage, we should have the necessary points for segmenting
  // the lobes: in the left lung, the right lung, or both. Before
  // proceeding, we need to verify this
  if ( roPoints.size() == 0 && rhPoints.size() == 0 && loPoints.size() == 0 )
    {
    std::cerr << "Insufficient fissure points specified. Exiting." << std::endl;
    return cip::INSUFFICIENTDATAFAILURE;
    }
  if ( (roPoints.size() == 0 && rhPoints.size() != 0) || (roPoints.size() != 0 && rhPoints.size() == 0) )
    {
    std::cerr << "Insufficient fissure points specified. Exiting." << std::endl;
    return cip::INSUFFICIENTDATAFAILURE;
    }
  

  // Now define the TPS surfaces for each of the three fissures
  cipThinPlateSplineSurface* roTPS = new cipThinPlateSplineSurface;
  if ( roPoints.size() != 0 )
    {
    std::cout << "Defining right oblique TPS surface..." << std::endl;
    roTPS->SetLambda( lambda );
    roTPS->SetSurfacePoints( &roPoints );
    }

  cipThinPlateSplineSurface* loTPS = new cipThinPlateSplineSurface;
  if ( loPoints.size() != 0 )
    {
    std::cout << "Defining left oblique TPS surface..." << std::endl;
    loTPS->SetLambda( lambda );
    loTPS->SetSurfacePoints( &loPoints );
    }

  cipThinPlateSplineSurface* rhTPS = new cipThinPlateSplineSurface;
  if ( rhPoints.size() != 0 )
    {
    std::cout << "Defining right horizontal TPS surface..." << std::endl;
    rhTPS->SetLambda( lambda );
    rhTPS->SetSurfacePoints( &rhPoints );
    }

  // Now segment the lobes
  std::cout << "Segmenting lobes..." << std::endl;
  LungLobeSegmentationType::Pointer lobeSegmenter = LungLobeSegmentationType::New();
    lobeSegmenter->SetInput( leftLungRightLungReader->GetOutput() );
  if ( roPoints.size() != 0 && rhPoints.size() != 0 )
    {
    lobeSegmenter->SetRightObliqueThinPlateSplineSurface( roTPS );
    lobeSegmenter->SetRightHorizontalThinPlateSplineSurface( rhTPS );
    }
  if ( loPoints.size() != 0 )
    {
    lobeSegmenter->SetLeftObliqueThinPlateSplineSurface( loTPS );
    }
    lobeSegmenter->Update();
      
  // Write the lung lobe label map 
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
