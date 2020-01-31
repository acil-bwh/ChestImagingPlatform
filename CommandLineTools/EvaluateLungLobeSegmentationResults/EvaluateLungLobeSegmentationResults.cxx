/** \file
 *  \ingroup commandLineTools 
 *  \details This program is used to evaluate lung lobe segmentation results.  
 *  There are various ways in which a user can specify the lobe segmentation 
 *  to evaluate. If lobe segmentation mask and a ground truth lobe segmentation
 *  mask are both specified, Dice scores will be calculated.
 *
 *  If the user specifies the particles that were used to generate the 
 *  automatic lobe segmentation, then the user must also specify a set of 
 *  ground truth points (e.g. particles) at which to measure the boundary
 *  discrepancies. The same lambda (smoothing) value used to generate the lobe
 *  boundaries should also be specified in order to reproduce the same 
 *  TPS boundary surface from the particles.
 *
 *  If the user specifies shape model files defining the lobe boundary, they
 *  will be used to generate the lobe boundaries to evaluate. Again, the user
 *  must indicate the set of ground truth points to use for evaluation.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "EvaluateLungLobeSegmentationResultsCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "cipLobeSurfaceModelIO.h"
#include "cipParticleToThinPlateSplineSurfaceMetric.h"
#include "cipThinPlateSplineSurface.h"
#include "cipNewtonOptimizer.h"
#include "vtkPolyDataReader.h"
#include "vtkSmartPointer.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"

typedef itk::ImageRegionIterator< cip::LabelMapType > LabelMapIteratorType;

double GetDistanceFromPointToThinPlateSplineSurface( double, double, double, const cipThinPlateSplineSurface& );
void PrintAndComputeDiceScores( cip::LabelMapType::Pointer, cip::LabelMapType::Pointer );
void PrintStats( std::vector< double > );
void ComputeAndPrintFullSurfaceDiscrepancies( const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&, 
					      const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&, 
					      cip::LabelMapType::Pointer );
void ComputeAndPrintPointWiseSurfaceDiscrepancies( const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&, 
						   vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData >, 
						   vtkSmartPointer< vtkPolyData >, cip::LabelMapType::Pointer );

int main( int argc, char *argv[] )
{
  //
  // The lambda value for determining the TPS surfaces
  //
  double lambda = 0.1;

  //
  // Begin by defining the arguments to be passed
  //

  std::string loShapeModelFileName = "NA";
  std::string roShapeModelFileName = "NA";
  std::string rhShapeModelFileName = "NA";


  PARSE_ARGS;     


  //
  // Instantiate conventions for general use
  //
  cip::ChestConventions conventions;

  //
  // Compute and print Dice scores if label map file names have been specified
  //
  std::cout << "Reading ground truth label map..." << std::endl;
  cip::LabelMapReaderType::Pointer gtReader = cip::LabelMapReaderType::New();
    gtReader->SetFileName( gtLabelMapFileName );
  try
    {
    gtReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading ground truth label map:";
    std::cerr << excp << std::endl;
    return cip::LABELMAPREADFAILURE;
    }
  
  std::cout << "Reading automatically segmented label map..." << std::endl;
  cip::LabelMapReaderType::Pointer autoReader = cip::LabelMapReaderType::New();
    autoReader->SetFileName( autoLabelMapFileName );
  try
    {
    autoReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading automatically segmented label map:";
    std::cerr << excp << std::endl;
    return cip::LABELMAPREADFAILURE;
    }
  
  std::cout << "Computing Dice scores..." << std::endl;
  PrintAndComputeDiceScores( gtReader->GetOutput(), autoReader->GetOutput() );

  //
  // Read GT particles
  //
  std::cout << "Reading left oblique ground truth particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > loGTParticlesReader = vtkPolyDataReader::New();
    loGTParticlesReader->SetFileName( loGTParticlesFileName.c_str() );
    loGTParticlesReader->Update();    

  std::cout << loGTParticlesReader->GetOutput()->GetNumberOfPoints() << std::endl;

  std::cout << "Reading right oblique ground truth particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > roGTParticlesReader = vtkPolyDataReader::New();
    roGTParticlesReader->SetFileName( roGTParticlesFileName.c_str() );
    roGTParticlesReader->Update();    

  std::cout << roGTParticlesReader->GetOutput()->GetNumberOfPoints() << std::endl;

  std::cout << "Reading right horizontal ground truth particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > rhGTParticlesReader = vtkPolyDataReader::New();
    rhGTParticlesReader->SetFileName( rhGTParticlesFileName.c_str());
    rhGTParticlesReader->Update();    

  std::cout << rhGTParticlesReader->GetOutput()->GetNumberOfPoints() << std::endl;

  //
  // Read particles for producing the automatic lobe segmentation boundaries
  //
  std::cout << "Reading left oblique particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > loParticlesReader = vtkPolyDataReader::New();
    loParticlesReader->SetFileName( loParticlesFileName.c_str());
    loParticlesReader->Update();    

  std::cout << loParticlesReader->GetOutput()->GetNumberOfPoints() << std::endl;

  std::cout << "Reading right oblique particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > roParticlesReader = vtkPolyDataReader::New();
    roParticlesReader->SetFileName( roParticlesFileName.c_str() );
    roParticlesReader->Update();    

  std::cout << roParticlesReader->GetOutput()->GetNumberOfPoints() << std::endl;

  std::cout << "Reading right horizontal ground truth particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > rhParticlesReader = vtkPolyDataReader::New();
    rhParticlesReader->SetFileName( rhParticlesFileName.c_str() );
    rhParticlesReader->Update();    

  std::cout << rhParticlesReader->GetOutput()->GetNumberOfPoints() << std::endl;

  //
  // Read the ground truth lobe boundary region and type points file
  //
  std::cout << "Reading region-type points..." << std::endl;
  cipChestRegionChestTypeLocationsIO regionTypePointsIO;
    regionTypePointsIO.SetFileName( regionAndTypePointsFileName );
    regionTypePointsIO.Read();

  //
  // Define the vectors of physical points needed for creating the TPS. Note
  // that the [ro,lo,rh]GTPoints are not to be confused with the GT particles.
  // The points will contain a comprehensive set of points needed to define the
  // ground truth boundaries between lobes. The GT particles are phyical points
  // on those boundaries that identify fissure locations only.
  //
  std::vector< cip::PointType > loGTPoints;
  std::vector< cip::PointType > roGTPoints;
  std::vector< cip::PointType > rhGTPoints;

  //
  // Loop through the points and identify those (if any) that
  // correspond to defined fissures
  //
  unsigned char cipRegion;
  unsigned char cipType;

  for ( unsigned int i=0; i<regionTypePointsIO.GetOutput()->GetNumberOfTuples(); i++ )
    {
      cipRegion = regionTypePointsIO.GetOutput()->GetChestRegionValue(i);
      cipType   = regionTypePointsIO.GetOutput()->GetChestTypeValue(i);
      
      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::LEFTLUNG ) ) )
	{
	  if ( cipType == static_cast< unsigned char >( cip::OBLIQUEFISSURE ) )
	    {
	      cip::PointType location(3);	      
	      regionTypePointsIO.GetOutput()->GetLocation( i, location );
	      loGTPoints.push_back( location );
	    }
        }
      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::RIGHTLUNG ) ) )
        {
        if ( cipType == static_cast< unsigned char >( cip::OBLIQUEFISSURE ) )
          {
	    cip::PointType location(3);
	    regionTypePointsIO.GetOutput()->GetLocation( i, location );
	    roGTPoints.push_back( location );
          }
        else if ( cipType == static_cast< unsigned char >( cip::HORIZONTALFISSURE ) )
          {
	    cip::PointType location(3);
	    regionTypePointsIO.GetOutput()->GetLocation( i, location );
	    rhGTPoints.push_back( location );
          }
        }
    }
 
  //
  // Now create the ground truth TPS boundary surfaces
  // 
  std::cout << "Defining ground truth right oblique TPS surface..." << std::endl;
  cipThinPlateSplineSurface roGTTPS;
    roGTTPS.SetLambda( lambda );
    roGTTPS.SetSurfacePoints( roGTPoints );

  std::cout << "Defining ground truth left oblique TPS surface..." << std::endl;
  cipThinPlateSplineSurface loGTTPS;
    loGTTPS.SetLambda( lambda );
    loGTTPS.SetSurfacePoints( loGTPoints );

  std::cout << "Defining ground truth right horizontal TPS surface..." << std::endl;
  cipThinPlateSplineSurface rhGTTPS;
    rhGTTPS.SetLambda( lambda );
    rhGTTPS.SetSurfacePoints( rhGTPoints );

  //
  // Now create the TPS boundary surfaces corresponding to the automatic 
  // segmentation result
  // 
  std::vector< cip::PointType > loPoints;
  std::vector< cip::PointType > roPoints;
  std::vector< cip::PointType > rhPoints;
  for ( unsigned int i=0; i<roParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
    {
      cip::PointType point(3);
        point[0] = roParticlesReader->GetOutput()->GetPoint(i)[0];
	point[1] = roParticlesReader->GetOutput()->GetPoint(i)[1];
	point[2] = roParticlesReader->GetOutput()->GetPoint(i)[2];

      roPoints.push_back( point );
    }
  for ( unsigned int i=0; i<loParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
    {
      cip::PointType point(3);
        point[0] = loParticlesReader->GetOutput()->GetPoint(i)[0];
	point[1] = loParticlesReader->GetOutput()->GetPoint(i)[1];
	point[2] = loParticlesReader->GetOutput()->GetPoint(i)[2];

      loPoints.push_back( point );
    }
  for ( unsigned int i=0; i<rhParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
    {
      cip::PointType point(3);
        point[0] = rhParticlesReader->GetOutput()->GetPoint(i)[0];
	point[1] = rhParticlesReader->GetOutput()->GetPoint(i)[1];
	point[2] = rhParticlesReader->GetOutput()->GetPoint(i)[2];
	
      rhPoints.push_back( point );
    }

  std::cout << "Defining right oblique TPS surface..." << std::endl;
  cipThinPlateSplineSurface roTPS;
    roTPS.SetLambda( lambda );
    roTPS.SetSurfacePoints( roPoints );

  std::cout << "Defining left oblique TPS surface..." << std::endl;
  cipThinPlateSplineSurface loTPS;
    loTPS.SetLambda( lambda );
    loTPS.SetSurfacePoints( loPoints );

  std::cout << "Defining right horizontal TPS surface..." << std::endl;
  cipThinPlateSplineSurface rhTPS;
    rhTPS.SetLambda( lambda );
    rhTPS.SetSurfacePoints( rhPoints );

  //
  // At this point we have the TPS surfaces we need: for the ground truth and
  // for the automatically generated. We now want to compute surface 
  // discrepancies in two ways: across the entire surface and for selected
  // points.
  //
  ComputeAndPrintFullSurfaceDiscrepancies( roTPS, rhTPS, loTPS, roGTTPS, 
  					   rhGTTPS, loGTTPS, autoReader->GetOutput() );

  ComputeAndPrintPointWiseSurfaceDiscrepancies( roTPS, rhTPS, loTPS, 
  						roGTParticlesReader->GetOutput(), 
  						rhGTParticlesReader->GetOutput(), 
  						loGTParticlesReader->GetOutput(), autoReader->GetOutput() );



  // std::cout << "Creating RO TPS..." << std::endl;
  // cipThinPlateSplineSurface* roTPS = new cipThinPlateSplineSurface;
  //   roTPS->SetSurfacePoints( roModelReader.GetOutput()->GetWeightedSurfacePoints() );

  // std::cout << "Creating LO TPS..." << std::endl;
  // cipThinPlateSplineSurface* loTPS = new cipThinPlateSplineSurface;
  //   loTPS->SetSurfacePoints( loModelReader.GetOutput()->GetWeightedSurfacePoints() );

  // std::cout << "Creating RH TPS..." << std::endl;
  // cipThinPlateSplineSurface* rhTPS = new cipThinPlateSplineSurface;
  //   rhTPS->SetSurfacePoints( rhModelReader.GetOutput()->GetWeightedSurfacePoints() );
  //   //rhTPS->SetSurfacePoints( rhModelReader.GetOutput()->GetMeanSurfacePoints() );
    
  // std::vector< double > rhDistances;
  // std::vector< double > roDistances;
  // std::vector< double > loDistances;

  // double dist;
  // for ( unsigned int i=0; i<roGTParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
  //   {
  //     double* point = new double[3];
  //     point = roGTParticlesReader->GetOutput()->GetPoint(i);
  //     dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], roTPS );
  //     roDistances.push_back( dist );
  //   }
  // for ( unsigned int i=0; i<rhGTParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
  //   {
  //     double* point = new double[3];
  //     point = rhGTParticlesReader->GetOutput()->GetPoint(i);
  //     dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], rhTPS );
  //     rhDistances.push_back( dist );
  //   }
  // for ( unsigned int i=0; i<loGTParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
  //   {
  //     double* point = new double[3];
  //     point = loGTParticlesReader->GetOutput()->GetPoint(i);
  //     dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], loTPS );
  //     loDistances.push_back( dist );
  //     std::cout << point[0] << "\t" << point[1] << "\t" << point[2] << "\t" << dist << std::endl;
  //   }

  // double dist;
  // for ( unsigned int i=0; i<numPoints; i++ )
  //   {
  //     std::string cipRegion = regionTypePointsIO.GetOutput()->GetChestRegionName( i ); 
  //     std::string cipType   = regionTypePointsIO.GetOutput()->GetChestTypeName( i ); 

  //     if ( (cipRegion.compare( "LEFTLUNG" ) == 0 || cipRegion.compare( "LEFTSUPERIORLOBE" ) == 0||
  // 	    cipRegion.compare( "LEFTINFERIORLOBE" ) == 0) && cipType.compare( "OBLIQUEFISSURE" ) == 0 )
  // 	{
  // 	  double* point = new double[3];
  // 	  regionTypePointsIO.GetOutput()->GetLocation( i, point );
  // 	  dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], loTPS );
  // 	  loDistances.push_back( dist );
  // 	}
  //     if ( cipRegion.compare( "RIGHTLUNG" ) == 0 || cipRegion.compare( "RIGHTSUPERIORLOBE" ) == 0 ||
  // 	   cipRegion.compare( "RIGHTINFERIORLOBE" ) == 0 || cipRegion.compare( "RIGHTMIDDLELOBE" ) == 0 )
  // 	{
  // 	  if ( cipType.compare( "OBLIQUEFISSURE" ) == 0 ) 
  // 	    {
  // 	      double* point = new double[3];
  // 	      regionTypePointsIO.GetOutput()->GetLocation( i, point );
  // 	      dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], roTPS );
  // 	      roDistances.push_back( dist );
  // 	    }
  // 	  if ( cipType.compare( "HORIZONTALFISSURE" ) == 0 ) 
  // 	    {
  // 	      double* point = new double[3];
  // 	      regionTypePointsIO.GetOutput()->GetLocation( i, point );
  // 	      dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], rhTPS );
  // 	      rhDistances.push_back( dist );
  // 	    }
  // 	}		
  //   }

  // std::cout << "------------------------" << std::endl;
  // std::cout << "RO Stats:" << std::endl;
  // PrintStats( roDistances );

  // std::cout << "------------------------" << std::endl;
  // std::cout << "LO Stats:" << std::endl;
  // PrintStats( loDistances );

  // std::cout << "------------------------" << std::endl;
  // std::cout << "RH Stats:" << std::endl;
  // PrintStats( rhDistances );

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


void PrintStats( std::vector< double > distances )
{
  double maxDist = 0.0;
  double sum     = 0.0;
  double rms     = 0.0;
  for ( unsigned int i=0; i<distances.size(); i++ )
    {
      rms += distances[i]*distances[i];

      if ( distances[i] > maxDist )
	{
	  maxDist = distances[i];
	}
      sum += distances[i];
    }

  double mean = sum/static_cast< double >( distances.size() );

  std::cout << "Mean:\t" << mean << std::endl;

  //
  // Compute standard deviation
  //
  double std = 0.0;

  for ( unsigned int i=0; i<distances.size(); i++ )
    {
    std += std::pow( distances[i]-mean, 2 );
    }
  std /= static_cast< double >( distances.size() );

  std = std::sqrt( std );
  std::cout << "STD:\t" << std << std::endl;

  std::cout << "Max:\t" << maxDist << std::endl;
  std::cout << "RMS:\t" << std::sqrt(rms/static_cast< double >( distances.size() )) << std::endl;
}


double GetDistanceFromPointToThinPlateSplineSurface( double x, double y, double z, const cipThinPlateSplineSurface& tps )
{
  cipParticleToThinPlateSplineSurfaceMetric particleToTPSMetric;
    particleToTPSMetric.SetThinPlateSplineSurface( tps );

  cipNewtonOptimizer< 2 >* optimizer = new cipNewtonOptimizer< 2 >();
    optimizer->SetMetric( particleToTPSMetric );

  cipNewtonOptimizer< 2 >::PointType* domainParams  = new cipNewtonOptimizer< 2 >::PointType( 2 );

  cip::PointType position(3);
    position[0] = x;
    position[1] = y;
    position[2] = z;

  //
  // Determine the domain location for which the particle is closest
  // to the TPS surface
  //
  particleToTPSMetric.SetParticle( position );

  //
  // The particle's x, and y location are a good place to initialize
  // the search for the domain locations that result in the smallest
  // distance between the particle and the TPS surface
  //
  (*domainParams)[0] = position[0]; 
  (*domainParams)[1] = position[1]; 

  //
  // Perform Newton line search to determine the closest point on
  // the current TPS surface
  //
  optimizer->SetInitialParameters( domainParams );
  optimizer->Update();

  //
  // Get the distance between the particle and the TPS surface. This
  // is just the square root of the objective function value
  // optimized by the Newton method.
  //
  double distance = std::sqrt( optimizer->GetOptimalValue() );

//  delete particleToTPSMetric;  
  delete optimizer;

  return distance;
}


void PrintAndComputeDiceScores( cip::LabelMapType::Pointer gtLabelMap, cip::LabelMapType::Pointer autoLabelMap )
{
  unsigned int autoLUL = 0;
  unsigned int autoLLL = 0;
  unsigned int autoRUL = 0;
  unsigned int autoRML = 0;
  unsigned int autoRLL = 0;

  unsigned int gtLUL = 0;
  unsigned int gtLLL = 0;
  unsigned int gtRUL = 0;
  unsigned int gtRML = 0;
  unsigned int gtRLL = 0;

  unsigned int intLUL = 0;
  unsigned int intLLL = 0;
  unsigned int intRUL = 0;
  unsigned int intRML = 0;
  unsigned int intRLL = 0;

  LabelMapIteratorType gtIt( gtLabelMap, gtLabelMap->GetBufferedRegion() );
  LabelMapIteratorType autoIt( autoLabelMap, autoLabelMap->GetBufferedRegion() );

  autoIt.GoToBegin();
  gtIt.GoToBegin();
  while ( !autoIt.IsAtEnd() )
    {
      if ( gtIt.Get() == cip::LEFTSUPERIORLOBE )
	{
	  gtLUL++;
	}
      if ( gtIt.Get() == cip::LEFTINFERIORLOBE )
	{
	  gtLLL++;
	}
      if ( gtIt.Get() == cip::RIGHTSUPERIORLOBE )
	{
	  gtRUL++;
	}
      if ( gtIt.Get() == cip::RIGHTMIDDLELOBE )
	{
	  gtRML++;
	}
      if ( gtIt.Get() == cip::RIGHTINFERIORLOBE )
	{
	  gtRLL++;
	}

      if ( autoIt.Get() == cip::LEFTSUPERIORLOBE )
	{
	  autoLUL++;
	}
      if ( autoIt.Get() == cip::LEFTINFERIORLOBE )
	{
	  autoLLL++;
	}
      if ( autoIt.Get() == cip::RIGHTSUPERIORLOBE )
	{
	  autoRUL++;
	}
      if ( autoIt.Get() == cip::RIGHTMIDDLELOBE )
	{
	  autoRML++;
	}
      if ( autoIt.Get() == cip::RIGHTINFERIORLOBE )
	{
	  autoRLL++;
	}

      if ( autoIt.Get() == cip::LEFTSUPERIORLOBE && gtIt.Get() == cip::LEFTSUPERIORLOBE )
	{
	  intLUL++;
	}
      if ( autoIt.Get() == cip::LEFTINFERIORLOBE && gtIt.Get() == cip::LEFTINFERIORLOBE )
	{
	  intLLL++;
	}
      if ( autoIt.Get() == cip::RIGHTSUPERIORLOBE && gtIt.Get() == cip::RIGHTSUPERIORLOBE )
	{
	  intRUL++;
	}
      if ( autoIt.Get() == cip::RIGHTMIDDLELOBE && gtIt.Get() == cip::RIGHTMIDDLELOBE )
	{
	  intRML++;
	}
      if ( autoIt.Get() == cip::RIGHTINFERIORLOBE && gtIt.Get() == cip::RIGHTINFERIORLOBE )
	{
	  intRLL++;
	}

      ++autoIt;
      ++gtIt;
    }

  std::cout << "RUL Dice:\t" << 2.0*static_cast< double >( intRUL )/static_cast< double >( autoRUL + gtRUL ) << std::endl;
  std::cout << "RML Dice:\t" << 2.0*static_cast< double >( intRML )/static_cast< double >( autoRML + gtRML ) << std::endl;
  std::cout << "RLL Dice:\t" << 2.0*static_cast< double >( intRLL )/static_cast< double >( autoRLL + gtRLL ) << std::endl;
  std::cout << "LUL Dice:\t" << 2.0*static_cast< double >( intLUL )/static_cast< double >( autoLUL + gtLUL ) << std::endl;
  std::cout << "LLL Dice:\t" << 2.0*static_cast< double >( intLLL )/static_cast< double >( autoLLL + gtLLL ) << std::endl;
}


void ComputeAndPrintFullSurfaceDiscrepancies( const cipThinPlateSplineSurface& roTPS, const cipThinPlateSplineSurface& rhTPS, const cipThinPlateSplineSurface& loTPS, 
					      const cipThinPlateSplineSurface& roGTTPS, const cipThinPlateSplineSurface& rhGTTPS, const cipThinPlateSplineSurface& loGTTPS, 
					      cip::LabelMapType::Pointer labelMap )
{
  cip::ChestConventions conventions;

  cip::LabelMapType::SizeType    size    = labelMap->GetBufferedRegion().GetSize();
  cip::LabelMapType::SpacingType spacing = labelMap->GetSpacing();
  cip::LabelMapType::PointType   origin  = labelMap->GetOrigin();

  cip::LabelMapType::PointType point;
  cip::LabelMapType::IndexType index;

  double loHeight, roHeight, rhHeight;
  double loGTHeight, roGTHeight, rhGTHeight;

  std::vector< double > loDistances;
  std::vector< double > roDistances;
  std::vector< double > rhDistances;

  unsigned char cipRegion;
  double dist;

  for ( unsigned int x=0; x<size[0]; x++ )
    {
      point[0] = origin[0] + static_cast< double >(x)*spacing[0];

      for ( unsigned int y=0; y<size[1]; y++ )
	{
	  point[1] = origin[1] + static_cast< double >(y)*spacing[1];

	  loHeight = loTPS.GetSurfaceHeight( point[0], point[1] );
	  roHeight = roTPS.GetSurfaceHeight( point[0], point[1] );
	  rhHeight = rhTPS.GetSurfaceHeight( point[0], point[1] );

	  point[2] = loHeight;
	  labelMap->TransformPhysicalPointToIndex( point, index );
	  if ( labelMap->GetBufferedRegion().IsInside( index ) )
	    {
	      cipRegion = conventions.GetChestRegionFromValue( labelMap->GetPixel( index ) );
	      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::LEFTLUNG ) ) )
		{
		  dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], loGTTPS );
		  loDistances.push_back( dist );
		}
	    }

	  point[2] = roHeight;
	  labelMap->TransformPhysicalPointToIndex( point, index );
	  if ( labelMap->GetBufferedRegion().IsInside( index ) )
	    {
	      cipRegion = conventions.GetChestRegionFromValue( labelMap->GetPixel( index ) );
	      if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::RIGHTLUNG ) ) )
		{
		  dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], roGTTPS );
		  roDistances.push_back( dist );
		}
	    }

	  if ( rhHeight > roHeight )
	    {
	      point[2] = rhHeight;
	      labelMap->TransformPhysicalPointToIndex( point, index );
	      if ( labelMap->GetBufferedRegion().IsInside( index ) )
		{
		  labelMap->TransformPhysicalPointToIndex( point, index );
		  cipRegion = conventions.GetChestRegionFromValue( labelMap->GetPixel( index ) );
		  if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast< unsigned char >( cip::RIGHTLUNG ) ) )
		    {
		      dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], roGTTPS );
		      rhDistances.push_back( dist );
		    }
		}
	    }
	}
    }

  std::cout << "--------------------------------------------------------" << std::endl;
  std::cout << "LO Full Surface Distances:" << std::endl;
  PrintStats( loDistances );
  std::cout << "--------------------------------------------------------" << std::endl;
  std::cout << "RO Full Surface Distances:" << std::endl;
  PrintStats( roDistances );
  std::cout << "--------------------------------------------------------" << std::endl;
  std::cout << "RH Full Surface Distances:" << std::endl;
  PrintStats( rhDistances );
}


void ComputeAndPrintPointWiseSurfaceDiscrepancies( const cipThinPlateSplineSurface& roTPS, const cipThinPlateSplineSurface& rhTPS, const cipThinPlateSplineSurface& loTPS, 
						   vtkSmartPointer< vtkPolyData > roParticles, vtkSmartPointer< vtkPolyData > rhParticles, 
						   vtkSmartPointer< vtkPolyData > loParticles, cip::LabelMapType::Pointer labelMap )
{
  std::vector< double > loDistances;
  std::vector< double > roDistances;
  std::vector< double > rhDistances;

  double point[3];
  double dist;
  
  for ( unsigned int i=0; i<roParticles->GetNumberOfPoints(); i++ )
    {
      point[0] = roParticles->GetPoint(i)[0];
      point[1] = roParticles->GetPoint(i)[1];
      point[2] = roParticles->GetPoint(i)[2];
      dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], roTPS );
      roDistances.push_back(dist);
    }
  for ( unsigned int i=0; i<rhParticles->GetNumberOfPoints(); i++ )
    {
      point[0] = rhParticles->GetPoint(i)[0];
      point[1] = rhParticles->GetPoint(i)[1];
      point[2] = rhParticles->GetPoint(i)[2];
      dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], rhTPS );
      rhDistances.push_back(dist);
    }
  for ( unsigned int i=0; i<loParticles->GetNumberOfPoints(); i++ )
    {
      point[0] = loParticles->GetPoint(i)[0];
      point[1] = loParticles->GetPoint(i)[1];
      point[2] = loParticles->GetPoint(i)[2];
      dist = GetDistanceFromPointToThinPlateSplineSurface( point[0], point[1], point[2], loTPS );
      loDistances.push_back(dist);
    }

  std::cout << "--------------------------------------------------------" << std::endl;
  std::cout << "LO Points Distances:" << std::endl;
  PrintStats( loDistances );
  std::cout << "--------------------------------------------------------" << std::endl;
  std::cout << "RO Points Distances:" << std::endl;
  PrintStats( roDistances );
  std::cout << "--------------------------------------------------------" << std::endl;
  std::cout << "RH Points Distances:" << std::endl;
  PrintStats( rhDistances );
}

#endif
