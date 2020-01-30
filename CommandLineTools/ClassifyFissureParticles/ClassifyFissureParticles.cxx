/** \file
 *  \ingroup commandLineTools
 *  \details This program is used to classify fissure particles using Fischer's
 *  Linear Discriminant. Left or right lung fissure particles are read in
 *  along with lobe boundary shape models for the left or right lung. For
 *  each particle, its distance and angle with respect to the lobe
 *  boundaries are computed. The weighted sum of these quantities is then
 *  computed and compared to a threshold value, and a classification
 *  decision is made (either fissure or noise). If particles in the right
 *  lung are being considered, a particle is classified according to which
 *  entity it is most like (noise, right horizontal or right oblique). The
 *  classified particles are then written to file.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipExceptionObject.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "itkNumericTraits.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "cipNewtonOptimizer.h"
#include "cipThinPlateSplineSurface.h"
#include "cipNelderMeadSimplexOptimizer.h"
#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "cipHelper.h"
#include "cipLobeSurfaceModelIO.h"
#include "ClassifyFissureParticlesCLP.h"

struct PARTICLEINFO
{
  std::vector< double > distance;
  std::vector< double > angle;
  unsigned char         cipType;
};

void GetParticleDistanceAndAngle( vtkPolyData*, unsigned int, const cipThinPlateSplineSurface&, double*, double* );
void TallyParticleInfo( vtkPolyData*, std::vector< cipThinPlateSplineSurface >, std::map< unsigned int, PARTICLEINFO >* );
void ClassifyParticles( std::map< unsigned int, PARTICLEINFO >*, double, double, double, double );
void WriteParticlesToFile( vtkSmartPointer< vtkPolyData >, std::map< unsigned int, PARTICLEINFO >, std::string, unsigned char );
void GetSurfacePointsFromLabelMap( cip::LabelMapType::Pointer, std::vector< cip::PointType >*, std::vector< cip::PointType >*,
				   std::vector< cip::PointType >* );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cipThinPlateSplineSurface loTPS;
  cipThinPlateSplineSurface roTPS;
  cipThinPlateSplineSurface rhTPS;  
  
  // Read complete particle in lung
  std::cout << "Reading lung particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( particlesFileName.c_str() );
    particlesReader->Update();
    
  if ( lungLobeLabelMapFileName.compare( "NA" ) != 0 )
    {
      cip::ChestConventions conventions;
      
      std::cout << "Reading lung lobe label map..." << std::endl;
      cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
        labelMapReader->SetFileName( lungLobeLabelMapFileName );
      try
	{
	labelMapReader->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught while reading label map:";
	std::cerr << excp << std::endl;
	  
	return cip::LABELMAPREADFAILURE;
	}

      cip::LabelMapType::IndexType index;
      cip::LabelMapType::PointType origin = labelMapReader->GetOutput()->GetOrigin();
      cip::LabelMapType::SpacingType spacing = labelMapReader->GetOutput()->GetSpacing();
      
      std::vector< cip::PointType > rightObliqueSurfacePoints;
      std::vector< cip::PointType > rightHorizontalSurfacePoints;
      std::vector< cip::PointType > leftObliqueSurfacePoints;

      std::cout << "Getting surface points..." << std::endl;
      GetSurfacePointsFromLabelMap(labelMapReader->GetOutput(), &rightObliqueSurfacePoints,
				   &rightHorizontalSurfacePoints, &leftObliqueSurfacePoints);

      std::cout << "Identifying particles region and isolating..." << std::endl;
      // March through the particles and identify left or right particles
      std::vector< vtkFloatArray* > leftArrayVec;
      std::vector< vtkFloatArray* > rightArrayVec;      

      unsigned int numberOfPointDataArrays = particlesReader->GetOutput()->GetPointData()->GetNumberOfArrays();
      for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
	{
	  vtkFloatArray* leftArray = vtkFloatArray::New();
	    leftArray->SetNumberOfComponents( particlesReader->GetOutput()->GetPointData()->GetArray(i)->GetNumberOfComponents() );
	    leftArray->SetName( particlesReader->GetOutput()->GetPointData()->GetArray(i)->GetName() );
	  
	  leftArrayVec.push_back( leftArray );

	  vtkFloatArray* rightArray = vtkFloatArray::New();
	    rightArray->SetNumberOfComponents( particlesReader->GetOutput()->GetPointData()->GetArray(i)->GetNumberOfComponents() );
	    rightArray->SetName( particlesReader->GetOutput()->GetPointData()->GetArray(i)->GetName() );
	  
	  rightArrayVec.push_back( rightArray );	  
	}
      
      vtkPoints* leftPoints  = vtkPoints::New();
      vtkPoints* rightPoints  = vtkPoints::New();
      
      vtkSmartPointer< vtkPolyData > leftParticles = vtkSmartPointer< vtkPolyData >::New();
      vtkSmartPointer< vtkPolyData > rightParticles = vtkSmartPointer< vtkPolyData >::New();

      cip::TransferFieldData( particlesReader->GetOutput(), leftParticles );
      cip::TransferFieldData( particlesReader->GetOutput(), rightParticles );      

      unsigned int leftInc = 0;
      unsigned int rightInc = 0;
      for ( unsigned int i=0; i<particlesReader->GetOutput()->GetNumberOfPoints(); i++ )
	{
	  index[0] = (unsigned int)((particlesReader->GetOutput()->GetPoint(i)[0] - origin[0])/spacing[0]);
	  index[1] = (unsigned int)((particlesReader->GetOutput()->GetPoint(i)[1] - origin[1])/spacing[1]);
	  index[2] = (unsigned int)((particlesReader->GetOutput()->GetPoint(i)[2] - origin[2])/spacing[2]);

	  unsigned char cipRegion = conventions.GetChestRegionFromValue( labelMapReader->GetOutput()->GetPixel( index ) );
	  if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)(cip::LEFTLUNG) ) )
	    {
	      leftPoints->InsertNextPoint( particlesReader->GetOutput()->GetPoint(i) );
	      for ( unsigned int k=0; k<numberOfPointDataArrays; k++ )
		{
		  leftArrayVec[k]->InsertTuple( leftInc, particlesReader->GetOutput()->GetPointData()->GetArray(k)->GetTuple(i) );
		}
	      leftInc++;	      
	    }
	  else if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)(cip::RIGHTLUNG) ) )
	    {
	      rightPoints->InsertNextPoint( particlesReader->GetOutput()->GetPoint(i) );
	      for ( unsigned int k=0; k<numberOfPointDataArrays; k++ )
		{
		  rightArrayVec[k]->InsertTuple( rightInc, particlesReader->GetOutput()->GetPointData()->GetArray(k)->GetTuple(i) );
		}
	      rightInc++;	      	      
	    }								   								   	 
	}

      leftParticles->SetPoints( leftPoints );
      rightParticles->SetPoints( rightPoints );      
      for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
	{
	  leftParticles->GetPointData()->AddArray( leftArrayVec[j] );
	  rightParticles->GetPointData()->AddArray( rightArrayVec[j] );	  
	}

      if ( leftParticles->GetNumberOfPoints() > 0 )
	{
	  loTPS.SetSurfacePoints( leftObliqueSurfacePoints );      
	  std::vector< cipThinPlateSplineSurface > tpsVecLeft;            
	  tpsVecLeft.push_back( loTPS );
	  
	  std::map< unsigned int, PARTICLEINFO >  leftParticleToInfoMap;
	  std::cout << "Tallying left particles info..." << std::endl;
	  TallyParticleInfo( leftParticles, tpsVecLeft, &leftParticleToInfoMap );
	  
	  std::cout << "Classifying particles..." << std::endl;
	  ClassifyParticles( &leftParticleToInfoMap, distanceWeight, angleWeight,
			     fischerThreshold, distanceThreshold );
	  
	  if ( loClassifiedFileName.compare( "NA" ) != 0 )
	    {
	      std::cout << "Writing left oblique particles to file..." << std::endl;
	      WriteParticlesToFile( leftParticles, leftParticleToInfoMap,
				    loClassifiedFileName, (unsigned char)( cip::OBLIQUEFISSURE ) );
	    }
	}

      if ( rightParticles->GetNumberOfPoints() > 0 )
	{      
	  roTPS.SetSurfacePoints( rightObliqueSurfacePoints );      
	  rhTPS.SetSurfacePoints( rightHorizontalSurfacePoints );      
	  std::vector< cipThinPlateSplineSurface > tpsVecRight;
	  tpsVecRight.push_back( roTPS );
	  tpsVecRight.push_back( rhTPS );

	  std::map< unsigned int, PARTICLEINFO >  rightParticleToInfoMap;
	  std::cout << "Tallying right particles info..." << std::endl;	  
	  TallyParticleInfo( rightParticles, tpsVecRight, &rightParticleToInfoMap );

	  std::cout << "Classifying particles..." << std::endl;
	  ClassifyParticles( &rightParticleToInfoMap, distanceWeight, angleWeight,
			     fischerThreshold, distanceThreshold );
	  
	  if ( roClassifiedFileName.compare( "NA" ) != 0 )
	    {
	      std::cout << "Writing right oblique particles to file..." << std::endl;
	      WriteParticlesToFile( rightParticles, rightParticleToInfoMap,
				    roClassifiedFileName, (unsigned char)( cip::OBLIQUEFISSURE ) );
	    }
	  if ( rhClassifiedFileName.compare( "NA" ) != 0 )
	    {
	      std::cout << "Writing right horizontal particles to file..." << std::endl;
	      WriteParticlesToFile( rightParticles, rightParticleToInfoMap,
				    rhClassifiedFileName, (unsigned char)( cip::HORIZONTALFISSURE ) );
	    }	  	  
	}
      std::cout << "DONE." << std::endl;

      return 0;
    }

  std::vector< cipThinPlateSplineSurface > tpsVec;

  // Read shape models
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
      std::cerr << "Exception caught reading right shape model:";
      std::cerr << excp << std::endl;
      }
      rightShapeModelIO->GetOutput()->SetRightLungSurfaceModel( true );

    roTPS.SetSurfacePoints( rightShapeModelIO->GetOutput()->GetRightObliqueWeightedSurfacePoints() );

    // Note ordering is important here. The RO needs to be pushed back
    // before the RH (assumed when we execute 'TallyParticleInfo')
    tpsVec.push_back( roTPS );

    rhTPS.SetSurfacePoints( rightShapeModelIO->GetOutput()->GetRightHorizontalWeightedSurfacePoints() );

    // Note ordering is important here. The RO needs to be pushed back
    // before the RH (assumed when we execute 'TallyParticleInfo')
    tpsVec.push_back( rhTPS );
    }
  else if ( leftShapeModelFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Reading left lung shape model..." << std::endl;
    cip::LobeSurfaceModelIO* leftShapeModelIO = new cip::LobeSurfaceModelIO();
      leftShapeModelIO->SetFileName( leftShapeModelFileName );
      leftShapeModelIO->Read();

    loTPS.SetSurfacePoints( leftShapeModelIO->GetOutput()->GetWeightedSurfacePoints() );

    tpsVec.push_back( loTPS );
    }
  else
    {
    std::cerr << "ERROR: No shape model specified. Exiting." << std::endl;
    return 1;
    }

  // Now we want to tally the component information, computing the
  // mean distance and the mean angle with respect to the fit surface
  std::map< unsigned int, PARTICLEINFO >  particleToInfoMap;

  TallyParticleInfo( particlesReader->GetOutput(), tpsVec, &particleToInfoMap );

  // Now classify the particles
  std::cout << "Classifying particles..." << std::endl;
  ClassifyParticles( &particleToInfoMap, distanceWeight, angleWeight, fischerThreshold, distanceThreshold );

  // Write the classified (fissure) particles to file
  if ( loClassifiedFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing left oblique particles to file..." << std::endl;
    WriteParticlesToFile( particlesReader->GetOutput(), particleToInfoMap, loClassifiedFileName, (unsigned char)( cip::OBLIQUEFISSURE ) );
    }
  if ( roClassifiedFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing right oblique particles to file..." << std::endl;
    WriteParticlesToFile( particlesReader->GetOutput(), particleToInfoMap, roClassifiedFileName, (unsigned char)( cip::OBLIQUEFISSURE ) );
    }
  if ( rhClassifiedFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing right horizontal particles to file..." << std::endl;
    WriteParticlesToFile( particlesReader->GetOutput(), particleToInfoMap, rhClassifiedFileName, (unsigned char)( cip::HORIZONTALFISSURE ) );
    }

  std::cout << "DONE." << std::endl;

  return 0;
}

void GetParticleDistanceAndAngle( vtkPolyData* particles, unsigned int whichParticle, const cipThinPlateSplineSurface& tps,
                                  double* distance, double* angle ) 
{
  cip::PointType position(3);
  cip::VectorType normal(3);
  cip::VectorType orientation(3);

  cipNewtonOptimizer< 2 >::PointType* domainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* optimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  position[0] = particles->GetPoint(whichParticle)[0];
  position[1] = particles->GetPoint(whichParticle)[1];
  position[2] = particles->GetPoint(whichParticle)[2];

  orientation[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple(whichParticle)[0];
  orientation[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple(whichParticle)[1];
  orientation[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple(whichParticle)[2];

  cipParticleToThinPlateSplineSurfaceMetric particleToTPSMetric;
    particleToTPSMetric.SetThinPlateSplineSurface( tps );
    particleToTPSMetric.SetParticle( position );

  (*domainParams)[0] = position[0];
  (*domainParams)[1] = position[1];

  cipNewtonOptimizer< 2 >* newtonOptimizer = new cipNewtonOptimizer< 2 >();
    newtonOptimizer->SetMetric( particleToTPSMetric );
    newtonOptimizer->SetInitialParameters( domainParams );
    newtonOptimizer->Update();
    newtonOptimizer->GetOptimalParameters( optimalParams );

  *distance = std::sqrt( newtonOptimizer->GetOptimalValue() );

  tps.GetSurfaceNormal( (*optimalParams)[0], (*optimalParams)[1], normal );

  *angle = cip::GetAngleBetweenVectors( normal, orientation, true );
}

// 'tpsVec' has either one (left) or two (right) elements. The
// convention is that if the 'tpsVec' contains surfaces for the right
// lung, the first element of the vector corresponds to the oblique,
// and the second element corresponds to the horizontal.
void TallyParticleInfo( vtkPolyData* particles, std::vector< cipThinPlateSplineSurface > tpsVec, 
			std::map< unsigned int, PARTICLEINFO >* particleToInfoMap )
{
  if ( tpsVec.size() == 1 )
    {
    // Dealing with the left lung
    for ( unsigned int i=0; i<particles->GetNumberOfPoints(); i++ )
      {
      PARTICLEINFO pInfo;

      double distance, angle;
      GetParticleDistanceAndAngle( particles, i, tpsVec[0], &distance, &angle );

      pInfo.distance.push_back( distance );
      pInfo.angle.push_back( angle );

      (*particleToInfoMap)[i] = pInfo;
      }
    }
  else
    {
    // Dealing with the right lung
    double roSurfaceHeight, rhSurfaceHeight;
    for ( unsigned int i=0; i<particles->GetNumberOfPoints(); i++ )
      {
      PARTICLEINFO pInfo;

      roSurfaceHeight = tpsVec[0].GetSurfaceHeight( particles->GetPoint(i)[0], particles->GetPoint(i)[1] );
      rhSurfaceHeight = tpsVec[1].GetSurfaceHeight( particles->GetPoint(i)[0], particles->GetPoint(i)[1] ); 

      if ( roSurfaceHeight > rhSurfaceHeight )
        {
        double distance, angle;
        GetParticleDistanceAndAngle( particles, i, tpsVec[0], &distance, &angle );

        pInfo.distance.push_back( distance );
        pInfo.angle.push_back( angle );

        (*particleToInfoMap)[i] = pInfo;
        }
      else
        {
        double roDistance, roAngle;
        GetParticleDistanceAndAngle( particles, i, tpsVec[0], &roDistance, &roAngle );

        pInfo.distance.push_back( roDistance );
        pInfo.angle.push_back( roAngle );

        double rhDistance, rhAngle;
        GetParticleDistanceAndAngle( particles, i, tpsVec[1], &rhDistance, &rhAngle );

        pInfo.distance.push_back( rhDistance );
        pInfo.angle.push_back( rhAngle );

        (*particleToInfoMap)[i] = pInfo;
        }
      }
    }
}

void ClassifyParticles( std::map< unsigned int, PARTICLEINFO >* particleToInfoMap, 
                        double distanceWeight, double angleWeight, double fischerThreshold, double distanceThreshold )
{
  std::map< unsigned int, PARTICLEINFO >::iterator it = (*particleToInfoMap).begin();

  while ( it != (*particleToInfoMap).end() )
    {
    std::vector< double > projection;

    for ( unsigned int i=0; i<(*it).second.distance.size(); i++)
      {
      projection.push_back( distanceWeight*(*it).second.distance[i] + angleWeight*(*it).second.angle[i] );
      }

    if ( projection.size() == 1 )
      {
      if ( projection[0] > fischerThreshold && (*it).second.distance[0] < distanceThreshold )
        {
	  (*it).second.cipType = (unsigned char)( cip::OBLIQUEFISSURE );
        }
      else
        {
	  (*it).second.cipType = (unsigned char)( cip::UNDEFINEDTYPE );
        }
      }
    else
      {
      // If here, we're necessarily talking about the right lung
      if ( projection[0] >= projection[1] )
        {
        if ( projection[0] > fischerThreshold && (*it).second.distance[0] < distanceThreshold )
          {
	    (*it).second.cipType = (unsigned char)( cip::OBLIQUEFISSURE );
          }
        else
          {
	    (*it).second.cipType = (unsigned char)( cip::UNDEFINEDTYPE );
          }
        }
      else
        {
        if ( projection[1] > fischerThreshold && (*it).second.distance[1] < distanceThreshold )
          {
	    (*it).second.cipType = (unsigned char)( cip::HORIZONTALFISSURE );
          }
        else
          {
	    (*it).second.cipType = (unsigned char)( cip::UNDEFINEDTYPE );
          }
        }
      }

    ++it;
    }
}

void WriteParticlesToFile( vtkSmartPointer< vtkPolyData > particles, std::map< unsigned int, PARTICLEINFO > particleToInfoMap,
                          std::string fileName, unsigned char cipType )
{
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  vtkPoints* outputPoints  = vtkPoints::New();

  std::vector< vtkFloatArray* > arrayVec;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
    vtkFloatArray* array = vtkFloatArray::New();
      array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }

  unsigned int inc = 0;
  for ( unsigned int i=0; i<particles->GetNumberOfPoints(); i++ )
    {
    if ( particleToInfoMap[i].cipType == cipType )
      {
      outputPoints->InsertNextPoint( particles->GetPoint(i) );

      for ( unsigned int k=0; k<numberOfPointDataArrays; k++ )
        {
        arrayVec[k]->InsertTuple( inc, particles->GetPointData()->GetArray(k)->GetTuple(i) );
        }
      inc++;
      }
    }

  vtkSmartPointer< vtkPolyData > outputParticles = vtkSmartPointer< vtkPolyData >::New();

  outputParticles->SetPoints( outputPoints );
  for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
    {
    outputParticles->GetPointData()->AddArray( arrayVec[j] );
    }

  // Transfer the input field data to the output
  cip::TransferFieldData( particles, outputParticles );

  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetInputData( outputParticles );
    writer->SetFileName( fileName.c_str() );
    writer->Write();
}

void GetSurfacePointsFromLabelMap( cip::LabelMapType::Pointer labelMap,
				   std::vector< cip::PointType >* rightObliqueSurfacePoints,
				   std::vector< cip::PointType >* rightHorizontalSurfacePoints,
				   std::vector< cip::PointType >* leftObliqueSurfacePoints )
{
  cip::ChestConventions conventions;

  // These point vectors will be used to internally collect the boundary
  // points. Below, we'll only record a subset of these so as not to swamp
  // the TPS computations.
  std::vector< cip::PointType > rightObliqueSurfacePointsInternal;
  std::vector< cip::PointType > rightHorizontalSurfacePointsInternal;
  std::vector< cip::PointType > leftObliqueSurfacePointsInternal;
  
  cip::LabelMapType::IndexType index;
  cip::LabelMapType::PointType origin = labelMap->GetOrigin();
  cip::LabelMapType::SpacingType spacing = labelMap->GetSpacing();

  typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType > LabelMapIteratorType;
  LabelMapIteratorType it( labelMap, labelMap->GetBufferedRegion() );
  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
      if ( it.Get() != 0 )
	{
	  unsigned char chestRegion = conventions.GetChestRegionFromValue( it.Get() );
	  if ( chestRegion == (unsigned char)(cip::RIGHTINFERIORLOBE) )
	    {
	      index[0] = it.GetIndex()[0];
	      index[1] = it.GetIndex()[1];
	      index[2] = it.GetIndex()[2] + 1;	      
	      unsigned char chestRegionAbove = conventions.GetChestRegionFromValue( labelMap->GetPixel( index ) );
	      
	      if ( chestRegionAbove == (unsigned char)(cip::RIGHTMIDDLELOBE) |
		   chestRegionAbove == (unsigned char)(cip::RIGHTSUPERIORLOBE) | 
		   chestRegionAbove == (unsigned char)(cip::RIGHTLUNG) )
		{
		  cip::PointType point(3);
		    point[0] = index[0]*spacing[0] + origin[0];
		    point[1] = index[1]*spacing[1] + origin[1];
		    point[2] = index[2]*spacing[2] + origin[2];

		  rightObliqueSurfacePointsInternal.push_back( point );
		}
	    }
	  if ( chestRegion == (unsigned char)(cip::RIGHTMIDDLELOBE) )
	    {
	      index[0] = it.GetIndex()[0];
	      index[1] = it.GetIndex()[1];
	      index[2] = it.GetIndex()[2] + 1;

	      unsigned char chestRegionAbove = conventions.GetChestRegionFromValue( labelMap->GetPixel( index ) );
	      if ( chestRegionAbove == (unsigned char)(cip::RIGHTSUPERIORLOBE) |
		   chestRegionAbove == (unsigned char)(cip::RIGHTLUNG) )
		{
		  cip::PointType point(3);
		    point[0] = index[0]*spacing[0] + origin[0];
		    point[1] = index[1]*spacing[1] + origin[1];
		    point[2] = index[2]*spacing[2] + origin[2];

		  rightHorizontalSurfacePointsInternal.push_back( point );
		}
	    }
	  if ( chestRegion == (unsigned char)(cip::LEFTINFERIORLOBE) )
	    {
	      index[0] = it.GetIndex()[0];
	      index[1] = it.GetIndex()[1];
	      index[2] = it.GetIndex()[2] + 1;

	      unsigned char chestRegionAbove = conventions.GetChestRegionFromValue( labelMap->GetPixel( index ) );
	      if ( chestRegionAbove == (unsigned char)(cip::LEFTSUPERIORLOBE) |
		   chestRegionAbove == (unsigned char)(cip::LEFTLUNG) )
		{
		  cip::PointType point(3);
		    point[0] = index[0]*spacing[0] + origin[0];
		    point[1] = index[1]*spacing[1] + origin[1];
		    point[2] = index[2]*spacing[2] + origin[2];

		  leftObliqueSurfacePointsInternal.push_back( point );
		}
	    }	  
	}
      
      ++it;
    }

  unsigned int incr = (unsigned int)(leftObliqueSurfacePointsInternal.size()/1000);
  for ( unsigned int i=0; i < leftObliqueSurfacePointsInternal.size(); i += incr )
    {
      (*leftObliqueSurfacePoints).push_back( leftObliqueSurfacePointsInternal[i] );
    }

  incr = (unsigned int)(rightObliqueSurfacePointsInternal.size()/1000);
  for ( unsigned int i=0; i < rightObliqueSurfacePointsInternal.size(); i += incr )
    {
      (*rightObliqueSurfacePoints).push_back( rightObliqueSurfacePointsInternal[i] );
    }

  incr = (unsigned int)(rightHorizontalSurfacePointsInternal.size()/1000);
  for ( unsigned int i=0; i < rightHorizontalSurfacePointsInternal.size(); i += incr )
    {
      (*rightHorizontalSurfacePoints).push_back( rightHorizontalSurfacePointsInternal[i] );
    }
}

#endif
