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
void ClassifyParticles( std::map< unsigned int, PARTICLEINFO >*, std::vector< cipThinPlateSplineSurface >, double, double, double );
void WriteParticlesToFile( vtkSmartPointer< vtkPolyData >, std::map< unsigned int, PARTICLEINFO >, std::string, unsigned char );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Read complete particle in lung
  std::cout << "Reading lung particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( particlesFileName.c_str() );
    particlesReader->Update();

  // Read shape models
  std::vector< cipThinPlateSplineSurface > tpsVec;

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

    cipThinPlateSplineSurface roTPS;
      roTPS.SetSurfacePoints( rightShapeModelIO->GetOutput()->GetRightObliqueWeightedSurfacePoints() );

    // Note ordering is important here. The RO needs to be pushed back
    // before the RH (assumed when we execute 'TallyParticleInfo')
    tpsVec.push_back( roTPS );

    cipThinPlateSplineSurface rhTPS;
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

    cipThinPlateSplineSurface loTPS;
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
  ClassifyParticles( &particleToInfoMap, tpsVec, distanceWeight, angleWeight, threshold );

  // Write the classified (fissure) particles to file
  if ( loClassifiedFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing left oblique particles to file..." << std::endl;
    WriteParticlesToFile( particlesReader->GetOutput(), particleToInfoMap, loClassifiedFileName, static_cast< unsigned char >( cip::OBLIQUEFISSURE ) );
    }
  if ( roClassifiedFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing right oblique particles to file..." << std::endl;
    WriteParticlesToFile( particlesReader->GetOutput(), particleToInfoMap, roClassifiedFileName, static_cast< unsigned char >( cip::OBLIQUEFISSURE ) );
    }
  if ( rhClassifiedFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing right horizontal particles to file..." << std::endl;
    WriteParticlesToFile( particlesReader->GetOutput(), particleToInfoMap, rhClassifiedFileName, static_cast< unsigned char >( cip::HORIZONTALFISSURE ) );
    }

  std::cout << "DONE." << std::endl;

  return 0;
}

void GetParticleDistanceAndAngle( vtkPolyData* particles, unsigned int whichParticle, const cipThinPlateSplineSurface& tps,
                                  double* distance, double* angle ) 
{
  cipParticleToThinPlateSplineSurfaceMetric particleToTPSMetric;
    particleToTPSMetric.SetThinPlateSplineSurface( tps );

  cipNewtonOptimizer< 2 >* newtonOptimizer = new cipNewtonOptimizer< 2 >();
    newtonOptimizer->SetMetric( particleToTPSMetric );

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

  particleToTPSMetric.SetParticle( position );

  (*domainParams)[0] = position[0];
  (*domainParams)[1] = position[1];

  newtonOptimizer->SetInitialParameters( domainParams );
  newtonOptimizer->Update();
  newtonOptimizer->GetOptimalParameters( optimalParams );

  *distance = vcl_sqrt( newtonOptimizer->GetOptimalValue() );

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

void ClassifyParticles( std::map< unsigned int, PARTICLEINFO >* particleToInfoMap, std::vector< cipThinPlateSplineSurface > tpsVec, 
                        double distanceWeight, double angleWeight, double threshold )
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
	std::cout << projection[0] << std::endl;
      if ( projection[0] > threshold )
        {
	  (*it).second.cipType = static_cast< unsigned char >( cip::OBLIQUEFISSURE );
	  std::cout << "lo" << std::endl;
        }
      else
        {
	  (*it).second.cipType = static_cast< unsigned char >( cip::UNDEFINEDTYPE );
	  std::cout << "left undefined" << std::endl;
        }
      }
    else
      {
      // If here, we're necessarily talking about the right lung
      if ( projection[0] >= projection[1] )
        {
        if ( projection[0] > threshold )
          {
	    (*it).second.cipType = static_cast< unsigned char >( cip::OBLIQUEFISSURE );
          }
        else
          {
	    (*it).second.cipType = static_cast< unsigned char >( cip::UNDEFINEDTYPE );
          }
        }
      else
        {
        if ( projection[1] > threshold )
          {
	    (*it).second.cipType = static_cast< unsigned char >( cip::HORIZONTALFISSURE );
          }
        else
          {
	    (*it).second.cipType = static_cast< unsigned char >( cip::UNDEFINEDTYPE );
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

  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetInputData( outputParticles );
    writer->SetFileName( fileName.c_str() );
    writer->Write();
}

#endif
