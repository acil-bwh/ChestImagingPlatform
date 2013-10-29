/** \file
 *  \ingroup commandLineTools 
 *  \details This program accepts as input multiple particle 
 *  datasets and merges them into one dataset for output. If 
 *  any of the inputs do not have ChestType and ChestRegion 
 *  fields defined this program will create them and initialize 
 *  entries to UNDEFINEDTYPE and UNDEFINEDREGION.
 *
 *  USAGE: 
 *
 *  MergeParticleDataSets  -i \<string\> ...  -o \<string\> [--] 
 *                        [--version]  [-h]
 *
 *  Where: 
 *
 *  -i \<string\>,  --in \<string\>  (accepted multiple times)
 *    (required)  Input particles file name
 *
 *  -o \<string\>,  --out \<string\>
 *    (required)  Output particles file name
 *
 *  --,  --ignore_rest
 *    Ignores the rest of the labeled arguments following this flag.
 *
 *  --version
 *    Displays version information and exits.
 *
 *  -h,  --help
 *    Displays usage information and exits.
 *
 *  $Date: 2012-10-23 10:46:23 -0400 (Tue, 23 Oct 2012) $
 *  $Author: jross $
 *  $Revision: 302 $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipConventions.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "MergeParticleDataSetsCLP.h"

void MergeParticles( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );
void CopyParticles( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );
void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > );


int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  vtkSmartPointer< vtkPolyData > mergedParticles = vtkSmartPointer< vtkPolyData >::New();

  for ( unsigned int i=0; i<inFileNamesVec.size(); i++ )
    {
    std::cout << "Reading particles..." << std::endl;
    vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
      particlesReader->SetFileName( inFileNamesVec[i].c_str() );
      particlesReader->Update();

    if ( i==0 )
      {
      std::cout << "Copying..." << std::endl;
      CopyParticles( particlesReader->GetOutput(), mergedParticles );
      }
    else
      {
      std::cout << "Merging..." << std::endl;
      MergeParticles( particlesReader->GetOutput(), mergedParticles );
      }
    }

  std::cout << "Writing merged particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetInput( mergedParticles );
    particlesWriter->SetFileName( outFileName.c_str() );
    particlesWriter->Write();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


void MergeParticles( vtkSmartPointer< vtkPolyData > particles, vtkSmartPointer< vtkPolyData > mergedParticles )
{
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();;

  vtkPoints* points = vtkPoints::New();

  std::vector< vtkFloatArray* > arrayVec;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
    vtkFloatArray* array = vtkFloatArray::New();
      array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }

  double point1[3];
  double point2[3];

  bool addPoint;

  unsigned int inc = 0;

  for ( int i=0; i<particles->GetNumberOfPoints(); i++ )
    {   
    addPoint = true;

    point1[0] = particles->GetPoint( i )[0];
    point1[1] = particles->GetPoint( i )[1];
    point1[2] = particles->GetPoint( i )[2];

    for ( int j=0; j<mergedParticles->GetNumberOfPoints(); j++ )
      {
      point2[0] = mergedParticles->GetPoint( j )[0];
      point2[1] = mergedParticles->GetPoint( j )[1];
      point2[2] = mergedParticles->GetPoint( j )[2];

      if ( point1[0] == point2[0] && point1[1] == point2[1] && point1[2] == point2[2] )
        {
        addPoint = false;
        break;
        }
      }
    if ( addPoint )
      {
      for ( unsigned int k=0; k<numberOfPointDataArrays; k++ )
        {
        mergedParticles->GetPointData()->GetArray(k)->InsertNextTuple( particles->GetPointData()->GetArray(k)->GetTuple(i) );
        }
      mergedParticles->GetPoints()->InsertNextPoint( point1 );
      }
    }
}


void CopyParticles( vtkSmartPointer< vtkPolyData > particles, vtkSmartPointer< vtkPolyData > mergedParticles )
{
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();;

  vtkPoints* points = vtkPoints::New();

  std::vector< vtkFloatArray* > arrayVec;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
    vtkFloatArray* array = vtkFloatArray::New();
      array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }

  vtkFloatArray* chestRegionArray = vtkFloatArray::New();
    chestRegionArray->SetNumberOfComponents( 1 );
    chestRegionArray->SetName( "ChestRegion" );

  vtkFloatArray* chestTypeArray = vtkFloatArray::New();
    chestTypeArray->SetNumberOfComponents( 1 );
    chestTypeArray->SetName( "ChestType" );

  unsigned int inc = 0;

  for ( int i=0; i<particles->GetNumberOfPoints(); i++ )
    {   
    for ( unsigned int k=0; k<numberOfPointDataArrays; k++ )
      {
      arrayVec[k]->InsertTuple( inc, particles->GetPointData()->GetArray(k)->GetTuple(i) );
      }
    inc++;
    points->InsertNextPoint( particles->GetPoint(i) );
    }

  mergedParticles->SetPoints( points );
  for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
    {
    mergedParticles->GetPointData()->AddArray( arrayVec[j] );
    }
}


//
// This function is used to verify that the input particles have
// 'ChestRegion' and 'ChestType' arrays. If the particles don't have
// these arrays, they are assigned with default entries
// 'UNDEFINEDREGION' and 'UNDEFINEDTYPE'
//
void AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > particles )
{
  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  bool foundChestRegionArray = false;
  bool foundChestTypeArray   = false;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
    std::string name( particles->GetPointData()->GetArray(i)->GetName() );

    if ( name.compare( "ChestRegion" ) == 0 )
      {
      foundChestRegionArray = true;
      }
    if ( name.compare( "ChestType" ) == 0 )
      {
      foundChestTypeArray = true;
      }
    }  

  if ( !foundChestRegionArray )
    {
    vtkSmartPointer< vtkFloatArray > chestRegionArray = vtkSmartPointer< vtkFloatArray >::New();
      chestRegionArray->SetNumberOfComponents( 1 );
      chestRegionArray->SetName( "ChestRegion" );

    particles->GetPointData()->AddArray( chestRegionArray );
    }
  if ( !foundChestTypeArray )
    {
    vtkSmartPointer< vtkFloatArray > chestTypeArray = vtkSmartPointer< vtkFloatArray >::New();
      chestTypeArray->SetNumberOfComponents( 1 );
      chestTypeArray->SetName( "ChestType" );

    particles->GetPointData()->AddArray( chestTypeArray );
    }

  float cipRegion = static_cast< float >( cip::UNDEFINEDREGION );
  float cipType   = static_cast< float >( cip::UNDEFINEDTYPE );
  if ( !foundChestRegionArray || !foundChestTypeArray )
    {
    for ( unsigned int i=0; i<numberParticles; i++ )
      {
      if ( !foundChestRegionArray )
        {
        particles->GetPointData()->GetArray( "ChestRegion" )->InsertTuple( i, &cipRegion );
        }
      if ( !foundChestTypeArray )
        {
        particles->GetPointData()->GetArray( "ChestType" )->InsertTuple( i, &cipType );
        }
      }
    }
}

#endif
