/** \file
 *  \ingroup commandLineTools
 *  \details This program accepts two inputs: a particles dataset and
 *  another particles dataset indicating which particles should be
 *  removed from the first dataset. The output if the set of particles
 *  in the first dataset but not in the second.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "cipHelper.h"
#include "RemoveParticlesFromParticlesDataSetCLP.h"

void RemoveParticles( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inFileName.c_str() );
    particlesReader->Update();

  std::cout << "Reading particles to remove..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > removeReader = vtkSmartPointer< vtkPolyDataReader >::New();
    removeReader->SetFileName( removeFileName.c_str() );
    removeReader->Update();

  vtkSmartPointer< vtkPolyData > cleanedParticles = vtkSmartPointer< vtkPolyData >::New();

  std::cout << "Removing particles..." << std::endl;
  RemoveParticles( particlesReader->GetOutput(), removeReader->GetOutput(), cleanedParticles );

  std::cout << "Writing airway particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetInputData( cleanedParticles );
    particlesWriter->SetFileName( outFileName.c_str() );
    particlesWriter->SetFileTypeToBinary();
    particlesWriter->Write();

  std::cout << "DONE." << std::endl;

  return 0;
}

void RemoveParticles( vtkSmartPointer< vtkPolyData > particles, vtkSmartPointer< vtkPolyData > removeParticles,
                      vtkSmartPointer< vtkPolyData > cleanedParticles )
{
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

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

    for ( int j=0; j<removeParticles->GetNumberOfPoints(); j++ )
      {
      point2[0] = removeParticles->GetPoint( j )[0];
      point2[1] = removeParticles->GetPoint( j )[1];
      point2[2] = removeParticles->GetPoint( j )[2];

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
        arrayVec[k]->InsertTuple( inc, particles->GetPointData()->GetArray(k)->GetTuple(i) );
        }
      inc++;
      points->InsertNextPoint( particles->GetPoint(i) );
      }
    }

  cleanedParticles->SetPoints( points );
  points->Delete();

  for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
    {
    cleanedParticles->GetPointData()->AddArray( arrayVec[j] );
    //Array has been added and register with container class. Delete the pointer
    arrayVec[j]->Delete();
    }
  
  cip::TransferFieldData( particles, cleanedParticles );
}

#endif
