/** \file
 *  \ingroup commandLineTools
 *  \details This program can used to perturb a particles
 *  dataset. This kind of operation can be useful for
 *  experimentation purposes. Currently, the program simply
 *  translates the particles dataset by a random offset in
 *  the x, y, and z directions. The user can control the
 *  magnitude of the random offset that is used.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "PerturbParticlesCLP.h"
#include "cipChestConventions.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkFieldData.h"
#include "vtkPolyData.h"
#include "vtkIndent.h"
#include "cipHelper.h"

#ifdef WIN32
#include <ctime>
#endif

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading VTK polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
    reader->SetFileName( inFileName.c_str() );
    reader->Update();

  unsigned int numberOfPoints = reader->GetOutput()->GetNumberOfPoints();
  unsigned int numberOfFieldDataArrays = reader->GetOutput()->GetFieldData()->GetNumberOfArrays();
  unsigned int numberOfPointDataArrays = reader->GetOutput()->GetPointData()->GetNumberOfArrays();

  vtkSmartPointer< vtkPoints > outputPoints = vtkSmartPointer< vtkPoints >::New();

  srand(seed);
  double starter = rand();
  srand(starter);
  double xOffset = offsetMagnitude*(double(rand())/double(RAND_MAX) - 0.5);
  double yOffset = offsetMagnitude*(double(rand())/double(RAND_MAX) - 0.5);
  double zOffset = offsetMagnitude*(double(rand())/double(RAND_MAX) - 0.5);

  for ( unsigned int i=0; i<numberOfPoints; i++ )
    {
      double* point = new double[3];

      point[0] = reader->GetOutput()->GetPoint(i)[0] + xOffset;
      point[1] = reader->GetOutput()->GetPoint(i)[1] + yOffset;
      point[2] = reader->GetOutput()->GetPoint(i)[2] + zOffset;

      outputPoints->InsertNextPoint( point );
    }

  // Create a new polydata to contain the output
  vtkSmartPointer< vtkPolyData > outPolyData = vtkSmartPointer< vtkPolyData >::New();
    outPolyData->SetPoints( outputPoints );

  // Add the field data to the output
  cip::TransferFieldData( reader->GetOutput(), outPolyData );

  // Add the point data to the output
  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
      outPolyData->GetPointData()->AddArray( reader->GetOutput()->GetPointData()->GetArray(i) );
    }

  std::cout << "Writing VTK polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetFileName( outFileName.c_str() );
    writer->SetInputData( outPolyData );
    writer->Update();

  std::cout << "DONE." << std::endl;

  return 0;
}

#endif
