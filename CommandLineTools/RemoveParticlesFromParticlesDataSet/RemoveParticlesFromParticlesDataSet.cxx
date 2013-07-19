/** \file
 *  \ingroup commandLineTools 
 *  \details This program accepts two inputs: a particles dataset and
 *  another particles dataset indicating which particles should be
 *  removed from the first dataset. The output if the set of particles
 *  in the first dataset but not in the second.
 *
 *  Usage: RemoveParticlesFromParticlesDataSet \<options\> where \<options\> is one or more of the following:\n
 *    \<-h\>     Display (this) usage information\n
 *    \<-i\>     Input particle data set file name. 
 *    \<-r\>     Particle data set file name indicating which
 *               particles should be removed from the data set
 *               specified by the -i flag.
 *    \<-o\>     Output particle data set file name. The particles in
 *               the output will consist of the particles in the data
 *               set specified with the -i flag minus the particles in
 *               the data set specified with the -r flag.
 */


//
// TODO
//
// Commenting in doxygen style needs to be shaped up
//

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"


void RemoveParticles( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );


void usage()
{
  std::cerr << "\n";
  std::cerr << "This program accepts two inputs: a particles dataset and another particles dataset\n";
  std::cerr << "indicating which particles should be removed from the first dataset. The output if\n";
  std::cerr << "the set of particles in the first dataset but not in the second.\n";
  std::cerr << std::endl;
  std::cerr << "Usage: RemoveParticlesFromParticlesDataSet <options> where <options> is one or more " << std::endl;
  std::cerr << "of the following:\n\n";
  std::cerr << "   <-h>     Display (this) usage information\n";
  std::cerr << "   <-i>     Input particle data set file name. \n";
  std::cerr << "   <-r>     Particle data set file name indicating which particles should be removed from\n";
  std::cerr << "            the data set specified by the -i flag.\n";
  std::cerr << "   <-o>     Output particle data set file name. The particles in the output will consist\n";
  std::cerr << "            of the particles in the data set specified with the -i flag minus the particles in\n";
  std::cerr << "            the data set specified with the -r flag.Output particle data set file name\n";

  exit(1);
}


int main( int argc, char *argv[] )
{
  bool ok;

  char* inFileName     = new char[512];   strcpy( inFileName, "q" );
  char* removeFileName = new char[512];   strcpy( removeFileName, "q" );
  char* outFileName    = new char[512];   strcpy( outFileName, "q" );

  while ( argc > 1 )
    {
    ok = false;

    if ((ok == false) && (strcmp(argv[1], "-h") == 0))
      {
      argc--; argv++;
      ok = true;
      usage();      
      }

    if ((ok == false) && (strcmp(argv[1], "-r") == 0))
      {
      argc--; argv++;
      ok = true;

      removeFileName = argv[1];      

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-i") == 0))
      {
      argc--; argv++;
      ok = true;

      inFileName = argv[1];      

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-o") == 0))
      {
      argc--; argv++;
      ok = true;
      
      outFileName = argv[1];

      argc--; argv++;
      }
    }

  std::cout << "Reading particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inFileName );
    particlesReader->Update();

  std::cout << "Reading particles to remove..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > removeReader = vtkSmartPointer< vtkPolyDataReader >::New();
  removeReader->SetFileName( removeFileName );
  removeReader->Update();

  vtkSmartPointer< vtkPolyData > cleanedParticles = vtkSmartPointer< vtkPolyData >::New();

  std::cout << "Removing particles..." << std::endl;
  RemoveParticles( particlesReader->GetOutput(), removeReader->GetOutput(), cleanedParticles );

  std::cout << "Writing airway particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetInput( cleanedParticles );
    particlesWriter->SetFileName( outFileName );
    particlesWriter->Write();

  std::cout << "DONE." << std::endl;

  return 0;
}


void RemoveParticles( vtkSmartPointer< vtkPolyData > particles, vtkSmartPointer< vtkPolyData > removeParticles,
                      vtkSmartPointer< vtkPolyData > cleanedParticles )
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
  for ( unsigned int j=0; j<numberOfPointDataArrays; j++ )
    {
    cleanedParticles->GetPointData()->AddArray( arrayVec[j] );
    }
}

#endif
