#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"
#include "cipFissureParticleConnectedComponentFilter.h"
#include "FilterFissureParticleDataCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
    reader->SetFileName( inParticlesFileName.c_str() );
    reader->Update();

  std::cout << "Filtering particles using connectedness..."  << std::endl;
  cipFissureParticleConnectedComponentFilter particleFilter;
    particleFilter.SetParticleDistanceThreshold( distanceThreshold );
    particleFilter.SetInput( reader->GetOutput() );
    particleFilter.SetMaximumComponentSize( maxComponentSize );	
    particleFilter.SetComponentSizeThreshold( componentSizeThreshold );
    particleFilter.SetParticleAngleThreshold( angleThreshold );
    particleFilter.Update();

  std::cout << "Writing filtered particles ..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > filteredWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    filteredWriter->SetFileName( outParticlesFileName.c_str() );
    filteredWriter->SetInputData( particleFilter.GetOutput() );
    filteredWriter->SetFileTypeToBinary();
    filteredWriter->Write();  
   
  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif

