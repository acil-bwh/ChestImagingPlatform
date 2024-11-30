#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipAirwayParticleConnectedComponentFilter.h"
#include "itkNumericTraits.h"
#include "cipChestConventions.h"
#include "vtkIndent.h"
#include "vtkSmartPointer.h"
#include "FilterAirwayParticleDataCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  unsigned int maxComponentSize       = (unsigned int) maxComponentSizeTemp;
  unsigned int componentSizeThreshold = (unsigned int) componentSizeThresholdTemp;

  std::cout << "Reading particles ..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > reader = vtkSmartPointer< vtkPolyDataReader >::New();
    reader->SetFileName( inParticlesFileName.c_str() );
    reader->Update();

  std::cout << "Filtering particles..." << std::endl;
  cipAirwayParticleConnectedComponentFilter filter;
    filter.SetComponentSizeThreshold( componentSizeThreshold );
    filter.SetParticleDistanceThreshold( maxAllowableDistance );
    filter.SetParticleAngleThreshold( particleAngleThreshold );
    filter.SetScaleRatioThreshold( scaleRatioThreshold );
    filter.SetMaximumComponentSize( maxComponentSize );
    filter.SetMaximumAllowableScale( maxAllowableScale );
    filter.SetMinimumAllowableScale( minAllowableScale );
    filter.SetInput( reader->GetOutput() );
    filter.Update();

  std::cout << "Writing filtered particles ..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > filteredWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    filteredWriter->SetFileName( outParticlesFileName.c_str() );
    filteredWriter->SetInputData( filter.GetOutput() );
    filteredWriter->SetFileTypeToBinary();
    filteredWriter->Write();
    
  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

