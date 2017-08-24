/** \file
 *  \ingroup commandLineTools
 *  \details This program reads vessels particles and filters them
 *  based on connected components analysis. Particles are placed in
 *  the same component provided they are sufficiently close to one
 *  another, have scale that is sufficiently similar, and sufficiently
 *  define a local cylinder (i.e. they are sufficiently parallel with the
 *  vector connecting the two paticle spatial locations). Only
 *  components that have cardinality greater than or equal to that
 *  specified by the user will be retained in the output. Furthermore,
 *  the output particles will have a defined "unmergedComponents"
 *  array that indicates the component label assigned to each particle.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipVesselParticleConnectedComponentFilter.h"
#include "itkNumericTraits.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "FilterVesselParticleDataCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading particles ..." << std::endl;
  vtkPolyDataReader* reader = vtkPolyDataReader::New();
    reader->SetFileName( inParticlesFileName.c_str() );
    reader->Update();

  std::cout << "Asserting chest-region chest-type existence..." << std::endl;
  cip::AssertChestRegionChestTypeArrayExistence( reader->GetOutput() );

  std::cout << "Filtering particles..." << std::endl;
  cipVesselParticleConnectedComponentFilter* filter = new cipVesselParticleConnectedComponentFilter();
    filter->SetComponentSizeThreshold( componentSizeThreshold );
    filter->SetParticleDistanceThreshold( maxAllowableDistance );
    filter->SetParticleAngleThreshold( particleAngleThreshold );
    filter->SetScaleRatioThreshold( scaleRatioThreshold );
    filter->SetMaximumComponentSize( maxComponentSize );
    filter->SetMaximumAllowableScale( maxAllowableScale );
    filter->SetMinimumAllowableScale( minAllowableScale );
    filter->SetInput( reader->GetOutput() );
    filter->Update();

  std::cout << "Writing filtered particles ..." << std::endl;
  vtkPolyDataWriter *filteredWriter = vtkPolyDataWriter::New();
    filteredWriter->SetFileName( outParticlesFileName.c_str() );
    filteredWriter->SetInputData( filter->GetOutput() );
    filteredWriter->SetFileTypeToBinary();
    filteredWriter->Write();

  reader->Delete();
  delete filter;
  filteredWriter->Delete();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
