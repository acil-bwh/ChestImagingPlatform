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
 *
 *  USAGE:
 *
 *  FilterVesselParticleData  [--spacing \<double\>] [-r \<double\>] 
 *                            [-a \<double\>] [-d \<double\>] 
 *                            [-m \<unsigned short\>] 
 *                            [-s \<unsigned short\>] 
 *                            -o \<string\> -i \<string\> [--] 
 *                            [--version] [-h]
 *
 *  Where:
 *
 *   --spacing \<double\>
 *     This value indicates the inter-particle spacing of the input data set
 *
 *   -r \<double\>,  --scaleRatio \<double\>
 *     Scale ratio threshold in the interval [0,1]. This value indicates the
 *     degree to which two particles can differ in scale and still be
 *     considered for connectivity. The higher the value, the more permisse
 *     the filter is with respect to scale differences.
 *
 *   -a \<double\>,  --angle \<double\>
 *     Particle angle threshold used to test the connectivity between two
 *     particles (in degrees). The vector connecting two particles is
 *     computed. The angle formed between the connecting vector and the
 *     particle Hessian eigenvector pointing in the direction of the vessel
 *     axis is then considered. For both particles, this angle must be below
 *     the specified threshold for the particles to be connected.
 *
 *   -d \<double\>,  --distance \<double\>
 *     Maximum inter-particle distance. Two particles must be at least this
 *     close together to be considered for connectivity
 *
 *   -m \<unsigned short\>,  --maxSize \<unsigned short\>
 *     Maximum component size. No component will be larger than the specified
 *     size
 *
 *   -s \<unsigned short\>,  --size \<unsigned short\>
 *     Component size cardinality threshold. Only components with this many
 *     particles or more will be retained in the output.
 *
 *   -o \<string\>,  --out \<string\>
 *     (required)  Output particles file name
 *
 *   -i \<string\>,  --in \<string\>
 *     (required)  Input particles file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 *  $Date: 2012-09-04 16:15:14 -0400 (Tue, 04 Sep 2012) $
 *  $Revision: 227 $
 *  $Author: rjosest $
 *
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
#include "FilterVesselParticleDataCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading particles ..." << std::endl;
  vtkPolyDataReader* reader = vtkPolyDataReader::New();
    reader->SetFileName( inParticlesFileName.c_str() );
    reader->Update();

  std::cout << "Filtering particles..." << std::endl;
  cipVesselParticleConnectedComponentFilter* filter = new cipVesselParticleConnectedComponentFilter();
    filter->SetInterParticleSpacing( interParticleSpacing );
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
    filteredWriter->SetInput( filter->GetOutput() );
    filteredWriter->Write();  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
