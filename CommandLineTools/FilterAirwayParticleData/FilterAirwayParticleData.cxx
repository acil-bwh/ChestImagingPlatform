/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads airway particles and filters them
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
 *  FilterAirwayParticleData  [--spacing \<double\>] [-r \<double\>] 
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
 *     considered for connectivity. The higher the value, the more permissive
 *     the filter is with respect to scale differences.
 *
 *   -a \<double\>,  --angle \<double\>
 *     Particle angle threshold used to test the connectivity between two
 *     particles (in degrees). The vector connecting two particles is
 *     computed. The angle formed between the connecting vector and the
 *     particle Hessian eigenvector pointing in the direction of the airway
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
 *  $Date: 2012-10-02 15:54:43 -0400 (Tue, 02 Oct 2012) $
 *  $Revision: 283 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipAirwayParticleConnectedComponentFilter.h"
#include "itkNumericTraits.h"
#include "cipConventions.h"
#include "vtkIndent.h"


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string inParticlesFileName  = "NA";
  std::string outParticlesFileName = "NA";

  double       interParticleSpacing   = 1.5;
  double       maxAllowableDistance   = 3.0;
  double       particleAngleThreshold = 20.0;
  double       scaleRatioThreshold    = 1.0;
  unsigned int componentSizeThreshold = 0;
  unsigned int maxComponentSize       = USHRT_MAX;

  //
  // Program and argument descriptions for user help
  //
  std::string programDesc = "This program reads airway particles and filters them \
based on connected components analysis. Particles are placed in \
the same component provided they are sufficiently close to one \
another, have scale that is sufficiently similar, and sufficiently \
define a local cylinder (i.e. they are sufficiently parallel with the \
vector connecting the two paticle spatial locations). Only \
components that have cardinality greater than or equal to that \
specified by the user will be retained in the output. Furthermore, \
the output particles will have a defined 'unmergedComponents' \
array that indicates the component label assigned to each particle.";

  std::string inParticlesFileNameDesc    = "Input particles file name";
  std::string outParticlesFileNameDesc   = "Output particles file name";
  std::string interParticleSpacingDesc   = "This value indicates the inter-particle spacing of the input data set";
  std::string maxAllowableDistanceDesc   = "Maximum inter-particle distance. Two particles must be at least this close \
together to be considered for connectivity";
  std::string particleAngleThresholdDesc = "Particle angle threshold used to test the connectivity between two particles (in degrees). \
The vector connecting two particles is computed. The angle formed between the connecting vector and the particle Hessian \
eigenvector pointing in the direction of the airway axis is then considered. For both particles, this angle must be below \
the specified threshold for the particles to be connected."; 
  std::string scaleRatioThresholdDesc    = "Scale ratio threshold in the interval [0,1]. This value indicates the degree to which \
two particles can differ in scale and still be considered for connectivity. The higher the value, the more permisse the filter is \
with respect to scale differences.";
  std::string componentSizeThresholdDesc = "Component size cardinality threshold. Only components with this many particles or more \
will be retained in the output.";
  std::string maxComponentSizeDesc       = "Maximum component size. No component will be larger than the specified size";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 283 $" );

    TCLAP::ValueArg<std::string> inParticlesFileNameArg( "i", "in", inParticlesFileNameDesc, true, inParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> outParticlesFileNameArg( "o", "out", outParticlesFileNameDesc, true, outParticlesFileName, "string", cl );
    TCLAP::ValueArg<unsigned short> componentSizeThresholdArg( "s", "size", componentSizeThresholdDesc, false, componentSizeThreshold, "unsigned short", cl );
    TCLAP::ValueArg<unsigned short> maxComponentSizedArg( "m", "maxSize", maxComponentSizeDesc, false, maxComponentSize, "unsigned short", cl );
    TCLAP::ValueArg<double> maxAllowableDistanceArg( "d", "distance", maxAllowableDistanceDesc, false, maxAllowableDistance, "double", cl );
    TCLAP::ValueArg<double> particleAngleThresholdArg( "a", "angle", particleAngleThresholdDesc, false, particleAngleThreshold, "double", cl );
    TCLAP::ValueArg<double> scaleRatioThresholdArg( "r", "scaleRatio", scaleRatioThresholdDesc, false, scaleRatioThreshold, "double", cl );
    TCLAP::ValueArg<double> interParticleSpacingArg( "", "spacing", interParticleSpacingDesc, false, interParticleSpacing, "double", cl );

    cl.parse( argc, argv );

    inParticlesFileName    = inParticlesFileNameArg.getValue();
    outParticlesFileName   = outParticlesFileNameArg.getValue();
    componentSizeThreshold = componentSizeThresholdArg.getValue();
    maxAllowableDistance   = maxAllowableDistanceArg.getValue();
    particleAngleThreshold = particleAngleThresholdArg.getValue();
    scaleRatioThreshold    = scaleRatioThresholdArg.getValue();
    maxComponentSize       = maxComponentSizedArg.getValue();
    interParticleSpacing   = interParticleSpacingArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  std::cout << "Reading particles ..." << std::endl;
  vtkPolyDataReader* reader = vtkPolyDataReader::New();
    reader->SetFileName( inParticlesFileName.c_str() );
    reader->Update();

  std::cout << "Filtering particles..." << std::endl;
  cipAirwayParticleConnectedComponentFilter* filter = new cipAirwayParticleConnectedComponentFilter();
    filter->SetInterParticleSpacing( interParticleSpacing );
    filter->SetComponentSizeThreshold( componentSizeThreshold );
    filter->SetParticleDistanceThreshold( maxAllowableDistance );
    filter->SetParticleAngleThreshold( particleAngleThreshold );
    filter->SetScaleRatioThreshold( scaleRatioThreshold );
    filter->SetMaximumComponentSize( maxComponentSize );
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
