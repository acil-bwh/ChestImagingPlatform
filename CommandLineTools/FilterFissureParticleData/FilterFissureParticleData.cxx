/** \file
 *  \ingroup commandLineTools 
 *  \details This program is used to filter fissure particles. In particular
 *  it is used as step in the overall lung lobe segmentation framework. 
 *  Default values for the input parameters are chosen for that purpose.
 *  The program provides an interface to the 
 *  'cipFissureParticleConnectedComponentFilter': connected components
 *  concepts are used to eliminate small particle groups and to retain
 *  those that form larger, sheet-like structures.
 *
 *  USAGE: 
 *
 *  FilterFissureParticleData  [-s \<unsigned int\>] [-a \<double\>] 
 *                             [-p \<double\>] [-d \<double\>] -o \<string\> 
 *                             -i \<string\> [--] [--version] [-h]
 *
 *  Where: 
 *
 *   -s \<unsigned int\>,  --size \<unsigned int\>
 *     The minimum cardinality of a set of component particles needed for
 *     that set to be passed through the filter
 *
 *   -a \<double\>,  --angleThresh \<double\>
 *     Particle angle threshold (degrees). The vector connecting two
 *     particles is compared to their respective orientation vectors
 *     (indicating the approximate normal vector to the local sheet they
 *     potentially lie on). If the angle between either of these vectors and
 *     the connecting vector is less than this threshold, the particles are
 *     considered to be disconnected
 *
 *   -p \<double\>,  --spacing \<double\>
 *     The inter-particle spacing that was used to generate the input
 *     particles
 *
 *   -d \<double\>,  --distanceThresh \<double\>
 *     Particle distance threshold (mm). A pair of particles must be at least
 *     this close together to be considered for connectivity.
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
 *
 *  $Date: 2012-08-28 18:56:56 -0400 (Tue, 28 Aug 2012) $
 *  $Revision: 218 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipFissureParticleConnectedComponentFilter.h"

int main( int argc, char *argv[] )
{
  //
  // Define arguments
  //
  std::string  inParticlesFileName    = "NA";
  std::string  outParticlesFileName   = "NA";
  double       distanceThreshold      = 3.0;
  double       interParticleSpacing   = 1.7;
  double       angleThreshold         = 70;
  unsigned int componentSizeThreshold = 110;

  //
  // Program and argument descriptions for user help
  //
  std::string programDesc = "This program is used to filter fissure particles. In particular \
it is used as step in the overall lung lobe segmentation framework. \
Default values for the input parameters are chosen for that purpose. \
The program provides an interface to the \
'cipFissureParticleConnectedComponentFilter': connected components \
concepts are used to eliminate small particle groups and to retain \
those that form larger, sheet-like structures.";

  std::string inParticlesFileNameDesc    = "Input particles file name";
  std::string outParticlesFileNameDesc   = "Output particles file name";
  std::string distanceThresholdDesc      = "Particle distance threshold (mm). A pair of particles must \
be at least this close together to be considered for connectivity.";
  std::string angleThresholdDesc         = "Particle angle threshold (degrees). The vector connecting \
two particles is compared to their respective orientation vectors (indicating the approximate normal \
vector to the local sheet they potentially lie on). If the angle between either of these vectors and \
the connecting vector is less than this threshold, the particles are considered to be disconnected";
  std::string interParticleSpacingDesc   = "The inter-particle spacing that was used to generate the input \
particles";
  std::string componentSizeThresholdDesc = "The minimum cardinality of a set of component particles needed \
for that set to be passed through the filter";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 218 $" );

    TCLAP::ValueArg<std::string>   inParticlesFileNameArg( "i", "in", inParticlesFileNameDesc, true, inParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string>   outParticlesFileNameArg( "o", "out", outParticlesFileNameDesc, true, outParticlesFileName, "string", cl );
    TCLAP::ValueArg<double>        distanceThresholdArg( "d", "distanceThresh", distanceThresholdDesc, false, distanceThreshold, "double", cl );
    TCLAP::ValueArg<double>        interParticleSpacingArg( "p", "spacing", interParticleSpacingDesc, false, interParticleSpacing, "double", cl );
    TCLAP::ValueArg<double>        angleThresholdArg( "a", "angleThresh", angleThresholdDesc, false, angleThreshold, "double", cl );
    TCLAP::ValueArg<unsigned int>  componentSizeThresholdArg( "s", "size", componentSizeThresholdDesc, false, componentSizeThreshold, "unsigned int", cl );

    cl.parse( argc, argv );

    inParticlesFileName    = inParticlesFileNameArg.getValue();
    outParticlesFileName   = outParticlesFileNameArg.getValue();
    distanceThreshold      = distanceThresholdArg.getValue();
    interParticleSpacing   = interParticleSpacingArg.getValue();
    angleThreshold         = angleThresholdArg.getValue();
    componentSizeThreshold = componentSizeThresholdArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  std::cout << "Reading polydata..." << std::endl;
  vtkPolyDataReader* reader = vtkPolyDataReader::New();
    reader->SetFileName( inParticlesFileName.c_str() );
    reader->Update();    

  std::cout << "Filtering particles using connectedness..."  << std::endl;
  cipFissureParticleConnectedComponentFilter particleFilter;
    particleFilter.SetInterParticleSpacing( interParticleSpacing );
    particleFilter.SetParticleDistanceThreshold( distanceThreshold );
    particleFilter.SetInput( reader->GetOutput() );
    particleFilter.SetComponentSizeThreshold( componentSizeThreshold );
    particleFilter.SetParticleAngleThreshold( angleThreshold );
    particleFilter.Update();

  std::cout << "Writing filtered particles ..." << std::endl;
  vtkPolyDataWriter *filteredWriter = vtkPolyDataWriter::New();
    filteredWriter->SetFileName( outParticlesFileName.c_str() );
    filteredWriter->SetInput( particleFilter.GetOutput() );
    filteredWriter->Write();  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif


