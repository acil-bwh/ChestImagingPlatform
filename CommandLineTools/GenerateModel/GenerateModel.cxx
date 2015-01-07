/** \file
 *  \ingroup commandLineTools
 *  \details This program will generate a 3D model from an input
 *  3D label map using the discrete marching cubes algorithm
 *
 *  USAGE:
 *
 *  GenerateModel  [-r \<float\>] [--origSp \<bool\>] [-l \<unsigned short\>]
 *                 [-s \<unsigned int\>] -o \<string\> -i \<string\> [--]
 *                 [--version] [-h]
 *
 *  Where:
 *
 *   -r \<float\>,  --reduc \<float\>
 *     Target reduction fraction for decimation
 *
 *   --origSp \<bool\>
 *     Set to 1 to used standard origin and spacing. Set to 0 by default.
 *
 *   -l \<unsigned short\>,  --label \<unsigned short\>
 *     Foreground label in the label map to be used for generating the model
 *
 *   -s \<unsigned int\>,  --smooth \<unsigned int\>
 *     Number of smoothing iterations
 *
 *   -o \<string\>,  --out \<string\>
 *     (required)  Output model file name
 *
 *   -i \<string\>,  --in \<string\>
 *     (required)  Input mask file name
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
 */

#include "cipChestConventions.h"
#include "vtkImageData.h"
#include "vtkPolyDataNormals.h"
#include "vtkDecimatePro.h"
#include "vtkDiscreteMarchingCubes.h"
#include "vtkWindowedSincPolyDataFilter.h"
#include "vtkPolyDataWriter.h"
#include "vtkPolyData.h"
#include "vtkImageImport.h"
#include "vtkSmartPointer.h"
#include "vtkNRRDReader.h"
#include "vtkNRRDWriter.h"
#include "vtkImageIterator.h"

#include "GenerateModelCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  unsigned int smootherIterations = (unsigned int) smootherIterationsTemp;

  std::cout << "Reading mask..." << std::endl;
  vtkSmartPointer< vtkNRRDReader > maskReader = vtkSmartPointer< vtkNRRDReader >::New();
    maskReader->SetFileName( maskFileName.c_str() );
    maskReader->Update();

  vtkImageIterator< unsigned short > it( maskReader->GetOutput(), maskReader->GetOutput()->GetExtent() );

  // If the user has not specified a foreground label, find the first
  // non-zero value and use that as the foreground value
  if ( foregroundLabel == -1 )
    {
      while( !it.IsAtEnd() )
	{
	  unsigned short* valIt = it.BeginSpan();
	  unsigned short* valEnd = it.EndSpan();
	  while ( valIt != valEnd )
	    {
	      if ( *valIt != 0 )
		{
		  foregroundLabel = int(*valIt);
		  break;
		}
	      ++valIt;
	    }

	  it.NextSpan();
	}
    }

  it.Initialize( maskReader->GetOutput(), maskReader->GetOutput()->GetExtent() );
  while ( !it.IsAtEnd() )
    {
      unsigned short* valIt = it.BeginSpan();
      unsigned short* valEnd = it.EndSpan();
      while ( valIt != valEnd )
	{
	  if ( int(*valIt) == foregroundLabel )
	    {
	      *valIt = 1;
	    }
	  else
	    {
	      *valIt = 0;
	    }
	  ++valIt;
	}
      
      it.NextSpan();
    }

  // Perform marching cubes on the reference binary image and then
  // decimate
  std::cout << "Running marching cubes..." << std::endl;
  vtkSmartPointer< vtkDiscreteMarchingCubes > cubes = vtkSmartPointer< vtkDiscreteMarchingCubes >::New();
    cubes->SetInputData( maskReader->GetOutput() );
    cubes->SetValue( 0, 1 );
    cubes->ComputeNormalsOff();
    cubes->ComputeScalarsOff();
    cubes->ComputeGradientsOff();
    cubes->Update();

  std::cout << "Smoothing model..." << std::endl;
  vtkSmartPointer< vtkWindowedSincPolyDataFilter > smoother = vtkSmartPointer< vtkWindowedSincPolyDataFilter >::New();
    smoother->SetInputConnection( cubes->GetOutputPort() );
    smoother->SetNumberOfIterations( smootherIterations );
    smoother->BoundarySmoothingOff();
    smoother->FeatureEdgeSmoothingOff();
    smoother->SetPassBand( 0.001 );
    smoother->NonManifoldSmoothingOn();
    smoother->NormalizeCoordinatesOn();
    smoother->Update();

  std::cout << "Decimating model..." << std::endl;
  vtkSmartPointer< vtkDecimatePro > decimator = vtkSmartPointer< vtkDecimatePro >::New();
    decimator->SetInputConnection( smoother->GetOutputPort() );
    decimator->SetTargetReduction( decimatorTargetReduction );
    decimator->PreserveTopologyOn();
    decimator->BoundaryVertexDeletionOff();
    decimator->Update();

  vtkSmartPointer< vtkPolyDataNormals > normals = vtkSmartPointer< vtkPolyDataNormals >::New();
    normals->SetInputConnection( decimator->GetOutputPort() );
    normals->SetFeatureAngle( 90 );
    normals->Update();

  std::cout << "Writing model..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > modelWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    modelWriter->SetFileName( outputModelFileName.c_str() );
    modelWriter->SetInputConnection( normals->GetOutputPort() );
    modelWriter->Write();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}
