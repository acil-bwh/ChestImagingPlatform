#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkImageData.h"
#include "vtkPolyDataNormals.h"
#include "vtkDecimatePro.h"
#include "vtkDiscreteMarchingCubes.h"
#include "vtkWindowedSincPolyDataFilter.h"
#include "vtkTransform.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkImageFlip.h"
#include "vtkPolyDataWriter.h"
#include "vtkPolyData.h"
#include "vtkImageImport.h"
#include "vtkSmartPointer.h"
#include "vtkImageIterator.h"
#include "itkImageToVTKImageFilter.h"

#include "GenerateModelCLP.h"

namespace
{
  typedef itk::ImageToVTKImageFilter< cip::LabelMapType > ConnectorType;
}

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  unsigned int smootherIterations = (unsigned int) smootherIterationsTemp;

  std::cout << "Reading mask..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName( maskFileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading mask:";
    std::cerr << excp << std::endl;

    return cip::NRRDREADFAILURE;
    }

  ConnectorType::Pointer connector = ConnectorType::New();
    connector->SetInput( reader->GetOutput() );
    connector->Update();

  vtkImageIterator< unsigned short > it( connector->GetOutput(), connector->GetOutput()->GetExtent() );

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

  it.Initialize( connector->GetOutput(), connector->GetOutput()->GetExtent() );
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

  // Correct origing to account for the fact that vtk is y-flip
  vtkSmartPointer< vtkImageFlip > flip = vtkSmartPointer < vtkImageFlip >::New();
  flip->SetInputData( connector->GetOutput() );
    flip->SetFilteredAxis(1);
    flip->Update();
  
  double *orig;
  orig=flip->GetOutput()->GetOrigin();
  std::cout<<orig[0]<<" "<<orig[1]<<" "<<orig[2]<<std::endl;
  
  // Perform marching cubes on the reference binary image and then
  // decimate
  
  
  std::cout << "Running marching cubes..." << std::endl;
  vtkSmartPointer< vtkDiscreteMarchingCubes > cubes = vtkSmartPointer< vtkDiscreteMarchingCubes >::New();
    cubes->SetInputData( flip->GetOutput() );
    cubes->SetValue( 0, 1 );
    cubes->ComputeNormalsOff();
    cubes->ComputeScalarsOff();
    cubes->ComputeGradientsOff();
    cubes->Update();
  
  //Transform points to correction origing and y-flip that is intrinsic to the VTK object
  //VTK Image object assumes that the origin is in the bottom top
  vtkSmartPointer< vtkTransform > transform  = vtkSmartPointer < vtkTransform >::New();
  transform->Identity();
  transform->GetMatrix()->SetElement(1,1,-1);
  transform->GetMatrix()->SetElement(2,2,-1);
  
  vtkSmartPointer< vtkTransformPolyDataFilter> tt_filter = vtkSmartPointer < vtkTransformPolyDataFilter >::New();
  tt_filter->SetInputConnection(cubes->GetOutputPort());
  tt_filter->SetTransform(transform);
  

  std::cout << "Smoothing model..." << std::endl;
  vtkSmartPointer< vtkWindowedSincPolyDataFilter > smoother = vtkSmartPointer< vtkWindowedSincPolyDataFilter >::New();
    smoother->SetInputConnection( tt_filter->GetOutputPort() );
    smoother->SetNumberOfIterations( smootherIterations );
    smoother->BoundarySmoothingOff();
    smoother->FeatureEdgeSmoothingOff();
    smoother->SetPassBand( 0.01 );
    smoother->SetFeatureAngle( 120.0 );
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
