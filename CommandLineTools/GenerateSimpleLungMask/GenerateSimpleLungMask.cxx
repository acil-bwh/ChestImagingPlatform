#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include <iostream>
#include <algorithm>
#include <string>

#include "vtkSmartPointer.h"
#include "vtkTimerLog.h"
#include "vtkImageData.h"
#include "GenerateSimpleLungMaskCLP.h"
#include "vtkSimpleLungMask.h"
#include "vtkImageLaplacian.h"
#include "vtkImageMedian3D.h"
#include "vtkImageResample.h"
#include "vtkMatrix4x4.h"
#include "itkImageToVTKImageFilter.h"
#include "itkVTKImageToImageFilter.h"
#include "cipHelper.h"
#include "itkCastImageFilter.h"
#include "cipChestConventions.h"

namespace
{
  typedef itk::ImageToVTKImageFilter< cip::CTType > ITKtoVTKType;
  typedef itk::VTKImageToImageFilter< cip::CTType > VTKtoITKType;
  typedef itk::CastImageFilter< cip::CTType, cip::LabelMapType > CastType;
}

int main( int argc, char * argv[] ){
  PARSE_ARGS;

  std::cout << "Reading image..." << std::endl;
  cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
    reader->SetFileName( inputVolume );
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

  ITKtoVTKType::Pointer connector = ITKtoVTKType::New();
    connector->SetInput( reader->GetOutput() );
    connector->Update();

  vtkMatrix4x4 *rasTovtk = vtkMatrix4x4::New();
  //rasTovtk->DeepCopy(connector->GetRasToIjkMatrix());
  for (int kk=0;kk<3;kk++) 
    {
      rasTovtk->SetElement(kk,1,-1.0*rasTovtk->GetElement(kk,1));
    }

  //Set some selection to downsample
  int dims[3];
  double sp[3];
  int ext[6];
  connector->GetOutput()->GetDimensions(dims);
  connector->GetOutput()->GetSpacing(sp);
  connector->GetOutput()->GetExtent(ext);
  int applyResampling=0;
  double downFactor=0.5;
  int fastMode=0;
  vtkImageData *vol;
  vtkImageResample *resampling = vtkImageResample::New();
  if (fastMode == 1 && sp[2]<2)
    {
      // Downsample in X-Y to speed things up a bit
      applyResampling=1;
      
      resampling->SetInputData(connector->GetOutput());
      resampling->SetDimensionality(3);
      resampling->SetAxisMagnificationFactor(0,downFactor);
      resampling->SetAxisMagnificationFactor(1,downFactor);
      resampling->SetAxisMagnificationFactor(2, downFactor);
      resampling->SetInterpolationModeToLinear();
      std::cout<<"Downsampling volume..."<<std::endl;
      resampling->Update();
      vol=resampling->GetOutput();
      
      rasTovtk->SetElement(0, 0, rasTovtk->GetElement(0,0)*downFactor);
      rasTovtk->SetElement(1, 1, rasTovtk->GetElement(1,1)*downFactor);
      rasTovtk->SetElement(2, 1, rasTovtk->GetElement(2,2)*downFactor);

    } 
  else 
    {
      vol=connector->GetOutput();
    }
  
  // Compute Laplacian to enhance edges
  //vtkImageLaplacian *laplacian = vtkImageLaplacian::New();
  //laplacian->SetInput(connector->GetOutput());
  //laplacian->SetDimensionality(3);
  //laplacian->Update();

  // Median Filtering to help lung extraction
  vtkImageMedian3D *filter = vtkImageMedian3D::New();
  vtkImageData *filtered;
  if ( lowDose == true) {
    std::cout << "Median filtering..." << std::endl;
      filter->SetInputData(vol);
      filter->SetKernelSize(3,3,1);
      filter->Update();
      filtered = filter->GetOutput();
  } else {
    filtered = vol;
  }

  // combine labels
  std::cout<<"Lung mask extraction..."<<std::endl;
  vtkSimpleLungMask *lungMask = vtkSimpleLungMask::New();
    lungMask->SetInputData( vol );
    lungMask->SetTracheaLabel(512);
    lungMask->SetRasToVtk(rasTovtk);
    lungMask->Update();

  // upsample
  resampling->Delete();
  resampling = vtkImageResample::New();

  if (applyResampling == 1) 
    {
      resampling->SetInputConnection(lungMask->GetOutputPort());
      resampling->SetDimensionality(3);
      for (int ii=0; ii<2;ii++)
      {
        resampling->SetAxisMagnificationFactor(ii,1/downFactor);
        //resampling->SetAxisMagnificationFactor(ii,1);

      }
      resampling->SetAxisMagnificationFactor(2,1/downFactor);
      resampling->SetInterpolationModeToNearestNeighbor();
      std::cout<<"Upsampling..."<<std::endl;
      resampling->Update();
      vol=resampling->GetOutput();
    } 
  else 
    {
      vol = lungMask->GetOutput();
    }

  VTKtoITKType::Pointer vtkToITK = VTKtoITKType::New();
    vtkToITK->SetInput( vol );
    vtkToITK->Update();

  CastType::Pointer caster = CastType::New();
    caster->SetInput( vtkToITK->GetOutput() );

  std::cout << "Writing simple lung mask..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetInput( caster->GetOutput() );
    writer->SetFileName( outputVolume );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing lung mask:";
    std::cerr << excp << std::endl;

    return cip::NRRDWRITEFAILURE;
    }

  //Delete everything
  rasTovtk->Delete();
  filter->Delete();
  resampling->Delete();
  lungMask->Delete();

  return EXIT_SUCCESS;
}
