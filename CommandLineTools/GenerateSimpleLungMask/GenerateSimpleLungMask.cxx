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

#include "vtkNRRDReaderCIP.h"
#include "vtkNRRDWriterCIP.h"
#include "vtkImageData.h"
#include "GenerateSimpleLungMaskCLP.h"

#include "vtkSimpleLungMask.h"
#include "vtkImageLaplacian.h"
#include "vtkImageMedian3D.h"
#include "vtkImageResample.h"
#include "vtkMatrix4x4.h"

int main( int argc, char * argv[] ){
  PARSE_ARGS;

  // Read in volume inputs
  vtkNRRDReaderCIP *readerA = vtkNRRDReaderCIP::New();
  readerA->SetFileName(inputVolume.c_str());
  readerA->Update();

  vtkMatrix4x4 *rasTovtk = vtkMatrix4x4::New();
  rasTovtk->DeepCopy(readerA->GetRasToIjkMatrix());
  for (int kk=0;kk<3;kk++) {
    rasTovtk->SetElement(kk,1,-1.0*rasTovtk->GetElement(kk,1));
  }

  for (int rr=0;rr<4;rr++) {
    for (int cc=0;cc<4;cc++) {
      std::cout<<readerA->GetRasToIjkMatrix()->GetElement(rr,cc)<<" ";
    }
    std::cout<<std::endl;
  }

  //Set some selection to downsample
  int dims[3];
  double sp[3];
  int ext[6];
  readerA->GetOutput()->GetDimensions(dims);
  readerA->GetOutput()->GetSpacing(sp);
  readerA->GetOutput()->GetExtent(ext);
  int applyResampling=0;
  double downFactor=1;
  vtkImageData *vol;
  vtkImageResample *resampling=vtkImageResample::New();
  if (sp[2]<2) {
    // Downsample in X-Y to speed things up a bit
    applyResampling=1;

    resampling->SetInputConnection(readerA->GetOutputPort());
    resampling->SetDimensionality(3);
    resampling->SetAxisMagnificationFactor(0,downFactor);
    resampling->SetAxisMagnificationFactor(1,downFactor);
    resampling->SetAxisMagnificationFactor(2,1);
    resampling->SetInterpolationModeToLinear();
    std::cout<<"Downsampling volume..."<<std::endl;
    resampling->Update();
    vol=resampling->GetOutput();

    rasTovtk->SetElement(0,0,rasTovtk->GetElement(0,0)*downFactor);
    rasTovtk->SetElement(1,1,rasTovtk->GetElement(1,1)*downFactor);
  } else {
    vol=readerA->GetOutput();
  }

  // Compute Laplacian to enhance edges
  //vtkImageLaplacian *laplacian = vtkImageLaplacian::New();
  //laplacian->SetInput(readerA->GetOutput());
  //laplacian->SetDimensionality(3);
  //laplacian->Update();

  // Median Filtering to help lung extraction
  vtkImageMedian3D *filter=vtkImageMedian3D::New();
  filter->SetInputData(vol);
  filter->SetKernelSize(3,3,1);
  std::cout<<"Median filtering..."<<std::endl;
  filter->Update();

  // combine labels
  vtkSimpleLungMask *lungMask = vtkSimpleLungMask::New();
  lungMask->SetInputData( filter->GetOutput() );
  lungMask->SetTracheaLabel(1);
  lungMask->SetRasToVtk(rasTovtk);
  std::cout<<"Lung mask extraction..."<<std::endl;
  lungMask->Update();

  // upsample
  resampling->Delete();
  resampling = vtkImageResample::New();

  if (applyResampling == 1) {
    resampling->SetInputConnection(lungMask->GetOutputPort());
    resampling->SetDimensionality(3);
    for (int ii=0; ii<2;ii++) {
      resampling->SetAxisMagnificationFactor(ii,1/downFactor);
    }
    resampling->SetAxisMagnificationFactor(2,1);
    resampling->SetInterpolationModeToNearestNeighbor();
    std::cout<<"Upsampling..."<<std::endl;
    resampling->Update();
    vol=resampling->GetOutput();
  } else {
    vol=lungMask->GetOutput();
  }

  // Output
  std::cout<<"Reade To Write File"<<endl;
  vtkNRRDWriterCIP *writer = vtkNRRDWriterCIP::New();
  writer->SetFileName(outputVolume.c_str());
  writer->SetInputData( vol );
  readerA->GetRasToIjkMatrix()->Invert();
  writer->SetIJKToRASMatrix( readerA->GetRasToIjkMatrix() );
  writer->Write();

  //Delete everything
  readerA->Delete();
  rasTovtk->Delete();
  filter->Delete();
  resampling->Delete();
  lungMask->Delete();
  writer->Delete();

  return EXIT_SUCCESS;
}
