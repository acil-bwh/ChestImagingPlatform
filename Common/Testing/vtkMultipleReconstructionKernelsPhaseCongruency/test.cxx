#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkMultipleReconstructionKernelsPhaseCongruency.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkImageCast.h>
#include <vtkImageToImageStencil.h>
#include <vtkImageStencilData.h>
#include <New/vtkNRRDExport.h>
#include <New/vtkImageTubularConfidence.h>
#include <New/vtkTubularScaleSelection.h>

// ---------------------------------------------------------------------------
// Testing vtkMultipleReconstructionKernelsPhaseCongruency
// Testing vtkNRRDExport
// Testing vtkImageTubularConfidence
// Testing vtkTubularScaleSelection
 
int main(int, char *[])
{
  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->SetWholeExtent(0, 32, 0, 32, 0, 0);
  source->Update();

  // Create a second image
  vtkSmartPointer<vtkImageSinusoidSource> source2 =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source2->SetWholeExtent(0, 32, 0, 32, 0, 0);
  source2->Update();

  vtkSmartPointer<vtkImageTubularConfidence> filter2 =
    vtkSmartPointer<vtkImageTubularConfidence>::New();
  filter2->SetInputConnection(source2->GetOutputPort());
  filter2->Update();

  vtkSmartPointer<vtkTubularScaleSelection> filter3 =
    vtkSmartPointer<vtkTubularScaleSelection>::New();
  filter3->SetInputConnection(source->GetOutputPort());
  filter3->Update();

  // Create stencil image
  vtkSmartPointer<vtkImageToImageStencil> imageToImageStencil = 
    vtkSmartPointer<vtkImageToImageStencil>::New();
  imageToImageStencil->SetInputConnection(source2->GetOutputPort());
  imageToImageStencil->ThresholdByUpper(122);
  imageToImageStencil->Update();
  vtkImageStencilData* stencil = imageToImageStencil->GetOutput();

  // Create vtkNRRDExport
  vtkSmartPointer<vtkNRRDExport> nrrdExport =
    vtkSmartPointer<vtkNRRDExport>::New();
  nrrdExport->SetInputData(source->GetOutput());
  Nrrd* nrrd = nrrdExport->GetNRRDPointer();

  vtkSmartPointer<vtkMultipleReconstructionKernelsPhaseCongruency> filter =
    vtkSmartPointer<vtkMultipleReconstructionKernelsPhaseCongruency>::New();

  filter->AddInputConnection(source->GetOutputPort());
  filter->AddInputConnection(filter2->GetOutputPort());
  filter->SetStencil(stencil);
  filter->Update();

  int num = filter->GetNumberOfInputConnections(0);
 
  // Create actors
  vtkSmartPointer<vtkImageActor> inputActor =
    vtkSmartPointer<vtkImageActor>::New();
  inputActor->GetMapper()->SetInputConnection(
    source->GetOutputPort());
 
  vtkSmartPointer<vtkImageActor> normalizedActor =
    vtkSmartPointer<vtkImageActor>::New();
  normalizedActor->GetMapper()->SetInputConnection(
    filter->GetOutputPort());
 
  // There will be one render window
  vtkSmartPointer<vtkRenderWindow> renderWindow =
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->SetSize(600, 300);
 
  // And one interactor
  vtkSmartPointer<vtkRenderWindowInteractor> interactor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor->SetRenderWindow(renderWindow);
 
  // Define viewport ranges
  // (xmin, ymin, xmax, ymax)
  double leftViewport[4] = {0.0, 0.0, 0.5, 1.0};
  double rightViewport[4] = {0.5, 0.0, 1.0, 1.0};
 
  // Setup both renderers
  vtkSmartPointer<vtkRenderer> leftRenderer =
    vtkSmartPointer<vtkRenderer>::New();
  renderWindow->AddRenderer(leftRenderer);
  leftRenderer->SetViewport(leftViewport);
  leftRenderer->SetBackground(.6, .5, .4);
 
  vtkSmartPointer<vtkRenderer> rightRenderer =
    vtkSmartPointer<vtkRenderer>::New();
  renderWindow->AddRenderer(rightRenderer);
  rightRenderer->SetViewport(rightViewport);
  rightRenderer->SetBackground(.4, .5, .6);
 
  leftRenderer->AddActor(inputActor);
  rightRenderer->AddActor(normalizedActor);
 
  leftRenderer->ResetCamera();
 
  rightRenderer->ResetCamera();
 
  renderWindow->Render();
  interactor->Start();
 
  return EXIT_SUCCESS;
}
