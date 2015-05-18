#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkLungIntensityCorrection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkImageCast.h>
 
// ---------------------------------------------------------------------------
// Testing vtkLungIntensityCorrection

int main(int, char *[])
{
  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->Update();
 
  vtkSmartPointer<vtkLungIntensityCorrection> filter =
    vtkSmartPointer<vtkLungIntensityCorrection>::New();
 
  filter->SetInputConnection(source->GetOutputPort());
  filter->Update();
 
  vtkSmartPointer<vtkImageCast> inputCastFilter =
    vtkSmartPointer<vtkImageCast>::New();
  inputCastFilter->SetInputConnection(source->GetOutputPort());
  inputCastFilter->SetOutputScalarTypeToUnsignedChar();
  inputCastFilter->Update();
 
  vtkSmartPointer<vtkImageCast> outputCastFilter =
    vtkSmartPointer<vtkImageCast>::New();
  outputCastFilter->SetInputConnection(filter->GetOutputPort());
  outputCastFilter->SetOutputScalarTypeToUnsignedChar();
  outputCastFilter->Update();
 
  // Create actors
  vtkSmartPointer<vtkImageActor> inputActor =
    vtkSmartPointer<vtkImageActor>::New();
  inputActor->GetMapper()->SetInputConnection(
    inputCastFilter->GetOutputPort());
 
  vtkSmartPointer<vtkImageActor> normalizedActor =
    vtkSmartPointer<vtkImageActor>::New();
  normalizedActor->GetMapper()->SetInputConnection(
    outputCastFilter->GetOutputPort());
 
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

