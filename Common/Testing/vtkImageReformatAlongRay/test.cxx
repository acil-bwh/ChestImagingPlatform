#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkImageReformatAlongRay.h>
#include <New/vtkImageReformatAlongRay2.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkImageCast.h>

// ---------------------------------------------------------------------------
// Testing vtkImageReformatAlongRay
// Testing vtkImageReformatAlongRay2
 
int main(int, char *[])
{
  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->Update();

  vtkSmartPointer<vtkImageReformatAlongRay> filter =
    vtkSmartPointer<vtkImageReformatAlongRay>::New();

  filter->SetInputConnection(source->GetOutputPort());
  filter->Update();
 
  vtkSmartPointer<vtkImageReformatAlongRay2> filter2 =
    vtkSmartPointer<vtkImageReformatAlongRay2>::New();

  filter2->SetSpacing(1.0);
  filter2->SetInputConnection(source->GetOutputPort());
  filter2->Update();

  // Create actors
  vtkSmartPointer<vtkImageActor> inputActor =
    vtkSmartPointer<vtkImageActor>::New();
  inputActor->GetMapper()->SetInputConnection(
    source->GetOutputPort());
 
  vtkSmartPointer<vtkImageActor> normalizedActor =
    vtkSmartPointer<vtkImageActor>::New();
  normalizedActor->GetMapper()->SetInputConnection(
    filter2->GetOutputPort());
 
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
