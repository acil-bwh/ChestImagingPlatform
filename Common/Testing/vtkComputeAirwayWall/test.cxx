#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkComputeAirwayWall.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkImageCast.h>

// ---------------------------------------------------------------------------
// Testing vtkComputeAirwayWall
 
int main(int, char *[])
{
  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->Update();

  vtkSmartPointer<vtkComputeAirwayWall> filter =
    vtkSmartPointer<vtkComputeAirwayWall>::New();

  filter->SetInputConnection(source->GetOutputPort());
  filter->Update();
 
  filter->Print(cout);

  return EXIT_SUCCESS;
}
