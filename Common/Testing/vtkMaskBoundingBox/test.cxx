#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkMaskBoundingBox.h>

// ---------------------------------------------------------------------------
// Testing vtkMaskBoundingBox
 
int main(int, char *[])
{
  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->Update();
 
  vtkSmartPointer<vtkMaskBoundingBox> boundingBoxFilter =
    vtkSmartPointer<vtkMaskBoundingBox>::New();
  
  boundingBoxFilter->SetInputData(source->GetOutput());
  boundingBoxFilter->Compute();

  return EXIT_SUCCESS;
}

