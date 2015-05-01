#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkGeneralizedQuadratureKernelSource.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkImageCast.h>

// ---------------------------------------------------------------------------
// Testing vtkGeneralizedQuadratureKernelSource
// Segmentation fault occurs which seems to be related to multi-threading
// (may not be due to porting...)
// source->DebugOn() was used to trace
 
int main(int, char *[])
{
  vtkSmartPointer<vtkGeneralizedQuadratureKernelSource> source =
    vtkSmartPointer<vtkGeneralizedQuadratureKernelSource>::New();
  cout << "----- source -----------------------------------------------" << endl;
  source->PrintSelf(cout, vtkIndent(2));

  //source->DebugOn(); // debug message causes thread deadlock
  source->Update();
  
  vtkImageKernel* imageKernel = source->GetOutput();
  cout << "----- output -----------------------------------------------" << endl;
  imageKernel->PrintSelf(cout, vtkIndent(2));
 
  return EXIT_SUCCESS;
}
