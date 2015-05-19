#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkSmoothLines.h>
#include <New/vtkTubularScalePolyDataFilter.h>
 
// ---------------------------------------------------------------------------
// Testing vtkSmoothLines
// Testing vtkTubularScalePolyDataFilter

int main(int, char *[])
{
  vtkSmartPointer<vtkPoints> points = 
    vtkSmartPointer<vtkPoints>::New();
  points->InsertNextPoint(0.0, 0.0, 0.0);
 
  vtkSmartPointer<vtkPolyData> inputPolydata =   
    vtkSmartPointer<vtkPolyData>::New();
  inputPolydata->SetPoints(points);
  
  inputPolydata->Print(std::cout);
 
  vtkSmartPointer<vtkSmoothLines> filter = 
    vtkSmartPointer<vtkSmoothLines>::New();

  filter->Print(std::cout);

  filter->SetInputData(inputPolydata);
  filter->Update();

  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->SetWholeExtent(0, 32, 0, 32, 0, 0);
  source->Update();

  vtkSmartPointer<vtkTubularScalePolyDataFilter> filter2 = 
    vtkSmartPointer<vtkTubularScalePolyDataFilter>::New();

  filter2->SetImageData(source->GetOutput());
  filter2->SetInputConnection(filter->GetOutputPort());
  filter2->Print(cout);
  filter2->Update();
 
  vtkPolyData* outputPolydata = filter2->GetOutput();
 
  outputPolydata->Print(std::cout);

  return EXIT_SUCCESS;
}

