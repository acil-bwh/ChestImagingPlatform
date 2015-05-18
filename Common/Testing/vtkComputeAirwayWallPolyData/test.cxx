#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkComputeAirwayWallPolyData.h>
 
// ---------------------------------------------------------------------------
// Testing vtkComputeAirwayWallPolyData

int main(int, char *[])
{
  vtkSmartPointer<vtkPoints> points = 
    vtkSmartPointer<vtkPoints>::New();
  points->InsertNextPoint(0.0, 0.0, 0.0);
  points->InsertNextPoint(1.0, 0.2, 0.0);
  points->InsertNextPoint(0.5, 0.3, 0.0);

  vtkSmartPointer<vtkPolyData> inputPolydata =   
    vtkSmartPointer<vtkPolyData>::New();
  inputPolydata->SetPoints(points);
  
  inputPolydata->Print(std::cout);
 
  vtkSmartPointer<vtkComputeAirwayWallPolyData> filter = 
    vtkSmartPointer<vtkComputeAirwayWallPolyData>::New();

  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->Update();

  filter->SetImage(source->GetOutput());
  filter->SetInputData(inputPolydata);
  filter->Print(std::cout);

  filter->DebugOn();
  filter->Update();
 
  vtkPolyData* outputPolydata = filter->GetOutput();
 
  outputPolydata->Print(std::cout);

  return EXIT_SUCCESS;
}

