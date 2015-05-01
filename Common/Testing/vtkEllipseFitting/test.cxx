#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <New/vtkEllipseFitting.h>
 
// ---------------------------------------------------------------------------
// Testing vtkEllipseFitting

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
 
  vtkSmartPointer<vtkEllipseFitting> filter = 
    vtkSmartPointer<vtkEllipseFitting>::New();

  filter->Print(std::cout);

  filter->SetInputData(inputPolydata);
  filter->DebugOn();
  filter->Update();
 
  vtkPolyData* outputPolydata = filter->GetOutput();
 
  outputPolydata->Print(std::cout);

  return EXIT_SUCCESS;
}

