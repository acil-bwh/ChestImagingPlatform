#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkImageSinusoidSource.h>
#include <New/vtkExtractAirwayTree.h>
 
// ---------------------------------------------------------------------------
// Testing vtkExtractAirwayTree

int main(int, char *[])
{
  // Create an image
  vtkSmartPointer<vtkImageSinusoidSource> source =
    vtkSmartPointer<vtkImageSinusoidSource>::New();
  source->SetWholeExtent(0, 255, 0, 255, 0, 1);
  source->Update();
 
  vtkImageData* input = vtkImageData::SafeDownCast(source->GetOutput());
  std::cout << "----- input --------------------------------------" << std::endl;
  input->PrintSelf(std::cout, vtkIndent(2));
 
  vtkSmartPointer<vtkExtractAirwayTree> filter = 
    vtkSmartPointer<vtkExtractAirwayTree>::New();

  filter->SetInputData(input);
  filter->Update();
 
  vtkPolyData* outputPolydata = filter->GetOutput();
 
  std::cout << "----- output -------------------------------------" << std::endl;
  outputPolydata->PrintSelf(std::cout, vtkIndent(2));

  return EXIT_SUCCESS;
}

