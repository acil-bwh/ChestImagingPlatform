#include "cipChestConventions.h"
#include "cipHelper.h"
#include "FitShapeModelTypes.h"
#include "ShapeModelInitializer.h"
#include "ShapeModelOptimizer.h"
#include "ShapeModelVisualizer.h"
#include "FitShapeModelCLP.h"

#include "ShapeModel.h"

#include <vtkOBJReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

#include <climits>
#include <float.h>
#include <ctime>

namespace
{
  class timer { // measures CPU time
  public:
    timer() { _begin = clock(); }
    double elapsed() { return (clock() - _begin) / (double)CLOCKS_PER_SEC; }
  private:
    clock_t _begin;
  };

  // -------------------------------------------------------------------------
  // main method

  int DoIT(int argc, char * argv[])
  {
    PARSE_ARGS;

    // create and load shape model (PCA data)
    ShapeModel shapeModel( shapeModelDir );

    // Read the input ct image
    std::cout << "Reading CT image..." << std::endl;
    ImageReaderType::Pointer imageReader = ImageReaderType::New();
    imageReader->SetFileName( imageFileName.c_str() );
    try
    {
      imageReader->Update();
    }
    catch (itk::ExceptionObject& e)
    {
      throw std::runtime_error( e.what() );
    }

    ImageType::Pointer image = imageReader->GetOutput();
    ImageType::SizeType sz = image->GetLargestPossibleRegion().GetSize();

    ImageType::IndexType originIndex = {0, 0, 0}, centerIndex = {sz[0]/2, sz[1]/2, sz[2]/2};
    ImageType::PointType originPoint, centerPoint;
    image->TransformIndexToPhysicalPoint(originIndex, originPoint);
    image->TransformIndexToPhysicalPoint(centerIndex, centerPoint);

    std::cout << "origin point: " << originPoint << std::endl;
    std::cout << "center point: " << centerPoint << std::endl;
    std::cout << "spacing: " << image->GetSpacing() << std::endl;

    // Read shape model data (including mesh, ASM)
    std::cout << "Reading mesh file..." << std::endl;
    MeshReaderType::Pointer meshReader = MeshReaderType::New();
    std::string meshFileName = shapeModelDir + "/mean-mesh.obj";
    meshReader->SetFileName( meshFileName.c_str() );
    try
    {
      meshReader->Update();
    }
    catch (itk::ExceptionObject& e)
    {
      throw std::runtime_error( e.what() );
    }

    MeshType::Pointer mesh = meshReader->GetOutput();

    // secondary reading mesh file using VTK
    vtkSmartPointer< vtkOBJReader > objReader = vtkSmartPointer< vtkOBJReader >::New();
    objReader->SetFileName( meshFileName.c_str() );
    objReader->Update();
    vtkSmartPointer< vtkPolyData > polydata = vtkSmartPointer< vtkPolyData >::New();
    polydata->DeepCopy( objReader->GetOutput() );

    std::cout << "VTK: number of mesh points: " << polydata->GetNumberOfPoints() << std::endl;

    shapeModel.setPolyData( polydata );

    ShapeModelVisualizer visualizer( shapeModel, mesh, image, outputFileName, outputGeometry );

    // initialize shape model (for scale & pose initialization using mean shape)
    ShapeModelInitializer initializer( shapeModel, image );
    initializer.run( offsetRL, offsetAP, offsetSI );

    if (runMode == "Alignment")
    {
      visualizer.update();
    }
    else if (runMode == "Fitting")
    {
      timer t;

      ShapeModelOptimizer optimizer( shapeModel, image );
      optimizer.run( searchLength,
                     sigma,
                     decayFactor,
                     maxIteration,
                     numModes,
                     visualizer );

      std::cout << "Fitting took " << t.elapsed() << " sec." << std::endl;

      visualizer.update();
    } // for fitting

    std::cout << "DONE." << std::endl;
    return cip::EXITSUCCESS;
  } // DoIT
} // end of anonymous namespace


int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  try
  {
    timer t;

    bool result = DoIT( argc, argv );

    std::cout << "Total module run time: " << t.elapsed() << " sec." << std::endl;
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return cip::EXITFAILURE;
  }

  return cip::EXITSUCCESS;
}
