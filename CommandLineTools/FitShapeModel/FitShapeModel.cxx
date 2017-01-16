#include "cipChestConventions.h"
#include "cipHelper.h"
#include "FitShapeModelTypes.h"
#include "ShapeModelImage.h"
#include "ShapeModelImageFactory.h"
#include "ShapeModelInitializer.h"
#include "ShapeModelOptimizerCT.h"
#include "ShapeModelOptimizerDT.h"
#include "ShapeModelVisualizer.h"
#include "ShapeModelFinalizer.h"
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

  int DoIT( int argc, char * argv[] )
  {
    PARSE_ARGS;
    
    ShapeModelImage* pimage = ShapeModelImageFactory::create( inputImageType );
    
    // Read the input ct image
    pimage->read( imageFileName );
    
    // create and load shape model (including mesh, PCA data)
    ShapeModel shapeModel( shapeModelDir );
    
    // by default geometry will be output in image space
    bool outputGeomInModelSpace = (inputImageType == "CT");

    ShapeModelVisualizer visualizer( shapeModel, 
                                     *pimage, 
                                     outputFileName, 
                                     outputGeometry,
                                     outputGeomInModelSpace );

    timer t;
                                     
    // initialize shape model (for scale & pose initialization using mean shape)
    ShapeModelInitializer initializer( shapeModel, *pimage );
    if (transformFileName.empty())
    {
      initializer.run( offsetRL, offsetAP, offsetSI );
    }
    else
    {
      initializer.run( transformFileName );
    }

    if (runMode == "Alignment")
    {
      visualizer.update( sigma );
    }
    else if (runMode == "Fitting" || runMode == "Segmentation")
    {
      ShapeModelOptimizer* poptimizer;
      if (inputImageType == "CT")
      {
        poptimizer = new ShapeModelOptimizerCT( shapeModel, *pimage );
      }
      else if (inputImageType == "DT") // distance transform
      {
        poptimizer = new ShapeModelOptimizerDT( shapeModel, *pimage );
      }
      else
      {
        throw std::invalid_argument( "Unsupported input image type: " + inputImageType );
      }

      poptimizer->run( imageFileName,
                       searchLength,
                       sigma,
                       decayFactor,
                       maxIteration,
                       poseOnlyIteration,
                       numModes,
                       verbose,
                       visualizer );

      if (runMode == "Segmentation")
      {
        ShapeModelFinalizer finalizer( shapeModel, *pimage, outputFileName, outputGeometry );
        finalizer.run();
      }
      else
      {
        visualizer.update( sigma );
      }
            
      delete poptimizer;
    } // for fitting
    
    std::cout << runMode << " took " << t.elapsed() << " sec." << std::endl;

    delete pimage;
    
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
