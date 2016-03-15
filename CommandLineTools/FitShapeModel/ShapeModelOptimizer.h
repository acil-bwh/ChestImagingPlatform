#ifndef _ShapeModelOptimizer_h_
#define _ShapeModelOptimizer_h_

#include "FitShapeModelTypes.h"
#include <vtkOBJReader.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkMatrix4x4.h>
#include <vtkPointData.h>
#include <vtkPolyDataNormals.h>
#include <vtkTransform.h>
#include <vtkLandmarkTransform.h>
#include <vtkTransformPolyDataFilter.h>

class ShapeModel;
class ShapeModelVisualizer;

class ShapeModelOptimizer
{
public:
  ShapeModelOptimizer( ShapeModel& shapeModel,
                       ImageType::Pointer image );
  virtual ~ShapeModelOptimizer();
  void run( double searchLength,
            double sigma,
            double decayFactor,
            int maxIteration,
            int poseOnlyIteration,
            int numModes,
            ShapeModelVisualizer& visualizer );

protected:
  void prepareGradientImages( double sigam );

private:
  ShapeModel& _shapeModel;
  ImageType::Pointer _image;
  CovImageType::Pointer _gradientImage;
  GradientInterpolatorType::Pointer _gradientInterpolator;
  bool _interpolateGradient;
};

#endif
