#ifndef _ShapeModelOptimizer_h_
#define _ShapeModelOptimizer_h_

#include "FitShapeModelTypes.h"
#include "ShapeModelObject.h"
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
class ShapeModelImage;
class ShapeModelVisualizer;

class ShapeModelOptimizer : public ShapeModelObject
{
public:
  ShapeModelOptimizer( ShapeModel& shapeModel,
                       ShapeModelImage& image );
  virtual ~ShapeModelOptimizer();
  // template method pattern
  virtual void run( const std::string& inputFileName,
                    double searchLength,
                    double sigma,
                    double decayFactor,
                    int maxIteration,
                    int poseOnlyIteration,
                    int numModes,
                    bool verbose,
                    ShapeModelVisualizer& visualizer );

protected:
  virtual void beforeOptimization( double sigam ) = 0;
  virtual double getSamplingFactor() const = 0;
  virtual bool transformPhysicalPointToIndex( const PointType& pt,
                                              IndexType& pixelIndex ) = 0;
  virtual bool updatePosition( const PointType& pt,
                               const IndexType& idx,
                               const PointType& prevPt,
                               double& prevEval,
                               PointType& qt, 
                               const CovPixelType& normal, 
                               double& maxEval,
                               double& minEval,
                               int j, int& minj ) = 0;
  virtual PointType determinePosition( unsigned int i, 
                                       const std::vector<PointType>& vecOrgPt, 
                                       const std::vector<PointType>& vecQt, 
                                       const std::vector<double>& vecMaxEval, 
                                       const std::vector<double>& vecMinEval,
                                       double maxMaxEval,
                                       double minMinEval ) = 0;
  
protected:
  ShapeModel& _shapeModel;
  ShapeModelImage& _image;
};

#endif
