#include "ShapeModelOptimizerCT.h"
#include "ShapeModel.h"
#include "ShapeModelImageShort.h"
#include "VNLVTKConverters.h"
#include "ShapeModelVisualizer.h"

#include <vtkMath.h>

#include <climits>
#include <float.h>
#include <ctime>
#include <algorithm>

ShapeModelOptimizerCT::ShapeModelOptimizerCT( ShapeModel& shapeModel,
                                              ShapeModelImage& image )
: ShapeModelOptimizer( shapeModel, image ),
  _interpolateGradient( false )
{
}

ShapeModelOptimizerCT::~ShapeModelOptimizerCT()
{
}

void
ShapeModelOptimizerCT::beforeOptimization( double sigma )
{
  // prepare gradient image and interpolator
  
  FT::GradientRecursiveGaussianImageFilterType::Pointer gradientFilter = 
    FT::GradientRecursiveGaussianImageFilterType::New();

  gradientFilter->SetInput( dynamic_cast< ShapeModelImageShort& >(_image).getImageITK() );
  gradientFilter->SetSigma( sigma );
  try
  {
    std::cout << "Running Gaussian gradient filter..." << std::endl;
    gradientFilter->Update();
    std::cout << "Done." << std::endl;
  }
  catch (itk::ExceptionObject& e)
  {
    throw std::runtime_error( e.what() );
  }
  _gradientImage = gradientFilter->GetOutput();

  if (_interpolateGradient)
  {
    _gradientInterpolator = FT::GradientInterpolatorType::New();
    _gradientInterpolator->SetInputImage( _gradientImage );
  }
}

bool
ShapeModelOptimizerCT::transformPhysicalPointToIndex( const PointType& pt,
                                                      IndexType& pixelIndex )
{
  return _gradientImage->TransformPhysicalPointToIndex( pt, pixelIndex );
}

double 
ShapeModelOptimizerCT::updatePosition( const PointType& pt,
                                       const IndexType& idx,
                                       const PointType& prevPt,
                                       double prevEval,
                                       PointType& qt, 
                                       const CovPixelType& normal, 
                                       double& maxEval,
                                       double& minEval,
                                       int j, int& minj )
{
  CovPixelType gradDir = (_interpolateGradient)
                         ? _gradientInterpolator->Evaluate( pt )
                         : _gradientImage->GetPixel( idx );
  double curEval = gradDir.GetNorm();
  if (normal * gradDir > 0 && curEval > maxEval)
  {
    maxEval = curEval;
    qt = pt;
  }
  return curEval;
}

PointType 
ShapeModelOptimizerCT::determinePosition( unsigned int i, 
                                          const std::vector<PointType>& vecOrgPt, 
                                          const std::vector<PointType>& vecQt, 
                                          const std::vector<double>& vecMaxEval, 
                                          const std::vector<double>& vecMinEval,
                                          double maxMaxEval,
                                          double minMinEval )
{
  PointType orgPt = vecOrgPt[i];
  PointType qt = vecQt[i];
  double maxEval = vecMaxEval[i];

  PointType::VectorType diff = qt - orgPt; // full offset
  PointType::VectorType::ValueType f = maxEval / std::max( maxMaxEval, 1.0 ); // fraction of full offset
  qt = orgPt + f * diff; // final target point
  
  return qt;
}
