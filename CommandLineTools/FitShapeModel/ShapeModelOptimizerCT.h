#ifndef _ShapeModelOptimizerCT_h_
#define _ShapeModelOptimizerCT_h_

#include "ShapeModelOptimizer.h"

class ShapeModelOptimizerCT : public ShapeModelOptimizer
{
public:
  typedef FitShapeModelType< short > FT;
  ShapeModelOptimizerCT( ShapeModel& shapeModel,
                         ShapeModelImage& image );
  virtual ~ShapeModelOptimizerCT();

protected:
  virtual void beforeOptimization( double sigam );
  virtual double getSamplingFactor() const { return 1.0; }
  virtual bool transformPhysicalPointToIndex( const PointType& pt,
                                              IndexType& pixelIndex );
  virtual bool updatePosition( const PointType& pt,
                               const IndexType& idx,
                               const PointType& prevPt,
                               double& prevEval,
                               PointType& qt, 
                               const CovPixelType& normal, 
                               double& maxEval,
                               double& minEval,
                               int j, int& minj );
  virtual PointType determinePosition( unsigned int i, 
                                       const std::vector<PointType>& vecOrgPt, 
                                       const std::vector<PointType>& vecQt, 
                                       const std::vector<double>& vecMaxEval, 
                                       const std::vector<double>& vecMinEval,
                                       double maxMaxEval,
                                       double minMinEval );

private:
  FT::CovImageType::Pointer _gradientImage;
  FT::GradientInterpolatorType::Pointer _gradientInterpolator;
  bool _interpolateGradient;
};

#endif
