#ifndef _ShapeModelOptimizerDT_h_
#define _ShapeModelOptimizerDT_h_

#include "ShapeModelOptimizer.h"

class ShapeModelOptimizerDT : public ShapeModelOptimizer
{
public:
  typedef FitShapeModelType< float > FT;
  ShapeModelOptimizerDT( ShapeModel& shapeModel,
                         ShapeModelImage& image );
  virtual ~ShapeModelOptimizerDT();

protected:
  virtual void beforeOptimization( double sigam );
  virtual double getSamplingFactor() const { return 2.0; } // super-sampling
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
  FT::ImageType::Pointer _itkImage;
  FT::ImageInterpolatorType::Pointer _imageInterpolator;
};

#endif
