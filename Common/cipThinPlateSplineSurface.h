/**
 *  \class cipThinPlateSplineSurface
 *  \ingroup common
 *  \brief This class is used to define a thin plate spline surface in
 *  3D given a set of 3D points
 *
 *  $Date: 2012-09-19 21:56:40 -0400 (Wed, 19 Sep 2012) $
 *  $Revision: 282 $
 *  $Author: jross $
 *
 *  TODO:
 *  1) Needs commenting
 *  2) Should not need to include itkImage.h, but currently
 *     we have to for compilation. Needs to be fixed.
 */

#ifndef __cipThinPlateSplineSurface_h
#define __cipThinPlateSplineSurface_h

#include <vector>
#include "itkImage.h"
#include "cipChestConventions.h"

class cipThinPlateSplineSurface
{
public:
  cipThinPlateSplineSurface();
  cipThinPlateSplineSurface( const std::vector< cip::PointType >& );
  ~cipThinPlateSplineSurface() {};

  double GetSurfaceHeight( double, double ) const;

  /**  */
  void SetSurfacePoints( const std::vector< cip::PointType >& );

  /**  */
  void SetSurfacePointWeights( const std::vector< double >* const );

  /**  */
  void ComputeThinPlateSplineVectors();

  /**  */
  void GetSurfaceNormal( double x, double y, cip::VectorType& normal ) const;

  /**  */
  void GetNonNormalizedSurfaceNormal( double, double, cip::VectorType& ) const;

  /** lambda is a parameter that controls smoothing. If set to 0,
      interpolation will be exact. As lambda increases, the TPS
      surface fit becomes looser and looser. Refer to
      http://elonen.iki.fi/code/tpsdemo/ */ 
  void SetLambda( double );
    
  /** */
  double GetLambda() const
    {
      return m_Lambda;
    }

  /** */
  double GetBendingEnergy() const;

  /** */
  const std::vector< double >& GetWVector() const
    {
      return m_w;
    };

  /** */
  const std::vector< double >& GetAVector() const
    {
      return m_a;
    };

  /** */
  const std::vector< cip::PointType >& GetSurfacePoints() const
    {
      return m_SurfacePoints;
    };

  /** */
  const unsigned int GetNumberSurfacePoints() const
    {
      return m_NumberSurfacePoints;
    };

private:
  std::vector< double >         m_a;
  std::vector< double >         m_w;
  std::vector< cip::PointType > m_SurfacePoints;
  std::vector< double >         m_SurfacePointWeights;
  double m_Lambda;
  unsigned int m_NumberSurfacePoints;
};

#endif
