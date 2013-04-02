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

class cipThinPlateSplineSurface
{
public:
  ~cipThinPlateSplineSurface();
  cipThinPlateSplineSurface();

  cipThinPlateSplineSurface( const std::vector< double* >* const );

  double GetSurfaceHeight( double, double );

  /**  */
  void SetSurfacePoints( const std::vector< double* >* const );

  /**  */
  void SetSurfacePointWeights( const std::vector< double >* const );

  /**  */
  void ComputeThinPlateSplineVectors();

  /**  */
  void GetSurfaceNormal( double, double, double* );

  /**  */
  void GetNonNormalizedSurfaceNormal( double, double, double* );

  /** lambda is a parameter that controls smoothing. If set to 0,
      interpolation will be exact. As lambda increases, the TPS
      surface fit becomes looser and looser. Refer to
      http://elonen.iki.fi/code/tpsdemo/ */ 
  void SetLambda( double );
    
  /** */
  double GetLambda()
    {
      return m_Lambda;
    }

  /** */
  double GetBendingEnergy();

  /** */
  const std::vector< double >* GetWVector()
    {
      return m_w;
    };

  /** */
  const std::vector< double >* GetAVector()
    {
      return m_a;
    };

  /** */
  const std::vector< const double* >* GetSurfacePoints()
    {
      return m_SurfacePoints;
    };

  /** */
  const unsigned int GetNumberSurfacePoints()
    {
      return m_SurfacePoints->size();
    };

private:
  void Init();

  std::vector< double >*         m_a;
  std::vector< double >*         m_w;
  std::vector< const double* >*  m_SurfacePoints;
  std::vector< double >*         m_SurfacePointWeights;
  double m_Lambda;

  unsigned int m_NumberSurfacePoints;
};

#endif
