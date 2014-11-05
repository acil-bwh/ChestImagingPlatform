/**
 *  \class cipLobeSurfaceModel
 *  \ingroup common
 *  \brief This class ...
 *
 *  Detailed description 
 *
 *  $Date: 2012-09-06 15:43:56 -0400 (Thu, 06 Sep 2012) $
 *  $Revision: 246 $
 *  $Author: jross $
 *
 *  TODO:
 *  
 */

#ifndef __cipLobeSurfaceModel_h
#define __cipLobeSurfaceModel_h

#include <vector>
#include "cipChestConventions.h"

class cipLobeSurfaceModel
{
public:
  ~cipLobeSurfaceModel();
  cipLobeSurfaceModel();

  /**  Pointers are assumed to have 3 elements */
  void SetImageOrigin( double const* );
  double const* GetImageOrigin() const;

  /**  Pointers are assumed to have 3 elements */
  void SetImageSpacing( double const* );
  double const* GetImageSpacing() const;

  /**  */
  void   SetEigenvalueSum( double );
  double GetEigenvalueSum() const;

  /**  double pointers are assumed to have 3 elements */
  void SetMeanSurfacePoints( const std::vector< cip::PointType >& );
  const std::vector< cip::PointType >& GetMeanSurfacePoints() const;

  /** Get surface points weighted according to the mode weights */
  const std::vector< cip::PointType >& GetWeightedSurfacePoints();

  /** Get the mean right horizontal surface points. Assumes that 
   *  the shape model corresponds to the right lung. */
  const std::vector< cip::PointType >& GetMeanRightHorizontalSurfacePoints();

  /** Get the mean right oblique surface points. Assumes that 
   *  the shape model corresponds to the right lung. */
  const std::vector< cip::PointType >& GetMeanRightObliqueSurfacePoints();

  /** Get right horizontal surface points weighted according to the 
   *  mode weights. Assumes that the shape model corresponds to the
   *  right lung. */
  const std::vector< cip::PointType >& GetRightHorizontalWeightedSurfacePoints();

  /** Get right oblique surface points weighted according to the 
   *  mode weights. Assumes that the shape model corresponds to the
   *  right lung. */
  const std::vector< cip::PointType >& GetRightObliqueWeightedSurfacePoints();

  /** */
  void SetEigenvalues( std::vector< double > const* );
  std::vector< double > const* GetEigenvalues() const;

  /**  */
  void SetEigenvectors( std::vector< std::vector< double > > const* );
  std::vector< std::vector< double > > const* GetEigenvectors() const;

  /** The 'Get' method return vector intentionally left
   * non-const. This should make it easier to modify and instance */
  void SetModeWeights( std::vector< double > const* );
  std::vector< double >* GetModeWeights();

  /** */
  void         SetNumberOfModes( unsigned int );
  unsigned int GetNumberOfModes() const;

  /** Set/Get whether the surface model corresponds to the right lung. 
   *  False by default */
  void SetRightLungSurfaceModel( bool );
  bool GetRightLungSurfaceModel();

  /** Set/Get whether the surface model corresponds to the left lung. 
   *  True by default */
  void SetLeftLungSurfaceModel( bool );
  bool GetLeftLungSurfaceModel();

private:
  void ComputeWeightedSurfacePoints();
  void ComputeRightHorizontalWeightedSurfacePoints();
  void ComputeRightObliqueWeightedSurfacePoints();

  //
  // The origin and spacing are of the image from which the shape
  // model is derived
  //
  double* ImageOrigin;
  double* ImageSpacing;

  //
  // The sum of all the PCA eigenvalues 
  //
  double EigenvalueSum;  

  //
  // The mean surface points are stored as a vector of 3D points
  // (physical coordinates). The mean is calculated from PCA.
  //
  std::vector< cip::PointType > MeanSurfacePoints;
  std::vector< cip::PointType > MeanRightHorizontalSurfacePoints;
  std::vector< cip::PointType > MeanRightObliqueSurfacePoints;

  //
  // The weighted surface points correspond to the mean surface points 
  // but weighted according to the stored weights for the various
  // principle components
  // 
  std::vector< cip::PointType > WeightedSurfacePoints;
  std::vector< cip::PointType > LeftObliqueWeightedSurfacePoints;
  std::vector< cip::PointType > RightObliqueWeightedSurfacePoints;
  std::vector< cip::PointType > RightHorizontalWeightedSurfacePoints;

  //
  // The eigenvalues and eigenvectors derived from PCA
  //
  std::vector< double >                Eigenvalues;
  std::vector< std::vector< double > > Eigenvectors;
  
  //
  // The shape model's mode weights. If all weights are zero, this
  // will correspond to the mean shape. Weighting the modes will alter
  // the shape of the lobe boundary model
  //
  std::vector< double > ModeWeights;

  //
  // The number of modes from PCA
  //
  unsigned int NumberOfModes;

  bool IsRightLungSurfaceModel;
  bool IsLeftLungSurfaceModel;
};

#endif
