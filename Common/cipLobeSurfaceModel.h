/**
 *  \class cipLobeBoundaryShapeModel
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

#ifndef __cipLobeBoundaryShapeModel_h
#define __cipLobeBoundaryShapeModel_h

#include <vector>

class cipLobeBoundaryShapeModel
{
public:
  ~cipLobeBoundaryShapeModel();
  cipLobeBoundaryShapeModel();

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
  void SetMeanSurfacePoints( std::vector< double* > const* );
  std::vector< double* > const* GetMeanSurfacePoints() const;

  /** Get surface points weighted according to the mode weights */
  std::vector< double* > const* GetWeightedSurfacePoints();

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

private:
  void ComputeWeightedSurfacePoints();

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
  std::vector< double* > MeanSurfacePoints;

  //
  // The weighted surface points correspond to the mean surface points 
  // but weighted according to the stored weights for the various
  // principle components
  // 
  std::vector< double* > WeightedSurfacePoints;

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
};

#endif
