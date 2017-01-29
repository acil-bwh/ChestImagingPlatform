#ifndef _ShapeModel_h_
#define _ShapeModel_h_

#include "ShapeModelObject.h"

#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vtkTransform.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

class ShapeModel : public ShapeModelObject
{
public:
  ShapeModel(const std::string& dataDir);
  ~ShapeModel();

  void project( const vnl_vector<double>& points, int numModes );
  const vnl_vector< double >& getMean() const { return _mean; }
  const vnl_matrix< double >& getEigVec() const { return _eigvec; }
  const vnl_vector< double >& getModel() const { return _model; }

  unsigned int getNumberOfPoints() const { return _mean.size() / 3; }
  unsigned int getNumberOfModes() const { return _eigvec.cols(); }
  unsigned int getNumberOfLeftPoints() const { return _numLeftPoints; }

  vtkSmartPointer< vtkPolyData > getPolyData() const { return _polydata; }
  vtkSmartPointer< vtkTransform > getTransform() const { return _transform; }
  vtkSmartPointer< vtkPolyData > getPolyDataModelSpace() const;
  vtkSmartPointer< vtkPolyData > transformToModelSpace( vtkSmartPointer< vtkPolyData > polydata ) const;
  void setTransform( vtkSmartPointer< vtkTransform > transform ) {
    _transform = transform;
    updatePolyData();
  }
  
  void setImagePoints( const vnl_vector<double>& imagePoints ) { _image = imagePoints; }
  vtkSmartPointer< vtkPolyData > getTargetPolyData() const;

protected:
  // throws std::runtime_error when it failed to load the model
  void load( const std::string& dataDir );
  // loading text data is depricated in favor of loading binary data
  void loadtxt( const std::string& dataDir );
  void updatePolyData();

private:
  vnl_matrix< double > _eigvecT;
  vnl_matrix< double > _eigvec;
  vnl_vector< double > _eigval;
  vnl_vector< double > _mean;
  vnl_vector< double > _model; // point model reconstructed from coef
  vnl_vector< double > _image; // target point model in image space 
  
  unsigned int _numLeftPoints; // number of points on the left ASM model (only for combined ASM)

  vtkSmartPointer< vtkPolyData > _polydata; // polydata to keep the surface & normal
  vtkSmartPointer< vtkTransform > _transform; // model to image transform (scale & pose)
};

#endif //_ShapeModel_h_
