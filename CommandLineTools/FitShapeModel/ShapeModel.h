#ifndef _ShapeModel_h_
#define _ShapeModel_h_

#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vtkTransform.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

class ShapeModel
{
public:
  ShapeModel(const std::string& dataDir);
  ~ShapeModel();

  void project( const vnl_vector<double>& points, int numModes );
  const vnl_vector< double >& getMean() const { return _mean; }
  const vnl_matrix< double >& getEigVec() const { return _eigvec; }
  const vnl_vector< double >& getModel() const { return _model; }

  unsigned int getNumberOfPoints() { return _mean.size() / 3; }

  vtkSmartPointer< vtkPolyData > getPolyData() const { return _polydata; }
  void setPolyData( vtkSmartPointer< vtkPolyData > polydata ) { _polydata = polydata; }
  vtkSmartPointer< vtkTransform > getTransform() const { return _transform; }
  void setTransform( vtkSmartPointer< vtkTransform > transform ) {
    _transform = transform;
    updatePolyData();
  }

protected:
  // throws std::runtime_error when it failed to load the model
  void load( const std::string& dataDir );
  void updatePolyData();

private:
  vnl_matrix< double > _eigvecT;
  vnl_matrix< double > _eigvec;
  vnl_vector< double > _eigval;
  vnl_vector< double > _mean;
  vnl_vector< double > _model; // point model reconstructed from coef

  vtkSmartPointer< vtkPolyData > _polydata; // polydata to keep the surface & normal
  vtkSmartPointer< vtkTransform > _transform; // model to image transform (scale & pose)
};

#endif
