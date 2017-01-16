#include "ShapeModel.h"
#include "VNLVTKConverters.h"
#include "cnpy.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vtkSmartPointer.h>
#include <vtkOBJReader.h>
#include <vtkTransformPolyDataFilter.h>

ShapeModel::ShapeModel( const std::string& dataDir )
{
  load( dataDir ); // load PCA data
  
  // Read shape model data (including mesh, ASM) using VTK
  std::cout << "Reading mean mesh file..." << std::endl;
  std::string meshFileName = dataDir + "/mean-mesh.obj";

  vtkSmartPointer< vtkOBJReader > objReader = vtkSmartPointer< vtkOBJReader >::New();
  objReader->SetFileName( meshFileName.c_str() );
  objReader->Update();
  _polydata = vtkSmartPointer< vtkPolyData >::New();
  _polydata->DeepCopy( objReader->GetOutput() );

  std::cout << "VTK: number of mesh points: " << _polydata->GetNumberOfPoints() << std::endl;
}

ShapeModel::~ShapeModel()
{
}

void
ShapeModel::project( const vnl_vector< double >& points, int numModes )
{
  vnl_vector<double> dev = points - _mean;

  vnl_vector<double> coef = _eigvecT * dev;

  //std::cout << "coef: " << coef << std::endl;

  // regularize shape
  for (unsigned int i = 0; i < coef.size(); i++)
  {
    if (i < numModes)
    {
      double sigma3 = 3 * sqrt(_eigval[i]);
      if (coef[i] > sigma3)
      {
        //std::cout << "adjusted from " << coef[i] << " to " << sigma3 << std::endl;
        coef[i] = sigma3;
      }
      else if (coef[i] < -sigma3)
      {
        //std::cout << "adjusted from " << coef[i] << " to " << -sigma3 << std::endl;
        coef[i] = -sigma3;
      }
    }
    else // cut off tailing coefficients
    {
      coef[i] = 0;
    }
  }

  // reconstruct from coef
  _model = _mean + _eigvec * coef;
}

void
ShapeModel::load( const std::string& dataDir )
{
  std::cout << "Loading PCA data..." << std::endl;
  
  // using binary numpy data loading module
  // https://github.com/rogersce/cnpy
  
  std::string filename = dataDir + "/lung-asm.npz";
  
  cnpy::npz_t dataz = cnpy::npz_load( filename );

  // load mean  
  cnpy::NpyArray& mean = dataz["mean"];
  unsigned int mean_len = mean.shape[0];
  
  std::cout << "mean data length: " << mean_len << std::endl;
  float* mean_data = reinterpret_cast< float* >( mean.data );
  
  _mean.set_size( mean_len );
  for (unsigned int i = 0; i < mean_len; i++)
  {
    _mean[i] = mean_data[i];
  }
  _model = _mean;
  
  // load variance (eigenvalues)
  cnpy::NpyArray& variance = dataz["variance"];
  unsigned int variance_len = variance.shape[0];
  
  std::cout << "eigenvalues count: " << variance_len << std::endl;
  float* variance_data = reinterpret_cast< float* >( variance.data );
  
  _eigval.set_size( variance_len );
  for (unsigned int i = 0; i < variance_len; i++)
  {
    _eigval[i] = variance_data[i];
  }
  
  // load modes (eigenvectors)
  cnpy::NpyArray& modes = dataz["modes"];
  unsigned int modes_len = modes.shape[0];
  unsigned int num_modes = modes.shape[1];
  
  std::cout << "eigenvectors: rows = " << modes_len << ", columns = " << num_modes << std::endl;
  
  if (mean_len != modes_len)
  {
    throw std::runtime_error("Lengths of mean vector and mode vector mismatch.");
  }

  float* modes_data = reinterpret_cast< float* >( modes.data );
  
  _eigvec.set_size( modes_len, num_modes );
  for (unsigned int i = 0; i < modes_len; i++)
  {
    for (unsigned int j = 0; j < num_modes; j++)
    {
      _eigvec( i, j ) = modes_data[ i * num_modes + j ];
    }
  }
  _eigvecT = _eigvec.transpose();
}

//
// loading text data is depricated in favor of loading binary data
//
void
ShapeModel::loadtxt( const std::string& dataDir )
{
  std::cout << "Loading PCA data in text format..." << std::endl;

  std::string eigvecFileName = dataDir + "/pca-modes.txt";
  std::string eigvalFileName = dataDir + "/pca-eigvals.txt";
  std::string meanFileName   = dataDir + "/pca-mean.txt";

  std::ifstream ifs;

  try
  {
    ifs.open(meanFileName.c_str());
    if (ifs.fail())
    {
      throw std::string(meanFileName);
    }
    _mean.read_ascii(ifs);
    ifs.close();
    std::cout << "mean data length: " << _mean.size() << std::endl;

    // initialize model with mean
    _model = _mean;

    ifs.open(eigvalFileName.c_str());
    if (ifs.fail())
    {
      throw std::string(eigvalFileName);
    }
    _eigval.read_ascii(ifs);
    ifs.close();
    std::cout << "eigenvalues count: " << _eigval.size() << std::endl;

    ifs.open(eigvecFileName.c_str());
    if (ifs.fail())
    {
      throw std::string(eigvecFileName);
    }
    _eigvec.read_ascii(ifs);
    ifs.close();
    std::cout << "eigenvectors: rows = " << _eigvec.rows() << ", columns = " << _eigvec.columns() << std::endl;
    _eigvecT = _eigvec.transpose();
  }
  catch (std::string& fname)
  {
    throw std::runtime_error("Failed to load shape model: failed to read " + fname);
  }
}

void
ShapeModel::updatePolyData()
{
  // to sync the polydata with the new model to image transform
  vtkSmartPointer< vtkTransformPolyDataFilter > modelToImageTransformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
  vnlVectorToVTKPoints( _model, _polydata->GetPoints() ); // copy model points to polydata
  modelToImageTransformFilter->SetInputData( _polydata ); // original mean data (model coordinate system)
  modelToImageTransformFilter->SetTransform( _transform ); // model coord -> image coord in the Sun's paper
  modelToImageTransformFilter->Update();
  _polydata = modelToImageTransformFilter->GetOutput(); // update input with output
}

vtkSmartPointer< vtkPolyData >
ShapeModel::getPolyDataModelSpace() const
{
  return transformToModelSpace( this->getPolyData() );
}

vtkSmartPointer< vtkPolyData >
ShapeModel::transformToModelSpace( vtkSmartPointer< vtkPolyData > polydata ) const
{
  vtkSmartPointer< vtkMatrix4x4 > matrix = vtkSmartPointer< vtkMatrix4x4 >::New();
  this->getTransform()->GetInverse( matrix );

  vtkSmartPointer< vtkTransform > imageToModelTransform = vtkSmartPointer< vtkTransform >::New();
  imageToModelTransform->SetMatrix( matrix );

  vtkSmartPointer< vtkTransformPolyDataFilter > imageToModelTransformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
  imageToModelTransformFilter->SetInputData( polydata ); // original mean data (model coordinate system)
  imageToModelTransformFilter->SetTransform( imageToModelTransform ); // model coord -> image coord in the Sun's paper
  imageToModelTransformFilter->Update();
  return imageToModelTransformFilter->GetOutput();
}

vtkSmartPointer< vtkPolyData > 
ShapeModel::getTargetPolyData() const 
{
  vtkSmartPointer< vtkPolyData > target_polydata = vtkSmartPointer< vtkPolyData >::New();
  /*
  vtkSmartPointer< vtkPoints > image_points = vtkSmartPointer< vtkPoints >::New();
  image_points->SetNumberOfPoints( this->getNumberOfPoints() );
  vnlVectorToVTKPoints( _image, image_points );
  target_polydata->SetPoints( image_points );
  */
  target_polydata->DeepCopy( _polydata );
  vnlVectorToVTKPoints( _image, target_polydata->GetPoints() );
  
  return target_polydata;
}
