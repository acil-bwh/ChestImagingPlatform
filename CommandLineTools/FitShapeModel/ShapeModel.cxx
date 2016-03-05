#include "ShapeModel.h"
#include "VNLVTKConverters.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vtkTransformPolyDataFilter.h>

ShapeModel::ShapeModel( const std::string& dataDir )
{
  load( dataDir );
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
