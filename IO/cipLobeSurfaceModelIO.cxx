#include "cipLobeSurfaceModelIO.h"
#include "cipExceptionObject.h"
#include <fstream>
#include <stdlib.h>
#include <iostream> //DEBUG

using namespace cip;

LobeSurfaceModelIO::LobeSurfaceModelIO()
{
  this->FileName = "NA";
  this->ShapeModel = new cipLobeSurfaceModel();
}


LobeSurfaceModelIO::~LobeSurfaceModelIO()
{
  delete this->ShapeModel;
}


void LobeSurfaceModelIO::SetFileName( std::string fileName )
{
  this->FileName = fileName;
}


void LobeSurfaceModelIO::SetInput( cipLobeSurfaceModel* shapeModel )
{
  this->ShapeModel = shapeModel;
}


void LobeSurfaceModelIO::Write() const
{
  // Check to see if the file name has been set to something other
  // than the default
  if ( this->FileName.compare("NA") != 0 )
    {
      std::ofstream file( this->FileName.c_str() );
      
      file << this->ShapeModel->GetImageOrigin()[0]  << "," 
	   << this->ShapeModel->GetImageOrigin()[1]  << "," 
           << this->ShapeModel->GetImageOrigin()[2]  << std::endl;

      file << this->ShapeModel->GetImageSpacing()[0] << "," 
	   << this->ShapeModel->GetImageSpacing()[1] << "," 
           << this->ShapeModel->GetImageSpacing()[2] << std::endl;

      file << this->ShapeModel->GetNumberOfModes() << std::endl;
      file << this->ShapeModel->GetMeanSurfacePoints().size() << std::endl;

      // Write the mean z-vector
      for ( unsigned int i=0; i<this->ShapeModel->GetMeanSurfacePoints().size(); i++ )
	{
	  file << this->ShapeModel->GetMeanSurfacePoints()[i][2] << ",";
	}
      file << std::endl;

      // Write the eigenvalues and mode weights
      for ( unsigned int i=0; i<this->ShapeModel->GetNumberOfModes(); i++ )
	{
	  file << (*this->ShapeModel->GetEigenvalues())[i] << ",";
	  file << (*this->ShapeModel->GetModeWeights())[i] << std::endl;
	}

      // Write the modes
      for ( unsigned int i=0; i<this->ShapeModel->GetNumberOfModes(); i++ )
	{
	  for ( unsigned int j=0; j<this->ShapeModel->GetMeanSurfacePoints().size(); j++ )
	    {
	      file << (*this->ShapeModel->GetEigenvectors())[i][j] << ",";
	    }
	  file << std::endl;
	}

      // Write the domain locations
      for ( unsigned int i=0; i<this->ShapeModel->GetMeanSurfacePoints().size(); i++ )
	{
	  file << this->ShapeModel->GetMeanSurfacePoints()[i][0] << ",";
	  file << this->ShapeModel->GetMeanSurfacePoints()[i][1] << ",";
	  file << 0 << std::endl;
	}
    }
}


void LobeSurfaceModelIO::Read()
{
  std::ifstream file( this->FileName.c_str() );

  if ( !file )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "LobeSurfaceModelIO::Read()", "Problem opening file" );
    }

   unsigned int commaLocOld, commaLocNew;
  
  // First get the image origin
  double origin[3];
  std::string originString;
  std::getline( file, originString );
  commaLocOld = 0;

  for ( unsigned int i=0; i<3; i++ )
    {
      if ( i == 0 )
        { 
  	  commaLocNew = originString.find( ',', 0 );
  	  origin[i] = atof( originString.substr( 0, commaLocNew-commaLocOld+1).c_str() );
        }
      else
        {
  	  commaLocNew = originString.find( ',', commaLocOld+1 );
  	  origin[i] = atof( originString.substr( commaLocOld+1, commaLocNew-commaLocOld-1).c_str() );
        }
      
      commaLocOld = commaLocNew;
    }
  this->ShapeModel->SetImageOrigin( origin );
  
  // Now get the image spacing
  double spacing[3];  
  std::string spacingString;
  std::getline( file, spacingString );
  commaLocOld = 0;
  
  for ( unsigned int i=0; i<3; i++ )
    {
      if ( i == 0 )
        {
  	  commaLocNew = spacingString.find( ',', 0 );
  	  spacing[i] = atof( spacingString.substr( 0, commaLocNew-commaLocOld+1).c_str() );
        }
      else
        {
  	  commaLocNew = spacingString.find( ',', commaLocOld+1 );
  	  spacing[i] = atof( spacingString.substr( commaLocOld+1, commaLocNew-commaLocOld-1).c_str() );
        }
      
      commaLocOld = commaLocNew;
    }
  this->ShapeModel->SetImageSpacing( spacing );
  
  // And now get the number of modes in the shape model
  std::string numModesString;
  std::getline( file, numModesString );
  this->ShapeModel->SetNumberOfModes( (unsigned int)( atoi( numModesString.c_str() ) ) );

  // Now get the number of z-values 
  std::string numZValsString;
  std::getline( file, numZValsString );
  unsigned int numZvals = atoi( numZValsString.c_str() );

  // Read in the mean z-vector
  std::vector< double > meanZValues;  
  std::string wholeLineString;
  std::getline( file, wholeLineString );  
  commaLocOld = 0;

  for ( unsigned int i=0; i<numZvals; i++ )
    {
      if ( i == 0 )
        {
  	  commaLocNew = wholeLineString.find( ',', 0 );
  	  meanZValues.push_back( atof( wholeLineString.substr( 0, commaLocNew-commaLocOld+1).c_str() ) );
        }
      else
        {
  	  commaLocNew = wholeLineString.find( ',', commaLocOld+1 );
  	  meanZValues.push_back( atof( wholeLineString.substr( commaLocOld+1, commaLocNew-commaLocOld-1).c_str() ) );
        }
      
      commaLocOld = commaLocNew;
    }

  // Read in the eigenvalues and mode weights
  double eigenvalueSum = 0.0;
  std::vector< double > eigenvalues;
  std::vector< double > modeWeights;

  for ( unsigned int i=0; i<this->ShapeModel->GetNumberOfModes(); i++ )
    {
      std::string valueAndWeightString;
      std::getline( file, valueAndWeightString );
      
      commaLocOld = 0;
      for ( unsigned int j=0; j<2; j++ )
        {
  	  if ( j == 0 )
  	    {
  	      commaLocNew = valueAndWeightString.find( ',', 0 );
  	      double eigenvalue = atof( valueAndWeightString.substr( 0, commaLocNew-commaLocOld+1).c_str() );
  	      eigenvalues.push_back( eigenvalue );
  	      eigenvalueSum += eigenvalue;
  	    }
  	  else
  	    {
  	      commaLocNew = valueAndWeightString.find( ',', commaLocOld+1 );
  	      modeWeights.push_back( atof( valueAndWeightString.substr( commaLocOld+1, commaLocNew-commaLocOld-1).c_str() ) );
  	    }
	  
  	  commaLocOld = commaLocNew;
        }
    }
  this->ShapeModel->SetEigenvalueSum( eigenvalueSum );
  this->ShapeModel->SetEigenvalues( &eigenvalues );
  this->ShapeModel->SetModeWeights( &modeWeights );

  // Read in each of the modes
  std::vector< std::vector< double > > eigenvectors;
  
  for ( unsigned int i=0; i<this->ShapeModel->GetNumberOfModes(); i++ )
    {
      std::string eigenvectorString;
      std::getline( file, eigenvectorString );
      std::vector< double > eigenvector;
      
      commaLocOld = 0;
      for ( unsigned int j=0; j<numZvals; j++ )
        {
  	  if ( j == 0 )
  	    {
  	      commaLocNew = eigenvectorString.find( ',', 0 );
  	      eigenvector.push_back( atof( eigenvectorString.substr( 0, commaLocNew-commaLocOld+1).c_str() ) );
  	    }
  	  else
  	    {
  	      commaLocNew = eigenvectorString.find( ',', commaLocOld+1 );
  	      eigenvector.push_back( atof( eigenvectorString.substr( commaLocOld+1, commaLocNew-commaLocOld-1).c_str() ) );
  	    }
	  
  	  commaLocOld = commaLocNew;
        }
      
      eigenvectors.push_back( eigenvector );
    }
  this->ShapeModel->SetEigenvectors( &eigenvectors );

  // Read the domain points and fill the mean surface points vec
  std::vector< cip::PointType > meanSurfacePoints;
  
  for ( unsigned int i=0; i<numZvals; i++ )
    {
      std::string domainString;
      std::getline( file, domainString );
 
      unsigned int commaLoc1 = domainString.find( ',', 0 );
      unsigned int commaLoc2 = domainString.find( ',', commaLoc1+1 );
      
      double x = atof( domainString.substr( 0, commaLoc1).c_str() );
      double y = atof( domainString.substr( commaLoc1+1, commaLoc2-commaLoc1-1).c_str() );
      double z = meanZValues[i];
      
      cip::PointType point(3);
        point[0] = x;
  	point[1] = y;
  	point[2] = z;

      meanSurfacePoints.push_back( point );
    }
  
  this->ShapeModel->SetMeanSurfacePoints( meanSurfacePoints );
  file.close(); 
}


cipLobeSurfaceModel* LobeSurfaceModelIO::GetOutput()
{
  return this->ShapeModel;
}
