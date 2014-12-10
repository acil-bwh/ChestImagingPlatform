#include "cipChestRegionChestTypeLocationsIO.h"
#include <fstream>
#include <stdlib.h>


cipChestRegionChestTypeLocationsIO::cipChestRegionChestTypeLocationsIO()
{
  this->FileName = "NA";
  this->RegionTypeLocations = new cipChestRegionChestTypeLocations();
}


cipChestRegionChestTypeLocationsIO::~cipChestRegionChestTypeLocationsIO()
{
  delete this->RegionTypeLocations;
}


void cipChestRegionChestTypeLocationsIO::SetFileName( std::string fileName )
{
  this->FileName = fileName;
}


bool cipChestRegionChestTypeLocationsIO::Read()
{
  std::ifstream file( this->FileName.c_str() );

  if ( !file )
    {
    return false;
    }

  std::string wholeLineString;
  std::getline( file, wholeLineString ); // Gobble header line

  while ( !file.eof() )
    {
      std::getline( file, wholeLineString );

      //check if the line is empty. If so, disregard
      if(wholeLineString.length() > 1)
	{
	  unsigned int commaLoc1 = wholeLineString.find( ',' );
	  unsigned int commaLoc2 = wholeLineString.find( ',', commaLoc1+1 );    
	  unsigned int commaLoc3 = wholeLineString.find( ',', commaLoc2+1 );
	  unsigned int commaLoc4 = wholeLineString.find( ',', commaLoc3+1 );

	  unsigned char cipRegion = this->Conventions.GetChestRegionValueFromName( wholeLineString.substr( 0, commaLoc1 ) );
	  unsigned char cipType   = this->Conventions.GetChestTypeValueFromName( wholeLineString.substr( commaLoc1+1, commaLoc2-commaLoc1-1 ) );

	  double* location = new double[3];
	  location[0] = static_cast< double >( atof( wholeLineString.substr( commaLoc2+1, commaLoc3-commaLoc2-1 ).c_str() ) );
	  location[1] = static_cast< double >( atof( wholeLineString.substr( commaLoc3+1, commaLoc4-commaLoc3-1 ).c_str() ) );
	  location[2] = static_cast< double >( atof( wholeLineString.substr( commaLoc4+1, wholeLineString.size()-commaLoc4-1 ).c_str() ) );
	  
	  this->RegionTypeLocations->SetChestRegionChestTypeLocation( cipRegion, cipType, location );
	}
    }
  file.close();

  return true;
}


void cipChestRegionChestTypeLocationsIO::Write() const
{
  std::ofstream file( this->FileName.c_str() );
  cip::PointType location(3);

  file << "Region,Type,X Location,Y Location, Z Location" << std::endl;
  for ( unsigned int i=0; i<this->RegionTypeLocations->GetNumberOfTuples(); i++ )
    {
    this->RegionTypeLocations->GetLocation( i, location );

    file << this->RegionTypeLocations->GetChestRegionName(i) << ",";
    file << this->RegionTypeLocations->GetChestTypeName(i) << ",";
    file << location[0] << "," << location[1] << "," << location[2] << std::endl;;
    }

  file.close();
}


cipChestRegionChestTypeLocations* cipChestRegionChestTypeLocationsIO::GetOutput()
{
  return this->RegionTypeLocations;
}


void cipChestRegionChestTypeLocationsIO::SetInput( cipChestRegionChestTypeLocations* const locations )
{
  this->RegionTypeLocations = locations;
}
