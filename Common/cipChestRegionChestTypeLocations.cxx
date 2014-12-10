#include "cipChestRegionChestTypeLocations.h"
#include "cipExceptionObject.h"

cipChestRegionChestTypeLocations::cipChestRegionChestTypeLocations()
{
  this->NumberOfTuples = 0;
}


cipChestRegionChestTypeLocations::~cipChestRegionChestTypeLocations()
{
  this->Locations.clear();
  this->ChestRegions.clear();
  this->ChestTypes.clear();
}

void cipChestRegionChestTypeLocations::SetChestRegionChestTypeLocation( unsigned char cipRegion, unsigned char cipType, double const* point )
{
  this->NumberOfTuples++;

  double* tempPoint = new double[3];
    tempPoint[0] = point[0];
    tempPoint[1] = point[1];
    tempPoint[2] = point[2];

  this->Locations.push_back( tempPoint );
  this->ChestRegions.push_back( cipRegion );
  this->ChestTypes.push_back( cipType );
}


void cipChestRegionChestTypeLocations::SetChestRegionChestTypeLocation( unsigned char cipRegion, unsigned char cipType, unsigned int const* index )
{
  this->NumberOfTuples++;

  double* tempPoint = new double[3];
    tempPoint[0] = static_cast< double >( index[0] );
    tempPoint[1] = static_cast< double >( index[1] );
    tempPoint[2] = static_cast< double >( index[2] );

  this->Locations.push_back( tempPoint );
  this->ChestRegions.push_back( cipRegion );
  this->ChestTypes.push_back( cipType );
}


void cipChestRegionChestTypeLocations::GetLocation( unsigned int whichPoint, cip::PointType& location ) const
{
  if ( whichPoint >= this->NumberOfTuples )
    {
     throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cipChestRegionChestTypeLocations::GetLocation( unsigned int, cip::PointType )", 
				  "Requested invalid point" );
    }

  location[0] = this->Locations[whichPoint][0];
  location[1] = this->Locations[whichPoint][1];
  location[2] = this->Locations[whichPoint][2];
}


void cipChestRegionChestTypeLocations::GetLocation( unsigned int whichIndex, unsigned int* location ) const
{
  if ( whichIndex >= this->NumberOfTuples )
    {
    location = NULL;
    }

  location[0] = static_cast< unsigned int >( this->Locations[whichIndex][0] );
  location[1] = static_cast< unsigned int >( this->Locations[whichIndex][1] );
  location[2] = static_cast< unsigned int >( this->Locations[whichIndex][2] );
}


unsigned char cipChestRegionChestTypeLocations::GetChestRegionValue( unsigned int whichTuple ) const
{
  if ( whichTuple >= this->NumberOfTuples )
    {
    return static_cast< unsigned char >( cip::UNDEFINEDREGION );
    }

  return this->ChestRegions[whichTuple];
}


unsigned char cipChestRegionChestTypeLocations::GetChestTypeValue( unsigned int whichTuple ) const
{
  if ( whichTuple >= this->NumberOfTuples )
    {
    return static_cast< unsigned char >( cip::UNDEFINEDTYPE );
    }

  return this->ChestTypes[whichTuple];
}


std::string cipChestRegionChestTypeLocations::GetChestRegionName( unsigned int whichTuple ) const
{
  if ( whichTuple >= this->NumberOfTuples )
    {
    return "UNDEFINEDREGION";
    }

  return this->Conventions.GetChestRegionName( this->ChestRegions[whichTuple] );
}


std::string cipChestRegionChestTypeLocations::GetChestTypeName( unsigned int whichTuple ) const
{
  if ( whichTuple >= this->NumberOfTuples )
    {
    return "UNDEFINEDTYPE";
    }

  return this->Conventions.GetChestTypeName( this->ChestTypes[whichTuple] );
}


void cipChestRegionChestTypeLocations::GetPolyDataFromChestRegionChestTypeDesignation( vtkSmartPointer< vtkPolyData > polyData, 
										       unsigned char cipRegion, unsigned char cipType )
{
  vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();

  for ( unsigned int i=0; i<this->Locations.size(); i++ )
    {
      if ( this->ChestRegions[i] == cipRegion && this->ChestTypes[i] == cipType )
	{
	  points->InsertNextPoint( this->Locations[i][0], this->Locations[i][1], this->Locations[i][2] );
	}
    }

  polyData->SetPoints( points ); 
}


void cipChestRegionChestTypeLocations::GetPolyDataFromChestRegionDesignation( vtkSmartPointer< vtkPolyData > polyData, unsigned char cipRegion )
{
  vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();

  for ( unsigned int i=0; i<this->Locations.size(); i++ )
    {
      if ( this->ChestRegions[i] == cipRegion )
	{
	  points->InsertNextPoint( this->Locations[i][0], this->Locations[i][1], this->Locations[i][2] );
	}
    }

  polyData->SetPoints( points ); 
}


void cipChestRegionChestTypeLocations::GetPolyDataFromChestTypeDesignation( vtkSmartPointer< vtkPolyData > polyData, unsigned char cipType )
{
  vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();

  for ( unsigned int i=0; i<this->Locations.size(); i++ )
    {
      if ( this->ChestTypes[i] == cipType )
	{
	  points->InsertNextPoint( this->Locations[i][0], this->Locations[i][1], this->Locations[i][2] );
	}
    }

  polyData->SetPoints( points ); 
}
