#ifndef _cipLabelMapToLungLobeLabelMapImageFilter_txx
#define _cipLabelMapToLungLobeLabelMapImageFilter_txx

#include "cipLabelMapToLungLobeLabelMapImageFilter.h"
#include "cipChestConventions.h"


cipLabelMapToLungLobeLabelMapImageFilter
::cipLabelMapToLungLobeLabelMapImageFilter()
{
  this->LeftObliqueThinPlateSplineSurface     = new cipThinPlateSplineSurface;
  this->RightObliqueThinPlateSplineSurface    = new cipThinPlateSplineSurface;
  this->RightHorizontalThinPlateSplineSurface = new cipThinPlateSplineSurface;
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::GenerateData()
{
  //
  // Allocate the output buffer
  //
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer( 0 );

  cip::ChestConventions conventions;

  InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
  InputImageType::PointType   origin  = this->GetInput()->GetOrigin();
  InputImageType::SizeType    size    = this->GetInput()->GetBufferedRegion().GetSize();  
  InputImageType::PointType   point;
  InputImageType::IndexType   index;

  int loZ, roZ, rhZ;  // The z index values for each of the fissures
  unsigned short newValue;
  unsigned char cipRegion, cipType;

  for ( int x=0; x < static_cast< int >( size[0] ); x++ )
    {
    for ( int y=0; y < static_cast< int >( size[1] ); y++ )
      {
      point[0] = static_cast< double >(x)*spacing[0] + origin[0];
      point[1] = static_cast< double >(y)*spacing[1] + origin[1];

      loZ = -1;   roZ = -1;   rhZ = -1;

      if ( this->LeftObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 )
        {
        point[2] = this->LeftObliqueThinPlateSplineSurface->GetSurfaceHeight( point[0], point[1] );
        loZ   = static_cast< unsigned int >( (point[2] - origin[2])/spacing[2] );
        }
      if ( this->RightObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 )
        {
        point[2] = this->RightObliqueThinPlateSplineSurface->GetSurfaceHeight( point[0], point[1] );
        roZ   = static_cast< unsigned int >( (point[2] - origin[2])/spacing[2] );
        }
      if ( this->RightHorizontalThinPlateSplineSurface->GetNumberSurfacePoints() > 0 )
        {
        point[2] = this->RightHorizontalThinPlateSplineSurface->GetSurfaceHeight( point[0], point[1] );
        rhZ   = static_cast< unsigned int >( (point[2] - origin[2])/spacing[2] );
        }

      for ( int z=0; z < static_cast< int >( size[2] ); z++ )
        {
        index[0] = x;   index[1] = y;   index[2] = z;

        this->GetOutput()->SetPixel( index, this->GetInput()->GetPixel( index ) );

        if ( this->GetInput()->GetPixel( index ) != 0 )
          {
          cipRegion = conventions.GetChestRegionFromValue( this->GetInput()->GetPixel( index ) );
          cipType   = conventions.GetChestTypeFromValue( this->GetInput()->GetPixel( index ) );

          if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast < unsigned char >( cip::LEFTLUNG ) ) )
            {
            if ( z < loZ )
              {
              cipRegion = static_cast< unsigned char >( cip::LEFTINFERIORLOBE );
              }
            else if ( z == loZ )
              {
              cipRegion = static_cast< unsigned char >( cip::LEFTLUNG );
              cipType   = static_cast< unsigned char >( cip::OBLIQUEFISSURE );
              }
            else
              {
              cipRegion = static_cast< unsigned char >( cip::LEFTSUPERIORLOBE );
              }

            newValue = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );

            this->GetOutput()->SetPixel( index, newValue );
            }
          else if ( conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, static_cast < unsigned char >( cip::RIGHTLUNG ) ) )
            {
            if ( z <= roZ )
              {
              cipRegion = static_cast< unsigned char >( cip::RIGHTINFERIORLOBE );
              }
            else if ( z > roZ && z <= rhZ )
              {
              cipRegion = static_cast< unsigned char >( cip::RIGHTMIDDLELOBE );
              }
            else
              {
              cipRegion = static_cast< unsigned char >( cip::RIGHTSUPERIORLOBE );
              }
            if ( z == roZ )
              {
              cipRegion = static_cast< unsigned char >( cip::RIGHTLUNG );
              cipType   = static_cast< unsigned char >( cip::OBLIQUEFISSURE );
              }
            if ( z == rhZ && cipRegion == static_cast< unsigned char >( cip::RIGHTMIDDLELOBE ) )
              {
              cipRegion = static_cast< unsigned char >( cip::RIGHTLUNG );
              cipType   = static_cast< unsigned char >( cip::HORIZONTALFISSURE );
              }

            newValue = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );

            this->GetOutput()->SetPixel( index, newValue );
            }
          }
        }
      }
    }
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetLeftObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* tps )
{
  this->LeftObliqueThinPlateSplineSurface = tps;
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* tps )
{
  this->RightObliqueThinPlateSplineSurface = tps;
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightHorizontalThinPlateSplineSurface( cipThinPlateSplineSurface* tps )
{
  this->RightHorizontalThinPlateSplineSurface = tps;
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetLeftObliqueFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->LeftObliqueFissureIndices.push_back( indices[i] );
    }
  if ( inputPtr.IsNotNull() )
    {
    InputImageType::PointType physicalPoint;

    for ( unsigned int i=0; i<indices.size(); i++ )
      {
      this->GetInput()->TransformIndexToPhysicalPoint( indices[i], physicalPoint );
    
      double* point = new double[3];
        point[0] = physicalPoint[0];
        point[1] = physicalPoint[1];
        point[2] = physicalPoint[2];

      this->LeftObliqueFissurePoints.push_back( point );
      }  

    this->LeftObliqueThinPlateSplineSurface->SetSurfacePoints( &this->LeftObliqueFissurePoints );
    }
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetLeftObliqueFissurePoints( std::vector< double* >* points )
{
  for ( unsigned int i=0; i<points->size(); i++ )
    {
    this->LeftObliqueFissurePoints.push_back( (*points)[i] );
    }

  this->LeftObliqueThinPlateSplineSurface->SetSurfacePoints( &this->LeftObliqueFissurePoints );
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightObliqueFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->RightObliqueFissureIndices.push_back( indices[i] );
    }
  if ( inputPtr.IsNotNull() )
    {
    InputImageType::PointType physicalPoint;

    for ( unsigned int i=0; i<indices.size(); i++ )
      {
      this->GetInput()->TransformIndexToPhysicalPoint( indices[i], physicalPoint );
    
      double* point = new double[3];
        point[0] = physicalPoint[0];
        point[1] = physicalPoint[1];
        point[2] = physicalPoint[2];

      this->RightObliqueFissurePoints.push_back( point );
      }  

    this->RightObliqueThinPlateSplineSurface->SetSurfacePoints( &this->RightObliqueFissurePoints );
    }
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightObliqueFissurePoints( std::vector< double* >* points )
{
  for ( unsigned int i=0; i<points->size(); i++ )
    {
    this->RightObliqueFissurePoints.push_back( (*points)[i] );
    }

  this->RightObliqueThinPlateSplineSurface->SetSurfacePoints( &this->RightObliqueFissurePoints );
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightHorizontalFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->RightHorizontalFissureIndices.push_back( indices[i] );
    }
  if ( inputPtr.IsNotNull() )
    {
    InputImageType::PointType physicalPoint;

    for ( unsigned int i=0; i<indices.size(); i++ )
      {
      this->GetInput()->TransformIndexToPhysicalPoint( indices[i], physicalPoint );
    
      double* point = new double[3];
        point[0] = physicalPoint[0];
        point[1] = physicalPoint[1];
        point[2] = physicalPoint[2];

      this->RightHorizontalFissurePoints.push_back( point );
      }  

    this->RightHorizontalThinPlateSplineSurface->SetSurfacePoints( &this->RightHorizontalFissurePoints );
    }
}


void
cipLabelMapToLungLobeLabelMapImageFilter
::SetRightHorizontalFissurePoints( std::vector< double* >* points )
{
  for ( unsigned int i=0; i<points->size(); i++ )
    {
    this->RightHorizontalFissurePoints.push_back( (*points)[i] );
    }

  this->RightHorizontalThinPlateSplineSurface->SetSurfacePoints( &this->RightHorizontalFissurePoints );
}


/**
 * Standard "PrintSelf" method
 */
void
cipLabelMapToLungLobeLabelMapImageFilter
::PrintSelf(
  std::ostream& os, 
  itk::Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

#endif
