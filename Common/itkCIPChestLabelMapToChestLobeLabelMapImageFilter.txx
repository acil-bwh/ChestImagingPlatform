#ifndef _itkCIPChestLabelMapToChestLobeLabelMapImageFilter_txx
#define _itkCIPChestLabelMapToChestLobeLabelMapImageFilter_txx

#include "itkCIPChestLabelMapToChestLobeLabelMapImageFilter.h"
#include "itkExtractLungLabelMapImageFilter.h"
#include "cipChestConventions.h"


namespace itk
{


CIPChestLabelMapToChestLobeLabelMapImageFilter
::CIPChestLabelMapToChestLobeLabelMapImageFilter()
{
  this->m_InputIsLeftLungRightLung = false;

  this->m_LeftObliqueThinPlateSplineSurface     = new ThinPlateSplineSurface;
  this->m_RightObliqueThinPlateSplineSurface    = new ThinPlateSplineSurface;
  this->m_RightHorizontalThinPlateSplineSurface = new ThinPlateSplineSurface;
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::GenerateData()
{
  //
  // Allocate the output buffer
  //
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer( 0 );

  LungConventions conventions;

  InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
  InputImageType::PointType   origin  = this->GetInput()->GetOrigin();
  InputImageType::SizeType    size    = this->GetInput()->GetBufferedRegion().GetSize();  
  InputImageType::PointType   point;
  InputImageType::IndexType   index;

  int loZ, roZ, rhZ;  // The z index values for each of the fissures
  unsigned short newValue;
  unsigned char lungRegion, lungType;

  for ( int x=0; x<size[0]; x++ )
    {
    for ( int y=0; y<size[1]; y++ )
      {
      point[0] = static_cast< double >(x)*spacing[0] + origin[0];
      point[1] = static_cast< double >(y)*spacing[1] + origin[1];

      loZ = -1;   roZ = -1;   rhZ = -1;

      if ( this->m_LeftObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 )
        {
        point[2] = this->m_LeftObliqueThinPlateSplineSurface->GetSurfaceHeight( point[0], point[1] );
        loZ   = static_cast< unsigned int >( (point[2] - origin[2])/spacing[2] );
        }
      if ( this->m_RightObliqueThinPlateSplineSurface->GetNumberSurfacePoints() > 0 )
        {
        point[2] = this->m_RightObliqueThinPlateSplineSurface->GetSurfaceHeight( point[0], point[1] );
        roZ   = static_cast< unsigned int >( (point[2] - origin[2])/spacing[2] );
        }
      if ( this->m_RightHorizontalThinPlateSplineSurface->GetNumberSurfacePoints() > 0 )
        {
        point[2] = this->m_RightHorizontalThinPlateSplineSurface->GetSurfaceHeight( point[0], point[1] );
        rhZ   = static_cast< unsigned int >( (point[2] - origin[2])/spacing[2] );
        }

      for ( int z=0; z<size[2]; z++ )
        {
        index[0] = x;   index[1] = y;   index[2] = z;

        this->GetOutput()->SetPixel( index, this->GetInput()->GetPixel( index ) );

        if ( this->GetInput()->GetPixel( index ) != 0 )
          {
          lungRegion = conventions.GetLungRegionFromValue( this->GetInput()->GetPixel( index ) );
          lungType   = conventions.GetLungTypeFromValue( this->GetInput()->GetPixel( index ) );

          if ( conventions.CheckSubordinateSuperiorLungRegionRelationship( lungRegion, static_cast < unsigned char >( LEFTLUNG ) ) )
            {
            if ( z < loZ )
              {
              lungRegion = static_cast< unsigned char >( LEFTINFERIORLOBE );
              }
            else if ( z == loZ )
              {
              lungRegion = static_cast< unsigned char >( LEFTLUNG );
              lungType   = static_cast< unsigned char >( OBLIQUEFISSURE );
              }
            else
              {
              lungRegion = static_cast< unsigned char >( LEFTSUPERIORLOBE );
              }

            newValue = conventions.GetValueFromLungRegionAndType( lungRegion, lungType );

            this->GetOutput()->SetPixel( index, newValue );
            }
          else if ( conventions.CheckSubordinateSuperiorLungRegionRelationship( lungRegion, static_cast < unsigned char >( RIGHTLUNG ) ) )
            {
            if ( z <= roZ )
              {
              lungRegion = static_cast< unsigned char >( RIGHTINFERIORLOBE );
              }
            else if ( z > roZ && z <= rhZ )
              {
              lungRegion = static_cast< unsigned char >( RIGHTMIDDLELOBE );
              }
            else
              {
              lungRegion = static_cast< unsigned char >( RIGHTSUPERIORLOBE );
              }
            if ( z == roZ )
              {
              lungRegion = static_cast< unsigned char >( RIGHTLUNG );
              lungType   = static_cast< unsigned char >( OBLIQUEFISSURE );
              }
            if ( z == rhZ && lungRegion == static_cast< unsigned char >( RIGHTMIDDLELOBE ) )
              {
              lungRegion = static_cast< unsigned char >( RIGHTLUNG );
              lungType   = static_cast< unsigned char >( HORIZONTALFISSURE );
              }

            newValue = conventions.GetValueFromLungRegionAndType( lungRegion, lungType );

            this->GetOutput()->SetPixel( index, newValue );
            }
          }
        }
      }
    }
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetLeftObliqueThinPlateSplineSurface( ThinPlateSplineSurface* tps )
{
  this->m_LeftObliqueThinPlateSplineSurface = tps;
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetRightObliqueThinPlateSplineSurface( ThinPlateSplineSurface* tps )
{
  this->m_RightObliqueThinPlateSplineSurface = tps;
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetRightHorizontalThinPlateSplineSurface( ThinPlateSplineSurface* tps )
{
  this->m_RightHorizontalThinPlateSplineSurface = tps;
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetLeftObliqueFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->m_LeftObliqueFissureIndices.push_back( indices[i] );
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

      this->m_LeftObliqueFissurePoints.push_back( point );
      }  

    this->m_LeftObliqueThinPlateSplineSurface->SetSurfacePoints( &this->m_LeftObliqueFissurePoints );
    }
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetLeftObliqueFissurePoints( std::vector< double* >* points )
{
  for ( unsigned int i=0; i<points->size(); i++ )
    {
    this->m_LeftObliqueFissurePoints.push_back( (*points)[i] );
    }

  this->m_LeftObliqueThinPlateSplineSurface->SetSurfacePoints( &this->m_LeftObliqueFissurePoints );
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetRightObliqueFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->m_RightObliqueFissureIndices.push_back( indices[i] );
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

      this->m_RightObliqueFissurePoints.push_back( point );
      }  

    this->m_RightObliqueThinPlateSplineSurface->SetSurfacePoints( &this->m_RightObliqueFissurePoints );
    }
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetRightObliqueFissurePoints( std::vector< double* >* points )
{
  for ( unsigned int i=0; i<points->size(); i++ )
    {
    this->m_RightObliqueFissurePoints.push_back( (*points)[i] );
    }

  this->m_RightObliqueThinPlateSplineSurface->SetSurfacePoints( &this->m_RightObliqueFissurePoints );
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetRightHorizontalFissureIndices( std::vector< InputImageType::IndexType > indices )
{
  Superclass::InputImagePointer inputPtr = const_cast< InputImageType* >( this->GetInput() );

  for ( unsigned int i=0; i<indices.size(); i++ )
    {
    this->m_RightHorizontalFissureIndices.push_back( indices[i] );
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

      this->m_RightHorizontalFissurePoints.push_back( point );
      }  

    this->m_RightHorizontalThinPlateSplineSurface->SetSurfacePoints( &this->m_RightHorizontalFissurePoints );
    }
}


void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::SetRightHorizontalFissurePoints( std::vector< double* >* points )
{
  for ( unsigned int i=0; i<points->size(); i++ )
    {
    this->m_RightHorizontalFissurePoints.push_back( (*points)[i] );
    }

  this->m_RightHorizontalThinPlateSplineSurface->SetSurfacePoints( &this->m_RightHorizontalFissurePoints );
}


/**
 * Standard "PrintSelf" method
 */
void
CIPChestLabelMapToChestLobeLabelMapImageFilter
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

} // end namespace itk

#endif
