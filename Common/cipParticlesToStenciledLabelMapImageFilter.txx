/*
 * $Date: 2012-06-11 17:58:50 -0700 (Mon, 11 Jun 2012) $
 * $Revision: 156 $
 * $Author: jross $
 *
 */

#ifndef __cipParticlesToStenciledLabelMapImageFilter_txx
#define __cipParticlesToStenciledLabelMapImageFilter_txx

#include "cipParticlesToStenciledLabelMapImageFilter.h"


template < class TInputImage >
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::cipParticlesToStenciledLabelMapImageFilter()
{
  this->SelectedParticleType               = VALLEYLINE; // Corresponds to airway
  this->ParticlesData                      = vtkSmartPointer< vtkPolyData >::New();
  this->ScaleStencilPatternByParticleScale = false;
  this->ScaleStencilPatternByParticleDNNRadius = false;
  this->DNNRadiusName                      = "";
  this->CTPointSpreadFunctionSigma         = 0.0;
}


template< class TInputImage >
void
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::SetParticlesData( vtkSmartPointer< vtkPolyData > particlesData )
{  
  this->ParticlesData = particlesData;
}


template< class TInputImage >
vtkSmartPointer< vtkPolyData >
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::GetParticlesData()
{  
  return this->ParticlesData;
}


template< class TInputImage >
void
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::SetStencil( cipStencil* st )
{  
  this->Stencil = st;
}


template< class TInputImage >
void
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::GenerateData()
{	
  // Set the label that will be used for the foreground value in the
  // output image
  cip::ChestConventions conventions;
  unsigned short foregroundLabel = 
    conventions.GetValueFromChestRegionAndType( static_cast< unsigned char >( cip::UNDEFINEDREGION ), 
                                                static_cast< unsigned char >( this->ChestParticleType ) );

  typename Superclass::InputImageConstPointer inputPtr = this->GetInput();

  typename InputImageType::SizeType size = inputPtr->GetBufferedRegion().GetSize();
  
  // Allocate space for the output image
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();    
    outputPtr->FillBuffer( 0 );
    outputPtr->SetSpacing( inputPtr->GetSpacing() );
    outputPtr->SetOrigin( inputPtr->GetOrigin() );

  // Create an ITK region that will be modified for each of our
  // particles   
  typename InputImageType::RegionType imageRegion;
  typename InputImageType::SizeType   regionSize;
  typename InputImageType::IndexType  regionStartIndex;
  typename InputImageType::IndexType  regionEndIndex;
  
  // The bounding box start and end points will be updated using the
  // stencil  
  double* boundingBoxStartPoint = new double[3];
  double* boundingBoxEndPoint   = new double[3];

  typename InputImageType::PointType itkPoint; //A temp container

  // TODO: Can the following be multithreaded? By what mechanism?
  for ( unsigned int i=0; i<this->ParticlesData->GetNumberOfPoints(); i++ ) 
    {
      unsigned short typeValue = conventions.GetChestTypeFromValue(ParticlesData->GetPointData()->GetArray("ChestRegionChestType")->GetTuple(i)[0]);
      unsigned short foregroundLabel = conventions.GetValueFromChestRegionAndType( static_cast< unsigned char >( cip::UNDEFINEDREGION ),
                                                                                   static_cast< unsigned char >( typeValue ) );
    //
    // Get the bounding box for the particle. We will create an ITK region
    // based on the extent of this bounding box.
    //
    this->Stencil->SetCenter( this->ParticlesData->GetPoint(i)[0], this->ParticlesData->GetPoint(i)[1], 
                              this->ParticlesData->GetPoint(i)[2] );

    //
    // Handling of the stencil pattern is adjusted based on the type
    // of structure represented (vessel, airway, or fissure).
    // 
    if ( this->ChestParticleType == cip::AIRWAY )
      {
      //
      // Following call has no effect if sphere stencil is used, but
      // is needed in case cylinder stencil is used
      //
      this->Stencil->SetOrientation( this->ParticlesData->GetPointData()->GetArray("hevec2")->GetTuple(i)[0], 
                                     this->ParticlesData->GetPointData()->GetArray("hevec2")->GetTuple(i)[1], 
                                     this->ParticlesData->GetPointData()->GetArray("hevec2")->GetTuple(i)[2] );

      if ( this->ScaleStencilPatternByParticleScale )
        {
          double scale = this->ParticlesData->GetPointData()->GetArray("scale")->GetTuple(i)[0];
          double tempRadius = std::sqrt(2.0)*std::sqrt( pow( scale, 2 ) + pow( this->CTPointSpreadFunctionSigma, 2 ) );

          this->Stencil->SetRadius( tempRadius );
        }

      if ( this->DNNRadiusName != "" )
        {
          double radius = this->ParticlesData->GetPointData()->GetArray(this->DNNRadiusName.c_str())->GetTuple(i)[0];
//          double tempRadius = std::sqrt(2.0) * std::sqrt(pow(radius, 2) + pow(this->CTPointSpreadFunctionSigma, 2));

          this->Stencil->SetRadius(radius);
        }
      }
    if ( this->ChestParticleType == cip::FISSURE )
      {
      //
      // Following call has no effect if sphere stencil is used, but
      // is needed in case cylinder stencil is used
      //
      this->Stencil->SetOrientation( this->ParticlesData->GetPointData()->GetArray("hevec1")->GetTuple(i)[0], 
                                     this->ParticlesData->GetPointData()->GetArray("hevec1")->GetTuple(i)[1], 
                                     this->ParticlesData->GetPointData()->GetArray("hevec1")->GetTuple(i)[2] );
      }
    if ( this->ChestParticleType == cip::VESSEL )
      {
      //
      // Following call has no effect if sphere stencil is used, but
      // is needed in case cylinder stencil is used
      //
      this->Stencil->SetOrientation( this->ParticlesData->GetPointData()->GetArray("hevec0")->GetTuple(i)[0], 
                                     this->ParticlesData->GetPointData()->GetArray("hevec0")->GetTuple(i)[1], 
                                     this->ParticlesData->GetPointData()->GetArray("hevec0")->GetTuple(i)[2] );        

      //
      // For vessels, both the cylinder and sphere radii can be scaled
      // using the particle scale according to an equation that
      // relates the particle scale and CT point spread function to
      // the actual vessel radius
      //
      if ( this->ScaleStencilPatternByParticleScale )
        {
        double scale = this->ParticlesData->GetPointData()->GetArray("scale")->GetTuple(i)[0];
        double tempRadius = std::sqrt(2.0)*std::sqrt( pow( scale, 2 ) + pow( this->CTPointSpreadFunctionSigma, 2 ) );

        this->Stencil->SetRadius( tempRadius );
        }

      if ( this->DNNRadiusName != "" )
       {
        double radius = this->ParticlesData->GetPointData()->GetArray(this->DNNRadiusName.c_str())->GetTuple(i)[0];
        double tempRadius = std::sqrt(2.0) * std::sqrt(pow(radius, 2) + pow(this->CTPointSpreadFunctionSigma, 2));

        this->Stencil->SetRadius(tempRadius);
        }
      }
    //
    // Must be AFTER we set the center, orientation, and radius
    //
    this->Stencil->GetStencilBoundingBox( boundingBoxStartPoint, boundingBoxEndPoint );

    //
    // Convert the physical bounding box coordinates to ITK indices 
    //
    itkPoint[0] = boundingBoxStartPoint[0];
    itkPoint[1] = boundingBoxStartPoint[1];
    itkPoint[2] = boundingBoxStartPoint[2];

    inputPtr->TransformPhysicalPointToIndex( itkPoint, regionStartIndex );     

    itkPoint[0] = boundingBoxEndPoint[0];
    itkPoint[1] = boundingBoxEndPoint[1];
    itkPoint[2] = boundingBoxEndPoint[2];

    inputPtr->TransformPhysicalPointToIndex( itkPoint, regionEndIndex ); 

    regionStartIndex[0] >= size[0] ? regionStartIndex[0] = size[0]-1 : false;
    regionStartIndex[1] >= size[1] ? regionStartIndex[1] = size[1]-1 : false;
    regionStartIndex[2] >= size[2] ? regionStartIndex[2] = size[2]-1 : false;

    regionEndIndex[0] >= size[0] ? regionEndIndex[0] = size[0]-1 : false;
    regionEndIndex[1] >= size[1] ? regionEndIndex[1] = size[1]-1 : false;
    regionEndIndex[2] >= size[2] ? regionEndIndex[2] = size[2]-1 : false;

    regionStartIndex[0] < 0 ? regionStartIndex[0] = 0 : false;
    regionStartIndex[1] < 0 ? regionStartIndex[1] = 0 : false;
    regionStartIndex[2] < 0 ? regionStartIndex[2] = 0 : false;

    regionEndIndex[0] < 0 ? regionEndIndex[0] = 0 : false;
    regionEndIndex[1] < 0 ? regionEndIndex[1] = 0 : false;
    regionEndIndex[2] < 0 ? regionEndIndex[2] = 0 : false;

    //
    // Set up the ITK region extent and start index
    //
    regionSize[0] = regionEndIndex[0]-regionStartIndex[0]+1;
    regionSize[1] = regionEndIndex[1]-regionStartIndex[1]+1;
    regionSize[2] = regionEndIndex[2]-regionStartIndex[2]+1;

    // 
    // Now create the ITK region of interest over which to perform
    // local segmentation and connected component
    // identification. We'll need to convert from the bounding box 
    //
    imageRegion.SetSize( regionSize );
    imageRegion.SetIndex( regionStartIndex );

    //
    // Now iterate over the region. If the physical point is inside
    // the stencil pattern, set the foreground value
    //
    IteratorType it( outputPtr, imageRegion );

    it.GoToBegin();
    while ( !it.IsAtEnd() )
      {     
      inputPtr->TransformIndexToPhysicalPoint( it.GetIndex(), itkPoint );
      
      if ( this->Stencil->IsInsideStencilPattern( itkPoint[0], itkPoint[1], itkPoint[2] ) )
        {
        outputPtr->SetPixel( it.GetIndex(), foregroundLabel );
        }
     
      ++it;
      }
    }
}


template< class TInputImage >
void
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::SetParticleType( unsigned int particleType )
{  
  this->SelectedParticleType = particleType;
}


template< class TInputImage >
unsigned int
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::GetParticleType()
{  
  return this->SelectedParticleType;
}


template< class TInputImage >
void
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::SetChestParticleType( unsigned int particleType )
{  
  this->ChestParticleType = particleType;

  if ( particleType == cip::AIRWAY )
    {
    this->SelectedParticleType = VALLEYLINE;
    }
  else if ( particleType == cip::VESSEL )
    {
    this->SelectedParticleType = RIDGELINE;
    }
  else if ( particleType == cip::FISSURE || particleType == cip::OBLIQUEFISSURE || particleType == cip::HORIZONTALFISSURE )
    {
    this->SelectedParticleType = RIDGESURFACE;
    }
}


template< class TInputImage >
unsigned int
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::GetChestParticleType()
{  
  if ( this->SelectedParticleType == RIDGESURFACE )
    {
    //
    // Our current scheme doesn't allow disambiguation of fissure
    // types: all three fissure types, when set, are mapped to
    // 'RIDGESURFACE', so the most general return value is 'FISSURE'
    //
    return cip::FISSURE;
    }
  if ( this->SelectedParticleType == RIDGELINE )
    {
    return cip::VESSEL;
    }
  
  //
  // We must return a valid ChestType. 'AIRWAY' is the only one left
  // at this point
  //
  return cip::AIRWAY;
}


template< class TInputImage >
void
cipParticlesToStenciledLabelMapImageFilter< TInputImage >
::PrintSelf( std::ostream& os, itk::Indent indent ) const
{
}


#endif
