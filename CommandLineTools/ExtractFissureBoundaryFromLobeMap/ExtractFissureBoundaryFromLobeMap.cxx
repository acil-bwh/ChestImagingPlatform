#include "cipHelper.cxx"
#include "ExtractFissureBoundaryFromLobeMapCLP.h"
#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"

int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  typedef itk::CastImageFilter<cip::LabelMapType,cip::LabelMapType> CastFilter;
  typedef itk::ConstNeighborhoodIterator<cip::LabelMapType> NeighborhoodIterator;
  typedef itk::ImageRegionIterator<cip::LabelMapType>       ImageIterator;
  
  // Read the CT image
  cip::LabelMapType::Pointer labelImage = cip::LabelMapType::New();

  if ( strcmp( inputLabelMap.c_str(), "NA") != 0 )
    {
    std::cout << "Reading Label Map from file..." << std::endl;
    labelImage = cip::ReadLabelMapFromFile( inputLabelMap );
    if (labelImage.GetPointer() == NULL)
        {
          return cip::NRRDREADFAILURE;
        }
    }
  else
    {
    std::cerr << "ERROR: No Label Map image specified" << std::endl;
    return cip::EXITFAILURE;
    }

  cip::ChestConventions *ChestConventions = new cip::ChestConventions();
  
  CastFilter::Pointer cast = CastFilter::New();
  cast->SetInput( labelImage );
  cast->Update();
  
  cip::LabelMapType::Pointer output_image = cast->GetOutput();
  
  //Set Radius of 1 in all directions
  NeighborhoodIterator::RadiusType radius;
  for (unsigned int i = 0; i < cip::LabelMapType::ImageDimension; ++i) radius[i] = 1;

  // Initializes the iterators on the input & output image regions
  NeighborhoodIterator nit(radius, labelImage,
                           output_image->GetRequestedRegion());
  ImageIterator out(output_image, output_image->GetRequestedRegion());
  
  // Iterates over the input and output
  for (nit.GoToBegin(), out.GoToBegin(); ! nit.IsAtEnd(); ++nit, ++out )
    {
      //Check just region info doing a casting to unsigned char
      unsigned char val = (unsigned char) out.Get();
      for (unsigned int i = 0; i < nit.Size(); ++i)
        {
           unsigned char testval = (unsigned char) nit.GetPixel(i);
           if (val != testval)
           {
             if ((val == cip::RIGHTSUPERIORLOBE && testval == cip::RIGHTMIDDLELOBE) ||
                 (val == cip::RIGHTMIDDLELOBE && testval == cip::RIGHTSUPERIORLOBE))
              {
                out.Set(ChestConventions->GetValueFromChestRegionAndType(val,cip::HORIZONTALFISSURE));
              }
             else if ( (val == cip::RIGHTSUPERIORLOBE && testval == cip::RIGHTINFERIORLOBE) ||
                      (val == cip::RIGHTINFERIORLOBE && testval == cip::RIGHTSUPERIORLOBE))
              {
               out.Set(ChestConventions->GetValueFromChestRegionAndType(val,cip::OBLIQUEFISSURE));
              }
             else if ( (val == cip::RIGHTMIDDLELOBE && testval == cip::RIGHTINFERIORLOBE) ||
                      (val == cip::RIGHTINFERIORLOBE && testval == cip::RIGHTMIDDLELOBE))
              {
                out.Set(ChestConventions->GetValueFromChestRegionAndType(val,cip::OBLIQUEFISSURE));
              }
          
             else if ( (val == cip::LEFTSUPERIORLOBE && testval == cip::LEFTINFERIORLOBE) ||
                      (val == cip::LEFTINFERIORLOBE && testval == cip::LEFTSUPERIORLOBE))
              {
                out.Set(ChestConventions->GetValueFromChestRegionAndType(val,cip::OBLIQUEFISSURE));
              }
    
            continue;
           }
        }
    }

  std::cout << "Writing fissure labeled image..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetInput( output_image );
    writer->SetFileName( outputLabelMap );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while writing fissure-labeled image:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}
