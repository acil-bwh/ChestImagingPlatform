#include "cipHelper.h"
#include "GenerateNLMFilteredImageCLP.h"
#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkPFNLMFilter.h"


int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  // Read the CT image
  cip::CTType::Pointer ctImage = cip::CTType::New();

  if ( strcmp( ctDir.c_str(), "NA") != 0 )
    {
    std::cout << "Reading CT from directory..." << std::endl;
    ctImage = cip::ReadCTFromDirectory( ctDir );
    if (ctImage.GetPointer() == NULL)
        {
        return cip::DICOMREADFAILURE;
        }
    }
  else if ( strcmp( ctFileName.c_str(), "NA") != 0 )
    {
    std::cout << "Reading CT from file..." << std::endl;
    ctImage = cip::ReadCTFromFile( ctFileName );
    if (ctImage.GetPointer() == NULL)
        {
          return cip::NRRDREADFAILURE;
        }
    }
  else
    {
    std::cerr << "ERROR: No CT image specified" << std::endl;
    return cip::EXITFAILURE;
    }
  
  typedef itk::PFNLMFilter< cip::CTType, cip::CTType > FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput( ctImage );
  
  /** SET PARAMETERS TO THE FILTER */
	// The power of noise:
  unsigned int DIMENSION=3;
	filter->SetSigma( iSigma );
	// The search radius
	FilterType::InputImageSizeType radius;
	for( unsigned int d=0; d<DIMENSION; ++d )
		radius[d] = iRadiusSearch[d];
	filter->SetRSearch( radius );
	// The comparison radius:
	for( unsigned int d=0; d<DIMENSION; ++d )
		radius[d] = iRadiusComp[d];
	filter->SetRComp( radius );
	// The "h" parameter:
	filter->SetH( iH );
	// The preselection threshold:
	filter->SetPSTh( iPs );
	
	// Run the filter:
	try
  {
		filter->Update();
  }
	catch ( itk::ExceptionObject & e )
  {
		std::cerr << "exception in filter" << std::endl;
		std::cerr << e.GetDescription() << std::endl;
		std::cerr << e.GetLocation() << std::endl;
		return cip::EXITFAILURE;
  }

  std::cout << "Writing filtered image..." << std::endl;
  cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
    writer->SetInput( filter->GetOutput() );
    writer->SetFileName( outputFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while writing filtered image:";
    std::cerr << excp << std::endl;
    return cip::EXITFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}
