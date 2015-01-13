/** \file
 *  \ingroup commandLineTools 
 *  \details 
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "itkImageRegionIterator.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "RemapLabelMapCLP.h"

typedef itk::ImageRegionIterator< cip::LabelMapType > IteratorType;

int main( int argc, char *argv[] )
{    
  PARSE_ARGS;  

  if ( inLabels.size() != outLabels.size() )
    {
      std::cerr << "Must have the same number of input and output labels" << std::endl;
      return cip::ARGUMENTPARSINGERROR;
    }

  std::map< unsigned short, unsigned short > mapper;
  for ( unsigned int i=0; i<inLabels.size(); i++ )
    {
      mapper[(unsigned short)(inLabels[i])] = (unsigned short)(outLabels[i]);
    }
  
  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName( inLabelMap );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    return cip::NRRDREADFAILURE;
    }

  std::cout << "Remapping..." << std::endl;
  IteratorType it( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
      if ( mapper.find((unsigned short)(it.Get())) != mapper.end() )
	{
	  it.Set( mapper[(unsigned short)(it.Get())] );
	}

      ++it;
    }

  std::cout << "Writing label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New(); 
    writer->SetFileName( outLabelMap );
    writer->SetInput( reader->GetOutput() );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    return cip::NRRDREADFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
