/** \file
 *  \ingroup sandbox 
 *  \details This simple program
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "itkImageRegionIteratorWithIndex.h"

typedef itk::ImageRegionIteratorWithIndex< cip::CTType > IteratorType;

int main( int argc, char *argv[] )
{
  // Begin by defining the arguments to be passed
  std::string reconFileName1 = "NA";
  std::string reconFileName2 = "NA";
  std::string outFileName    = "NA";

  // Input argument descriptions for user help
  std::string programDesc = "This simple program.";

  std::string reconFileName1Desc = "File name for the first recon-kernel image.";
  std::string reconFileName2Desc = "File name for the second recon-kernel image.";
  std::string outFileNameDesc = "File name for the filtered output.";

  // Parse the input arguments
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision$" );

    TCLAP::ValueArg<std::string> reconFileName1Arg( "", "r1", reconFileName1Desc, true, reconFileName1, "string", cl );
    TCLAP::ValueArg<std::string> reconFileName2Arg( "", "r2", reconFileName2Desc, true, reconFileName2, "string", cl );
    TCLAP::ValueArg<std::string> outFileNameArg( "o", "", outFileNameDesc, true, outFileName, "string", cl );

    cl.parse( argc, argv );

    reconFileName1 = reconFileName1Arg.getValue();
    reconFileName2 = reconFileName2Arg.getValue();
    outFileName    = outFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  std::cout << "Reading recon 1..." << std::endl;
  cip::CTReaderType::Pointer reader1 = cip::CTReaderType::New();
    reader1->SetFileName( reconFileName1 );
  try
    {
    reader1->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading recon 1:";
    std::cerr << excp << std::endl;
    }

  cip::CTType::SizeType size = reader1->GetOutput()->GetBufferedRegion().GetSize();

  std::cout << "Reading recon 2..." << std::endl;
  cip::CTReaderType::Pointer reader2 = cip::CTReaderType::New();
    reader2->SetFileName( reconFileName2 );
  try
    {
    reader2->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading recon 2:";
    std::cerr << excp << std::endl;
    }

  // Instead of allocating memory for a new image, just filter and overwrite
  // recon 1
  std::cout << "Merging recon kernels..." << std::endl;
  IteratorType it1( reader1->GetOutput(), reader1->GetOutput()->GetBufferedRegion() );
  IteratorType it2( reader2->GetOutput(), reader2->GetOutput()->GetBufferedRegion() );

  it1.GoToBegin();
  it2.GoToBegin();
  while ( !it1.IsAtEnd() )
    {
    double v1 = double(it1.Get());
    double v2 = double(it2.Get());
    double filtered = 0.5*v1 + 0.5*v2;

    it1.Set( short(filtered) );

    ++it1;
    ++it2;
    }

  // Now median filter. Filter over a diamond structuring element (ITK does
  // not permit this structuring element).
  cip::CTType::Pointer medianFiltered = cip::CTType::New();
    medianFiltered->SetRegions( reader1->GetOutput()->GetBufferedRegion().GetSize() );
    medianFiltered->Allocate();
    medianFiltered->FillBuffer( -1024 );
    medianFiltered->SetOrigin( reader1->GetOutput()->GetOrigin() );
    medianFiltered->SetSpacing( reader1->GetOutput()->GetSpacing() );

  cip::CTType::IndexType index;

  IteratorType mIt( medianFiltered, medianFiltered->GetBufferedRegion() );
  IteratorType fIt( reader1->GetOutput(), reader1->GetOutput()->GetBufferedRegion() );

  std::cout << "Median filtering..." << std::endl;

  unsigned int totVoxels = size[0]*size[1]*size[2];
  unsigned int inc = 0;

  mIt.GoToBegin();
  fIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    std::list< short > valueList;  

    if ( mIt.GetIndex()[0] > 0 && mIt.GetIndex()[0] < size[0] -2 &&
	 mIt.GetIndex()[1] > 0 && mIt.GetIndex()[1] < size[1] -2 &&
	 mIt.GetIndex()[2] > 0 && mIt.GetIndex()[2] < size[2] -2 )
      {
	index[0] = mIt.GetIndex()[0];      index[1] = mIt.GetIndex()[1];      index[2] = mIt.GetIndex()[2];
	valueList.push_back( reader1->GetOutput()->GetPixel( index ) );
	index[0] = mIt.GetIndex()[0] - 1;  index[1] = mIt.GetIndex()[1];      index[2] = mIt.GetIndex()[2];
	valueList.push_back( reader1->GetOutput()->GetPixel( index ) );
	index[0] = mIt.GetIndex()[0] + 1;  index[1] = mIt.GetIndex()[1];      index[2] = mIt.GetIndex()[2];
	valueList.push_back( reader1->GetOutput()->GetPixel( index ) );
	index[0] = mIt.GetIndex()[0];      index[1] = mIt.GetIndex()[1] - 1;  index[2] = mIt.GetIndex()[2];
	valueList.push_back( reader1->GetOutput()->GetPixel( index ) );
	index[0] = mIt.GetIndex()[0];      index[1] = mIt.GetIndex()[1] + 1;  index[2] = mIt.GetIndex()[2];
	valueList.push_back( reader1->GetOutput()->GetPixel( index ) );
	index[0] = mIt.GetIndex()[0];      index[1] = mIt.GetIndex()[1];      index[2] = mIt.GetIndex()[2] - 1;
	valueList.push_back( reader1->GetOutput()->GetPixel( index ) );
	index[0] = mIt.GetIndex()[0];      index[1] = mIt.GetIndex()[1];      index[2] = mIt.GetIndex()[2] + 1;
	valueList.push_back( reader1->GetOutput()->GetPixel( index ) );

	valueList.sort();
	
	std::list< short >::iterator listIt = valueList.begin();
	++listIt; ++listIt; ++listIt;

	mIt.Set( *listIt );
      }

    ++mIt;
    ++fIt;

    inc++;
    if ( inc%10000000 == 0 )
      {
	std::cout << double(inc)/double(totVoxels) << std::endl;
      }
    }

  std::cout << "Writing filtered image..." << std::endl;
  cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
    writer->SetInput( medianFiltered );
    writer->SetFileName( outFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing filtered image:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
