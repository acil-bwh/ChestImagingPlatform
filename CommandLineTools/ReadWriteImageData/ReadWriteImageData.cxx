/** \file
 *  \ingroup commandLineTools 
 *  \details This simple program reads and writes images, either 
 *  label maps or CT images. It is useful for renaming (obviating 
 *  the need to need to manually modify headers)
 *
 *  USAGE: 
 *
 *  ReadWriteImageData  [--oct <string>] [--ict <string>] [--ol <string>]
 *                      [--il <string>] [--] [--version] [-h]
 *
 *  Where: 
 *
 *   --oct <string>
 *     Output CT file name
 *
 *   --ict <string>
 *     Input CT file name
 *
 *   --ol <string>
 *     Output label map file name
 *
 *   --il <string>
 *     Input label map file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "ReadWriteImageDataCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
    

  if (inLabelMapFileName.compare("q") != 0)
    {
    std::cout << "Reading label map..." << std::endl;
    cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
      labelMapReader->SetFileName(inLabelMapFileName);
    try
      {
      labelMapReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading label map:";
      std::cerr << excp << std::endl;
      }

    if (outLabelMapFileName.compare("q") != 0)
      {
      std::cout << "Writing label map..." << std::endl;
      cip::LabelMapWriterType::Pointer labelMapWriter = cip::LabelMapWriterType::New();
        labelMapWriter->SetFileName(outLabelMapFileName);
	labelMapWriter->SetInput(labelMapReader->GetOutput());
	labelMapWriter->UseCompressionOn();
      try
	{
	labelMapWriter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught writing label map:";
	std::cerr << excp << std::endl;
	}
      }
    }
  
  if (inCTFileName.compare("q") != 0)
    {
    std::cout << "Reading CT..." << std::endl;
    cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
      ctReader->SetFileName(inCTFileName);
    try
      {
      ctReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading CT:";
      std::cerr << excp << std::endl;
      }

    if (outCTFileName.compare("q") != 0)
      {
      std::cout << "Writing CT..." << std::endl;
      cip::CTWriterType::Pointer ctWriter = cip::CTWriterType::New();
        ctWriter->SetFileName(outCTFileName);
	ctWriter->SetInput(ctReader->GetOutput());
	ctWriter->UseCompressionOn();
      try
	{
	ctWriter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught writing CT:";
	std::cerr << excp << std::endl;
	}
      }
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
