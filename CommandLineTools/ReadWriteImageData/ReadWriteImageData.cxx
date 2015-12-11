#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "ReadWriteImageDataCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;    

  if (inLabelMapFileName.compare("NA") != 0)
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

    if (outLabelMapFileName.compare("NA") != 0)
      {
      std::cout << "Writing label map..." << std::endl;
      cip::LabelMapWriterType::Pointer labelMapWriter = cip::LabelMapWriterType::New();
	labelMapWriter->SetInput(labelMapReader->GetOutput());
	labelMapWriter->SetFileName(outLabelMapFileName);
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
  
  if (inCTFileName.compare("NA") != 0)
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

    if (outCTFileName.compare("NA") != 0)
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
