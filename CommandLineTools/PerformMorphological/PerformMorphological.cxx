/** \file
 *  \ingroup commandLineTools 
 *  \details This ... 
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "cipHelper.h"

int main(int argc, char *argv[])
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string inFileName = "NA";
  std::string outFileName = "NA";
  bool close = false;
  bool open = false;
  bool dilate = false;
  bool erode = false;
  unsigned int kernelRadiusX = 1;
  unsigned int kernelRadiusY = 1;
  unsigned int kernelRadiusZ = 1;

  std::vector<unsigned char> regionsVec;
  std::vector<unsigned char> typesVec;

  //
  // Input argument descriptions for user help
  //
  std::string programDesc = "This simple";

  std::string inFileNameDesc = "Input label map file name";
  std::string outFileNameDesc = "Output label map file name";
  std::string regionsVecDesc = "Users must specify chest-region chest-type pairs \
only labels corresponding to the pairs will be operated on. Use this flag to specify \
the chest region of a given pair. Each time this flag is used, the -t or --type flag \
should also be used to specify the corresponding type";
  std::string typesVecDesc = "Users must specify chest-region chest-type pairs; \
only labels corresponding to the pairs will be operated on. Use this flag to specify \
the chest type of a given pair. Each time this flag is used, the -r or --region flag \
should also be used to specify the corresponding region";
  std::string dilateDesc = "Set to 1 to perform morphological dilation (0 by default). \
Only one morphological operation can be specified. Operations are considered in the \
following order: dilate, erode, open, close. E.g., if dilation is specified, only it will be \
performed regardless of whether the other operations have requested.";
  std::string erodeDesc = "Set to 1 to perform morphological erosion (0 by default). \
Only one morphological operation can be specified. Operations are considered in the \
following order: dilate, erode, open, close. E.g., if dilation is specified, only it will be \
performed regardless of whether the other operations have requested.";
  std::string openDesc = "Set to 1 to perform morphological opening (0 by default). \
Only one morphological operation can be specified. Operations are considered in the \
following order: dilate, erode, open, close. E.g., if dilation is specified, only it will be \
performed regardless of whether the other operations have requested.";
  std::string closeDesc = "Set to 1 to perform morphological closing (0 by default). \
Only one morphological operation can be specified. Operations are considered in the \
following order: dilate, erode, open, close. E.g., if dilation is specified, only it will be \
performed regardless of whether the other operations have requested.";
  std::string kernelRadiusXDesc = "Radius of morphology kernel in the x-direction.";
  std::string kernelRadiusYDesc = "Radius of morphology kernel in the y-direction.";
  std::string kernelRadiusZDesc = "Radius of morphology kernel in the z-direction.";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision$" );

    TCLAP::ValueArg<std::string> inFileNameArg("i", "in", inFileNameDesc, true, inFileName, "string", cl);
    TCLAP::ValueArg<std::string> outFileNameArg("o", "out", outFileNameDesc, true, outFileName, "string", cl);
    TCLAP::MultiArg<unsigned int> regionsVecArg("r", "region", regionsVecDesc, true, "int", cl);
    TCLAP::MultiArg<unsigned int> typesVecArg("t", "type", typesVecDesc, true, "int", cl);
    TCLAP::SwitchArg dilateArg("d", "dilate", dilateDesc, cl, false);
    TCLAP::SwitchArg erodeArg("e", "erode", erodeDesc, cl, false);
    TCLAP::SwitchArg openArg("", "op", openDesc, cl, false);
    TCLAP::SwitchArg closeArg("", "cl", closeDesc, cl, false);
    TCLAP::ValueArg<unsigned int> kernelRadiusXArg("", "radx", kernelRadiusXDesc, false, kernelRadiusX, "", cl);
    TCLAP::ValueArg<unsigned int> kernelRadiusYArg("", "rady", kernelRadiusYDesc, false, kernelRadiusY, "", cl);
    TCLAP::ValueArg<unsigned int> kernelRadiusZArg("", "radz", kernelRadiusZDesc, false, kernelRadiusZ, "", cl);

    cl.parse(argc, argv);

    inFileName  = inFileNameArg.getValue();
    outFileName = outFileNameArg.getValue();
    dilate = dilateArg.getValue();
    erode = erodeArg.getValue();
    open = openArg.getValue();
    close = closeArg.getValue();
    kernelRadiusX = kernelRadiusXArg.getValue();
    kernelRadiusY = kernelRadiusYArg.getValue();
    kernelRadiusZ = kernelRadiusZArg.getValue();

    if (regionsVecArg.getValue().size() != typesVecArg.getValue().size())
      {
      std::cerr << "Error: must specify same number of chest regions and chest types" << std::endl;
      return cip::ARGUMENTPARSINGERROR;
      }

    for (unsigned int i=0; i<regionsVecArg.getValue().size(); i++)
      {
      regionsVec.push_back((unsigned char)regionsVecArg.getValue()[i]);
      }
    for (unsigned int i=0; i<typesVecArg.getValue().size(); i++)
      {
      typesVec.push_back((unsigned char)typesVecArg.getValue()[i]);
      }
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName(inFileName);
  try
    {
    reader->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  if (dilate)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {      
      std::cout << "Dilating..." << std::endl;
      cip::DilateLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i], 
			  kernelRadiusX, kernelRadiusY, kernelRadiusZ );
      }
    }
  else if (erode)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {
      std::cout << "Eroding..." << std::endl;      
      cip::ErodeLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i],
  			 kernelRadiusX, kernelRadiusY, kernelRadiusZ );
      }
    }
  else if (open)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {      
      std::cout << "Opening..." << std::endl;
      cip::OpenLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i],
  			kernelRadiusX, kernelRadiusY, kernelRadiusZ );
      }
    }
  else if (close)
    {
    for (unsigned int i=0; i<regionsVec.size(); i++)
      {      
      std::cout << "Closing..." << std::endl;
      cip::CloseLabelMap(reader->GetOutput(), regionsVec[i], typesVec[i], 
  			 kernelRadiusX, kernelRadiusY, kernelRadiusZ );
      }
    }

  std::cout << "Writing label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetFileName(outFileName);
    writer->SetInput(reader->GetOutput());
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
