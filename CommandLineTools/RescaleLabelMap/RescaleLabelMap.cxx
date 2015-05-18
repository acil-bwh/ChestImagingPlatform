#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "RescaleLabelMapCLP.h"
#include "itkCastImageFilter.h"

namespace
{   
  template <unsigned int TDimension>
  int DoIT(int argc, char * argv[])
  {    
    PARSE_ARGS;

    typedef itk::Image< unsigned short, TDimension >                     LabelMapType;
    typedef itk::ImageFileReader< LabelMapType >                         LabelMapReaderType;
    typedef itk::ImageFileWriter< LabelMapType >                         LabelMapWriterType;
    typedef itk::CastImageFilter< LabelMapType, cip::LabelMapType >      CasterTempTo3dType;
    typedef itk::CastImageFilter< cip::LabelMapType, LabelMapType >      Caster3dToTempType;
    typedef itk::CastImageFilter< LabelMapType, cip::LabelMapSliceType > CasterTempTo2dType;
    typedef itk::CastImageFilter< cip::LabelMapSliceType, LabelMapType > Caster2dToTempType;

    std::cout << "Reading label map..." << std::endl;
    typename LabelMapReaderType::Pointer reader = LabelMapReaderType::New();
      reader->SetFileName( labelMapFileName );
    try
      {
      reader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading label map:";
      std::cerr << excp << std::endl;
    
      return cip::LABELMAPREADFAILURE;
      }

    typename LabelMapType::Pointer outLabelMap = LabelMapType::New();

    if ( TDimension == 3 )
      {
	typename CasterTempTo3dType::Pointer caster = CasterTempTo3dType::New();
	  caster->SetInput( reader->GetOutput() );
	  caster->Update();

	cip::LabelMapType::Pointer tmp = cip::LabelMapType::New();
	if ( downScale > 1 )
	  {
	    tmp = cip::DownsampleLabelMap( downScale, caster->GetOutput() );
	  }
	else if ( upScale > 1 )
	  {
	    tmp = cip::UpsampleLabelMap( upScale, caster->GetOutput() );
	  }

	typename Caster3dToTempType::Pointer tmpCaster = Caster3dToTempType::New();
	  tmpCaster->SetInput( tmp );
	  tmpCaster->Update();
	
	outLabelMap = tmpCaster->GetOutput();
      }
    else if ( TDimension == 2 )
      {
	typename CasterTempTo2dType::Pointer caster = CasterTempTo2dType::New();
	  caster->SetInput( reader->GetOutput() );
	  caster->Update();

	cip::LabelMapSliceType::Pointer tmp = cip::LabelMapSliceType::New();
	if ( downScale > 1 )
	  {
	    tmp = cip::DownsampleLabelMapSlice( downScale, caster->GetOutput() );
	  }
	else if ( upScale > 1 )
	  {
	    tmp = cip::UpsampleLabelMapSlice( upScale, caster->GetOutput() );
	  }

	typename Caster2dToTempType::Pointer tmpCaster = Caster2dToTempType::New();
	  tmpCaster->SetInput( tmp );
	  tmpCaster->Update();
	
	outLabelMap = tmpCaster->GetOutput();
      }
  
    // Write the resampled label map to file
    std::cout << "Writing rescaled label map..." << std::endl;
    typename LabelMapWriterType::Pointer writer = LabelMapWriterType::New();
      writer->SetFileName( rescaledFileName );
      writer->UseCompressionOn();
      writer->SetInput( outLabelMap );
    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught writing label map:";
      std::cerr << excp << std::endl;
    
      return cip::LABELMAPWRITEFAILURE;
      }

    std::cout << "DONE." << std::endl;   
    return cip::EXITSUCCESS;
  }
  
} // end of anonymous namespace

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
 
  switch(dimension)
    {
    case 2:
      {
	DoIT<2>( argc, argv );
	break;
      }
    case 3:
      {
	DoIT<3>( argc, argv );
	break;
      }
    default:
      {
	std::cerr << "Bad dimensions:";
	return cip::EXITFAILURE;
      }
    }

  return cip::EXITSUCCESS;
}
