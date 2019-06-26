#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "RescaleLabelMapCLP.h"
#include "itkCastImageFilter.h"

namespace
{   


  template <unsigned int TDimension> typename itk::Image<unsigned short, TDimension>::Pointer
  UpDownSample (typename itk::Image<unsigned short, TDimension>::Pointer inputImage, 
    unsigned int downScale, unsigned int upScale)
  {
    return NULL;
  }

  template<>
  typename itk::Image<unsigned short, 2>::Pointer
  UpDownSample<2> (typename itk::Image<unsigned short, 2>::Pointer inputImage, 
    unsigned int downScale, unsigned int upScale)
  {
    typedef itk::Image<unsigned short, 2> LabelMapType;
    typedef itk::CastImageFilter< LabelMapType, cip::LabelMapSliceType > CasterTempTo2dType;
    typedef itk::CastImageFilter< cip::LabelMapSliceType, LabelMapType > Caster2dToTempType;

    CasterTempTo2dType::Pointer caster = CasterTempTo2dType::New();
	  caster->SetInput( inputImage );
	  caster->Update();

	  cip::LabelMapSliceType::Pointer tmp = cip::LabelMapSliceType::New();
    tmp = caster->GetOutput();

	  if ( downScale > 1 )
	  {
	    tmp = cip::DownsampleLabelMapSlice( downScale, tmp );
	  }
	  else if ( upScale > 1 )
	  {
	    tmp = cip::UpsampleLabelMapSlice( upScale, tmp );
	  }

	  Caster2dToTempType::Pointer tmpCaster = Caster2dToTempType::New();
	  tmpCaster->SetInput( tmp );
	  tmpCaster->Update();
	
	  return tmpCaster->GetOutput();
  }

  template<>
  typename itk::Image<unsigned short, 3>::Pointer
  UpDownSample<3> (typename itk::Image<unsigned short, 3>::Pointer inputImage, 
    unsigned int downScale, unsigned int upScale)
  {
    typedef itk::Image<unsigned short, 3> LabelMapType;
    typedef itk::CastImageFilter< LabelMapType, cip::LabelMapType >      CasterTempTo3dType;
    typedef itk::CastImageFilter< cip::LabelMapType, LabelMapType >      Caster3dToTempType;

    CasterTempTo3dType::Pointer caster = CasterTempTo3dType::New();
	  caster->SetInput( inputImage );
	  caster->Update();

	  cip::LabelMapType::Pointer tmp = cip::LabelMapType::New();
    tmp = caster->GetOutput();

	  if ( downScale > 1 )
	  {
	    tmp = cip::DownsampleLabelMap( downScale, tmp );
	  }
  	else if ( upScale > 1 )
	  {
	    tmp = cip::UpsampleLabelMap( upScale, tmp );
	  } 
    
	  Caster3dToTempType::Pointer tmpCaster = Caster3dToTempType::New();
	  tmpCaster->SetInput( tmp );
	  tmpCaster->Update();
	
	  return tmpCaster->GetOutput();
  }
  

  template <unsigned int TDimension>
  int DoIT(int argc, char * argv[])
  {
    PARSE_ARGS;

    typedef itk::Image< unsigned short, TDimension >                     LabelMapType;
    typedef itk::ImageFileReader< LabelMapType >                         LabelMapReaderType;
    typedef itk::ImageFileWriter< LabelMapType >                         LabelMapWriterType;
    
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
    
    outLabelMap = UpDownSample<TDimension>(reader->GetOutput(), downScale, upScale);
              
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
