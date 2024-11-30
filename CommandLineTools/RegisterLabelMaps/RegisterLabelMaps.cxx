#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include <fstream>
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImageRegistrationMethod.h"
#include "itkCenteredTransformInitializer.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkCastImageFilter.h"

//xml
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h> 
#undef reference // to use vtklibxml2

#include "RegisterLabelMapsCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include <sstream>

namespace
{
#define MY_ENCODING "ISO-8859-1"
    
  struct REGIONTYPEPAIR
  {
    unsigned char region;
    unsigned char type;
  };
  
  struct REGISTRATION_XML_DATA
  {
    std::string registrationID;
    float similarityValue;
    std::string transformationLink;
    std::string sourceID;
    std::string destID;
    std::string similarityMeasure;
    std::string image_type;
    int transformationIndex;
  };
    
  template <unsigned int TDimension> void 
  WriteTransformFile( typename itk::AffineTransform< double, TDimension >::Pointer transform, char* fileName )
  {
    itk::TransformFileWriter::Pointer transformWriter = itk::TransformFileWriter::New();
      transformWriter->SetInput( transform );
      transformWriter->SetFileName( fileName );
    try
      {
      transformWriter->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught while updating transform writer:";
      std::cerr << excp << std::endl;
      }
  }
    
  template <unsigned int TDimension> typename itk::Image< unsigned short, TDimension >::Pointer 
  ReadLabelMapFromFile( std::string labelMapFileName )
  {
    std::cout << "Reading label map..." << std::endl;
    typedef itk::Image< unsigned short, TDimension >  LabelMapType;
    typedef itk::ImageFileReader< LabelMapType >      LabelMapReaderType;
    
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
      }
    
    return reader->GetOutput();
  }
  
  void WriteRegistrationXML(const char *file, REGISTRATION_XML_DATA &theXMLData)
  {
    std::cout<<"Writing registration XML file"<<std::endl;
    xmlDocPtr doc = NULL;       /* document pointer */
    xmlNodePtr root_node = NULL; /* Node pointers */
    xmlDtdPtr dtd = NULL;       /* DTD pointer */
    
    doc = xmlNewDoc(BAD_CAST "1.0");
    root_node = xmlNewNode(NULL, BAD_CAST "Registration");
    xmlDocSetRootElement(doc, root_node);
    
    dtd = xmlCreateIntSubset(doc, BAD_CAST "root", NULL, BAD_CAST "RegistrationOutput_v2.dtd");
    
    time_t timer;
    time( &timer );
    std::stringstream tempStream;
    tempStream << timer;
    theXMLData.registrationID.assign("Registration");
    theXMLData.registrationID.append(tempStream.str());
    theXMLData.registrationID.append("_");
    theXMLData.registrationID.append(theXMLData.sourceID.c_str());
    theXMLData.registrationID.append("_to_");
    theXMLData.registrationID.append(theXMLData.destID.c_str());
    
    xmlNewProp(root_node, BAD_CAST "Registration_ID", BAD_CAST (theXMLData.registrationID.c_str()));
    
    // xmlNewChild() creates a new node, which is "attached"
    // as child node of root_node node.
    std::ostringstream similaritString;
    //std::string tempsource;
    similaritString << theXMLData.similarityValue;
    std::ostringstream transformationIndexString;    
    transformationIndexString << theXMLData.transformationIndex;
    
    xmlNewChild(root_node, NULL, BAD_CAST "image_type", BAD_CAST
                (theXMLData.image_type.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "transformation", BAD_CAST
		(theXMLData.transformationLink.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "transformation_index", BAD_CAST
                (transformationIndexString.str().c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "movingID", BAD_CAST
		(theXMLData.sourceID.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "fixedID", BAD_CAST
		(theXMLData.destID.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "SimilarityMeasure", BAD_CAST
		(theXMLData.similarityMeasure.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "SimilarityValue", BAD_CAST
		(similaritString.str().c_str()));
    xmlSaveFormatFileEnc(file, doc, "UTF-8", 1);
    xmlFreeDoc(doc);
  }  

  template < unsigned int TDimension > typename itk::Image<unsigned short, TDimension>::Pointer
  DownsampleLabelMap (typename itk::Image<unsigned short, TDimension>::Pointer labelMap, unsigned int downsampleFactor) 
  {
    return NULL;
  }

  // 2D specialization
  template <> 
  itk::Image<unsigned short, 2>::Pointer DownsampleLabelMap<2> (typename itk::Image<unsigned short, 2>::Pointer labelMap, 
    unsigned int downsampleFactor) 
  {
    typedef itk::Image<unsigned short, 2>                                                LabelMapType;
    typedef itk::CastImageFilter< LabelMapType, cip::LabelMapSliceType >                   CasterTempToCIPType;
    typedef itk::CastImageFilter< cip::LabelMapSliceType, LabelMapType >                   CasterCIPToTempType;

    // First down-sample the fixed label map
	  CasterTempToCIPType::Pointer tempToCIPCaster = CasterTempToCIPType::New();
	  tempToCIPCaster->SetInput( labelMap );
	  tempToCIPCaster->Update();

	  cip::LabelMapSliceType::Pointer tmp = cip::LabelMapSliceType::New();
	  tmp = cip::DownsampleLabelMapSlice( downsampleFactor, tempToCIPCaster->GetOutput() );

	  CasterCIPToTempType::Pointer CIPToTempCaster = CasterCIPToTempType::New();
	  CIPToTempCaster->SetInput( tmp );
	  CIPToTempCaster->Update();
	
	  return CIPToTempCaster->GetOutput();
  }
  
  // 3D specialization
  template <> 
  itk::Image<unsigned short, 3>::Pointer DownsampleLabelMap<3> (typename itk::Image<unsigned short, 3>::Pointer labelMap, 
    unsigned int downsampleFactor) 
  {
    typedef itk::Image<unsigned short, 3>                                                LabelMapType;
    typedef itk::CastImageFilter< LabelMapType, cip::LabelMapType >                        CasterTempToCIPType;
    typedef itk::CastImageFilter< cip::LabelMapType, LabelMapType >                        CasterCIPToTempType;

  	CasterTempToCIPType::Pointer tempToCIPCaster = CasterTempToCIPType::New();
	  tempToCIPCaster->SetInput( labelMap );
	  tempToCIPCaster->Update();

	  cip::LabelMapType::Pointer tmp = cip::LabelMapType::New();
		tmp = cip::DownsampleLabelMap( downsampleFactor, tempToCIPCaster->GetOutput() );

	  CasterCIPToTempType::Pointer CIPToTempCaster = CasterCIPToTempType::New();
	  CIPToTempCaster->SetInput( tmp );
	  CIPToTempCaster->Update();
	
	  return CIPToTempCaster->GetOutput();
  }


  template <unsigned int TDimension>
  int DoIT(int argc, char * argv[])
  {
    PARSE_ARGS;
    
    //dimension specific typedefs
    typedef itk::Image< unsigned short, TDimension >                                       LabelMapType;
    typedef itk::ResampleImageFilter< LabelMapType, LabelMapType >                         ResampleFilterType;
    typedef itk::ImageFileWriter< LabelMapType >                                           LabelMapWriterType;
    typedef itk::RegularStepGradientDescentOptimizer                                       OptimizerType;
    typedef itk::ImageRegistrationMethod< LabelMapType, LabelMapType >                     RegistrationType;
    typedef itk::KappaStatisticImageToImageMetric< LabelMapType,LabelMapType >             MetricType;
    typedef itk::NearestNeighborInterpolateImageFunction< LabelMapType, double >           InterpolatorType;
    typedef OptimizerType::ScalesType                                                      OptimizerScalesType;
    typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                              IteratorType;
    typedef itk::RegionOfInterestImageFilter< LabelMapType, LabelMapType >                 RegionOfInterestType;
    typedef itk::ResampleImageFilter<LabelMapType,LabelMapType >                           ResampleType;
    typedef itk::IdentityTransform< double, TDimension >                                   IdentityType;
    typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                              LabelMapIteratorType;
    typedef itk::AffineTransform<double, TDimension >                                      TransformType;
    typedef itk::CenteredTransformInitializer< TransformType, LabelMapType, LabelMapType > InitializerType;
    typedef itk::ImageFileReader<LabelMapType >                                            LabelMapReaderType;
    typedef itk::CIPExtractChestLabelMapImageFilter  <TDimension>                          LabelMapExtractorType;
        
    //Read in fixed image label map from file and subsample
    typename LabelMapType::Pointer fixedLabelMap = LabelMapType::New();

    if ( strcmp( fixedImageFileName.c_str(), "NA" ) != 0 )
      {
	std::cout << "Reading label map from file..." << std::endl;
	
	fixedLabelMap = ReadLabelMapFromFile< TDimension >( fixedImageFileName );
	if ( fixedLabelMap.GetPointer() == NULL )
	  {
	    return cip::LABELMAPREADFAILURE;
	  }	
      }
    else
      {
	std::cerr << "Error: No lung label map specified" << std::endl;
	return cip::EXITFAILURE;
      }

    //Read in moving image label map from file and subsample
    typename LabelMapType::Pointer movingLabelMap = LabelMapType::New();
    if ( strcmp( movingImageFileName.c_str(), "NA") != 0 )
      {
	std::cout << "Reading label map from file..." << std::endl;
	movingLabelMap = ReadLabelMapFromFile<TDimension>( movingImageFileName );
	
	if (movingLabelMap.GetPointer() == NULL)
	  {
	    return cip::LABELMAPREADFAILURE;
	  }
      }
    else
      {
	std::cerr << "Error: No lung label map specified" << std::endl;
	return cip::EXITFAILURE;
      }

    // Sub-sample the fixed and moving label maps if required
    typename LabelMapType::Pointer subSampledFixedImage = LabelMapType::New();
    typename LabelMapType::Pointer subSampledMovingImage = LabelMapType::New();

    std::cout << "Subsampling fixed image by a factor of " << downsampleFactor << "..." << std::endl;
    subSampledFixedImage = DownsampleLabelMap<TDimension>(fixedLabelMap, downsampleFactor);
    
    std::cout << "Subsampling moving image by a factor of " << downsampleFactor << "..." << std::endl;
    subSampledMovingImage = DownsampleLabelMap<TDimension>(movingLabelMap, downsampleFactor);
        
    LabelMapIteratorType fit( subSampledFixedImage, subSampledFixedImage->GetBufferedRegion() );    
    fit.GoToBegin();
    while ( !fit.IsAtEnd() )
      {
	if ( fit.Get() != 0 )
	  {
	    fit.Set( 1 );
	  }
	++fit;
      }
    
    LabelMapIteratorType mit( subSampledMovingImage, subSampledMovingImage->GetBufferedRegion() );    
    mit.GoToBegin();
    while ( !mit.IsAtEnd() )
      {
	if ( mit.Get() != 0 )
	  {
	    mit.Set( 1 );
	  }	
	++mit;
      }
    
    typename MetricType::Pointer metric = MetricType::New();
      metric->SetForegroundValue( 1 ); 
    
    typename TransformType::Pointer transform = TransformType::New();

    std::cout << "Initializing transform..." << std::endl;
    typename InitializerType::Pointer initializer = InitializerType::New();
      initializer->SetTransform( transform );
      initializer->SetFixedImage( subSampledFixedImage );
      initializer->SetMovingImage( subSampledMovingImage );
      initializer->MomentsOn();
    try
      {
      initializer->InitializeTransform();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught initializing transform:";
      std::cerr << excp << std::endl;
      }

    OptimizerScalesType optimizerScales( transform->GetNumberOfParameters());
      optimizerScales[0]  =  1.0;   
      optimizerScales[1]  =  1.0;
      optimizerScales[2]  =  1.0;
      optimizerScales[3]  =  1.0;   
      optimizerScales[4]  =  1.0;
      optimizerScales[5]  =  1.0;
      optimizerScales[6]  =  1.0;   
      optimizerScales[7]  =  1.0;
      optimizerScales[8]  =  1.0;
      optimizerScales[9]  =  translationScale;
      optimizerScales[10] =  translationScale;
      optimizerScales[11] =  translationScale;

    typename OptimizerType::Pointer optimizer = OptimizerType::New();
      optimizer->SetScales( optimizerScales );
      optimizer->SetMaximumStepLength( maxStepLength );
      optimizer->SetMinimumStepLength( minStepLength );
      optimizer->SetNumberOfIterations( numberOfIterations );

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    
    std::cout << "Starting registration..." << std::endl;
    typename RegistrationType::Pointer registration = RegistrationType::New();
      registration->SetMetric( metric );
      registration->SetOptimizer( optimizer );
      registration->SetInterpolator( interpolator );
      registration->SetTransform( transform );          
      registration->SetFixedImage( subSampledFixedImage );
      registration->SetMovingImage( subSampledMovingImage );
      registration->SetFixedImageRegion( subSampledFixedImage->GetBufferedRegion() );
      registration->SetInitialTransformParameters( transform->GetParameters());      
    try
      {
	registration->Initialize();
	registration->Update();
      }
    catch( itk::ExceptionObject &excp )
      {
	std::cerr << "ExceptionObject caught while executing registration" << std::endl;
	std::cerr << excp << std::endl;
      }
    
    // Get all params to output to file
    numberOfIterations = optimizer->GetCurrentIteration();

    // The value of the image metric corresponding to the last set of parameters
    const double bestValue = optimizer->GetValue();
    std::cout << "Final metric value: " << optimizer->GetValue() << std::endl;
    typename OptimizerType::ParametersType finalParams = registration->GetLastTransformParameters();
    
    typename TransformType::Pointer finalTransform = TransformType::New();
      finalTransform->SetParameters( finalParams );
      finalTransform->SetCenter( transform->GetCenter() );
    
    if ( strcmp(outputTransformFileName.c_str(), "NA") != 0 )
      {
	std::string infoFilename = outputTransformFileName;
	int result = infoFilename.find_last_of('.');
	if (std::string::npos != result)
	  {
	    infoFilename.erase(result);
	  }
	// append extension:
	infoFilename.append(".xml");
	
	REGISTRATION_XML_DATA labelMapRegistrationXMLData;
	labelMapRegistrationXMLData.similarityValue = (float)(bestValue);
	const char *similarity_type = metric->GetNameOfClass();
	labelMapRegistrationXMLData.similarityMeasure.assign(similarity_type);
	
	//labelMapRegistrationXMLData.image_type.assign("leftLungRightLung");
	labelMapRegistrationXMLData.transformationIndex = 0;
	
	// TODO: See if xml file is empty and change the transformation
	// index accordingly
	
	//if the patient IDs are specified  as args, use them,
	//otherwise, NA
	
	int pathLength = 0, pos=0, next=0;
	
	if ( strcmp(movingImageID.c_str(), "NA") != 0 )
	  {
	    labelMapRegistrationXMLData.sourceID.assign(movingImageID);
	  }
	else
	  {
	    labelMapRegistrationXMLData.sourceID.assign("N/A");
	  }	
	
	if ( strcmp(fixedImageID.c_str(), "NA") != 0 )
	  {
	    labelMapRegistrationXMLData.destID.assign(fixedImageID);
	  }
	else
	  {
	    labelMapRegistrationXMLData.destID.assign("N/A");
	  }
	
	// Remove path from output transformation file before storing in xml
	pos=0;
	next=0;
	for (int i = 0; i < (pathLength);i++)
	  {
	    pos = next+1;
	    next = outputTransformFileName.find('/', next+1);
	  }
	
	labelMapRegistrationXMLData.transformationLink.assign(outputTransformFileName.c_str());
	labelMapRegistrationXMLData.transformationLink.erase(0,pos);
	WriteRegistrationXML(infoFilename.c_str(),labelMapRegistrationXMLData);
	
	std::cout << "Writing transform..." << std::endl;
	itk::TransformFileWriter::Pointer transformWriter = itk::TransformFileWriter::New();
          transformWriter->SetInput( finalTransform );   
	  transformWriter->SetFileName( outputTransformFileName );
	  transformWriter->Update();
      }
    
    if ( strcmp(resampledLabelMapFileName.c_str(), "NA") != 0 )
      {
	std::cout << "Resampling moving image..." << std::endl;        
	typename ResampleFilterType::Pointer resample = ResampleFilterType::New();
          resample->SetTransform( finalTransform );
	  resample->SetInput( movingLabelMap);
	  resample->SetSize(fixedLabelMap->GetLargestPossibleRegion().GetSize());
	  resample->SetOutputOrigin( fixedLabelMap->GetOrigin() );
	  resample->SetOutputSpacing( fixedLabelMap->GetSpacing());
	  resample->SetOutputDirection( fixedLabelMap->GetDirection() );
	  resample->SetInterpolator( interpolator );
	  resample->SetDefaultPixelValue( 0 );
	  resample->Update();
	
	typename LabelMapWriterType::Pointer writer = LabelMapWriterType::New();
          writer->SetInput(resample->GetOutput());
	  writer->SetFileName( resampledLabelMapFileName );
	  writer->UseCompressionOn();
	try
	  {
	  writer->Update();
	  }
	catch ( itk::ExceptionObject &excp )
	  {
	  std::cerr << "Exception caught writing output image:";
	  std::cerr << excp << std::endl;
	  }
      }
    
    std::cout << "DONE." << std::endl;
    return cip::EXITSUCCESS;    
  }
  
} //end namespace

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
