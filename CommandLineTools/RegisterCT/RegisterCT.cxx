/** \file
 *  \ingroup commandLineTools
 *  \details This program registers 2 label maps, source and target, and
 * a transformation file as well as the transformed image
 *
 *  USAGE: ./RegisterLabelMaps --regionVec 1 -m
 
 /net/th914_nas.bwh.harvard.edu/mnt/array1/share/Processed/COPDGene/11622T/11622T_INSP_STD_HAR_COPD/11622T_INSP_STD_HAR_COPD_leftLungRightLung.nhdr
 -f
 /net/th914_nas.bwh.harvard.edu/mnt/array1/share/Processed/COPDGene/10393Z/10393Z_INSP_STD_HAR_COPD/10393Z_INSP_STD_HAR_COPD_leftLungRightLung.nhdr
 --outputImage /projects/lmi/people/rharmo/projects/dataInfo/testoutput.nrrd
 --outputTransform output_transform_file -d 12
 
 *
 *  $Date: $
 *  $Revision: $
 *  $Author:  $
 *
 */

//image
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include <fstream>
#include "itkImageRegionIterator.h"
#include "itkNormalizeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkImageMaskSpatialObject.h"

//xml
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>

//registration
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkGradientDescentOptimizer.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkImageRegistrationMethod.h"
#include "itkCenteredTransformInitializer.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkRigid2DTransform.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkBinaryThresholdImageFilter.h"
//#include "itkRegularStepGradientDescentOptimizerv4.h"

#include "RegisterCTCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include <sstream>
#include "cipExceptionObject.h"


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
  
  template <unsigned int TDimension> typename itk::Image< unsigned short, TDimension >::Pointer ReadLabelMapFromFile( std::string labelMapFileName )
  {
    std::cout << "Reading label map..." << std::endl;
    typedef itk::Image< unsigned short, TDimension >                              LabelMapType;
    typedef itk::ImageFileReader< LabelMapType >                                  LabelMapReaderType;
    
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
  
  
  
  template <unsigned int TDimension> typename itk::Image< short, TDimension >::Pointer ReadCTFromFile( std::string fileName )
  {
    typedef itk::Image< short, TDimension >                              ShortImageType;
    typedef itk::ImageFileReader< ShortImageType >                                                  ShortReaderType;
    
    if(strcmp( fileName.c_str(), "q") == 0 )
      {
	throw cip::ExceptionObject( __FILE__, __LINE__, "RegisterLabelMaps::main()", " No lung label map specified" );
      }
    
    typename ShortReaderType::Pointer reader = ShortReaderType::New();
    reader->SetFileName( fileName );
    try
      {
	reader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
	std::cerr << "Exception caught reading CT image:";
	std::cerr << excp << std::endl;
	return NULL;
      }
    
    return reader->GetOutput();
  }
  
  void WriteRegistrationXML(const char *file, REGISTRATION_XML_DATA
			    &theXMLData)
  {
    std::cout<<"Writing registration XML file"<<std::endl;
    xmlDocPtr doc = NULL;       /* document pointer */
    xmlNodePtr root_node = NULL; /* Node pointers */
    xmlDtdPtr dtd = NULL;       /* DTD pointer */
    
    doc = xmlNewDoc(BAD_CAST "1.0");
    root_node = xmlNewNode(NULL, BAD_CAST "Registration");
    xmlDocSetRootElement(doc, root_node);
    
    dtd = xmlCreateIntSubset(doc, BAD_CAST "root", NULL, BAD_CAST
			     "RegistrationOutput_v2.dtd");
    
    time_t timer;
    time(&timer);
    std::stringstream tempStream;
    tempStream<<timer;
    theXMLData.registrationID.assign("Registration");
    theXMLData.registrationID.append(tempStream.str());
    theXMLData.registrationID.append("_");
    theXMLData.registrationID.append(theXMLData.sourceID.c_str());
    theXMLData.registrationID.append("_to_");
    theXMLData.registrationID.append(theXMLData.destID.c_str());
    
    
    xmlNewProp(root_node, BAD_CAST "Registration_ID", BAD_CAST
	       (theXMLData.registrationID.c_str()));
    
    // xmlNewChild() creates a new node, which is "attached"
    // as child node of root_node node.
    std::ostringstream similaritString;
    //std::string tempsource;
    similaritString <<theXMLData.similarityValue;
    std::ostringstream transformationIndexString;    
    transformationIndexString <<theXMLData.transformationIndex;
    
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
  
  
} //end namespace

template <unsigned int TDimension>
int DoIT2(int argc, char * argv[])
{  
  std::vector< unsigned char >  regionVec;
  std::vector< unsigned char >  typeVec;
  std::vector< unsigned char >  regionPairVec;
  std::vector< unsigned char >  typePairVec; 
  
  PARSE_ARGS;
  
  typedef itk::Image< unsigned short, TDimension >                     LabelMapType;
  typedef itk::Image< short, TDimension >                              ShortImageType;
  
  typedef itk::ImageRegionIterator< ShortImageType >                                   CTImageIteratorType;  
  typedef itk::GDCMImageIO                                                                          ImageIOType;
  typedef itk::GDCMSeriesFileNames                                                                  NamesGeneratorType;
  typedef itk::ImageFileReader< LabelMapType >                                                    LabelMapReaderType;
  typedef itk::ImageFileReader< ShortImageType >                                                  ShortReaderType;
  typedef itk::ImageFileWriter< ShortImageType >                                                  ImageWriterType;
  
  typedef itk::CIPExtractChestLabelMapImageFilter< TDimension >                                                    LabelMapExtractorType;
  typedef itk::ImageMaskSpatialObject< TDimension >                                                          MaskType;
  typedef itk::Image< unsigned char, TDimension >                                                            ImageMaskType;
  typedef itk::ImageFileReader< ImageMaskType >                                                     MaskReaderType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                                       LabelMapIteratorType;
  
  typedef itk::DiscreteGaussianImageFilter< ShortImageType, ShortImageType >                    GaussianFilterType;  
  typedef itk::NormalizeImageFilter<ShortImageType,ShortImageType>                              FixedNormalizeFilterType;
  typedef itk::NormalizeImageFilter<ShortImageType,ShortImageType>                              MovingNormalizeFilterType;
  typedef itk::BinaryThresholdImageFilter< ShortImageType, ShortImageType >  FilterType;
  
  //similarity
  typedef itk::NormalizedCorrelationImageToImageMetric<ShortImageType, ShortImageType  >        ncMetricType;
  typedef itk::MeanSquaresImageToImageMetric<ShortImageType, ShortImageType  >        meanSquaredMetricType;
  
  //transformations   
  typedef itk::IdentityTransform< double, TDimension >                                                       IdentityType;  
  typedef itk::AffineTransform<double, TDimension >                                                          AffineTransformType;
  typedef itk::Similarity2DTransform< double>                                                       SimilarityTransformType;
  
  typedef itk::RegularStepGradientDescentOptimizer  OptimizerType;
  
  typedef itk::GradientDescentOptimizer                                                             GradOptimizerType;
  typedef OptimizerType::ScalesType                                                                 OptimizerScalesType;
  
  typedef itk::ImageRegistrationMethod<ShortImageType,ShortImageType >                          CTRegistrationType;
  typedef itk::LinearInterpolateImageFunction< ShortImageType, double >                  CTInterpolatorType;
  typedef itk::ResampleImageFilter< ShortImageType, ShortImageType >                            ResampleFilterType;
  
  
  //Read in region and type pair
  std::vector< REGIONTYPEPAIR > regionTypePairVec;
  
  for ( unsigned int i=0; i<regionVecArg.size(); i++ )
    {
      regionVec.push_back(regionVecArg[i]);
    }
  for ( unsigned int i=0; i<typeVecArg.size(); i++ )
    {
      typeVec.push_back( typeVecArg[i] );
    }
  if (regionPairVec.size() == typePairVecArg.size())
    {
      for ( unsigned int i=0; i<regionPairVecArg.size(); i++ )
	{
          REGIONTYPEPAIR regionTypePairTemp;
	  
          regionTypePairTemp.region = regionPairVecArg[i];
          argc--; argv++;
          regionTypePairTemp.type   = typePairVecArg[i];
	  
          regionTypePairVec.push_back( regionTypePairTemp );
	}
    }
  
  //Read in fixed image label map from file and subsample
  typename  LabelMapType::Pointer fixedLabelMap = LabelMapType::New();
  typename  LabelMapType::Pointer movingLabelMap =LabelMapType::New();
  typename  ShortImageType::Pointer fixedCT = ShortImageType::New();
  typename ShortImageType::Pointer movingCT = ShortImageType::New();
  
  try 
    {
      std::cout << "Reading CT from file..." << std::endl;
      
      fixedCT = ReadCTFromFile< TDimension > ( fixedImageFileName );
      if (fixedCT.GetPointer() == NULL)
	{
	  return cip::LABELMAPREADFAILURE;
	}
      
    }
  catch (cip::ExceptionObject &exep)
    {
      std::cerr << "Error: No CT specified"<< std::endl;
      std::cerr << exep << std::endl;
      return cip::EXITFAILURE;
    }
  
  //Read in moving image label map from file and subsample
  
  try
    {
      std::cout << "Reading CT from file..." << std::endl;
      
      movingCT = ReadCTFromFile< TDimension > ( movingImageFileName );
      
      if (movingCT.GetPointer() == NULL)
	{
	  return cip::LABELMAPREADFAILURE;
	}
      
    }
  catch (cip::ExceptionObject &exep)
    {
      std::cerr <<"Error: No CT specified"<< std::endl;
      std::cerr << exep << std::endl;
      return cip::EXITFAILURE;
    }
    
  typename ncMetricType::Pointer nc_metric_affine = ncMetricType::New();
  typename meanSquaredMetricType::Pointer nc_metric = meanSquaredMetricType::New();
  
  typename MaskType::Pointer  spatialObjectMask = MaskType::New();
  typename MaskReaderType::Pointer  maskReader = MaskReaderType::New();
  
  if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
    {
      std::cout<<"reading moving label map "<<movingLabelmapFileName.c_str() <<std::endl;
      maskReader->SetFileName(movingLabelmapFileName.c_str() );
      
      try
        {
	  maskReader->Update();
        }
      catch( itk::ExceptionObject & err )
        {
	  std::cerr << "ExceptionObject caught !" << std::endl;
	  std::cerr << err << std::endl;
	  return EXIT_FAILURE;
        }
      spatialObjectMask->SetImage(maskReader->GetOutput());
      nc_metric_affine->SetMovingImageMask( spatialObjectMask );
      //nc_metric->SetMovingImageMask( spatialObjectMask );
      
    }
  
  
  std::cout<<"initializing transform"<<std::endl;
  
  typedef itk::Rigid2DTransform< double>            RigidTransformType;
  typedef itk::CenteredTransformInitializer< RigidTransformType, ShortImageType, ShortImageType >  RigidInitializerTypeIntensity;
  
  typename RigidTransformType::Pointer rigidTransform = RigidTransformType::New();
  typename CTInterpolatorType::Pointer CTinterpolator = CTInterpolatorType::New();  
  
  //initial registration with a rigid transform      
  //create a similarity transform to experiment
  typename SimilarityTransformType::Pointer similarityTransform = SimilarityTransformType::New();
  
  OptimizerScalesType rigidOptimizerScales(rigidTransform->GetNumberOfParameters());
  
  rigidOptimizerScales[0] =  1.0;
  rigidOptimizerScales[1] =  1.0;
  translationScale = 1.0 / 100.0;
  rigidOptimizerScales[2] =  translationScale;
  rigidOptimizerScales[3] =  translationScale;
  
  rigidOptimizerScales[0] =  1.0;
  translationScale = 1.0 / 100.0;
  rigidOptimizerScales[1] =  translationScale;
  rigidOptimizerScales[2] =  translationScale;
  
  //create a mask for the body only in order to initialize the registration
  const short lowerThreshold =  -1024 ;
  const short intensity_offset =  1024 ;
  
  CTImageIteratorType   fixedctIt ( fixedCT, fixedCT->GetBufferedRegion() );
  CTImageIteratorType   movingctIt ( movingCT, movingCT->GetBufferedRegion() );
  fixedctIt.GoToBegin();
  movingctIt.GoToBegin();
  while ( !fixedctIt.IsAtEnd() )
    {
      short original_value = fixedctIt.Get();
      if(original_value < lowerThreshold)
	{
	  original_value = lowerThreshold;
	}
      fixedctIt.Set(original_value+=intensity_offset);
      ++fixedctIt;
    }
  while ( !movingctIt.IsAtEnd() )
    {
      short original_value = movingctIt.Get();
      if(original_value < lowerThreshold)
	{
	  original_value = lowerThreshold;
	}
      movingctIt.Set(original_value+=intensity_offset);
      ++movingctIt;
    }
  
  typename RigidInitializerTypeIntensity::Pointer rigidInitializer = RigidInitializerTypeIntensity::New(); //NAMIC 2015 good
  rigidInitializer->SetTransform( rigidTransform ); //NAMIC 2015 good
  rigidInitializer->SetFixedImage(fixedCT);  //NAMIC 2015 good
  rigidInitializer->SetMovingImage(movingCT); //NAMIC 2015 good
  //rigidInitializer->SetMovingImageMask(filter_moving->GetOutput());
  rigidInitializer->MomentsOn(); //NAMIC 2015 good
  rigidInitializer->InitializeTransform();  //NAMIC 2015 good  
  
  typename OptimizerType::Pointer rigid_optimizer = OptimizerType::New();
  rigid_optimizer->SetScales(rigidOptimizerScales);
  rigid_optimizer->SetMaximumStepLength(0.2); 
  rigid_optimizer->SetMinimumStepLength(0.001);
  rigid_optimizer->SetNumberOfIterations(1000);
  
  typename CTRegistrationType::Pointer registration = CTRegistrationType::New();  
  
  std::cout<< " setting registration parameters "<<std::endl;
  typename ShortImageType::RegionType fixedRegion = fixedCT->GetBufferedRegion();
  
  registration->SetMetric( nc_metric );
  registration->SetFixedImage(fixedCT  ); //fixedSmoother->GetOutput()
  registration->SetMovingImage(movingCT); //movingSmoother->GetOutput()
  registration->SetOptimizer( rigid_optimizer );
  registration->SetInterpolator( CTinterpolator );
  registration->SetTransform( rigidTransform ); //NAMIC 2015 good
  
  registration->SetInitialTransformParameters( rigidTransform->GetParameters());  //NAMIC 2015 good

  std::cout<< "  registrating "<<std::endl;    
  try
    {
      registration->Initialize();
      registration->Update();
    }
  catch( itk::ExceptionObject &excp )
    {
      std::cerr << "ExceptionObject caught while executing registration" <<
	std::endl;
      std::cerr << excp << std::endl;
    }
  
  std::cout << "Optimizer stop condition = "
	    << registration->GetOptimizer()->GetStopConditionDescription()
	    << std::endl;
  
  rigidTransform->SetParameters( registration->GetLastTransformParameters() );  


  /****
       Now for the affine registration
  ****/
  
  std::cout<<"affine registration new" <<std::endl;
  
  typename AffineTransformType::Pointer affineTransform = AffineTransformType::New();
  
  affineTransform->SetCenter( rigidTransform->GetCenter() );  //NAMIC 2015 good
  affineTransform->SetTranslation( rigidTransform->GetTranslation() );  //NAMIC 2015 good
  affineTransform->SetMatrix( rigidTransform->GetMatrix() );  //NAMIC 2015 good
 
  std::cout<<"affine registration 2" <<std::endl;
  typename CTRegistrationType::Pointer registration_affine = CTRegistrationType::New();  

  registration_affine->SetMetric( nc_metric );
  registration_affine->SetFixedImage(fixedCT  ); //fixedSmoother->GetOutput()
  registration_affine->SetMovingImage(movingCT); //movingSmoother->GetOutput()
  
  OptimizerType::Pointer affine_optimizer = OptimizerType::New();
  OptimizerScalesType optimizerScales(affineTransform->GetNumberOfParameters());
  //OptimizerScalesType similarityOptimizerScales(similarityTransform->GetNumberOfParameters());
  

  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = 1.0;
  translationScale = 1/1000.0;
  optimizerScales[4]  = translationScale;
  optimizerScales[5] = translationScale;

  
  affine_optimizer->SetScales( optimizerScales );
  affine_optimizer->SetMaximumStepLength( 0.2000  );
  affine_optimizer->SetMinimumStepLength( 0.0001 );
  affine_optimizer->SetNumberOfIterations( 300);//300 );
  registration_affine->SetOptimizer( affine_optimizer );
  registration_affine->SetInterpolator( CTinterpolator );
  
  registration_affine->SetTransform( affineTransform );
  registration_affine->SetInitialTransformParameters( affineTransform->GetParameters() );
  
  std::cout<<"affine registration 6" <<std::endl;  
  //
  // The Affine transform has 12 parameters we use therefore a more samples to run
  // this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  //nc_metric->SetNumberOfSpatialSamples( 50000L );
  
  std::cout << "Starting CT affine registration..." << std::endl;
  
  try
    {
      registration_affine->Initialize();
      registration_affine->Update();
    }
  catch( itk::ExceptionObject &excp )
    {
      std::cerr << "ExceptionObject caught while executing registration" <<
	std::endl;
      std::cerr << excp << std::endl;
    }
  
  
  affineTransform->SetParameters( registration_affine->GetLastTransformParameters());
  //affineTransform2D->SetParameters( registration_affine->GetInitialTransformParameters());
  
  /*
    GradOptimizerType::ParametersType finalParams;
    finalParams =registration->GetLastTransformParameters();
    
    std::cout <<finalParams<<std::endl;
    
    AffineTransformType2D::Pointer finalTransform2D = AffineTransformType2D::New();
    finalTransform2D->SetParameters( finalParams);//affineTransform2D->GetParameters());//finalParams );
    finalTransform2D->SetCenter( rigidTransform->GetCenter() );
  */
  std::cout<<"writing final transform"<<std::endl;
  
  if ( strcmp(outputTransformFileName.c_str(), "q") != 0 )
    {
      std::string infoFilename = outputTransformFileName;
      int result = infoFilename.find_last_of('.');
      if (std::string::npos != result)
	infoFilename.erase(result);
      // append extension:
      infoFilename.append(".xml");
      
      REGISTRATION_XML_DATA labelMapRegistrationXMLData;
      //labelMapRegistrationXMLData.similarityValue = (float)(bestValue);
      const char *similarity_type = nc_metric->GetNameOfClass();
      labelMapRegistrationXMLData.similarityMeasure.assign(similarity_type);
      
      //labelMapRegistrationXMLData.image_type.assign("leftLungRightLung");
      labelMapRegistrationXMLData.transformationIndex = 0;
      
      
      //if the patient IDs are specified  as args, use them,
      //otherwise,NA
      
      int pathLength = 0, pos=0, next=0;
      
      if ( strcmp(movingImageID.c_str(), "q") != 0 )
	{
	  labelMapRegistrationXMLData.sourceID.assign(movingImageID);
	}
      else
	{
	  labelMapRegistrationXMLData.sourceID.assign("N/A");
	}	
      
      if ( strcmp(fixedImageID.c_str(), "q") != 0 )
	{
	  labelMapRegistrationXMLData.destID.assign(fixedImageID);
	}
      else
	{
	  labelMapRegistrationXMLData.destID.assign("N/A");
	}
      
      //remove path from output transformation file before storing in xml
      std::cout<<"outputtransform filename="<<outputTransformFileName.c_str()<<std::endl;
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
      transformWriter->SetInput( affineTransform );
      
      
      transformWriter->SetFileName( outputTransformFileName );
      transformWriter->Update();
    }
  
  std::cout << "DONE." << std::endl;
  
  return 0;
}




template <unsigned int TDimension>
int DoIT3(int argc, char * argv[])
{  
  std::vector< unsigned char >  regionVec;
  std::vector< unsigned char >  typeVec;
  std::vector< unsigned char >  regionPairVec;
  std::vector< unsigned char >  typePairVec; 
  
  PARSE_ARGS;
  
  typedef itk::Image< unsigned short, TDimension >                     LabelMapType;
  typedef itk::Image< short, TDimension >                              ShortImageType;
  
  typedef itk::ImageRegionIterator< ShortImageType >                                   CTImageIteratorType;  
  typedef itk::GDCMImageIO                                                                          ImageIOType;
  typedef itk::GDCMSeriesFileNames                                                                  NamesGeneratorType;
  typedef itk::ImageFileReader< LabelMapType >                                                    LabelMapReaderType;
  typedef itk::ImageFileReader< ShortImageType >                                                  ShortReaderType;
  typedef itk::ImageFileWriter< ShortImageType >                                                  ImageWriterType;
  
  typedef itk::CIPExtractChestLabelMapImageFilter< TDimension >                                                    LabelMapExtractorType;
  typedef itk::ImageMaskSpatialObject< TDimension >                                                          MaskType;
  typedef itk::Image< unsigned char, TDimension >                                                            ImageMaskType;
  typedef itk::ImageFileReader< ImageMaskType >                                                     MaskReaderType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                                       LabelMapIteratorType;
  
  typedef itk::DiscreteGaussianImageFilter< ShortImageType, ShortImageType >                    GaussianFilterType;  
  typedef itk::NormalizeImageFilter<ShortImageType,ShortImageType>                              FixedNormalizeFilterType;
  typedef itk::NormalizeImageFilter<ShortImageType,ShortImageType>                              MovingNormalizeFilterType;
  typedef itk::BinaryThresholdImageFilter< ShortImageType, ShortImageType >  FilterType;
  
  //similarity
  typedef itk::NormalizedCorrelationImageToImageMetric<ShortImageType, ShortImageType  >        ncMetricType;
  typedef itk::MeanSquaresImageToImageMetric<ShortImageType, ShortImageType  >        meanSquaredMetricType;
  
  //transformations   
  typedef itk::IdentityTransform< double, TDimension >                                                       IdentityType;  
  typedef itk::AffineTransform<double, TDimension >                                                          AffineTransformType;
  typedef itk::Similarity2DTransform< double>                                                       SimilarityTransformType;
  
  typedef itk::RegularStepGradientDescentOptimizer  OptimizerType;
  
  typedef itk::GradientDescentOptimizer                                                             GradOptimizerType;
  typedef OptimizerType::ScalesType                                                                 OptimizerScalesType;
  
  typedef itk::ImageRegistrationMethod<ShortImageType,ShortImageType >                          CTRegistrationType;
  typedef itk::LinearInterpolateImageFunction< ShortImageType, double >                  CTInterpolatorType;
  typedef itk::ResampleImageFilter< ShortImageType, ShortImageType >                            ResampleFilterType;
  
  
  //Read in region and type pair
  std::vector< REGIONTYPEPAIR > regionTypePairVec;
  
  for ( unsigned int i=0; i<regionVecArg.size(); i++ )
    {
      regionVec.push_back(regionVecArg[i]);
    }
  for ( unsigned int i=0; i<typeVecArg.size(); i++ )
    {
      typeVec.push_back( typeVecArg[i] );
    }
  if (regionPairVec.size() == typePairVecArg.size())
    {
      for ( unsigned int i=0; i<regionPairVecArg.size(); i++ )
	{
          REGIONTYPEPAIR regionTypePairTemp;
	  
          regionTypePairTemp.region = regionPairVecArg[i];
          argc--; argv++;
          regionTypePairTemp.type   = typePairVecArg[i];
	  
          regionTypePairVec.push_back( regionTypePairTemp );
	}
    }
  
  //Read in fixed image label map from file and subsample
  typename  LabelMapType::Pointer fixedLabelMap = LabelMapType::New();
  typename  LabelMapType::Pointer movingLabelMap =LabelMapType::New();
  typename  ShortImageType::Pointer fixedCT = ShortImageType::New();
  typename ShortImageType::Pointer movingCT = ShortImageType::New();
  
  try 
    {
      std::cout << "Reading CT from file..." << std::endl;
      
      fixedCT = ReadCTFromFile< TDimension > ( fixedImageFileName );
      if (fixedCT.GetPointer() == NULL)
	{
	  return cip::LABELMAPREADFAILURE;
	}
      
    }
  catch (cip::ExceptionObject &exep)
    {
      std::cerr << "Error: No CT specified"<< std::endl;
      std::cerr << exep << std::endl;
      return cip::EXITFAILURE;
    }
  
  //Read in moving image label map from file and subsample
  
  try
    {
      std::cout << "Reading CT from file..." << std::endl;
      
      movingCT = ReadCTFromFile< TDimension > ( movingImageFileName );
      
      if (movingCT.GetPointer() == NULL)
	{
	  return cip::LABELMAPREADFAILURE;
	}
      
    }
  catch (cip::ExceptionObject &exep)
    {
      std::cerr <<"Error: No CT specified"<< std::endl;
      std::cerr << exep << std::endl;
      return cip::EXITFAILURE;
    }
    
  typename ncMetricType::Pointer nc_metric_affine = ncMetricType::New();
  typename meanSquaredMetricType::Pointer nc_metric = meanSquaredMetricType::New();
  
  typename MaskType::Pointer  spatialObjectMask = MaskType::New();
  typename MaskReaderType::Pointer  maskReader = MaskReaderType::New();
  
  if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
    {
      std::cout<<"reading moving label map "<<movingLabelmapFileName.c_str() <<std::endl;
      maskReader->SetFileName(movingLabelmapFileName.c_str() );
      
      try
        {
	  maskReader->Update();
        }
      catch( itk::ExceptionObject & err )
        {
	  std::cerr << "ExceptionObject caught !" << std::endl;
	  std::cerr << err << std::endl;
	  return EXIT_FAILURE;
        }
      spatialObjectMask->SetImage(maskReader->GetOutput());
      nc_metric_affine->SetMovingImageMask( spatialObjectMask );
      //nc_metric->SetMovingImageMask( spatialObjectMask );
      
    }
  
  
  std::cout<<"initializing transform"<<std::endl;
  

  typedef itk::Rigid3DTransform< double>            RigidTransformType;
  typedef itk::CenteredTransformInitializer< RigidTransformType, ShortImageType, ShortImageType >  RigidInitializerTypeIntensity;
 
  typename RigidTransformType::Pointer rigidTransform = RigidTransformType::New();
  typename CTInterpolatorType::Pointer CTinterpolator = CTInterpolatorType::New();  
  
  //initial registration with a rigid transform      
  //create a similarity transform to experiment
  typename SimilarityTransformType::Pointer similarityTransform = SimilarityTransformType::New();
  
  OptimizerScalesType rigidOptimizerScales(rigidTransform->GetNumberOfParameters());
  
  rigidOptimizerScales[0] =  1.0;
  rigidOptimizerScales[1] =  1.0;
  translationScale = 1.0 / 100.0;
  rigidOptimizerScales[2] =  translationScale;
  rigidOptimizerScales[3] =  translationScale;
  
  rigidOptimizerScales[0] =  1.0;
  translationScale = 1.0 / 100.0;
  rigidOptimizerScales[1] =  translationScale;
  rigidOptimizerScales[2] =  translationScale;
  
  //create a mask for the body only in order to initialize the registration
  const short lowerThreshold =  -1024 ;
  const short intensity_offset =  1024 ;
  
  CTImageIteratorType   fixedctIt ( fixedCT, fixedCT->GetBufferedRegion() );
  CTImageIteratorType   movingctIt ( movingCT, movingCT->GetBufferedRegion() );
  fixedctIt.GoToBegin();
  movingctIt.GoToBegin();
  while ( !fixedctIt.IsAtEnd() )
    {
      short original_value = fixedctIt.Get();
      if(original_value < lowerThreshold)
	{
	  original_value = lowerThreshold;
	}
      fixedctIt.Set(original_value+=intensity_offset);
      ++fixedctIt;
    }
  while ( !movingctIt.IsAtEnd() )
    {
      short original_value = movingctIt.Get();
      if(original_value < lowerThreshold)
	{
	  original_value = lowerThreshold;
	}
      movingctIt.Set(original_value+=intensity_offset);
      ++movingctIt;
    }
  
  typename RigidInitializerTypeIntensity::Pointer rigidInitializer = RigidInitializerTypeIntensity::New(); //NAMIC 2015 good
  rigidInitializer->SetTransform( rigidTransform ); //NAMIC 2015 good
  rigidInitializer->SetFixedImage(fixedCT);  //NAMIC 2015 good
  rigidInitializer->SetMovingImage(movingCT); //NAMIC 2015 good
  //rigidInitializer->SetMovingImageMask(filter_moving->GetOutput());
  rigidInitializer->MomentsOn(); //NAMIC 2015 good
  rigidInitializer->InitializeTransform();  //NAMIC 2015 good  
  
  typename OptimizerType::Pointer rigid_optimizer = OptimizerType::New();
  rigid_optimizer->SetScales(rigidOptimizerScales);
  rigid_optimizer->SetMaximumStepLength(0.2); 
  rigid_optimizer->SetMinimumStepLength(0.001);
  rigid_optimizer->SetNumberOfIterations(1000);
  
  typename CTRegistrationType::Pointer registration = CTRegistrationType::New();  
  
  std::cout<< " setting registration parameters "<<std::endl;
  typename ShortImageType::RegionType fixedRegion = fixedCT->GetBufferedRegion();
  
  registration->SetMetric( nc_metric );
  registration->SetFixedImage(fixedCT  ); //fixedSmoother->GetOutput()
  registration->SetMovingImage(movingCT); //movingSmoother->GetOutput()
  registration->SetOptimizer( rigid_optimizer );
  registration->SetInterpolator( CTinterpolator );
  registration->SetTransform( rigidTransform ); //NAMIC 2015 good
  
  registration->SetInitialTransformParameters( rigidTransform->GetParameters());  //NAMIC 2015 good

  std::cout<< "  registrating "<<std::endl;    
  try
    {
      registration->Initialize();
      registration->Update();
    }
  catch( itk::ExceptionObject &excp )
    {
      std::cerr << "ExceptionObject caught while executing registration" <<
	std::endl;
      std::cerr << excp << std::endl;
    }
  
  std::cout << "Optimizer stop condition = "
	    << registration->GetOptimizer()->GetStopConditionDescription()
	    << std::endl;
  
  rigidTransform->SetParameters( registration->GetLastTransformParameters() );  


  /****
       Now for the affine registration
  ****/
  
  std::cout<<"affine registration new" <<std::endl;
  
  typename AffineTransformType::Pointer affineTransform = AffineTransformType::New();
  
  affineTransform->SetCenter( rigidTransform->GetCenter() );  //NAMIC 2015 good
  affineTransform->SetTranslation( rigidTransform->GetTranslation() );  //NAMIC 2015 good
  affineTransform->SetMatrix( rigidTransform->GetMatrix() );  //NAMIC 2015 good
 
  std::cout<<"affine registration 2" <<std::endl;
  typename CTRegistrationType::Pointer registration_affine = CTRegistrationType::New();  

  registration_affine->SetMetric( nc_metric );
  registration_affine->SetFixedImage(fixedCT  ); //fixedSmoother->GetOutput()
  registration_affine->SetMovingImage(movingCT); //movingSmoother->GetOutput()
  
  OptimizerType::Pointer affine_optimizer = OptimizerType::New();
  OptimizerScalesType optimizerScales(affineTransform->GetNumberOfParameters());
  //OptimizerScalesType similarityOptimizerScales(similarityTransform->GetNumberOfParameters());
  
      
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = 1.0;
  optimizerScales[4] = 1.0;
  optimizerScales[5] = 1.0;
  optimizerScales[6] = 1.0;
  optimizerScales[7] = 1.0;
  optimizerScales[8] = 1.0;
  optimizerScales[9]  = translationScale;
  optimizerScales[10] = translationScale;
  optimizerScales[11] = translationScale; 
  translationScale = 1/1000.0;

  
  affine_optimizer->SetScales( optimizerScales );
  affine_optimizer->SetMaximumStepLength( 0.2000  );
  affine_optimizer->SetMinimumStepLength( 0.0001 );
  affine_optimizer->SetNumberOfIterations( 300);//300 );
  registration_affine->SetOptimizer( affine_optimizer );
  registration_affine->SetInterpolator( CTinterpolator );
  
  registration_affine->SetTransform( affineTransform );
  registration_affine->SetInitialTransformParameters( affineTransform->GetParameters() );
  
  std::cout<<"affine registration 6" <<std::endl;  
  //
  // The Affine transform has 12 parameters we use therefore a more samples to run
  // this stage.
  //
  // Regulating the number of samples in the Metric is equivalent to performing
  // multi-resolution registration because it is indeed a sub-sampling of the
  // image.
  //nc_metric->SetNumberOfSpatialSamples( 50000L );
  
  std::cout << "Starting CT affine registration..." << std::endl;
  
  try
    {
      registration_affine->Initialize();
      registration_affine->Update();
    }
  catch( itk::ExceptionObject &excp )
    {
      std::cerr << "ExceptionObject caught while executing registration" <<
	std::endl;
      std::cerr << excp << std::endl;
    }
  
  
  affineTransform->SetParameters( registration_affine->GetLastTransformParameters());
  //affineTransform2D->SetParameters( registration_affine->GetInitialTransformParameters());
  
  /*
    GradOptimizerType::ParametersType finalParams;
    finalParams =registration->GetLastTransformParameters();
    
    std::cout <<finalParams<<std::endl;
    
    AffineTransformType2D::Pointer finalTransform2D = AffineTransformType2D::New();
    finalTransform2D->SetParameters( finalParams);//affineTransform2D->GetParameters());//finalParams );
    finalTransform2D->SetCenter( rigidTransform->GetCenter() );
  */
  std::cout<<"writing final transform"<<std::endl;
  
  if ( strcmp(outputTransformFileName.c_str(), "q") != 0 )
    {
      std::string infoFilename = outputTransformFileName;
      int result = infoFilename.find_last_of('.');
      if (std::string::npos != result)
	infoFilename.erase(result);
      // append extension:
      infoFilename.append(".xml");
      
      REGISTRATION_XML_DATA labelMapRegistrationXMLData;
      //labelMapRegistrationXMLData.similarityValue = (float)(bestValue);
      const char *similarity_type = nc_metric->GetNameOfClass();
      labelMapRegistrationXMLData.similarityMeasure.assign(similarity_type);
      
      //labelMapRegistrationXMLData.image_type.assign("leftLungRightLung");
      labelMapRegistrationXMLData.transformationIndex = 0;
      
      
      //if the patient IDs are specified  as args, use them,
      //otherwise,NA
      
      int pathLength = 0, pos=0, next=0;
      
      if ( strcmp(movingImageID.c_str(), "q") != 0 )
	{
	  labelMapRegistrationXMLData.sourceID.assign(movingImageID);
	}
      else
	{
	  labelMapRegistrationXMLData.sourceID.assign("N/A");
	}	
      
      if ( strcmp(fixedImageID.c_str(), "q") != 0 )
	{
	  labelMapRegistrationXMLData.destID.assign(fixedImageID);
	}
      else
	{
	  labelMapRegistrationXMLData.destID.assign("N/A");
	}
      
      //remove path from output transformation file before storing in xml
      std::cout<<"outputtransform filename="<<outputTransformFileName.c_str()<<std::endl;
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
      transformWriter->SetInput( affineTransform );
      
      
      transformWriter->SetFileName( outputTransformFileName );
      transformWriter->Update();
    }
  
  std::cout << "DONE." << std::endl;
  
  return 0;
}




int main( int argc, char *argv[] )
{
  
  PARSE_ARGS;
  switch(dimension)
    {
    case 2:
      {
	DoIT2<2>( argc, argv);
	break;
      }
      case 3:
      {
	DoIT3<3>( argc, argv);
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
