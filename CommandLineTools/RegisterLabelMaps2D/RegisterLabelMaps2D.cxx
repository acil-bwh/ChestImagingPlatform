
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
#include "itkAmoebaOptimizer.h"
#include "itkImageRegistrationMethod.h"
#include "itkCenteredTransformInitializer.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkQuaternionRigidTransform.h"

#include "RegisterLabelMaps2DCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include <sstream>
#include "cipExceptionObject.h"


namespace
{
#define MY_ENCODING "ISO-8859-1"


  typedef itk::IdentityTransform< double, 2 >                                                   IdentityType;
  typedef itk::AffineTransform<double, 2 >                                                      TransformType2D; 

  typedef itk::Image< unsigned short, 2 >                                                       LabelMapType2D; 
  typedef itk::CIPExtractChestLabelMapImageFilter                                               LabelMapExtractorType; 
  typedef itk::GDCMImageIO                                                                      ImageIOType;
  typedef itk::GDCMSeriesFileNames                                                              NamesGeneratorType; 
  typedef itk::Image< short, 2 >                                                                ShortImageType2D;
  typedef itk::ResampleImageFilter< LabelMapType2D, LabelMapType2D >                            ResampleFilterType;
  typedef itk::ImageFileReader< LabelMapType2D >                                                LabelMap2DReaderType;    

  typedef itk::ImageRegionIteratorWithIndex< LabelMapType2D >                                   IteratorType;
  typedef itk::ImageFileWriter< LabelMapType2D >                                                ImageWriterType;
  typedef itk::ResampleImageFilter< LabelMapType2D, LabelMapType2D >                            ResampleType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType2D >                                   LabelMapIteratorType;       
  typedef itk::NormalizeImageFilter<ShortImageType2D,ShortImageType2D>                          FixedNormalizeFilterType;
  typedef itk::NormalizeImageFilter<ShortImageType2D,ShortImageType2D>                          MovingNormalizeFilterType;
  typedef itk::ImageMaskSpatialObject< 2 >                                                      MaskType;   
  typedef itk::Image< unsigned char, 2 >                                                        ImageMaskType;
  typedef itk::ImageFileReader< ImageMaskType >                                                 MaskReaderType;

  typedef itk::RegularStepGradientDescentOptimizer                                              OptimizerType;    
  typedef itk::GradientDescentOptimizer                                                         GradOptimizerType; 
  typedef itk::AmoebaOptimizer                                                                  AmoebaOptimizerType;   
  typedef OptimizerType::ScalesType                                                             OptimizerScalesType; 
  typedef itk::ImageRegistrationMethod<LabelMapType2D,LabelMapType2D >                          RegistrationType;    
  typedef itk::KappaStatisticImageToImageMetric<LabelMapType2D, LabelMapType2D >                MetricType;    
  typedef itk::NearestNeighborInterpolateImageFunction< LabelMapType2D, double >                InterpolatorType;
  typedef itk::LinearInterpolateImageFunction< ShortImageType2D, double >                         LinearInterpolatorType;
  typedef itk::CenteredTransformInitializer< TransformType2D, LabelMapType2D, LabelMapType2D >  InitializerType2D;

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

  void WriteTransformFile( TransformType2D::Pointer transform, char* fileName )
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
 
  LabelMapType2D::Pointer ReadLabelMap2DFromFile( std::string labelMapFileName )
  {
    if(strcmp( labelMapFileName.c_str(), "q") == 0 )
      {
      throw cip::ExceptionObject( __FILE__, __LINE__, "RegisterLabelMaps2D::main()", " No lung label map specified" );
      }
    std::cout << "Reading label map..." << std::endl;
    LabelMap2DReaderType::Pointer reader = LabelMap2DReaderType::New();
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

int main( int argc, char *argv[] )
{

  std::vector< unsigned char >  regionVec;
  std::vector< unsigned char >  typeVec;
  std::vector< unsigned char >  regionPairVec;
  std::vector< unsigned char >  typePairVec; 

  PARSE_ARGS;
   
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
  LabelMapType2D::Pointer fixedLabelMap2D = LabelMapType2D::New();
  LabelMapType2D::Pointer movingLabelMap2D =LabelMapType2D::New();
  
  try
    {
      std::cout << "Reading label map from file..." << std::endl;

      fixedLabelMap2D = ReadLabelMap2DFromFile( fixedImageFileName );
      if (fixedLabelMap2D.GetPointer() == NULL)
	{
	  return cip::LABELMAPREADFAILURE;
	}
    }
  catch (cip::ExceptionObject &exep)
    {
      std::cerr << "Error: No lung label map specified" << std::endl;
      std::cerr << exep << std::endl;
      return cip::EXITFAILURE;
    }
 
  //Read in moving image label map
  try
    {
      std::cout << "Reading label map from file..." << std::endl;
      movingLabelMap2D = ReadLabelMap2DFromFile( movingImageFileName );
      
      if (movingLabelMap2D.GetPointer() == NULL)
	{
	  throw cip::ExceptionObject( __FILE__, __LINE__, "RegisterLabelMaps2D::main()", "Problem opening file" );
	  return cip::LABELMAPREADFAILURE;
	}
    }
  catch (cip::ExceptionObject &exep)
    {
      std::cerr << "Error: No lung label map specified" << std::endl;
      std::cerr << exep << std::endl;
      return cip::EXITFAILURE;
    }      

  LabelMapIteratorType fIt( fixedLabelMap2D, fixedLabelMap2D->GetBufferedRegion() );
  LabelMapIteratorType mIt( movingLabelMap2D, movingLabelMap2D->GetBufferedRegion() );
  
  fIt.GoToBegin();
  mIt.GoToBegin();
  while ( !fIt.IsAtEnd() )
    {
      if ( (fIt.Get() != 0 && fIt.Get() != 1) || (mIt.Get() != 0 && mIt.Get() != 1 ))
	{
	  std::cout << "oops!" << std::endl;
	}

      ++fIt;
      ++mIt;
    }
  std::cout << "whew" << std::endl;

  MetricType::Pointer metric = MetricType::New();
    metric->SetForegroundValue( 1 );    //because we are minimizing as opposed to maximizing

  MaskType::Pointer  spatialObjectMask = MaskType::New();
  MaskReaderType::Pointer  maskReader = MaskReaderType::New();
          
  TransformType2D::Pointer transform2D = TransformType2D::New();

  std::cout << "initializing transform" << std::endl;
  InitializerType2D::Pointer initializer2D = InitializerType2D::New();
    initializer2D->SetTransform( transform2D );
    initializer2D->SetFixedImage(fixedLabelMap2D );
    initializer2D->SetMovingImage( movingLabelMap2D);
    initializer2D->MomentsOn();
    initializer2D->InitializeTransform(); 

  std::cout << "Initial transform..." << std::endl;
  std::cout << transform2D << std::endl;

  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  GradOptimizerType::Pointer grad_optimizer = GradOptimizerType::New();
  typedef OptimizerType::ScalesType OptimizerScalesType; 
  OptimizerScalesType optimizerScales(transform2D->GetNumberOfParameters());     
    optimizerScales[0] =  1.0; 
    optimizerScales[1] =  1.0; 
    optimizerScales[2] =  1.0; 
    optimizerScales[3] =  1.0; 
    optimizerScales[4] =  translationScale; 
    optimizerScales[5] =  translationScale; 

   
  OptimizerType::Pointer optimizer = OptimizerType::New(); 
    optimizer->SetScales(optimizerScales);
    optimizer->SetMaximumStepLength(0.1); 
    optimizer->SetMinimumStepLength(0.0001); 
    unsigned int maxNumberOfIterations = 300;
    optimizer->SetNumberOfIterations(maxNumberOfIterations);
       
    /*
  AmoebaOptimizerType::Pointer optimizer = AmoebaOptimizerType::New();
  optimizer->SetScales( optimizerScales );
  optimizer->SetParametersConvergenceTolerance( 0.00001 ); // reasonable defaults
  optimizer->SetFunctionConvergenceTolerance( 0.0001 );
    
  double steplength = 0.1;

  unsigned int maxNumberOfIterations = 1500;

  optimizer->SetMaximumNumberOfIterations( maxNumberOfIterations );
    */
  optimizer->MinimizeOn();


  RegistrationType::Pointer registration = RegistrationType::New();  
    registration->SetMetric( metric );
    registration->SetFixedImage( fixedLabelMap2D );
    registration->SetMovingImage( movingLabelMap2D );    
    registration->SetOptimizer( optimizer  );
    registration->SetInterpolator( interpolator );
    registration->SetTransform( transform2D );      
    registration->SetInitialTransformParameters( transform2D->GetParameters());      
  try
    {
    registration->Initialize();
    registration->Update();
    }
  catch( itk::ExceptionObject &excp )
    {
    std::cerr << "ExceptionObject caught while executing registration" <<std::endl;
    std::cerr << excp << std::endl;
    }
      
  //get all params to output to file
  double bestValue;
  
  numberOfIterations = maxNumberOfIterations;
  bestValue = optimizer->GetValue();
  
  std::cout <<" best similarity value, amoeba = " <<bestValue<<std::endl;
  
  OptimizerType::ParametersType finalParams;
    finalParams = registration->GetLastTransformParameters();

  TransformType2D::Pointer finalTransform2D = TransformType2D::New();
    finalTransform2D->SetParameters(finalParams );
    finalTransform2D->SetCenter( transform2D->GetCenter() );

  std::cout << finalTransform2D << std::endl;  
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
      labelMapRegistrationXMLData.similarityValue = (float)(bestValue);
      const char *similarity_type = metric->GetNameOfClass();
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
      transformWriter->SetInput( finalTransform2D );

   
      transformWriter->SetFileName( outputTransformFileName );
      transformWriter->Update();
    }

  std::cout << "DONE." << std::endl;

  return 0;
}
