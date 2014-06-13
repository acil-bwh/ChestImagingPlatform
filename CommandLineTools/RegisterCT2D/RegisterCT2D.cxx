
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
#include "itkImageRegistrationMethod.h"
#include "itkCenteredTransformInitializer.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkLBFGSOptimizer.h"

#include "RegisterCT2DCLP.h"
#include "cipConventions.h"
#include "cipHelper.h"
#include <sstream>
#include "cipExceptionObject.h"


namespace
{
#define MY_ENCODING "ISO-8859-1"
  typedef itk::Image< unsigned short, 2 >                                                           LabelMapType2D; 
  //typedef itk::RegularStepGradientDescentOptimizer                                                  OptimizerType;
  typedef itk::LBFGSOptimizer       OptimizerType;  
  typedef itk::GradientDescentOptimizer GradOptimizerType;
    
  typedef OptimizerType::ScalesType                                                                 OptimizerScalesType;
 
  typedef itk::IdentityTransform< double, 2 >                                                       IdentityType;
  typedef itk::CIPExtractChestLabelMapImageFilter                                                   LabelMapExtractorType;
 
  typedef itk::GDCMImageIO                                                                          ImageIOType;
  typedef itk::GDCMSeriesFileNames                                                                  NamesGeneratorType;
 
  typedef itk::Image< short, 2 >                                                                    ShortImageType2D;
  typedef itk::ResampleImageFilter< LabelMapType2D, LabelMapType2D >                                ResampleFilterType;
  
    
  typedef itk::AffineTransform<double, 2 >                                                          TransformType2D;
  //typedef itk::QuaternionRigidTransform<double>                                                     RigidTransformType;

  typedef itk::CenteredTransformInitializer< TransformType2D, LabelMapType2D, LabelMapType2D >      InitializerType2D;
  typedef itk::CenteredTransformInitializer< TransformType2D, ShortImageType2D, ShortImageType2D >  InitializerType2DIntensity;
    
  typedef itk::ImageFileReader< LabelMapType2D >                                               LabelMap2DReaderType;
  typedef itk::ImageFileReader< ShortImageType2D >                                               ShortReaderType;
    
  typedef itk::ImageRegistrationMethod<LabelMapType2D,LabelMapType2D >                         RegistrationType;
  typedef itk::ImageRegistrationMethod<ShortImageType2D,ShortImageType2D >                         CTRegistrationType;

    
  typedef itk::KappaStatisticImageToImageMetric<LabelMapType2D, LabelMapType2D >               MetricType;
  typedef itk::NormalizedCorrelationImageToImageMetric<ShortImageType2D, ShortImageType2D  >       ncMetricType;
    
  typedef itk::NearestNeighborInterpolateImageFunction< LabelMapType2D, double >               InterpolatorType;
  typedef itk::NearestNeighborInterpolateImageFunction< ShortImageType2D, double >               CTInterpolatorType;
    
    
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType2D >                                  IteratorType;
  typedef itk::ImageFileWriter< LabelMapType2D >                                               ImageWriterType;
  typedef itk::ResampleImageFilter< LabelMapType2D, LabelMapType2D >                           ResampleType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType2D >                                  LabelMapIteratorType;
    
  typedef itk::DiscreteGaussianImageFilter< ShortImageType2D, ShortImageType2D > GaussianFilterType;
    
  typedef itk::NormalizeImageFilter<ShortImageType2D,ShortImageType2D> FixedNormalizeFilterType;
  typedef itk::NormalizeImageFilter<ShortImageType2D,ShortImageType2D> MovingNormalizeFilterType;
  typedef itk::ImageMaskSpatialObject< 2 >   MaskType;
   


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
    itk::TransformFileWriter::Pointer transformWriter =
      itk::TransformFileWriter::New();
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
 

  LabelMapType2D::Pointer ReadLabelMap2DFromFile( std::string
						  labelMapFileName )
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

    
  ShortImageType2D::Pointer ReadCTFromFile( std::string fileName )
  {
    if(strcmp( fileName.c_str(), "q") == 0 )
      {
      throw cip::ExceptionObject( __FILE__, __LINE__, "RegisterLabelMaps2D::main()", " No lung label map specified" );
      }

    ShortReaderType::Pointer reader = ShortReaderType::New();
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
  ShortImageType2D::Pointer fixedCT2D = ShortImageType2D::New();
  ShortImageType2D::Pointer movingCT2D = ShortImageType2D::New();
    
  if(isIntensity != true)
    {
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
    }
    
  else //intensity based registration, load CT images
    {
      try 
        {
	  std::cout << "Reading CT from file..." << std::endl;
            
	  fixedCT2D = ReadCTFromFile( fixedImageFileName );
	  if (fixedCT2D.GetPointer() == NULL)
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
            
	  movingCT2D = ReadCTFromFile( movingImageFileName );
            
	  if (movingCT2D.GetPointer() == NULL)
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
    }
    

  MetricType::Pointer metric = MetricType::New();
  metric->SetForegroundValue( 1);    //because we are minimizing as opposed to maximizing
  ncMetricType::Pointer nc_metric = ncMetricType::New();

  typedef itk::Image< unsigned char, 2 >   ImageMaskType;
  MaskType::Pointer  spatialObjectMask = MaskType::New();
  typedef itk::ImageFileReader< ImageMaskType >    MaskReaderType;
  MaskReaderType::Pointer  maskReader = MaskReaderType::New();
    
    if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
    {
      std::cout<<"reading fixed label map "<<movingLabelmapFileName.c_str() <<std::endl;
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
      nc_metric->SetMovingImageMask( spatialObjectMask );
        
    }
  
        
    
  TransformType2D::Pointer transform2D = TransformType2D::New();
 
  std::cout<<"initializing transform"<<std::endl;
    

    
  InitializerType2D::Pointer initializer2D = InitializerType2D::New();
  InitializerType2DIntensity::Pointer initializer2DIntensity = InitializerType2DIntensity::New();
   
    
  OptimizerType::Pointer optimizer = OptimizerType::New();
  GradOptimizerType::Pointer grad_optimizer = GradOptimizerType::New();
   
  FixedNormalizeFilterType::Pointer fixedNormalizer =FixedNormalizeFilterType::New();
  MovingNormalizeFilterType::Pointer movingNormalizer =MovingNormalizeFilterType::New();
    
  GaussianFilterType::Pointer fixedSmoother  = GaussianFilterType::New();
  GaussianFilterType::Pointer movingSmoother = GaussianFilterType::New();
  fixedSmoother->SetVariance( 1.5 ); 
  movingSmoother->SetVariance( 1.5 );
  //fixedSmoother->SetVariance( 0.0 );
  //movingSmoother->SetVariance( 0.0 );
  if(isIntensity != true)
    {
      initializer2D->SetTransform( transform2D );
      initializer2D->SetFixedImage(fixedLabelMap2D );
      initializer2D->SetMovingImage( movingLabelMap2D);
      initializer2D->MomentsOn();
      initializer2D->InitializeTransform(); //this makes it work
    }
  else
    {
      std::cout<<"initializing intensity transform.."<<std::endl;
      fixedSmoother->SetInput( fixedCT2D );
      fixedSmoother->Update();
      movingSmoother->SetInput( movingCT2D);
      movingSmoother->Update();
      
      fixedNormalizer->SetInput(  fixedSmoother->GetOutput() );
      movingNormalizer->SetInput( movingSmoother->GetOutput() );
      fixedNormalizer->Update();
      movingNormalizer->Update();
      initializer2DIntensity->SetTransform( transform2D );
      initializer2DIntensity->SetFixedImage(fixedNormalizer->GetOutput());//fixedCT2D);//normalizer ruins it fixedNormalizer->GetOutput());
    initializer2DIntensity->SetMovingImage(movingNormalizer->GetOutput());//movingCT2D);// movingNormalizer->GetOutput()  );
      //initializer2DIntensity->MomentsOn(); // needed to make it work better for 10465Y_INSP_STD_NJC_COPD , needed to remove it to work
      // 10476D_INSP_STD_UAB_COPD 
      initializer2DIntensity->InitializeTransform();
    }

  typedef OptimizerType::ScalesType OptimizerScalesType; 
  OptimizerScalesType optimizerScales(transform2D->GetNumberOfParameters());     
    optimizerScales[0] =  1.0; 
    optimizerScales[1] =  1.0; 
    optimizerScales[2] =  1.0; 
    optimizerScales[3] =  1.0; 
    optimizerScales[4] =  translationScale; 
    optimizerScales[5] =  translationScale; 

    /*
    optimizer->SetScales(optimizerScales);
    optimizer->SetMaximumStepLength(0.1); 
    //optimizer->SetMinimumStepLength(0.0001); 
    optimizer->SetNumberOfIterations(200);
    */

  optimizer->SetGradientConvergenceTolerance( 1.6 );
  optimizer->SetLineSearchAccuracy( 0.9 );
  optimizer->SetDefaultStepLength( .08 );
  optimizer->TraceOn();
  optimizer->SetMaximumNumberOfFunctionEvaluations( 1000 ); 

  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  CTInterpolatorType::Pointer CTinterpolator = CTInterpolatorType::New();      

  std::cout << "Starting registration..." << std::endl;

  RegistrationType::Pointer registration = RegistrationType::New();
  CTRegistrationType::Pointer CTregistration = CTRegistrationType::New();
  double bestValue;
  TransformType2D::Pointer finalTransform2D = TransformType2D::New();
    
  if (isIntensity == true)
    {
      CTregistration->SetMetric( nc_metric );
      CTregistration->SetFixedImage( fixedSmoother->GetOutput() );
      CTregistration->SetMovingImage(movingSmoother->GetOutput());
      
      CTregistration->SetOptimizer( optimizer );
      CTregistration->SetInterpolator( CTinterpolator );
      CTregistration->SetTransform( transform2D );
      
      CTregistration->SetInitialTransformParameters( transform2D->GetParameters());
      
      try
	{
          CTregistration->Initialize();
          CTregistration->Update();
	}
      catch( itk::ExceptionObject &excp )
	{
          std::cerr << "ExceptionObject caught while executing registration" <<
	    std::endl;
          
          std::cerr << excp << std::endl;
	}
      
      //get all params to output to file
      numberOfIterations = 0;//optimizer->GetCurrentIteration();
      bestValue = optimizer->GetValue();
      
      std::cout <<" best similarity value = " <<bestValue<<std::endl;
      
      GradOptimizerType::ParametersType finalParams;
      finalParams =CTregistration->GetLastTransformParameters();
      finalTransform2D->SetParameters( finalParams );
    }
      
  else
    {
      registration->SetMetric( metric );
      registration->SetFixedImage( fixedLabelMap2D );
      registration->SetMovingImage(movingLabelMap2D);
      
      registration->SetOptimizer( optimizer );
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
      //numberOfIterations = optimizer->GetCurrentIteration();
      bestValue = optimizer->GetValue();
      
      std::cout <<" best similarity value = " <<bestValue<<std::endl;
      
      OptimizerType::ParametersType finalParams;
      finalParams =registration->GetLastTransformParameters();
      finalTransform2D->SetParameters( finalParams );
    }



  finalTransform2D->SetCenter( transform2D->GetCenter() );
    
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
