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
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkMeanSquaresImageToImageMetric.h"
//#include "itkRegularStepGradientDescentOptimizerv4.h"

#include "RegisterCT2DCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include <sstream>
#include "cipExceptionObject.h"


namespace
{
#define MY_ENCODING "ISO-8859-1"

  //image
  typedef itk::Image< unsigned short, 2 >                                                           LabelMapType2D; 
  typedef itk::Image< short, 2 >                                                                    ShortImageType2D;

  typedef itk::GDCMImageIO                                                                          ImageIOType;
  typedef itk::GDCMSeriesFileNames                                                                  NamesGeneratorType;
  typedef itk::ImageFileReader< LabelMapType2D >                                                    LabelMap2DReaderType;
  typedef itk::ImageFileReader< ShortImageType2D >                                                  ShortReaderType;
  typedef itk::ImageFileWriter< ShortImageType2D >                                                  ImageWriterType;

  typedef itk::CIPExtractChestLabelMapImageFilter                                                   LabelMapExtractorType;
  typedef itk::ImageMaskSpatialObject< 2 >                                                          MaskType;
  typedef itk::Image< unsigned char, 2 >                                                            ImageMaskType;
  typedef itk::ImageFileReader< ImageMaskType >                                                     MaskReaderType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType2D >                                       LabelMapIteratorType;
    
  typedef itk::DiscreteGaussianImageFilter< ShortImageType2D, ShortImageType2D >                    GaussianFilterType;  
  typedef itk::NormalizeImageFilter<ShortImageType2D,ShortImageType2D>                              FixedNormalizeFilterType;
  typedef itk::NormalizeImageFilter<ShortImageType2D,ShortImageType2D>                              MovingNormalizeFilterType;


  //similarity
  typedef itk::NormalizedCorrelationImageToImageMetric<ShortImageType2D, ShortImageType2D  >        ncMetricType;
  typedef itk::MeanSquaresImageToImageMetric<ShortImageType2D, ShortImageType2D  >        meanSquaredMetricType;

  //typedef itk::MeanSquaresImageToImageMetricv4<
  //                                  ShortImageType2D,
  //                                  ShortImageType2D >         ncMetricType;


  //transformations   
  typedef itk::IdentityTransform< double, 2 >                                                       IdentityType;  
  typedef itk::AffineTransform<double, 2 >                                                          AffineTransformType2D;
  typedef itk::Similarity2DTransform< double>                                                       RigidTransformType;
  //typedef itk::VersorRigid2DTransform< double > RigidTransformType;
  typedef itk::CenteredTransformInitializer< RigidTransformType, ShortImageType2D, ShortImageType2D >  RigidInitializerType2DIntensity;
  //typedef itk::CenteredTransformInitializer< SimilarityTransformType2D, ShortImageType2D, ShortImageType2D >  SimilarityInitializerType2DIntensity;
  
  //registration
  //typedef itk::RegularStepGradientDescentOptimizer                                                  OptimizerType;   
  typedef itk::RegularStepGradientDescentOptimizer  OptimizerType;

  typedef itk::GradientDescentOptimizer                                                             GradOptimizerType;
  typedef OptimizerType::ScalesType                                                                 OptimizerScalesType;

  typedef itk::ImageRegistrationMethod<ShortImageType2D,ShortImageType2D >                          CTRegistrationType;
  typedef itk::NearestNeighborInterpolateImageFunction< ShortImageType2D, double >                  CTInterpolatorType;
  typedef itk::ResampleImageFilter< ShortImageType2D, ShortImageType2D >                            ResampleFilterType;


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

  void WriteTransformFile( AffineTransformType2D::Pointer transform, char* fileName )
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
    
    
  ncMetricType::Pointer nc_metric_affine = ncMetricType::New();
  meanSquaredMetricType::Pointer nc_metric = meanSquaredMetricType::New();

  MaskType::Pointer  spatialObjectMask = MaskType::New();
  MaskReaderType::Pointer  maskReader = MaskReaderType::New();
    
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
  /*
  GaussianFilterType::Pointer fixedSmoother  = GaussianFilterType::New();
  fixedSmoother->SetVariance( 1.5 );
  fixedSmoother->SetInput( fixedCT2D );
  fixedSmoother->Update();

  GaussianFilterType::Pointer movingSmoother = GaussianFilterType::New();
  movingSmoother->SetVariance( 1.5 );
  movingSmoother->SetInput( movingCT2D);
  movingSmoother->Update();

  FixedNormalizeFilterType::Pointer fixedNormalizer =FixedNormalizeFilterType::New();      
    fixedNormalizer->SetInput(  fixedSmoother->GetOutput() );
    fixedNormalizer->Update();

  MovingNormalizeFilterType::Pointer movingNormalizer =MovingNormalizeFilterType::New();
  movingNormalizer->SetInput( movingSmoother->GetOutput() );
  movingNormalizer->Update();
  */
  //initial registration with a similarity transform
  RigidTransformType::Pointer  rigidTransform = RigidTransformType::New();
  RigidInitializerType2DIntensity::Pointer rigidInitializer = RigidInitializerType2DIntensity::New();

  rigidInitializer->SetTransform( rigidTransform );
  rigidInitializer->SetFixedImage(fixedCT2D);//fixedNormalizer->GetOutput());
  rigidInitializer->SetMovingImage(movingCT2D);//movingNormalizer->GetOutput()  );
  rigidInitializer->MomentsOn();
  rigidInitializer->InitializeTransform();

  OptimizerScalesType rigidOptimizerScales(rigidTransform->GetNumberOfParameters());     
    rigidOptimizerScales[0] =  1.0; 
    rigidOptimizerScales[1] =  1.0; 
    rigidOptimizerScales[2] =  1.0; 
    translationScale = 1.0 / 1000.0;
    rigidOptimizerScales[3] =  1.0;//translationScale; 
    rigidOptimizerScales[4] =  translationScale; 
    rigidOptimizerScales[5] =  translationScale; 



  OptimizerType::Pointer rigid_optimizer = OptimizerType::New();
  //GradOptimizerType::Pointer grad_optimizer = GradOptimizerType::New();    
    rigid_optimizer->SetScales(rigidOptimizerScales);
    rigid_optimizer->SetMaximumStepLength(0.2); 
    rigid_optimizer->SetMinimumStepLength(0.0001); 
    rigid_optimizer->SetNumberOfIterations(500);

  CTInterpolatorType::Pointer CTinterpolator = CTInterpolatorType::New();  
  CTRegistrationType::Pointer registration = CTRegistrationType::New();  
  //nc_metric->SetNumberOfSpatialSamples( 10000L );
    
  std::cout<< " setting registration parameters "<<std::endl;
  ShortImageType2D::RegionType fixedRegion = fixedCT2D->GetBufferedRegion();

  registration->SetMetric( nc_metric );
  registration->SetFixedImage(fixedCT2D  ); //fixedSmoother->GetOutput()
  registration->SetMovingImage(movingCT2D); //movingSmoother->GetOutput()
  //registration->SetFixedImageRegion( fixedRegion );
  registration->SetOptimizer( rigid_optimizer );
  registration->SetInterpolator( CTinterpolator );
  registration->SetTransform( rigidTransform );
      
  registration->SetInitialTransformParameters( rigidTransform->GetParameters());
  /*
  CTRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize( 3 );
  shrinkFactorsPerLevel[0] = 4;
  shrinkFactorsPerLevel[1] = 2;
  shrinkFactorsPerLevel[2] = 1;
  CTRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize( 3 );
  smoothingSigmasPerLevel[0] = 8;
  smoothingSigmasPerLevel[1] = 4;
  smoothingSigmasPerLevel[2] = 1;
  registration->SetNumberOfLevels ( numberOfLevels );
  registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
  registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
  */
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
  
  AffineTransformType2D::Pointer affineTransform2D = AffineTransformType2D::New();
    affineTransform2D->SetCenter( rigidTransform->GetCenter() );
    affineTransform2D->SetTranslation( rigidTransform->GetTranslation() );
  affineTransform2D->SetMatrix( rigidTransform->GetMatrix() );

  std::cout<<"affine registration 2" <<std::endl;
  CTRegistrationType::Pointer registration_affine = CTRegistrationType::New();  


  registration_affine->SetMetric( nc_metric );
  registration_affine->SetFixedImage(fixedCT2D  ); //fixedSmoother->GetOutput()
  registration_affine->SetMovingImage(movingCT2D); //movingSmoother->GetOutput()

   OptimizerType::Pointer affine_optimizer = OptimizerType::New();
  std::cout<<"affine registration 3" <<std::endl;  
  //translationScale = 0.000001;
  OptimizerScalesType optimizerScales(affineTransform2D->GetNumberOfParameters());     
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = 1.0;
  /*optimizerScales[4] = 1.0;
  optimizerScales[5] = 1.0;
  optimizerScales[6] = 1.0;
  optimizerScales[7] = 1.0;
  optimizerScales[8] = 1.0;
  optimizerScales[9]  = translationScale;
  optimizerScales[10] = translationScale;
  optimizerScales[11] = translationScale; */
  optimizerScales[4]  = translationScale;
  optimizerScales[5] = translationScale;

 std::cout<<"affine registration 4" <<std::endl;  
  affine_optimizer->SetScales( optimizerScales );
  affine_optimizer->SetMaximumStepLength( 0.2000  );
std::cout<<"affine registration 5" <<std::endl;  
  affine_optimizer->SetMinimumStepLength( 0.0001 );
  affine_optimizer->SetNumberOfIterations( 300 );

  registration_affine->SetOptimizer( affine_optimizer );
  registration_affine->SetInterpolator( CTinterpolator );

   registration_affine->SetTransform( affineTransform2D );
   registration_affine->SetInitialTransformParameters( affineTransform2D->GetParameters() );

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
      std::cout<<"affine registration 7" <<std::endl;  
      registration_affine->Update();
      std::cout<<"affine registration 8" <<std::endl;  
    }
  catch( itk::ExceptionObject &excp )
    {
      std::cerr << "ExceptionObject caught while executing registration" <<
      std::endl;
      std::cerr << excp << std::endl;
     }
  
   //get all params to output to file
   numberOfIterations = rigid_optimizer->GetCurrentIteration();
   double bestValue;
   bestValue = rigid_optimizer->GetValue();
      
   std::cout <<" best similarity value = " <<bestValue<<std::endl;
   affineTransform2D->SetParameters( registration_affine->GetLastTransformParameters());
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
      labelMapRegistrationXMLData.similarityValue = (float)(bestValue);
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
      transformWriter->SetInput( affineTransform2D );

   
      transformWriter->SetFileName( outputTransformFileName );
      transformWriter->Update();
    }

  std::cout << "DONE." << std::endl;

  return 0;
}
