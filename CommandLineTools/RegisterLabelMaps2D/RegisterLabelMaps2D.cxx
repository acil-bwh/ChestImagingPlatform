
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

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegistrationMethod.h"
#include "itkCenteredTransformInitializer.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "RegisterLabelMaps2DCLP.h"
#include "cipConventions.h"
#include "cipHelper.h"
#include <sstream>
#include "itkQuaternionRigidTransform.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include <fstream>
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
//#include <uuid/uuid.h>

namespace
{
#define MY_ENCODING "ISO-8859-1"
  typedef itk::Image< unsigned short, 2 >       LabelMapType2D; 
  typedef itk::RegularStepGradientDescentOptimizer                                                  OptimizerType;
  typedef OptimizerType::ScalesType                                                                 OptimizerScalesType;
 
  typedef itk::IdentityTransform< double, 2 >                                                       IdentityType;
  typedef itk::CIPExtractChestLabelMapImageFilter                                                   LabelMapExtractorType;
 
  typedef itk::GDCMImageIO                                                                          ImageIOType;
  typedef itk::GDCMSeriesFileNames                                                                  NamesGeneratorType;
  //typedef itk::ImageFileReader< cip::CTType >                                                       CTFileReaderType;
 

  typedef itk::ResampleImageFilter< LabelMapType2D, LabelMapType2D >                          ResampleFilterType;
  
    
  typedef itk::AffineTransform<double, 2 >  TransformType2D;
    typedef itk::QuaternionRigidTransform<double>  RigidTransformType;
  typedef itk::CenteredTransformInitializer< TransformType2D, LabelMapType2D, LabelMapType2D >  InitializerType2D;
  typedef itk::ImageFileReader< LabelMapType2D >  LabelMap2DReaderType;
  typedef itk::ImageRegistrationMethod<LabelMapType2D,LabelMapType2D >                         RegistrationType;
  typedef itk::KappaStatisticImageToImageMetric<LabelMapType2D, LabelMapType2D >               MetricType;
  typedef itk::NearestNeighborInterpolateImageFunction< LabelMapType2D, double >               InterpolatorType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType2D >                                  IteratorType;
  typedef itk::ImageFileWriter< LabelMapType2D >                                               ImageWriterType;
  typedef itk::RegionOfInterestImageFilter< LabelMapType2D, LabelMapType2D >                   RegionOfInterestType;
  typedef itk::ResampleImageFilter< LabelMapType2D, LabelMapType2D >                           ResampleType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType2D >                                  LabelMapIteratorType;

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

  std::cout << "agrs parsed, new" << std::endl;

    
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
  cip::LabelMapType::Pointer fixedLabelMap = cip::LabelMapType::New();
  cip::LabelMapType::Pointer subSampledFixedImage = cip::LabelMapType::New();

  LabelMapType2D::Pointer fixedLabelMap2D = LabelMapType2D::New();

  if ( strcmp( fixedImageFileName.c_str(), "q") != 0 )
    {
      std::cout << "Reading label map from file..." << std::endl;

 	fixedLabelMap2D = ReadLabelMap2DFromFile( fixedImageFileName );
        if (fixedLabelMap2D.GetPointer() == NULL)
	  {
	    return cip::LABELMAPREADFAILURE;
	  }

    }
  else
    {
      std::cerr <<"Error: No lung label map specified"<< std::endl;
      return cip::EXITFAILURE;
    }

  std::cout << "Subsampling fixed image with factor..."<<downsampleFactor<< std::endl;

 


  //Read in moving image label map from file and subsample
  cip::LabelMapType::Pointer movingLabelMap = cip::LabelMapType::New();
  LabelMapType2D::Pointer movingLabelMap2D =LabelMapType2D::New();
  cip::LabelMapType::Pointer subSampledMovingImage = cip::LabelMapType::New();
  if ( strcmp( movingImageFileName.c_str(), "q") != 0 )
    {
      std::cout << "Reading label map from file..." << std::endl;

      movingLabelMap2D = ReadLabelMap2DFromFile( movingImageFileName );

      if (movingLabelMap2D.GetPointer() == NULL)
	{
	  return cip::LABELMAPREADFAILURE;
	}

    }
  else
    {
      std::cerr <<"Error: No lung label map specified"<< std::endl;
      return cip::EXITFAILURE;
    }


  

  MetricType::Pointer metric = MetricType::New();
  //because we are minimizing as opposed to maximizing
  metric->SetForegroundValue( 1);

  TransformType2D::Pointer transform2D = TransformType2D::New();
  RigidTransformType::Pointer rigidTransform = RigidTransformType::New();
  std::cout<<"initializing transform"<<std::endl;
    
  InitializerType2D::Pointer initializer2D = InitializerType2D::New();

      initializer2D->SetTransform( transform2D );
      initializer2D->SetFixedImage(fixedLabelMap2D );
      initializer2D->SetMovingImage( movingLabelMap2D);
      initializer2D->MomentsOn();
      initializer2D->InitializeTransform();
  

     
      /*
    
  OptimizerScalesType optimizerScales( transform2D->GetNumberOfParameters());
  optimizerScales[0] =  1.0;   optimizerScales[1] =  1.0;
  optimizerScales[2] =  1.0;
  optimizerScales[3] =  1.0;   optimizerScales[4] =  1.0;
  optimizerScales[5] =  1.0;
  optimizerScales[6] =  1.0;   optimizerScales[7] =  1.0;
  optimizerScales[8] =  1.0;
  optimizerScales[9]  =  translationScale;
  optimizerScales[10] =  translationScale;
  optimizerScales[11] =  translationScale;
      */
  OptimizerType::Pointer optimizer = OptimizerType::New();

  //optimizer->SetScales( optimizerScales );
  /* optimizer->SetMaximumStepLength( maxStepLength );
  optimizer->SetMinimumStepLength( minStepLength );
  optimizer->SetNumberOfIterations( numberOfIterations );
  */

  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  std::cout << "Starting registration..." << std::endl;
    /*
  RegistrationType::Pointer rigidRegistration = RegistrationType::New();
  rigidRegistration->SetMetric( metric );
  rigidRegistration->SetOptimizer( optimizer );
  rigidRegistration->SetInterpolator( interpolator );
  rigidRegistration->SetTransform( rigidTransform );
    
    try
    {
        rigidRegistration->StartRegistration();
        //registration->Update(); for ITKv4
    }
    catch( itk::ExceptionObject &excp )
    {
        std::cerr << "ExceptionObject caught while executing rigid registration" <<
        std::endl;
        
        std::cerr << excp << std::endl;
    }
    */
  RegistrationType::Pointer registration = RegistrationType::New();

  registration->SetMetric( metric );
  registration->SetOptimizer( optimizer );
  registration->SetInterpolator( interpolator );
  registration->SetTransform( transform2D );
  
  
  registration->SetFixedImage( fixedLabelMap2D );
  registration->SetMovingImage(movingLabelMap2D);
 

  registration->SetInitialTransformParameters( transform2D->GetParameters()); 

  
  try
    {
      //registration->StartRegistration();
      registration->Initialize();
      registration->Update();
    }
  catch( itk::ExceptionObject &excp )
    {
      std::cerr << "ExceptionObject caught while executing registration" <<
	std::endl;

      std::cerr << excp << std::endl;
    }

  //get all params to output to file
  numberOfIterations = optimizer->GetCurrentIteration();

  //  The value of the image metric corresponding to the last set of
  // parameters
  //  can be obtained with the \code{GetValue()} method of the optimizer.

  const double bestValue = optimizer->GetValue();

  std::cout<<"similarity output = "<< optimizer->GetValue() <<" best value = " <<bestValue<<std::endl;

  OptimizerType::ParametersType finalParams =registration->GetLastTransformParameters();



    TransformType2D::Pointer finalTransform2D = TransformType2D::New();
    finalTransform2D->SetParameters( finalParams );
    finalTransform2D->SetCenter( transform2D->GetCenter() );

  if ( strcmp(outputTransformFileName.c_str(), "q") != 0 )
    {
      std::string infoFilename = outputTransformFileName;
      int result = infoFilename.find_last_of('.');
      // Does new_filename.erase(std::string::npos) working here in place of
      //this following test?
      if (std::string::npos != result)
	infoFilename.erase(result);
      // append extension:
      infoFilename.append(".xml");

      REGISTRATION_XML_DATA labelMapRegistrationXMLData;
      labelMapRegistrationXMLData.similarityValue = (float)(bestValue);
      const char *similarity_type = metric->GetNameOfClass();
      labelMapRegistrationXMLData.similarityMeasure.assign(similarity_type);
        
      labelMapRegistrationXMLData.image_type.assign("leftLungRightLung");
      labelMapRegistrationXMLData.transformationIndex = 0;
        
      // TODO: See if xml file is empty and change the transformation
      // index accordingly
        
      //if the patient IDs are specified  as args, use them,
      //otherwise, extract from patient path

      int pathLength = 0, pos=0, next=0;

      if ( strcmp(movingImageID.c_str(), "q") != 0 )
	{
	  labelMapRegistrationXMLData.sourceID.assign(movingImageID);
	}
      else
	{
	  //first find length of path
	  next=1;
	  while(next>=1)
	    {
	      next = movingImageFileName.find("/", next+1);
	      pathLength++;
	    }
	  pos=0;
	  next=0;
       
	  std::string tempSourceID;
	  for (int i = 0; i < (pathLength-1);i++)
	    {
	      pos= next+1;
	      next = movingImageFileName.find("/", next+1);
	    }
       
	  labelMapRegistrationXMLData.sourceID.assign(movingImageFileName.c_str());
	  labelMapRegistrationXMLData.sourceID.erase(next,labelMapRegistrationXMLData.sourceID.length()-1);
	  labelMapRegistrationXMLData.sourceID.erase(0, pos);
	}
    
      if ( strcmp(fixedImageID.c_str(), "q") != 0 )
	{
	  labelMapRegistrationXMLData.destID =fixedImageID.c_str();
	}
      else
	{
	  pos=0;
	  next=0;
	  for (int i = 0; i < (pathLength-1);i++)
	    {
	      pos = next+1;
	      next = fixedImageFileName.find('/', next+1);
	    }

	  labelMapRegistrationXMLData.destID.assign(fixedImageFileName.c_str());//
	  //=tempSourceID.c_str();//movingImageFileName.substr(pos, next-1).c_str();

	  labelMapRegistrationXMLData.destID.erase(next,labelMapRegistrationXMLData.destID.length()-1);
	  labelMapRegistrationXMLData.destID.erase(0, pos);

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
  /*
      if ( strcmp(outputImageFileName.c_str(), "q") != 0 )
	{
	  std::cout << "Resampling moving image..." << std::endl;
        
	  ResampleFilterType::Pointer resample = ResampleFilterType::New();

            resample->SetTransform( finalTransform2D );

	  resample->SetInput( movingLabelMap2D);//movingExtractor->GetOutput());//movingLabelMap );
	  resample->SetSize(fixedLabelMap2D->GetLargestPossibleRegion().GetSize());
	  resample->SetOutputOrigin(  fixedLabelMap2D->GetOrigin() );
	  resample->SetOutputSpacing( fixedLabelMap2D->GetSpacing());
	  resample->SetOutputDirection( fixedLabelMap2D->GetDirection() );
	  resample->SetInterpolator( interpolator );
	  resample->SetDefaultPixelValue( 0 );
	  resample->Update();


	  //ImageType::Pointer upsampledImage = ImageType::New();

	  //std::cout << "Upsampling to original size..." << std::endl;
	  //ResampleImage( resample->GetOutput(), upsampledImage,1.0/downsampleFactor );
	  //upsampledImage=cip::UpsampleLabelMap(downsampleFactor,resample->GetOutput());

	  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
	  writer->SetInput(resample->GetOutput());//resample->GetOutput());// movingExtractor->GetOutput()
	  writer->SetFileName( outputImageFileName );
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
  */
      std::cout << "DONE." << std::endl;

      return 0;
    }
