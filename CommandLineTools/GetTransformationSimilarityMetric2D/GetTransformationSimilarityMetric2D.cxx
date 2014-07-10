//GetTransformationSimilarityMetric

/** \file
 *  \ingroup commandLineTools 
 *  \details This program registers 2 label maps, source and target, and 
 * a transformation file as well as the transformed image 
 *
 *  USAGE: 
./GetTransformationSimilarityMetric --fixedLabelMapFileName ~/Documents/Data/COPDGene/14388A/14388A_INSP_STD_HAR_COPD/14388A_INSP_STD_HAR_COPD_leftLungRightLung.nhdr --movingLabelMapFileName ~/Documents/Data/COPDGene/14988Y/14988Y_INSP_STD_UAB_COPD/14988Y_INSP_STD_UAB_COPD_leftLungRightLung.nhdr --regionVec 1 --fixedCTFileName ~/Documents/Data/COPDGene/14388A/14388A_INSP_STD_HAR_COPD/14388A_INSP_STD_HAR_COPD.nhdr --movingCTFileName ~/Documents/Data/COPDGene/14988Y/14988Y_INSP_STD_UAB_COPD/14988Y_INSP_STD_UAB_COPD.nhdr --inputTransform ~/Documents/Data/COPDGene/14988Y/14988Y_INSP_STD_UAB_COPD/registrationData/14988Y_INSP_STD_UAB_COPD_to_14388A_INSP_STD_HAR_COPD.tfm

 *
 *  $Date: $
 *  $Revision: $
 *  $Author:  $
 *
 */


//Image
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkImageMaskSpatialObject.h"

//registration
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImageRegistrationMethod.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkAffineTransform.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include <itkCompositeTransform.h>
#include <itkAffineTransform.h>
#include "itkCenteredTransformInitializer.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"

//similarity
#include "itkMutualInformationImageToImageMetric.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkGradientDifferenceImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"


//xml
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>

#include "GetTransformationSimilarityMetric2DCLP.h"
#include "cipConventions.h"
#include "cipHelper.h"
#include <sstream>
#include <fstream>

namespace
{
#define MY_ENCODING "ISO-8859-1"

  //images
typedef itk::Image< unsigned short, 2 >       LabelMapType2D;
typedef itk::ImageFileReader< LabelMapType2D >  LabelMapReaderType2D;
typedef itk::Image< short, 2 >                ShortImageType;
typedef itk::ImageFileReader< ShortImageType > ShortReaderType;
typedef itk::ImageFileReader< ShortImageType >                                                            CTFileReaderType;

  //registration
typedef itk::RegularStepGradientDescentOptimizer                                                       OptimizerType;
typedef itk::ImageRegistrationMethod< ShortImageType, ShortImageType >                                 RegistrationType;
typedef itk::LinearInterpolateImageFunction< ShortImageType, double >                         InterpolatorType;
typedef itk::AffineTransform<double, 2 >                                                               TransformType;
typedef itk::CenteredTransformInitializer< TransformType, ShortImageType, ShortImageType >             InitializerType;
typedef OptimizerType::ScalesType                                                                      OptimizerScalesType;
typedef itk::IdentityTransform< double, 2 >                                                            IdentityType;
typedef itk::CompositeTransform< double, 2 > CompositeTransformType;
typedef itk::TranslationTransform< double, 2 >  TranslationTransformType;
typedef itk::ResampleImageFilter< ShortImageType,ShortImageType >           ResampleType;

  //similarity metrics
typedef itk::MutualInformationImageToImageMetric<ShortImageType, ShortImageType >                      MIMetricType;
typedef itk::NormalizedMutualInformationHistogramImageToImageMetric< ShortImageType, ShortImageType >  NMIMetricType;
typedef itk::MeanSquaresImageToImageMetric<  ShortImageType, ShortImageType  >                         msqrMetricType;
typedef itk::NormalizedCorrelationImageToImageMetric<ShortImageType, ShortImageType  >                 ncMetricType;
typedef itk::GradientDifferenceImageToImageMetric<ShortImageType, ShortImageType  >                  gdMetricType;



struct REGIONTYPEPAIR
{
  unsigned char region;
  unsigned char type;
};

//struct for saving the xml file wih similarity information
  struct SIMILARITY_XML_DATA
  {
    double similarityValue;
    std::string transformationLink[5];
    std::string transformation_isInverse[5];
    unsigned int transformation_order[5];
    std::string fixedID;
    std::string movingID;
    std::string similarityMeasure;
    std::string regionAndTypeUsed;
    std::string fixedMask;
    std::string movingMask;
    std::string extention;
  };

 //Reads a .tfm file and returns an itkTransform
  TransformType::Pointer GetTransformFromFile( std::string fileName )
  {
    itk::TransformFileReader::Pointer transformReader = itk::TransformFileReader::New();
    transformReader->SetFileName( fileName );
    try
      {
	transformReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
	std::cerr << "Exception caught reading transform:";
	std::cerr << excp << std::endl;
      }
        
    itk::TransformFileReader::TransformListType::const_iterator it;
        
    it = transformReader->GetTransformList()->begin();
        
    TransformType::Pointer transform = static_cast< TransformType* >( (*it).GetPointer() );
        
    return transform;
  }
    
  LabelMapType2D::Pointer ReadLabelMapFromFile( std::string labelMapFileName )
  {
    LabelMapReaderType2D::Pointer reader = LabelMapReaderType2D::New(); 
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


  ShortImageType::Pointer ReadCTFromFile( std::string fileName )
  {
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

  //writes the similarity measures to an xml file
  // the similarity info is stored in the SIMILARITY_XML_DATA struct
   void WriteMeasuresXML(const char *file, SIMILARITY_XML_DATA &theXMLData)
  {      
    std::cout<<"Writing similarity XML file"<<std::endl;
    xmlDocPtr doc = NULL;       /* document pointer */
    xmlNodePtr root_node = NULL; /* Node pointers */
    xmlDtdPtr dtd = NULL;       /* DTD pointer */
    xmlAttrPtr newattr;

    xmlNodePtr transfo_node[5];
    
    for(int i = 0; i< 5; i++)
      transfo_node[i]= NULL;


    doc = xmlNewDoc(BAD_CAST "1.0");
    root_node = xmlNewNode(NULL, BAD_CAST "Inter_Subject_Measure");
    xmlDocSetRootElement(doc, root_node);

    dtd = xmlCreateIntSubset(doc, BAD_CAST "root", NULL, BAD_CAST "InterSubjectMeasures_v3.dtd");
 
    // xmlNewChild() creates a new node, which is "attached"
    // as child node of root_node node. 
    std::ostringstream similaritString;
    similaritString <<theXMLData.similarityValue;
  
    xmlNewChild(root_node, NULL, BAD_CAST "movingID", BAD_CAST (theXMLData. movingID.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "fixedID", BAD_CAST (theXMLData.fixedID.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "movingMask", BAD_CAST (theXMLData. movingMask.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "fixedMask", BAD_CAST (theXMLData.fixedMask.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "SimilarityMeasure", BAD_CAST (theXMLData.similarityMeasure.c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "SimilarityValue", BAD_CAST (similaritString.str().c_str()));
    xmlNewChild(root_node, NULL, BAD_CAST "ImageExtension", BAD_CAST (theXMLData.extention.c_str()));



    for(int i = 0; i<5; i++)
      {
        if (!theXMLData.transformationLink[i].empty())
	  {
            transfo_node[i] = xmlNewChild(root_node, NULL, BAD_CAST "transformation", BAD_CAST (""));
            xmlNewChild(transfo_node[i], NULL, BAD_CAST "file", BAD_CAST (theXMLData.transformationLink[i].c_str()));
            newattr = xmlNewProp(transfo_node[i], BAD_CAST "isInverse", BAD_CAST (theXMLData.transformation_isInverse[i].c_str()));
            std::ostringstream order_string;
            order_string << theXMLData.transformation_order[i];
            newattr = xmlNewProp(transfo_node[i], BAD_CAST "order",  BAD_CAST (order_string.str().c_str()));
	  }
      }    
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
   const char* similarity_type;
  
  PARSE_ARGS;

  //fill out the isInvertTransform vector
  bool *isInvertTransform = new bool[inputTransformFileName.size()];
  for ( unsigned int i=0; i<inputTransformFileName.size(); i++ )
    {
      std::cout<<inputTransformFileName[i]<<std::endl;
      isInvertTransform[i] = false;
    }   
  std::cout<<invertTransform.size()<<std::endl;
  for ( unsigned int i=0; i<invertTransform.size(); i++ )
    {
      isInvertTransform[invertTransform[i]] = true;
    }  
   
    //Read the CT images
  ShortImageType::Pointer ctFixedImage = ShortImageType::New();
  ctFixedImage = ReadCTFromFile( fixedCTFileName );
  if (ctFixedImage.GetPointer() == NULL)
      {
        return cip::NRRDREADFAILURE;
      }

  ShortImageType::Pointer ctMovingImage = ShortImageType::New();
  ctMovingImage = ReadCTFromFile( movingCTFileName);
  if (ctMovingImage.GetPointer() == NULL)
      {
        return cip::NRRDREADFAILURE;
      }

  //Set mask  object
  typedef itk::ImageMaskSpatialObject< 2 >   MaskType;
  MaskType::Pointer  fixedSpatialObjectMask = MaskType::New();
  MaskType::Pointer  movingSpatialObjectMask = MaskType::New();

  typedef itk::Image< unsigned char, 2 >   ImageMaskType;
  typedef itk::ImageFileReader< ImageMaskType >    MaskReaderType;
  MaskReaderType::Pointer  fixedMaskReader = MaskReaderType::New();
  MaskReaderType::Pointer  movingMaskReader = MaskReaderType::New();
  
 if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
    {
      movingMaskReader->SetFileName(movingLabelmapFileName.c_str() );

      try
	{
	  movingMaskReader->Update();
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << "ExceptionObject caught !" << std::endl;
	  std::cerr << err << std::endl;
	  return EXIT_FAILURE;
	}
      movingSpatialObjectMask->SetImage(movingMaskReader->GetOutput());
    }
  else
    {
      std::cout <<"No moving label map specified"<< std::endl;
    }
  if ( strcmp( fixedLabelmapFileName.c_str(), "q") != 0 )
    {
      fixedMaskReader->SetFileName(fixedLabelmapFileName.c_str() );

      try
	{
	  fixedMaskReader->Update();
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << "ExceptionObject caught !" << std::endl;
	  std::cerr << err << std::endl;
	  return EXIT_FAILURE;
	}
      fixedSpatialObjectMask->SetImage(fixedMaskReader->GetOutput());
    }
  else
    {
      std::cout <<"No fixed label map specified"<< std::endl;
    }
  
  //parse transform arg  and join transforms together
  
   //last transform applied first, so make last transform
  TransformType::Pointer transformTemp = TransformType::New();
  TransformType::Pointer transformTempInv = TransformType::New();

  CompositeTransformType::Pointer test_inverse_transform = CompositeTransformType::New();
  CompositeTransformType::Pointer transform = CompositeTransformType::New();
   transform->SetAllTransformsToOptimizeOn();

  for ( unsigned int i=0; i<inputTransformFileName.size(); i++ )
    {
     
      transformTemp = GetTransformFromFile(inputTransformFileName[i] );
      // Invert the transformation if specified by command like argument. Only inverting the first transformation
      if(isInvertTransform[i] == true)
        {
	  std::cout<<"inverting transform "<<inputTransformFileName[i]<<std::endl;
          transformTemp->SetMatrix( transformTemp->GetInverseMatrix());
          transform->AddTransform(transformTemp);
        }          
      else
	{
	transform->AddTransform(transformTemp);
	transformTempInv->SetMatrix( transformTemp->GetInverseMatrix());
	test_inverse_transform->AddTransform(transformTempInv);
	}
      transform->SetAllTransformsToOptimizeOn();
      test_inverse_transform->SetAllTransformsToOptimizeOn();
    }
 
  CompositeTransformType::Pointer transform_forsim = CompositeTransformType::New();
  TransformType::Pointer id_transform = TransformType::New();
  id_transform->SetIdentity();
  transform_forsim->AddTransform(id_transform);
  transform_forsim->SetAllTransformsToOptimizeOn();

  std::cout << " transforms read. "<<std::endl;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  double similarityValue;

  std::cout << "Resampling..." << std::endl;
  ResampleType::Pointer resampler = ResampleType::New();
    resampler->SetTransform( transform );
    resampler->SetInterpolator( interpolator );
    resampler->SetInput( ctMovingImage);
    resampler->SetSize( ctFixedImage->GetLargestPossibleRegion().GetSize() );
    resampler->SetOutputSpacing( ctFixedImage->GetSpacing() );
    resampler->SetOutputOrigin( ctFixedImage->GetOrigin() );
    resampler->SetOutputDirection( ctFixedImage->GetDirection() );
  try
    {
    resampler->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught resampling:";
    std::cerr << excp << std::endl;

    return cip::RESAMPLEFAILURE;
    }

  //initialize metric 
  if (similarityMetric =="nc")
      {
         ncMetricType::Pointer metric = ncMetricType::New();
	 transform->SetAllTransformsToOptimizeOn();
	 std::cout<<transform<<std::endl;
	 metric->SetInterpolator( interpolator );
	 metric->SetTransform(transform_forsim);//transform);
	 metric->SetFixedImage( ctFixedImage );
	 metric->SetMovingImage( resampler->GetOutput() );
	 ShortImageType::RegionType fixedRegion = ctFixedImage->GetBufferedRegion();
	 metric->SetFixedImageRegion(fixedRegion);
	 metric->Initialize();
	 
	 std::cout<<metric<<std::endl;
	 if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
	   metric->SetMovingImageMask( movingSpatialObjectMask );
	 if ( strcmp( fixedLabelmapFileName.c_str(), "q") != 0 )
	 metric->SetFixedImageMask( fixedSpatialObjectMask );  

	 ncMetricType::TransformParametersType zero_params( transform->GetNumberOfParameters() );
	 zero_params = transform_forsim->GetParameters();
	 //should be set to identity, not zero

	 //for(int i = 0; i< transform->GetNumberOfParameters(); i++)
	 //    zero_params[i]=0;

	 similarityValue = metric->GetValue(zero_params );
	 std::cout<<"the ncc value is: "<<similarityValue<<std::endl;

	 similarity_type = "NCC";
      }
  else if (similarityMetric =="NMI")
    {
      NMIMetricType::Pointer metric = NMIMetricType::New();
      NMIMetricType::HistogramType::SizeType histogramSize;
      histogramSize.SetSize(2);
      histogramSize[0] = 20;
      histogramSize[1] = 20;
      metric->SetHistogramSize( histogramSize );

      if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
	metric->SetMovingImageMask( movingSpatialObjectMask );
      if ( strcmp( fixedLabelmapFileName.c_str(), "q") != 0 )
	metric->SetFixedImageMask( fixedSpatialObjectMask );     
     
      similarity_type = "NMI";
    }
    else if (similarityMetric =="msqr")
      {
	msqrMetricType::Pointer metric = msqrMetricType::New();
        if ( strcmp( fixedLabelmapFileName.c_str(), "q") != 0 )
            metric->SetFixedImageMask( fixedSpatialObjectMask );   

	similarity_type = "msqr";
      }
     
    else if (similarityMetric =="gd")
      {
	gdMetricType::Pointer metric = gdMetricType::New();
      if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
	metric->SetMovingImageMask( movingSpatialObjectMask );
      if ( strcmp( fixedLabelmapFileName.c_str(), "q") != 0 )
	metric->SetFixedImageMask( fixedSpatialObjectMask );       
      similarity_type = "gd";
      }
   else //MI is default
     {
     
      MIMetricType::Pointer metric = MIMetricType::New();
      metric->SetFixedImageStandardDeviation( 13.5 );
      metric->SetMovingImageStandardDeviation( 13.5 );
      if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
	metric->SetMovingImageMask( movingSpatialObjectMask );
      if ( strcmp( fixedLabelmapFileName.c_str(), "q") != 0 )
	metric->SetFixedImageMask( fixedSpatialObjectMask );      
      similarity_type = "MI";
     }
 
 

  //Write data to xml file if necessary
  if ( strcmp(outputXMLFileName.c_str(), "q") != 0 ) 
    {


      SIMILARITY_XML_DATA ctSimilarityXMLData;
      ctSimilarityXMLData.regionAndTypeUsed.assign("N/A");
      ctSimilarityXMLData.similarityValue = (float)(similarityValue);
      ctSimilarityXMLData.similarityMeasure.assign(similarity_type);
      if ( strcmp( movingLabelmapFileName.c_str(), "q") != 0 )
	{
	  ctSimilarityXMLData.movingMask.assign(movingLabelmapFileName);
	}
      else
	{
	  ctSimilarityXMLData.movingMask.assign("N/A");
	}
      if ( strcmp( fixedLabelmapFileName.c_str(), "q") != 0 )
	{
	  ctSimilarityXMLData.fixedMask.assign(fixedLabelmapFileName);
	}
      else
	{
	  ctSimilarityXMLData.fixedMask.assign("N/A");
	}
      
      if ( strcmp(movingImageID.c_str(), "q") != 0 ) 
	{
	  ctSimilarityXMLData.movingID.assign(movingImageID);
	}
      else
	{
	  ctSimilarityXMLData.movingID.assign("N/A");
	}
  
      
      if ( strcmp(fixedImageID.c_str(), "q") != 0 ) 
	{
	  ctSimilarityXMLData.fixedID =fixedImageID.c_str();
	}
      else
	{
	  ctSimilarityXMLData.fixedID.assign("N/A");
	}
      //extract the extension from the filename
      ctSimilarityXMLData.extention.assign(fixedCTFileName);
      int pathLength = 0, pos=0, next=0; 
      next = ctSimilarityXMLData.extention.find('.', next+1);
      ctSimilarityXMLData.extention.erase(0,next);
      std::cout<<ctSimilarityXMLData.extention<<std::endl;
	
      //remove path from output transformation file before storing in xml
      //For each transformation       
      for ( unsigned int i=0; i<inputTransformFileName.size(); i++ )
	{ 
	  pos=0;
	  next=0;
	  for (int ii = 0; ii < (pathLength);ii++)
	    {
	      pos = next+1;
	      next = inputTransformFileName[0].find('/', next+1);
	    }
	  ctSimilarityXMLData.transformationLink[i].assign(inputTransformFileName[i].c_str());
	  ctSimilarityXMLData.transformation_order[i]= i;
	  if(isInvertTransform[i] == true)
	    ctSimilarityXMLData.transformation_isInverse[i].assign("True");
	  else
	    ctSimilarityXMLData.transformation_isInverse[i].assign("False");
	  ctSimilarityXMLData.transformationLink[i].erase(0,pos);
	}
   
      
      std::cout<<"saving output to: "<<outputXMLFileName.c_str()<<std::endl;
      WriteMeasuresXML(outputXMLFileName.c_str(), ctSimilarityXMLData);

    }
 
  return 0;

}
