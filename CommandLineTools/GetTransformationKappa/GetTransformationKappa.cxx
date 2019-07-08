//Image
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkImageMaskSpatialObject.h"

//registration

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkAffineTransform.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include <itkCompositeTransform.h>
#include <itkAffineTransform.h>
#include "itkResampleImageFilter.h"


//similarity

#include "itkKappaStatisticImageToImageMetric.h"

//xml
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#undef reference // to use vtklibxml2

#include "GetTransformationKappaCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include <sstream>
#include <fstream>

namespace
{
#define MY_ENCODING "ISO-8859-1"


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
  template <unsigned int TDimension> typename itk::AffineTransform< double, TDimension >::Pointer GetTransformFromFile( std::string fileName )
  {

    typedef itk::AffineTransform< double, TDimension >  TransformType;

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
 
    typename TransformType::Pointer transform = static_cast< TransformType* >( (*it).GetPointer() ); 
    return transform;
  }
  
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

template <unsigned int TDimension>
int DoIT(int argc, char * argv[])
{  

  
  PARSE_ARGS;

  typedef itk::Image< unsigned short, TDimension >       LabelMapType;
  typedef itk::ImageFileReader< LabelMapType >  LabelMapReaderType;
  
  //image transformation
  typedef itk::NearestNeighborInterpolateImageFunction< LabelMapType, double >                         InterpolatorType;
  typedef itk::AffineTransform<double, TDimension >                                                               TransformType;
  typedef itk::IdentityTransform< double, TDimension >                                                            IdentityType;
  typedef itk::CompositeTransform< double, TDimension > CompositeTransformType;
  typedef itk::ResampleImageFilter< LabelMapType,LabelMapType >           ResampleType;
  
  //similarity metrics
  typedef itk::KappaStatisticImageToImageMetric<LabelMapType, LabelMapType >                         metricType;
  //fill out the isInvertTransform vector which is a boolean vector.
  // each entry corresponds to a transform and indicates whether the transform 
  // is to be inverted.
  bool *isInvertTransform = new bool[inputTransformFileName.size()];
  for ( unsigned int i=0; i<inputTransformFileName.size(); i++ )
    {
      isInvertTransform[i] = false;
    }   
  for ( unsigned int i=0; i<invertTransform.size(); i++ )
    {
      isInvertTransform[invertTransform[i]] = true;
    }  
   
    //Read the labelmaps
  typename LabelMapType::Pointer fixedImage = LabelMapType::New();
  fixedImage = ReadLabelMapFromFile<TDimension>( fixedCTFileName );
  if (fixedImage.GetPointer() == NULL)
      {
        return cip::LABELMAPREADFAILURE;
      }

  typename LabelMapType::Pointer movingImage = LabelMapType::New();
  movingImage = ReadLabelMapFromFile<TDimension>( movingCTFileName);
  if (movingImage.GetPointer() == NULL)
      {
        return cip::LABELMAPREADFAILURE;
      }


  //parse transform arg  and join transforms into a composite one
   //last transform applied first
  typename TransformType::Pointer transformTemp = TransformType::New();
  typename CompositeTransformType::Pointer transform = CompositeTransformType::New();
   transform->SetAllTransformsToOptimizeOn();
   if ( strcmp( inputTransformFileName[0].c_str(), "q") != 0 )
    {
      for ( unsigned int i=0; i<inputTransformFileName.size(); i++ )
	{
	  
	  transformTemp = GetTransformFromFile<TDimension>(inputTransformFileName[i] );
	  // Invert the transformation if specified by command like argument.
	  if(isInvertTransform[i] == true)
	    {
	      std::cout<<"inverting transform "<<inputTransformFileName[i]<<std::endl;
        transform->AddTransform(transformTemp->GetInverseTransform());
	    }          
	  else
	    {
	      transform->AddTransform(transformTemp);
	    }
	  transform->SetAllTransformsToOptimizeOn();
	}
    }
   else
     {
       // Set to identity by default
       transformTemp->SetIdentity();
       transform->AddTransform(transformTemp);
     }
   transform->SetAllTransformsToOptimizeOn();

   
   typename TransformType::Pointer id_transform = TransformType::New();
     id_transform->SetIdentity();
   typename CompositeTransformType::Pointer transform_forsim = CompositeTransformType::New();
     transform_forsim->AddTransform(id_transform);
     transform_forsim->SetAllTransformsToOptimizeOn();

   typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
   typename ResampleType::Pointer resampler = ResampleType::New();
    resampler->SetTransform( transform );
    resampler->SetInterpolator( interpolator );
    resampler->SetInput( movingImage);
    resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    resampler->SetOutputSpacing( fixedImage->GetSpacing() );
    resampler->SetOutputOrigin( fixedImage->GetOrigin() );
    resampler->SetOutputDirection( fixedImage->GetDirection() );
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
  

  const char* similarity_type;
    similarity_type = "Kappa";
  typename metricType::Pointer metric = metricType::New();
  metric->SetForegroundValue( 1 );    
  metric->SetInterpolator( interpolator );
  metric->SetTransform(id_transform);
  metric->SetFixedImage( fixedImage );
  metric->SetMovingImage( resampler->GetOutput() );
    
  typename LabelMapType::RegionType fixedRegion = fixedImage->GetBufferedRegion();
  metric->SetFixedImageRegion(fixedRegion);
  metric->Initialize();

  typename metricType::TransformParametersType zero_params(id_transform->GetNumberOfParameters() );
    zero_params = id_transform->GetParameters();
  double similarityValue;
    similarityValue = metric->GetValue(zero_params );
  std::cout<<"the kappa value is: "<<similarityValue<<std::endl;
  

  //Write data to xml file if necessary
  if ( strcmp(outputXMLFileName.c_str(), "q") != 0 ) 
    {

      std::cout<<outputXMLFileName.c_str()<<std::endl;
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
	
      //remove path from output transformation file before storing in xml
      //For each transformation       
      std::cout<<"about to write"<<std::endl;
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


int main( int argc, char *argv[] )
{
  
  PARSE_ARGS;
  switch(dimension)
    {
    case 2:
      {
	DoIT<2>( argc, argv);
	break;
      }
      case 3:
      {
	DoIT<3>( argc, argv);
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
