//GetTransformationSimilarityMetric

/** \file
 *  \ingroup commandLineTools 
 *  \details This program registers 2 label maps, source and target, and 
 * a transformation file as well as the transformed image 
 *
 *  USAGE: ./GetTransformationSimilarityMetric --regionVec 1 -m /net/th914_nas.bwh.harvard.edu/mnt/array1/share/Processed/COPDGene/11622T/11622T_INSP_STD_HAR_COPD/11622T_INSP_STD_HAR_COPD_leftLungRightLung.nhdr -f /net/th914_nas.bwh.harvard.edu/mnt/array1/share/Processed/COPDGene/10393Z/10393Z_INSP_STD_HAR_COPD/10393Z_INSP_STD_HAR_COPD_leftLungRightLung.nhdr --outputImage /projects/lmi/people/rharmo/projects/dataInfo/testoutput.nrrd --outputTransform output_transform_file 

>./GetTransformationSimilarityMetric --fixedLabelMapFileName COPDGene/11622T/11622T_INSP_STD_HAR_COPD/11622T_INSP_STD_HAR_COPD_leftLungRightLung.nhdr --movingLabelMapFileName COPDGene/10393Z/10393Z_INSP_STD_HAR_COPD/10393Z_INSP_STD_HAR_COPD_leftLungRightLung.nhdr --regionVec 1 --fixedCTFileName COPDGene/11622T/11622T_INSP_STD_HAR_COPD/11622T_INSP_STD_HAR_COPD_leftLungRightLung.nhdr --movingCTFileName COPDGene/10393Z/10393Z_INSP_STD_HAR_COPD/10393Z_INSP_STD_HAR_COPD_leftLungRightLung.nhdr --inputTransform
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
#include "GetTransformationSimilarityMetricCLP.h"
#include "cipConventions.h"
#include "cipHelper.h"
#include <sstream>

#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include <fstream>
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include "itkTransformFactoryBase.h"

namespace
{
#define MY_ENCODING "ISO-8859-1"

typedef itk::Image< unsigned short, 3 >                                             UnsignedShortImageType;
typedef itk::Image< short, 3 >                                                      ShortImageType;
typedef itk::ImageFileReader< ShortImageType >                                      ShortReaderType;
typedef itk::ImageFileReader< UnsignedShortImageType >                              ImageReaderType;
typedef itk::RegularStepGradientDescentOptimizer                                    OptimizerType;
typedef itk::ImageRegistrationMethod< ShortImageType, ShortImageType >                        RegistrationType;
typedef itk::NearestNeighborInterpolateImageFunction< ShortImageType, double >           InterpolatorType;
typedef itk::AffineTransform<double, 3 >                                            TransformType;
typedef itk::CenteredTransformInitializer< TransformType, ShortImageType, ShortImageType >    InitializerType;
typedef OptimizerType::ScalesType                                                   OptimizerScalesType;
typedef itk::ImageRegionIteratorWithIndex< UnsignedShortImageType >                              IteratorType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >                              CTIteratorType;
typedef itk::RegionOfInterestImageFilter< ShortImageType, ShortImageType >                    RegionOfInterestType;
typedef itk::ResampleImageFilter< ShortImageType, ShortImageType >                            ResampleType;
typedef itk::IdentityTransform< double, 3 >                                         IdentityType;
typedef itk::CIPExtractChestLabelMapImageFilter                                     LabelMapExtractorType;
typedef itk::ImageSeriesReader< cip::CTType >                                       CTSeriesReaderType;
typedef itk::GDCMImageIO                                                            ImageIOType;
typedef itk::GDCMSeriesFileNames                                                    NamesGeneratorType;
typedef itk::ImageFileReader< cip::CTType >                                         CTFileReaderType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                      LabelMapIteratorType;
typedef itk::MattesMutualInformationImageToImageMetric<ShortImageType, 
                                                        ShortImageType >            MetricType;

struct REGIONTYPEPAIR
{
  unsigned char region;
  unsigned char type;
};

struct SIMILARITY_XML_DATA
{
  float similarityValue;
  std::string transformationLink;
  std::string sourceID;
  std::string destID;
  std::string similarityMeasure;
  std::string regionAndTypeUsed;
};

void ReadTransformFromFile( TransformType::Pointer transform, const char* fileName )
{
  itk::TransformFileReader::Pointer reader =
    itk::TransformFileReader::New();
  reader->SetFileName(fileName);
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while updating transform reader:";
    std::cerr << excp << std::endl;
    }

  typedef itk::TransformFileReader::TransformListType * TransformListType;
  TransformListType transforms = reader->GetTransformList();

  itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();

  transform = static_cast<TransformType*>((*it).GetPointer());
  transform->Print(std::cout);
 
}


cip::LabelMapType::Pointer ReadLabelMapFromFile( std::string labelMapFileName )
{
  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
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

void WriteMeasuresXML(const char *file, SIMILARITY_XML_DATA &theXMLData)
{      
  std::cout<<"Writing similarity XML file"<<std::endl;
  xmlDocPtr doc = NULL;       /* document pointer */
  xmlNodePtr root_node = NULL; /* Node pointers */
  xmlDtdPtr dtd = NULL;       /* DTD pointer */

  doc = xmlNewDoc(BAD_CAST "1.0");
  root_node = xmlNewNode(NULL, BAD_CAST "Inter_Subject_Measure");
  xmlDocSetRootElement(doc, root_node);

  dtd = xmlCreateIntSubset(doc, BAD_CAST "root", NULL, BAD_CAST "InterSubjectMeasures_v1.dtd");
 
  // xmlNewChild() creates a new node, which is "attached"
  // as child node of root_node node. 
  std::ostringstream similaritString;
  //std::string tempsource;
  similaritString <<theXMLData.similarityValue;
  xmlNewChild(root_node, NULL, BAD_CAST "transformation", BAD_CAST (theXMLData.transformationLink.c_str()));
  xmlNewChild(root_node, NULL, BAD_CAST "movingID", BAD_CAST (theXMLData.sourceID.c_str()));
  xmlNewChild(root_node, NULL, BAD_CAST "fixedID", BAD_CAST (theXMLData.destID.c_str()));
  xmlNewChild(root_node, NULL, BAD_CAST "SimilarityMeasure", BAD_CAST (theXMLData.similarityMeasure.c_str()));
  xmlNewChild(root_node, NULL, BAD_CAST "SimilarityValue", BAD_CAST (similaritString.str().c_str()));
  xmlNewChild(root_node, NULL, BAD_CAST "regionAndTypeUsed", BAD_CAST (theXMLData.regionAndTypeUsed.c_str()));
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

  std::cout<< "agrs parsed, new"<<std::endl;

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
  if ( strcmp( fixedCTFileName.c_str(), "q") != 0 )
    {
    std::cout << "Reading label map from file..." << std::endl;
    fixedLabelMap = ReadLabelMapFromFile( fixedLabelmapFileName );

    if (fixedLabelMap.GetPointer() == NULL)
      {
      return cip::LABELMAPREADFAILURE;
      }
    }
  else
    {
    std::cerr <<"Error: No lung label map specified"<< std::endl;
    return cip::EXITFAILURE;
    }


  //Read in moving image label map from file and subsample
  cip::LabelMapType::Pointer movingLabelMap = cip::LabelMapType::New();
  cip::LabelMapType::Pointer subSampledMovingImage = cip::LabelMapType::New();
  if ( strcmp( movingCTFileName.c_str(), "q") != 0 )
    {
    std::cout << "Reading label map from file..." << std::endl;
    movingLabelMap = ReadLabelMapFromFile( movingLabelmapFileName );

    if (movingLabelMap.GetPointer() == NULL)
      {
      return cip::LABELMAPREADFAILURE;
      }
    }
  else
    {
    std::cerr <<"Error: No lung label map specified"<< std::endl;
    return cip::EXITFAILURE;
    }

  // Extract fixed Image region that we want
  std::cout << "Extracting region and type..." << std::endl;
  LabelMapExtractorType::Pointer fixedExtractor = LabelMapExtractorType::New();
  fixedExtractor->SetInput( fixedLabelMap );

  LabelMapExtractorType::Pointer movingExtractor = LabelMapExtractorType::New();
  movingExtractor->SetInput( movingLabelMap );

  for ( unsigned int i=0; i<regionVec.size(); i++ )
    { 
    fixedExtractor->SetChestRegion(regionVec[i]);
    movingExtractor->SetChestRegion(regionVec[i]);
    }
  if (typeVec.size()>0)
    {
    for ( unsigned int i=0; i<typeVec.size(); i++ )
      {
      fixedExtractor->SetChestType( typeVec[i] );
      movingExtractor->SetChestType( typeVec[i] );
      }
    }
  if (regionTypePairVec.size()>0)
    {
    for ( unsigned int i=0; i<regionTypePairVec.size(); i++ )
      {
      fixedExtractor->SetRegionAndType( regionTypePairVec[i].region, regionTypePairVec[i].type );
      movingExtractor->SetRegionAndType( regionTypePairVec[i].region, regionTypePairVec[i].type );
      }
      }

  fixedExtractor->Update();
  movingExtractor->Update();
  
    //Read the CT images
  ShortImageType::Pointer ctFixedImage = ShortImageType::New();
  ctFixedImage = ReadCTFromFile( fixedCTFileName );
  if (ctFixedImage.GetPointer() == NULL)
      {
        return cip::NRRDREADFAILURE;
      }

  ShortImageType::Pointer ctMovingImage = ShortImageType::New();
  ctMovingImage = ReadCTFromFile( movingCTFileName );
  if (ctMovingImage.GetPointer() == NULL)
      {
        return cip::NRRDREADFAILURE;
      }
  
  std::cout << "Isolating region and type of interest..." << std::endl;
  LabelMapIteratorType it( fixedExtractor->GetOutput(), fixedExtractor->GetOutput()->GetBufferedRegion() );
  CTIteratorType itct( ctFixedImage, ctFixedImage->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
	itct.Set( itct.Get() );
      }
    else
      {
	itct.Set(0 );
      }
     ++it;
     ++itct;
        }

  LabelMapIteratorType itmoving( movingExtractor->GetOutput(), movingExtractor->GetOutput()->GetBufferedRegion() );
  CTIteratorType itmovingct( ctMovingImage, ctMovingImage->GetBufferedRegion() );
  itmoving.GoToBegin();
  while ( !itmoving.IsAtEnd() )
    {
    if ( itmoving.Get() != 0 )
      {
	itmovingct.Set( itmovingct.Get() );
      }
    else
      {
	itmovingct.Set(0 );
      }
     ++itmoving;
     ++itmovingct;
    }


  //Read in the transform file
  TransformType::Pointer transform = TransformType::New();
  ReadTransformFromFile(transform, inputTransformFileName.c_str() );


  std::cout<<"initializing transform"<<std::endl;
  InitializerType::Pointer initializer = InitializerType::New();
  initializer->SetTransform( transform ); 
  initializer->SetFixedImage(  ctFixedImage );
  initializer->SetMovingImage( ctMovingImage );
  initializer->MomentsOn();
  initializer->InitializeTransform();

  OptimizerScalesType optimizerScales( transform->GetNumberOfParameters() );
  optimizerScales[0] =  1.0;   optimizerScales[1] =  1.0;   optimizerScales[2] =  1.0;
  optimizerScales[3] =  1.0;   optimizerScales[4] =  1.0;   optimizerScales[5] =  1.0;
  optimizerScales[6] =  1.0;   optimizerScales[7] =  1.0;   optimizerScales[8] =  1.0;
  optimizerScales[9]  =  0.001;
  optimizerScales[10] =  0.001;
  optimizerScales[11] =  0.001;


  MetricType::Pointer metric = MetricType::New(); 
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetScales( optimizerScales );
  optimizer->SetMaximumStepLength( 1.0 );
  optimizer->SetMinimumStepLength( 0.001 );
  optimizer->SetNumberOfIterations( 0 );
 
  std::cout << "Starting registration for mutual information calculation..." << std::endl;
  RegistrationType::Pointer registration = RegistrationType::New();
  registration->SetMetric( metric );
  registration->SetOptimizer( optimizer );
  registration->SetInterpolator( interpolator );
  registration->SetTransform( transform );
  registration->SetFixedImage( ctFixedImage);
  registration->SetMovingImage(ctMovingImage );
  registration->SetFixedImageRegion(
  	 fixedExtractor->GetOutput()->GetBufferedRegion() );
  registration->SetInitialTransformParameters( transform->GetParameters());
  try
    {
      registration->StartRegistration();
    }
  catch( itk::ExceptionObject &excp )
    {
    std::cerr << "ExceptionObject caught while executing registration" <<
std::endl;
    std::cerr << excp << std::endl;
    }


  //The following is just for debugging purposes, to see that the transformation has not changed
  OptimizerType::ParametersType finalParams = registration->GetLastTransformParameters();
  //get the Mutual information value between the registered CT images
  int numberOfIterations2 = optimizer->GetCurrentIteration();
  const double mutualInformationValue = optimizer->GetValue();
  std::cout<<"number mutual iteration = "<<numberOfIterations2<<"  MI="<<mutualInformationValue<<std::endl;

  SIMILARITY_XML_DATA ctSimilarityXMLData;
  ctSimilarityXMLData.regionAndTypeUsed.assign("");
  for ( unsigned int i=0; i<regionVecArg.size(); i++ )
    {
      std::ostringstream regionArgs;
      regionArgs <<regionVecArg[i];
      ctSimilarityXMLData.regionAndTypeUsed.append(regionArgs.str().c_str());
      ctSimilarityXMLData.regionAndTypeUsed.append(", ");
    }
  for ( unsigned int i=0; i<typeVecArg.size(); i++ )
    {
      std::ostringstream regionArgs;
      regionArgs <<regionVecArg[i];
      ctSimilarityXMLData.regionAndTypeUsed.append(regionArgs.str().c_str());
      ctSimilarityXMLData.regionAndTypeUsed.append(", ");
    }
  if (regionPairVec.size() == typePairVecArg.size())
    {
    for ( unsigned int i=0; i<regionPairVecArg.size(); i++ )
      {
      std::ostringstream regionArgs;
      regionArgs <<regionVecArg[i];
      ctSimilarityXMLData.regionAndTypeUsed.append(regionArgs.str().c_str());
      ctSimilarityXMLData.regionAndTypeUsed.append(", ");
      }
    }

         
    ctSimilarityXMLData.similarityValue = (float)(mutualInformationValue);
    const char *similarity_type = metric->GetNameOfClass();
    ctSimilarityXMLData.similarityMeasure.assign(similarity_type);

  if ( strcmp(inputTransformFileName.c_str(), "q") != 0 )
    {
    std::string infoFilename = inputTransformFileName;
    int result = infoFilename.find_last_of('.');
    if (std::string::npos != result)
      infoFilename.erase(result);
    // append extension:
    infoFilename.append("_measures.xml");
                

 
    int pathLength = 0, pos=0, next=0;   

    if ( strcmp(movingImageID.c_str(), "q") != 0 ) 
      ctSimilarityXMLData.sourceID.assign(movingImageID);
    else

      {       
	//first find length of path
	next=1;
	while(next>=1)
	  {
	    next = movingCTFileName.find("/", next+1);    
	    pathLength++;
	  }
	pos=0;
	next=0;

	std::string tempSourceID;
	for (int i = 0; i < (pathLength-1);i++)
	  {
	    pos= next+1;
	    next = movingCTFileName.find("/", next+1);
	  }               
      ctSimilarityXMLData.sourceID.assign(movingCTFileName.c_str());
      ctSimilarityXMLData.sourceID.erase(next,ctSimilarityXMLData.sourceID.length()-1);
      ctSimilarityXMLData.sourceID.erase(0, pos);
      }
    

    if ( strcmp(fixedImageID.c_str(), "q") != 0 ) 
      ctSimilarityXMLData.destID =fixedImageID.c_str();
    else
      { 
      pos=0;
      next=0;
      for (int i = 0; i < (pathLength-1);i++)
        { 
        pos = next+1;
        next = fixedCTFileName.find('/', next+1);  
        }
      ctSimilarityXMLData.destID.assign(fixedCTFileName.c_str());// =tempSourceID.c_str();//movingImageFileName.substr(pos, next-1).c_str();
      ctSimilarityXMLData.destID.erase(next,ctSimilarityXMLData.destID.length()-1);
      ctSimilarityXMLData.destID.erase(0, pos);

      }	

    //remove path from output transformation file before storing in xml
    std::cout<<"outputtransform filename ="<<inputTransformFileName.c_str()<<std::endl;
      pos=0;
      next=0;
      for (int i = 0; i < (pathLength);i++)
        { 
        pos = next+1;
        next = inputTransformFileName.find('/', next+1);  
        }
      ctSimilarityXMLData.transformationLink.assign(inputTransformFileName.c_str());
      ctSimilarityXMLData.transformationLink.erase(0,pos);
      //WriteRegistrationXML(infoFilename.c_str(), ctSimilarityXMLData);

    }
  std::cout << "DONE." << std::endl;

  return 0;
}

