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
#include "RegisterLabelMapsCLP.h"
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
    
    typedef itk::Image< unsigned short, 3 >
    ImageType;
    typedef itk::ResampleImageFilter< ImageType, ImageType >
    ResampleFilterType;
    typedef itk::ImageFileReader< ImageType >
    ImageReaderType;
    typedef itk::ImageFileWriter< ImageType >
    ImageWriterType;
    typedef itk::RegularStepGradientDescentOptimizer
    OptimizerType;
    typedef itk::ImageRegistrationMethod< ImageType, ImageType >
    RegistrationType;
    typedef itk::KappaStatisticImageToImageMetric< ImageType, ImageType >
    MetricType;
    typedef itk::NearestNeighborInterpolateImageFunction< ImageType, double >
    InterpolatorType;
    typedef itk::AffineTransform<double, 3 >
    TransformType;
    typedef itk::CenteredTransformInitializer< TransformType, ImageType,
    ImageType >    InitializerType;
    typedef OptimizerType::ScalesType
    OptimizerScalesType;
    typedef itk::ImageRegionIteratorWithIndex< ImageType >
    IteratorType;
    typedef itk::RegionOfInterestImageFilter< ImageType, ImageType >
    RegionOfInterestType;
    typedef itk::ResampleImageFilter< ImageType, ImageType >
    ResampleType;
    typedef itk::IdentityTransform< double, 3 >
    IdentityType;
    typedef itk::CIPExtractChestLabelMapImageFilter
    LabelMapExtractorType;
    typedef itk::ImageSeriesReader< cip::CTType >
    CTSeriesReaderType;
    typedef itk::GDCMImageIO
    ImageIOType;
    typedef itk::GDCMSeriesFileNames
    NamesGeneratorType;
    typedef itk::ImageFileReader< cip::CTType >
    CTFileReaderType;
    typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >
    LabelMapIteratorType;
    
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
    };
    
    void WriteTransformFile( TransformType::Pointer transform, char* fileName )
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
    
    
    cip::LabelMapType::Pointer ReadLabelMapFromFile( std::string
                                                    labelMapFileName )
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
                                 "RegistrationOutput_v1.dtd");
        
        //ID: attribute
        /* uuid not working on cluster
         uuid_t registration_id;
         char uuid_string[37];
         uuid_generate(registration_id);
         uuid_unparse(registration_id, uuid_string);
         std::string temp_string(uuid_string);;
         theXMLData.registrationID.assign(temp_string);
         */
        
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
        xmlNewChild(root_node, NULL, BAD_CAST "transformation", BAD_CAST
                    (theXMLData.transformationLink.c_str()));
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
    cip::LabelMapType::Pointer subSampledFixedImage =
    cip::LabelMapType::New();
    if ( strcmp( fixedImageFileName.c_str(), "q") != 0 )
    {
        std::cout << "Reading label map from file..." << std::endl;
        fixedLabelMap = ReadLabelMapFromFile( fixedImageFileName );
        
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
    
    std::cout << "Subsampling fixed image with factor..."
    <<downsampleFactor<< std::endl;
    // ResampleImage(fixedLabelMap, subSampledFixedImage, downsampleFactor );
    
    subSampledFixedImage=cip::DownsampleLabelMap(downsampleFactor,fixedLabelMap);
    
    
    //Read in moving image label map from file and subsample
    cip::LabelMapType::Pointer movingLabelMap = cip::LabelMapType::New();
    cip::LabelMapType::Pointer subSampledMovingImage =
    cip::LabelMapType::New();
    if ( strcmp( movingImageFileName.c_str(), "q") != 0 )
    {
        std::cout << "Reading label map from file..." << std::endl;
        movingLabelMap = ReadLabelMapFromFile( movingImageFileName );
        
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
    
    std::cout << "Subsampling moving image..." << std::endl;
    
    subSampledMovingImage=cip::DownsampleLabelMap(downsampleFactor,movingLabelMap);
    
    // Extract fixed Image region that we want
    std::cout << "Extracting region and type..." << std::endl;
    LabelMapExtractorType::Pointer fixedExtractor =
    LabelMapExtractorType::New();
    fixedExtractor->SetInput( subSampledFixedImage );
    
    LabelMapExtractorType::Pointer movingExtractor =
    LabelMapExtractorType::New();
    movingExtractor->SetInput( subSampledMovingImage );
    
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
            fixedExtractor->SetRegionAndType( regionTypePairVec[i].region,
                                             regionTypePairVec[i].type );
            movingExtractor->SetRegionAndType( regionTypePairVec[i].region,
                                              regionTypePairVec[i].type );
        }
    }
    
    fixedExtractor->Update();
    movingExtractor->Update();
    
    std::cout << "Isolating region and type of interest..." << std::endl;
    LabelMapIteratorType it( fixedExtractor->GetOutput(),
                            fixedExtractor->GetOutput()->GetBufferedRegion() );
    
    it.GoToBegin();
    while ( !it.IsAtEnd() )
    {
        if ( it.Get() != 0 )
        {
            it.Set( 1 );
        }
        ++it;
    }
    
    LabelMapIteratorType itmoving( movingExtractor->GetOutput(),
                                  movingExtractor->GetOutput()->GetBufferedRegion() );
    
    itmoving.GoToBegin();
    while ( !itmoving.IsAtEnd() )
    {
        if ( itmoving.Get() != 0 )
        {
            itmoving.Set( 1 );
        }
        
        ++itmoving;
    }
    
    MetricType::Pointer metric = MetricType::New();
    //because we are minimizing as opposed to maximizing
    metric->SetForegroundValue( 1);
    
    
    TransformType::Pointer transform = TransformType::New();
    std::cout<<"initializing transform"<<std::endl;
    InitializerType::Pointer initializer = InitializerType::New();
    initializer->SetTransform( transform );
    initializer->SetFixedImage(  fixedExtractor->GetOutput() );
    initializer->SetMovingImage( movingExtractor->GetOutput() );
    initializer->MomentsOn();
    initializer->InitializeTransform();
    
    OptimizerScalesType optimizerScales( transform->GetNumberOfParameters()
                                        );
    optimizerScales[0] =  1.0;   optimizerScales[1] =  1.0;
    optimizerScales[2] =  1.0;
    optimizerScales[3] =  1.0;   optimizerScales[4] =  1.0;
    optimizerScales[5] =  1.0;
    optimizerScales[6] =  1.0;   optimizerScales[7] =  1.0;
    optimizerScales[8] =  1.0;
    optimizerScales[9]  =  translationScale;
    optimizerScales[10] =  translationScale;
    optimizerScales[11] =  translationScale;
    
    OptimizerType::Pointer optimizer = OptimizerType::New();
    optimizer->SetScales( optimizerScales );
    optimizer->SetMaximumStepLength( maxStepLength );
    optimizer->SetMinimumStepLength( minStepLength );
    optimizer->SetNumberOfIterations( numberOfIterations );
    
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    
    std::cout << "Starting registration..." << std::endl;
    RegistrationType::Pointer registration = RegistrationType::New();
    registration->SetMetric( metric );
    registration->SetOptimizer( optimizer );
    registration->SetInterpolator( interpolator );
    registration->SetTransform( transform );
    registration->SetFixedImage( fixedExtractor->GetOutput() );
    registration->SetMovingImage( movingExtractor->GetOutput() );
    registration->SetFixedImageRegion(
                                      fixedExtractor->GetOutput()->GetBufferedRegion() );
    registration->SetInitialTransformParameters( transform->GetParameters()
                                                );
    try
    {
        registration->StartRegistration();
        //registration->Update(); for ITKv4
    }
    catch( itk::ExceptionObject &excp )
    {
        std::cerr << "ExceptionObject caught while executing registration" <<
        std::endl;
        std::cerr << excp << std::endl;
    }
    
    //get all params to output to file
    numberOfIterations = optimizer->GetCurrentIteration();
    
    //  The value of the image metric corresponding to the last set of parameters
    //  can be obtained with the \code{GetValue()} method of the optimizer.
    
    const double bestValue = optimizer->GetValue();
    
    std::cout<<"similarity output = "<< optimizer->GetValue() <<" best value= " <<bestValue<<std::endl;
    OptimizerType::ParametersType finalParams = registration->GetLastTransformParameters();
    
    TransformType::Pointer finalTransform = TransformType::New();
    finalTransform->SetParameters( finalParams );
    finalTransform->SetCenter( transform->GetCenter() );
    
    
    if ( strcmp(outputTransformFileName.c_str(), "q") != 0 )
    {
        std::string infoFilename = outputTransformFileName;
        int result = infoFilename.find_last_of('.');
        // Does new_filename.erase(std::string::npos) working here in place ofthis following test?
        if (std::string::npos != result)
            infoFilename.erase(result);
        // append extension:
        infoFilename.append(".xml");
        
        REGISTRATION_XML_DATA labelMapRegistrationXMLData;
        labelMapRegistrationXMLData.similarityValue = (float)(bestValue);
        const char *similarity_type = metric->GetNameOfClass();
        labelMapRegistrationXMLData.similarityMeasure.assign(similarity_type);
        //if the patient IDs are specified  as args, use them,
        //otherwise, extract from patient path
        
        int pathLength = 0, pos=0, next=0;
        
        if ( strcmp(movingImageID.c_str(), "q") != 0 )
            labelMapRegistrationXMLData.sourceID.assign(movingImageID);
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
            labelMapRegistrationXMLData.destID =fixedImageID.c_str();
        else
        {
            pos=0;
            next=0;
            for (int i = 0; i < (pathLength-1);i++)
            {
                pos = next+1;
                next = fixedImageFileName.find('/', next+1);
            }
            
            labelMapRegistrationXMLData.destID.assign(fixedImageFileName.c_str());//=tempSourceID.c_str();//movingImageFileName.substr(pos, next-1).c_str();
            
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
        WriteRegistrationXML(infoFilename.c_str(),
                             labelMapRegistrationXMLData);
        
        
        
        std::cout << "Writing transform..." << std::endl;
        itk::TransformFileWriter::Pointer transformWriter =
        itk::TransformFileWriter::New();
        transformWriter->SetInput( finalTransform );
        transformWriter->SetFileName( outputTransformFileName );
        transformWriter->Update();
        
    }
    
    if ( strcmp(outputImageFileName.c_str(), "q") != 0 )
    {
        std::cout << "Resampling moving image..." << std::endl;
        
        ResampleFilterType::Pointer resample = ResampleFilterType::New();
        resample->SetTransform( finalTransform );
        resample->SetInput( movingExtractor->GetOutput() );
        resample->SetInput( movingExtractor->GetOutput() );
        resample->SetSize(
                          fixedExtractor->GetOutput()->GetBufferedRegion().GetSize() );
        resample->SetOutputOrigin(  fixedExtractor->GetOutput()->GetOrigin() );
        resample->SetOutputSpacing( fixedExtractor->GetOutput()->GetSpacing()
                                   );
        resample->SetInterpolator( interpolator );
        resample->SetDefaultPixelValue( 0 );
        resample->Update();
        
        
        ImageType::Pointer upsampledImage = ImageType::New();
        
        std::cout << "Upsampling to original size..." << std::endl;
        //ResampleImage( resample->GetOutput(), upsampledImage,1.0/downsampleFactor );
        upsampledImage=cip::UpsampleLabelMap(downsampleFactor,
                                             resample->GetOutput());
        ImageWriterType::Pointer writer = ImageWriterType::New();
        writer->SetInput(upsampledImage);// movingExtractor->GetOutput()
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
    
    std::cout << "DONE." << std::endl;
    
    return 0;
}
