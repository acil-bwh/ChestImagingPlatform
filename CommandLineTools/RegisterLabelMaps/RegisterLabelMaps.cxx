/** \file
*  \ingroup commandLineTools 
*  \details This program reads atlas lung images and generates a
*
*  USAGE:
*
*  $Date: $
*  $Revision: $
*  $Author:  $
*
*/

//
//RegisterLabelMaps --regionVec 3 -m D:\Postdoc\Data\Processed\lola11-04\lola11-04_oriented_leftLungRightLung.nrrd -f D:\Postdoc\Data\Processed\lola11-04\lola11-04_oriented_leftLungRightLung.nrrd --outputImage D:\Postdoc\Data\Processed\lola11-04\lola11-5to4reg_oriented_leftLungRightLung.nrrd

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

namespace
{
         #define MY_ENCODING "ISO-8859-1"

	typedef itk::Image< unsigned short, 3 >                                             ImageType;
	typedef itk::ResampleImageFilter< ImageType, ImageType >                            ResampleFilterType;
	typedef itk::ImageFileReader< ImageType >                                           ImageReaderType;
	typedef itk::ImageFileWriter< ImageType >                                           ImageWriterType;
	typedef itk::RegularStepGradientDescentOptimizer                                    OptimizerType;
	typedef itk::ImageRegistrationMethod< ImageType, ImageType >                        RegistrationType;
	typedef itk::KappaStatisticImageToImageMetric< ImageType, ImageType >               MetricType;
	typedef itk::NearestNeighborInterpolateImageFunction< ImageType, double >           InterpolatorType;
	typedef itk::AffineTransform<double, 3 >                                            TransformType;
	typedef itk::CenteredTransformInitializer< TransformType, ImageType, ImageType >    InitializerType;
	typedef OptimizerType::ScalesType                                                   OptimizerScalesType;
	typedef itk::ImageRegionIteratorWithIndex< ImageType >                              IteratorType;
	typedef itk::RegionOfInterestImageFilter< ImageType, ImageType >                    RegionOfInterestType;
	typedef itk::ResampleImageFilter< ImageType, ImageType >                            ResampleType;
	typedef itk::IdentityTransform< double, 3 >                                         IdentityType;
	typedef itk::CIPExtractChestLabelMapImageFilter                                     LabelMapExtractorType;
	typedef itk::ImageSeriesReader< cip::CTType >                                       CTSeriesReaderType;
	typedef itk::GDCMImageIO                                                            ImageIOType;
	typedef itk::GDCMSeriesFileNames                                                    NamesGeneratorType;
	typedef itk::ImageFileReader< cip::CTType >                                         CTFileReaderType;
	typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                      LabelMapIteratorType;

	struct REGIONTYPEPAIR
	{
		unsigned char region;
		unsigned char type;
	};

        struct REGISTRATION_XML_DATA
        {
          char registrationID[256];
          float similarityValue;
          char transformationLink[256];
          char sourceID[256];
          char destID[256];
        };

	void WriteTransformFile( TransformType::Pointer transform, char* fileName )
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


	void ResampleImage( ImageType::Pointer image, ImageType::Pointer subsampledROIImage, float downsampleFactor )
	{
		ImageType::SizeType inputSize = image->GetBufferedRegion().GetSize();

		ImageType::SpacingType inputSpacing = image->GetSpacing();

		ImageType::SpacingType outputSpacing;
		outputSpacing[0] = inputSpacing[0]*downsampleFactor;
		outputSpacing[1] = inputSpacing[1]*downsampleFactor;
		outputSpacing[2] = inputSpacing[2]*downsampleFactor;

		ImageType::SizeType outputSize;
		outputSize[0] = static_cast< unsigned int >( static_cast< double >( inputSize[0] )/downsampleFactor );
		outputSize[1] = static_cast< unsigned int >( static_cast< double >( inputSize[1] )/downsampleFactor );
		outputSize[2] = static_cast< unsigned int >( static_cast< double >( inputSize[2] )/downsampleFactor );

		InterpolatorType::Pointer interpolator = InterpolatorType::New();

		IdentityType::Pointer transform = IdentityType::New();
		transform->SetIdentity();

		ResampleType::Pointer resampler = ResampleType::New();
		resampler->SetTransform( transform );
		resampler->SetInterpolator( interpolator );
		resampler->SetInput( image );
		resampler->SetSize( outputSize );
		resampler->SetOutputSpacing( outputSpacing );
		resampler->SetOutputOrigin( image->GetOrigin() );
		try
		{
			resampler->Update();
		}
		catch ( itk::ExceptionObject &excp )
		{
			std::cerr << "Exception caught down sampling:";
			std::cerr << excp << std::endl;
		}

		subsampledROIImage->SetRegions( resampler->GetOutput()->GetBufferedRegion().GetSize() );
		subsampledROIImage->Allocate();
		subsampledROIImage->FillBuffer( 0 );
		subsampledROIImage->SetSpacing( outputSpacing );
		subsampledROIImage->SetOrigin( image->GetOrigin() );

		IteratorType rIt( resampler->GetOutput(), resampler->GetOutput()->GetBufferedRegion() );
		IteratorType sIt( subsampledROIImage, subsampledROIImage->GetBufferedRegion() );

		rIt.GoToBegin();
		sIt.GoToBegin();
		while ( !sIt.IsAtEnd() )
		{
			sIt.Set( rIt.Get() );

			++rIt;
			++sIt;
		}
	} 

	cip::CTType::Pointer ReadCTFromDirectory( std::string ctDir )
	{
		ImageIOType::Pointer gdcmIO = ImageIOType::New();

		std::cout << "---Getting file names..." << std::endl;
		NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
		namesGenerator->SetInputDirectory( ctDir );

		const CTSeriesReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();

		std::cout << "---Reading DICOM image..." << std::endl;
		CTSeriesReaderType::Pointer dicomReader = CTSeriesReaderType::New();
		dicomReader->SetImageIO( gdcmIO );
		dicomReader->SetFileNames( filenames );
		try
		{
			dicomReader->Update();
		}
		catch (itk::ExceptionObject &excp)
		{
			std::cerr << "Exception caught while reading dicom:";
			std::cerr << excp << std::endl;
			return NULL;
		}  

		return dicomReader->GetOutput();
	}


	cip::CTType::Pointer ReadCTFromFile( std::string fileName )
	{
		CTFileReaderType::Pointer reader = CTFileReaderType::New();
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


void WriteRegistrationXML(const char *file, REGISTRATION_XML_DATA theXMLData)
        {      

             xmlDocPtr doc = NULL;       /* document pointer */
             xmlNodePtr root_node = NULL, node = NULL, node1 = NULL;/* node pointers */
             xmlDtdPtr dtd = NULL;       /* DTD pointer */
             char buff[256];
             int i, j;
             //http://www.xmlsoft.org/examples/tree2.c
    
             doc = xmlNewDoc(BAD_CAST "1.0");
             root_node = xmlNewNode(NULL, BAD_CAST "Registration");
             xmlDocSetRootElement(doc, root_node);

             dtd = xmlCreateIntSubset(doc, BAD_CAST "root", NULL, BAD_CAST "RegistrationOutput.dtd");

             // xmlNewChild() creates a new node, which is "attached"
             // as child node of root_node node. 
     
             xmlNewChild(root_node, NULL, BAD_CAST "SimilarityValue", BAD_CAST (char)(theXMLData.similarityValue));

             xmlSaveFormatFileEnc("testxmlout3.xml", doc, "UTF-8", 1);
             xmlFreeDoc(doc);

         }

} //end namespace

int main( int argc, char *argv[] )
{
	  std::vector< unsigned char >  regionVec;
      std::vector< unsigned char >  typeVec;
      std::vector< unsigned char >  regionPairVec;
      std::vector< unsigned char >  typePairVec;


	std::cout<<"parsing args"<<std::endl;
	PARSE_ARGS;
	//ChestConventions conventions;
	std::cout<<"args parsed"<<std::endl;


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

	std::cout << "Subsampling fixed image..." << std::endl;
	ResampleImage(fixedLabelMap, subSampledFixedImage, downsampleFactor );

	//Read in moving image label map from file and subsample
	cip::LabelMapType::Pointer movingLabelMap = cip::LabelMapType::New();
	cip::LabelMapType::Pointer subSampledMovingImage = cip::LabelMapType::New();
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
	ResampleImage(movingLabelMap, subSampledMovingImage, downsampleFactor );

	std::cout<<"region vec size="<<regionVec.size()<<std::endl;

	// Extract fixed Image region that we want
	std::cout << "Extracting region and type..." << std::endl;
	LabelMapExtractorType::Pointer fixedExtractor = LabelMapExtractorType::New();
	fixedExtractor->SetInput( subSampledFixedImage );

	LabelMapExtractorType::Pointer movingExtractor = LabelMapExtractorType::New();
	movingExtractor->SetInput( subSampledMovingImage );

	for ( unsigned int i=0; i<regionVec.size(); i++ )
	{ 
		fixedExtractor->SetChestRegion(  static_cast< unsigned char >( regionVec[i]) );
		movingExtractor->SetChestRegion( static_cast< unsigned char >(regionVec[i]));
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

	std::cout << "Isolating region and type of interest..." << std::endl;
	LabelMapIteratorType it( fixedExtractor->GetOutput(), fixedExtractor->GetOutput()->GetBufferedRegion() );

	it.GoToBegin();
	while ( !it.IsAtEnd() )
	{
		if ( it.Get() != 0 )
		{
			it.Set( 1 );
		}
		++it;
	}

        LabelMapIteratorType itmoving( movingExtractor->GetOutput(), movingExtractor->GetOutput()->GetBufferedRegion() );

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
	metric->ComplementOn();
	metric->SetForegroundValue( 1);

	TransformType::Pointer transform = TransformType::New();

	InitializerType::Pointer initializer = InitializerType::New();
	initializer->SetTransform( transform ); 
	initializer->SetFixedImage(  fixedExtractor->GetOutput() );
	initializer->SetMovingImage( movingExtractor->GetOutput() );
	initializer->MomentsOn();
	initializer->InitializeTransform();

	OptimizerScalesType optimizerScales( transform->GetNumberOfParameters() );
	optimizerScales[0] =  1.0;   optimizerScales[1] =  1.0;   optimizerScales[2] =  1.0;
	optimizerScales[3] =  1.0;   optimizerScales[4] =  1.0;   optimizerScales[5] =  1.0;
	optimizerScales[6] =  1.0;   optimizerScales[7] =  1.0;   optimizerScales[8] =  1.0;
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
	registration->SetFixedImageRegion( fixedExtractor->GetOutput()->GetBufferedRegion() );
	registration->SetInitialTransformParameters( transform->GetParameters() );
	try 
	{ 
		registration->StartRegistration(); 
	} 
	catch( itk::ExceptionObject &excp ) 
	{ 
		std::cerr << "ExceptionObject caught while executing registration" << std::endl; 
		std::cerr << excp << std::endl; 
	} 

	//get all params to output to file
	numberOfIterations = optimizer->GetCurrentIteration();

	//  The value of the image metric corresponding to the last set of parameters
	//  can be obtained with the \code{GetValue()} method of the optimizer.

	const double bestValue = optimizer->GetValue();


	OptimizerType::ParametersType finalParams = registration->GetLastTransformParameters();

	TransformType::Pointer finalTransform = TransformType::New();
	finalTransform->SetParameters( finalParams );
	finalTransform->SetCenter( transform->GetCenter() );
/*
        struct REGISTRATION_XML_DATA
        {
          char registrationID[256];
          float similarityValue;
          char transformationLink[256];
          char sourceID[256];
          char destID[256];
        }
*/
        if ( strcmp(outputTransformFileName.c_str(), "q") != 0 )
	{
		std::string infoFilename = outputTransformFileName;
		int result = infoFilename.find_last_of('.');
		// Does new_filename.erase(std::string::npos) working here in place of this following test?
		if (std::string::npos != result)
			infoFilename.erase(result);
		// append extension:
		infoFilename.append("xml");
                
                REGISTRATION_XML_DATA labelMapRegistrationXMLData;
                labelMapRegistrationXMLData.registrationID = "1";
                labelMapRegistrationXMLData.similarityValue = bestValue;
                labelMapRegistrationXMLData.transformationLink = outputTransformFileName.c_str();
                
		/*std::ofstream infoFile( infoFilename.c_str() );
		infoFile<<"Transformation Filename: " <<outputTransformFileName.c_str()<<std::endl;
		infoFile<<"Transformation: "<<std::endl;
		infoFile<<"Similarity Metric: " <<std::endl;
		infoFile<<"optimizer: "  <<std::endl;
		infoFile<<"Number of iterations used: " <<numberOfIterations<<std::endl;
		infoFile<<"Final metric value: " <<bestValue<<std::endl;
                */
                WriteRegistrationXML(infoFilename.c_str(), labelMapRegistrationXMLData);


		std::cout << "Writing transform..." << std::endl;
		itk::TransformFileWriter::Pointer transformWriter = itk::TransformFileWriter::New();
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
		resample->SetSize( fixedExtractor->GetOutput()->GetBufferedRegion().GetSize() );
		resample->SetOutputOrigin(  fixedExtractor->GetOutput()->GetOrigin() );
		resample->SetOutputSpacing( fixedExtractor->GetOutput()->GetSpacing() );
		resample->SetInterpolator( interpolator );
		resample->SetDefaultPixelValue( 0 );
		resample->Update();


		ImageType::Pointer upsampledImage = ImageType::New();

		std::cout << "Upsampling to original size..." << std::endl;
		ResampleImage( resample->GetOutput(), upsampledImage, 1.0/downsampleFactor );

		ImageWriterType::Pointer writer = ImageWriterType::New();
		writer->SetInput(upsampledImage);// movingExtractor->GetOutput() );//upsampledImage );
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

	//Write the registration information file


	std::cout << "DONE." << std::endl;

	return 0;
}

