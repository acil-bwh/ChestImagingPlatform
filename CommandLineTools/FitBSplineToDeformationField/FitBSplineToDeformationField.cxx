//#if defined(_MSC_VER)
//#pragma warning ( disable : 4786 )
//#endif

#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTransformFileWriter.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkBSplineTransform.h"
#include "itkCastImageFilter.h"
#include "itkImageAdaptor.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkIdentityTransform.h"
#include "itkResampleImageFilter.h"
#include "FitBSplineToDeformationFieldCLP.h"


template< class TPixel >
class DFValueAccessor
{
public:
	typedef TPixel                     InternalType;
	typedef float                      ExternalType;
  
	inline ExternalType Get( const InternalType & input ) const 
	{
		return static_cast<ExternalType>( input[m_DFIdx] );
	}

	void SetDFIdx( unsigned int i )
	{
		this->m_DFIdx = i;
	}
  
private:
	unsigned int m_DFIdx;
};

int main(int argc, char *argv[] )
{

  PARSE_ARGS;
	
	const    unsigned int    ImageDimension = 3;
	typedef  float           PixelType;

	const unsigned int SpaceDimension = ImageDimension;
	const unsigned int SplineOrder = 3;
	
	std::cout << ""  << std::endl;
	std::cout << "Using BSpline Order:  " << SplineOrder << std::endl;
	
	typedef float CoordinateRepType;
    
	typedef itk::BSplineTransform<CoordinateRepType,
                              SpaceDimension,
                              SplineOrder >     TransformType;
	unsigned int numberOfGridNodes = NumberOfNodes;
							  
	typedef TransformType::ParametersType      ParametersType;
	typedef TransformType::ImageType 		   ParametersImageType;
	
	
	//Read deformation field 
	std::cout << "Reading displacement field..." << std::endl;
	
	typedef itk::Vector< PixelType, ImageDimension >          VectorPixelType;
	typedef itk::Image< VectorPixelType, ImageDimension > DisplacementFieldImageType;
	typedef itk::ImageFileReader<DisplacementFieldImageType> DFReaderType;
  
	DFReaderType::Pointer dfreader = DFReaderType::New();
	dfreader->SetFileName(deformationFileName);
  
  try
  {
    dfreader->Update();
  }
  catch ( itk::ExceptionObject &excp )
  {
    std::cerr << "Exception caught reading deformation field image";
    std::cerr << excp << std::endl;
    return cip::NRRDREADFAILURE;
  }
  
	DisplacementFieldImageType::Pointer defField = dfreader->GetOutput();
	
	// Initiate transform
	TransformType::Pointer  transform = TransformType::New();
	numberOfGridNodes = NumberOfNodes;
    
	TransformType::MeshSizeType             meshSize;
	TransformType::OriginType               fixedOrigin;
	TransformType::PhysicalDimensionsType   fixedPhysicalDimensions;
		
	for( unsigned int i=0; i< SpaceDimension; i++ )
	{
		fixedOrigin[i] = defField->GetOrigin()[i];
		fixedPhysicalDimensions[i] = defField->GetSpacing()[i] *
			static_cast<double>(
				defField->GetLargestPossibleRegion().GetSize()[i] - 1 );
	}
    
	// Define BSpline mesh size and set all the parameters based on the reference image
	meshSize.Fill( numberOfGridNodes - SplineOrder );
	
	transform->SetTransformDomainOrigin( fixedOrigin );
	transform->SetTransformDomainPhysicalDimensions(
		fixedPhysicalDimensions );
	transform->SetTransformDomainMeshSize( meshSize );
	transform->SetTransformDomainDirection( defField->GetDirection() );
    
	ParametersType parameters( transform->GetNumberOfParameters() );
	parameters.Fill( 0.0 );
	
	// Calculates the B-Spline coefficients of an image. Spline order may be from 0 to 5.
	// The input image is the displacement field found during the registration process (BSplineSyn)
	std::cout << "Adjusting BSpline to the Displacement field..." << std::endl;
	
	unsigned int counter = 0;
	for ( unsigned int k = 0; k < SpaceDimension; k++ )
	{
	
		//Extract deformation in each direction from the deformation field 
		typedef itk::FixedArray< PixelType, ImageDimension > DFArrayType;
		typedef itk::ImageAdaptor< DisplacementFieldImageType, DFValueAccessor< DFArrayType > > 		ImageAdaptorType;
		typedef itk::Image< PixelType, ImageDimension > 												EachDFValueImageType;
		typedef itk::CastImageFilter< ImageAdaptorType, ParametersImageType >							CastImageFilterType;
	
		ImageAdaptorType::Pointer dfAdaptor = ImageAdaptorType::New();
		DFValueAccessor< DFArrayType > accessor;
		accessor.SetDFIdx( k );
		dfAdaptor->SetImage( defField );
		dfAdaptor->SetPixelAccessor( accessor );

		CastImageFilterType::Pointer caster = CastImageFilterType::New();
		caster->SetInput( dfAdaptor );
		caster->Update();
		
			// QC step to see if we are spliting each component of the DF propertly
			// Write each component of the displacement field
			// typedef itk::ImageFileWriter< EachDFValueImageType  > ImageFileWriterType;
			// ImageFileWriterType::Pointer  writeImage  = ImageFileWriterType::New();	
		
			// To verify that we are extracting each component of the displacement field correctly
			//writeImage->SetInput(caster->GetOutput());
			//std::string name = outputPrefix + "comp" + std::to_string(k+1) + ".nrrd";
			//writeImage->SetFileName( name );
			//writeImage->Update();			
		
		// Down sample the displacement field to the resolution of the desired bspline
	    typedef itk::ResampleImageFilter<ParametersImageType,ParametersImageType> ResamplerType;
	    ResamplerType::Pointer downsampler = ResamplerType::New();
		
		typedef itk::BSplineResampleImageFunction<ParametersImageType,double> FunctionType;
		    FunctionType::Pointer function = FunctionType::New();

	    typedef itk::IdentityTransform<double,SpaceDimension> IdentityTransformType;
	    IdentityTransformType::Pointer identity = IdentityTransformType::New();

	    downsampler->SetInput( caster->GetOutput() );
	    downsampler->SetInterpolator( function );
	    downsampler->SetTransform( identity );
	    downsampler->SetSize( transform->GetCoefficientImages()[k]->
	      GetLargestPossibleRegion().GetSize() );
	    downsampler->SetOutputSpacing(
	      transform->GetCoefficientImages()[k]->GetSpacing() );
	    downsampler->SetOutputOrigin(
	      transform->GetCoefficientImages()[k]->GetOrigin() );
	    downsampler->SetOutputDirection( defField->GetDirection() );
		
		//Adjust the BSpline to the image and extract the coefficients		 
		typedef itk::BSplineDecompositionImageFilter<ParametersImageType,ParametersImageType> DecompositionType;
		DecompositionType::Pointer decomposition = DecompositionType::New();
		decomposition->SetSplineOrder( SplineOrder );
		decomposition->SetInput( downsampler->GetOutput() );
		decomposition->Update();
		
		// // QC step to see what image we are adjusting
		// typedef itk::ImageFileWriter< EachDFValueImageType  > ImageFileWriterType;
		// ImageFileWriterType::Pointer  writeImage  = ImageFileWriterType::New();
		// writeImage->SetInput(decomposition->GetOutput());
		// std::string name = outputPrefix + "bspline" + std::to_string(k+1) + ".nrrd";
		// writeImage->SetFileName( name );
		// writeImage->Update();

		ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();
		
		// Copy the coefficients into the parameter array
		typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
		Iterator it( newCoefficients,
		transform->GetCoefficientImages()[k]->GetLargestPossibleRegion() );
		while ( !it.IsAtEnd() )
		{
			parameters[ counter++ ] = it.Get();
			++it;
		}
	}
	
	transform->SetParameters( parameters);
	 

	//Write tranform
	std::cout << "Saving new BSpline transform..." << std::endl;
	
#if (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 5) || ITK_VERSION_MAJOR > 4
	itk::TransformFileWriterTemplate<float>::Pointer writer =
		itk::TransformFileWriterTemplate<float>::New();
#else
	itk::TransformFileWriter::Pointer writer = itk::TransformFileWriter::New();
#endif
	writer->SetInput(transform);
	//writer->SetFileName(outputPrefix +".tfm");
  writer->SetFileName(bsplineFileName);
  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject &excp )
  {
    std::cerr << "Exception caught writing output Bspline image";
    std::cerr << excp << std::endl;
    return cip::NRRDWRITEFAILURE;
  }

	std::cout << "Done" << std::endl;
	
  return cip::EXITSUCCESS;
	
}
