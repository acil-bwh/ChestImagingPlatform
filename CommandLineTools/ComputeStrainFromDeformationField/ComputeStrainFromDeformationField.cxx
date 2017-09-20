#include "cipHelper.h"
#include "ComputeStrainFromDeformationFieldCLP.h"
#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTransformFileReader.h"
#include "itkTransformFactoryBase.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkCompositeTransform.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkCastImageFilter.h"
#include "itkImageAdaptor.h"
#include "itkStrainImageFilter.h"

// Eigenvalue pixel accessor to access vector of eigen value pixels
// as individual images
template< class TPixel >
class EigenValueAccessor
{
public:
    typedef TPixel                     InternalType;
    typedef float                      ExternalType;
    
    inline ExternalType Get( const InternalType & input ) const
    {
        return static_cast<ExternalType>( input[m_EigenIdx] );
    }
    
    void SetEigenIdx( unsigned int i )
    {
        this->m_EigenIdx = i;
    }
    
private:
    unsigned int m_EigenIdx;
};


int main( int argc, char * argv[] )
{
    PARSE_ARGS;
  
    //Read deformation field
    
    const unsigned int Dimension = 3;
    typedef float PixelType;

    typedef itk::Image< PixelType, Dimension >         ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typedef itk::Transform<PixelType, Dimension, Dimension> TransformType;
    typedef itk::CompositeTransform<PixelType, Dimension> CompositeTransformType;
    typedef TransformType::Pointer transform;

    //Read deformation field
    typedef itk::Vector< PixelType, Dimension >          VectorPixelType;
    typedef itk::Image< VectorPixelType, Dimension > DisplacementFieldImageType;
    typedef itk::ImageFileReader<DisplacementFieldImageType> DFReaderType;
  
    DFReaderType::Pointer dfreader = DFReaderType::New();
    dfreader->SetFileName(deformationFileName);

  
    try
    {
      dfreader->Update();
      
    }
    catch( itk::ExceptionObject & excp )
    {

      std::cerr << "Error while reading deformation field" << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }

  
    // Generate a strain field image from a displacement field image. ------------------------------------------------------
    // Strain is a symmetric second rank tensor
    // Green Lagrangian: tracking a material point
    // Eulerian-Almansi: tracking a spatial point
    // The output image is defined as a symmetric second rank tensor image

    typedef itk::Vector< PixelType, Dimension > DisplacementVectorType;
    typedef itk::Image< DisplacementVectorType, Dimension > InputImageType;
    typedef itk::StrainImageFilter< InputImageType, PixelType, PixelType > StrainFilterType;
    StrainFilterType::Pointer strainFilter = StrainFilterType::New();
    strainFilter->SetInput( dfreader->GetOutput() );
    
    if (StrainType == "Lagrangian") {
        
        strainFilter->SetStrainForm( StrainFilterType::GREENLAGRANGIAN );
        
    } else if (StrainType == "Almansi") {
        
        strainFilter->SetStrainForm( StrainFilterType::EULERIANALMANSI  );
        
    } else if (StrainType == "Infinitesimal") {
        
        strainFilter->SetStrainForm( StrainFilterType::INFINITESIMAL  );
    } else {
        
        std::cerr << "Missing the strain type: infinitesimal, lagrangian or almansi should be specified" << std::endl;
        
        return cip::EXITFAILURE;
    }
  
    if (DeformationTensor) {
      strainFilter->SetDeformationTensor(true);
    } else {
      strainFilter->SetDeformationTensor(false);
    }
    
   
    strainFilter->Update();
    itk::Image<itk::SymmetricSecondRankTensor<PixelType, Dimension>, Dimension> * strainTensor;
    strainTensor = strainFilter->GetOutput();
  
    //Once the strainTensor is calculated we delete the reader of the displacement field to deallocate memory
    dfreader = NULL;
  
    //Write the strain tensor
    //typedef itk::ImageFileWriter< StrainFilterType::OutputImageType >  tensorWriterType;
    //tensorWriterType::Pointer tensorWriter = tensorWriterType::New();
    //tensorWriter->SetInput( strainTensor );
    //tensorWriter->SetFileName(outputPrefix + "strainTensor_output.nii.gz");
    //tensorWriter->Update();
    
    // Compute egigen Analysis -----------------------------------------------------------------------------------------
    // SymmetricEigenAnalysisImageFilter applies pixel-wise the invokation for computing the
    // eigen-values and eigen-vectors of the symmetric matrix corresponding to every input pixel.
    // 	EigenValueOrderType {
    //	OrderByValue = 1, OrderByValue: lambda_1 < lambda_2 < lambda_3
    //	OrderByMagnitude, OrderByMagnitude: |lambda_1| < |lambda_2| < |lambda_3|
    //	DoNotOrder
    // The default operation is to order eigen values in ascending order.
    typedef   itk::FixedArray< PixelType, Dimension >  EigenValueArrayType;
    typedef   itk::Image< EigenValueArrayType, Dimension >	EigenValueImageType;
    typedef itk::SymmetricEigenAnalysisImageFilter< StrainFilterType::OutputImageType, EigenValueImageType > EigenFilterType;
    
    EigenFilterType::Pointer eigenFilter = EigenFilterType::New();
    eigenFilter->SetDimension(Dimension);
    eigenFilter->SetInput(strainTensor);
    eigenFilter->OrderEigenValuesBy( EigenFilterType::FunctorType::OrderByValue );
    std::cout << "Performing eigenanalysis..." << std::endl;
    std::cout << "OrderByValue: lambda_1 < lambda_2 < lambda_3 " << std::endl;
    eigenFilter->Update();
    std::cout << "Done" << std::endl;
    
    //Once the eigenvalues are delete the strain tensor pointer to deallocate memory
    strainTensor = NULL;
    
    //Extract each eigen value of the Strain Matrix	and save it as a new image -----------------------------------
    typedef itk::ImageAdaptor< EigenValueImageType, EigenValueAccessor< EigenValueArrayType > > 	ImageAdaptorType;
    typedef itk::Image< PixelType, Dimension >
      EachEigenValueImageType;
    typedef itk::CastImageFilter< ImageAdaptorType, EachEigenValueImageType >
      CastImageFilterType;
    typedef itk::Image< PixelType, Dimension >
      OutputImageType;
    typedef itk::ImageFileWriter< OutputImageType >
      WriterType;
    
    
    std::map<int,std::string> outputFileNames;
    outputFileNames[0]=outputLambda1;
    outputFileNames[1]=outputLambda2;
    outputFileNames[2]=outputLambda3;
    
    for ( unsigned int k = 0; k < Dimension; k++ )
    {
        // First eigenvalue
        ImageAdaptorType::Pointer eigenAdaptor = ImageAdaptorType::New();
        EigenValueAccessor< EigenValueArrayType > accessor;
        accessor.SetEigenIdx( k );
        eigenAdaptor->SetImage( eigenFilter->GetOutput() );
        eigenAdaptor->SetPixelAccessor( accessor );
        
        CastImageFilterType::Pointer caster = CastImageFilterType::New();
        caster->SetInput( eigenAdaptor );
        caster->Update();
        
        WriterType::Pointer   writer = WriterType::New();
        writer->SetInput( caster->GetOutput() );
        //std::string eigenName = outputPrefix + "eigenValue" + std::to_string(k+1) + ".nrrd";
        if(outputFileNames[k].length()>0)
        {
            writer->SetFileName( outputFileNames[k] );
            writer->Update();
        }
        
    }

    return cip::EXITSUCCESS;
}
