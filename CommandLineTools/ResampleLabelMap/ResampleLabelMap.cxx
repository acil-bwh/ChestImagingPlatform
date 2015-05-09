#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkMatrix.h"
#include "itkTransformFileReader.h"
#include "itkResampleImageFilter.h"
#include "ResampleLabelMapCLP.h"
#include <itkCompositeTransform.h>

namespace
{
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
    
    template <unsigned int TDimension>
    int DoIT(int argc, char * argv[])
    {
        PARSE_ARGS;
        
        //dimension specific typedefs
        typedef itk::Image< unsigned short, TDimension >                              LabelMapType;
        typedef itk::NearestNeighborInterpolateImageFunction< LabelMapType, double >  InterpolatorType;
        typedef itk::ResampleImageFilter< LabelMapType,LabelMapType >                 ResampleType;
        typedef itk::AffineTransform< double, TDimension >                            TransformType;
        typedef itk::CompositeTransform< double, TDimension >                         CompositeTransformType;
        typedef itk::ImageFileReader< LabelMapType >                                  LabelMapReaderType;
        typedef itk::ImageFileWriter< LabelMapType >                                  LabelMapWriterType;
        
        // Read the destination image information for spacing, origin, and
        // size information (neede for the resampling process).
        typename LabelMapType::SpacingType spacing;
        typename LabelMapType::SizeType    size;
        typename LabelMapType::PointType   origin;
        
        std::cout << "Reading destination information..." << std::endl;
        typename LabelMapReaderType::Pointer destinationReader = LabelMapReaderType::New();
        destinationReader->SetFileName( destinationFileName.c_str() );
        try
        {
            destinationReader->Update();
        }
        catch ( itk::ExceptionObject &excp )
        {
            std::cerr << "Exception caught reading label map:";
            std::cerr << excp << std::endl;
            
            return cip::LABELMAPREADFAILURE;
        }
        
        spacing = destinationReader->GetOutput()->GetSpacing();
        size    = destinationReader->GetOutput()->GetLargestPossibleRegion().GetSize();
        origin  = destinationReader->GetOutput()->GetOrigin();
        
        // Read the label map image
        std::cout << "Reading label map image..." << std::endl;
        typename LabelMapReaderType::Pointer labelMapReader = LabelMapReaderType::New();
        labelMapReader->SetFileName( labelMapFileName.c_str() );
        try
        {
            labelMapReader->Update();
        }
        catch ( itk::ExceptionObject &excp )
        {
            std::cerr << "Exception caught reading label map:";
            std::cerr << excp << std::endl;
            
            return cip::LABELMAPREADFAILURE;
        }
        
        // Read the transform (last transform applied first)
        typename CompositeTransformType::Pointer transform = CompositeTransformType::New();
        
        for ( unsigned int i=0; i<transformFileName.size(); i++ )
        {
            // Invert the transformation if specified by command like argument.
            // Only inverting the first transformation
            std::cout << "Adding transform: " << i << std::endl;
            typename TransformType::Pointer transformTemp = TransformType::New();
            transformTemp = GetTransformFromFile<TDimension>((transformFileName[i]).c_str() );
            if( i==0 && isInvertTransformation == true )
            {
                transformTemp->GetInverse( transformTemp );
            }
            transform->AddTransform(transformTemp);
        }
        transform->SetAllTransformsToOptimizeOn();
        
        // Resample the label map
        typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
        
        std::cout << "Resampling..." << std::endl;
        typename ResampleType::Pointer resampler = ResampleType::New();
        resampler->SetTransform( transform );
        resampler->SetInterpolator( interpolator );
        resampler->SetInput( labelMapReader->GetOutput() );
        resampler->SetSize( size );
        resampler->SetOutputSpacing( spacing );
        resampler->SetOutputOrigin( origin );
        resampler->SetOutputDirection( destinationReader->GetOutput()->GetDirection() );
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
        
        // Write the resampled label map to file
        std::cout << "Writing resampled label map..." << std::endl;
        typename LabelMapWriterType::Pointer writer = LabelMapWriterType::New();
        writer->SetFileName( resampledFileName.c_str());
        writer->UseCompressionOn();
        writer->SetInput( resampler->GetOutput() );
        try
        {
            writer->Update();
        }
        catch ( itk::ExceptionObject &excp )
        {
            std::cerr << "Exception caught writing label map:";
            std::cerr << excp << std::endl;
            
            return cip::LABELMAPWRITEFAILURE;
        }
        
        std::cout << "DONE." << std::endl;   
        return cip::EXITSUCCESS;
    }
    
} // end of anonymous namespace


int main( int argc, char *argv[] )
{
    PARSE_ARGS;
    
    switch(dimension)
    {
        case 2:
        {
            DoIT<2>( argc, argv );
            break;
        }
        case 3:
        {
            DoIT<3>( argc, argv );
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
