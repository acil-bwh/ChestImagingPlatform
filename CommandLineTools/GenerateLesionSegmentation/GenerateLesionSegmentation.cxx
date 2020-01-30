
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkEventObject.h"
#include "itkOrientImageFilter.h"
#include "itkFixedArray.h"
#include "itkPasteImageFilter.h"
#include "itkResampleImageFilter.h"
// This needs to come after the other includes to prevent the global definitions
// of PixelType to be shadowed by other declarations.
#include "itkLesionSegmentationImageFilter8.h"
#include "itkNearestNeighborExtrapolateImageFunction.h"

#include <GenerateLesionSegmentationCLP.h>

typedef short PixelType;
const static unsigned int ImageDimension = 3;
typedef itk::Image< PixelType, ImageDimension > InputImageType;
typedef itk::Image< float, ImageDimension > RealImageType;

typedef itk::LandmarkSpatialObject< 3 >    SeedSpatialObjectType;
typedef SeedSpatialObjectType::LandmarkPointType   LandmarkPointType;
typedef SeedSpatialObjectType::LandmarkPointListType   LandmarkPointListType;
typedef InputImageType::IndexType IndexType;

typedef itk::ImageFileReader< InputImageType > InputReaderType;
typedef itk::ImageFileWriter< RealImageType > OutputWriterType;
typedef itk::LesionSegmentationImageFilter8< InputImageType, RealImageType > SegmentationFilterType;

// For resampling the output
typedef itk::PasteImageFilter<RealImageType,RealImageType> PasteImageFilterType;
typedef itk::ResampleImageFilter<RealImageType,RealImageType> ResampleImageFilterType;
typedef itk::IdentityTransform<double, ImageDimension> TransformType;
typedef itk::NearestNeighborExtrapolateImageFunction<RealImageType,double> ExtrapolatorType;

LandmarkPointListType GetSeeds(std::vector<std::vector<float> > seeds,InputImageType *image)
{
  
  const unsigned int nb_of_markers = seeds.size();
  LandmarkPointListType seedsList(seeds.size());
  
  for (unsigned int i = 0; i < seeds.size(); i++)
  {
    seeds[i][0];
    InputImageType::PointType pointSeed;
    pointSeed[0] = seeds[i][0];
    pointSeed[1] = seeds[i][1];
    pointSeed[2] = seeds[i][2];
    IndexType indexSeed;
    image->TransformPhysicalPointToIndex(pointSeed, indexSeed);
    if (!image->GetBufferedRegion().IsInside(indexSeed))
    {
      std::cerr << "Seed with pixel units of index: " <<
      indexSeed << " does not lie within the image. The images extents are"
      << image->GetBufferedRegion() << std::endl;
      
    } else {
      seedsList[i].SetPositionInWorldSpace(pointSeed);
      
    }
  }
  return seedsList;
  
}


// --------------------------------------------------------------------------
int main( int argc, char * argv[] )
{

  PARSE_ARGS;


  // Read the volume
  InputReaderType::Pointer reader = InputReaderType::New();
  InputImageType::Pointer image;

  reader->SetFileName(inputImage);
  
  try {
    reader->Update();
  } catch (itk::ExceptionObject &excp) {
    std::cerr << "Exception caught while reading image:";
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  image=reader->GetOutput();

  //To make sure the tumor polydata aligns with the image volume during
  //vtk rendering in ViewImageAndSegmentationSurface(),
  //reorient image so that the direction matrix is an identity matrix.
  typedef itk::ImageFileReader< InputImageType > InputReaderType;
  itk::OrientImageFilter<InputImageType,InputImageType>::Pointer orienter =
  itk::OrientImageFilter<InputImageType,InputImageType>::New();
  orienter->UseImageDirectionOn();
  InputImageType::DirectionType direction;
  direction.SetIdentity();
  orienter->SetDesiredCoordinateDirection (direction);
  orienter->SetInput(image);
  orienter->Update();
  image = orienter->GetOutput();


  // Compute the ROI region
  InputImageType::RegionType region = image->GetLargestPossibleRegion();

  if (roi.size()!=6) {
    std::cerr <<"ROI should have six elements"<<std::endl;
    return EXIT_FAILURE;
  }

  if (roi[0] == 0 && roi[1] == 0 && roi[2]==0 &&
      roi[3] == 0 && roi[4] ==0 && roi[5]==0){
    //Compute ROI from seed and maximum Radius
    LandmarkPointListType seeds=GetSeeds(seedsFiducials,image);
    seeds[0];
    for (int i=0; i < 3; i++)
    {
      roi[2*i]= seeds[0].GetPositionInWorldSpace()[i] - maximumRadius;
      roi[2*i+1]= seeds[0].GetPositionInWorldSpace()[i] + maximumRadius;
    }
  }
  
  // convert bounds into region indices
  InputImageType::PointType p1, p2;
  InputImageType::IndexType pi1, pi2;
  InputImageType::IndexType startIndex;
  for (unsigned int i = 0; i < ImageDimension; i++)
    {
    p1[i] = roi[2*i];
    p2[i] = roi[2*i+1];
    }

  image->TransformPhysicalPointToIndex(p1, pi1);
  image->TransformPhysicalPointToIndex(p2, pi2);

  InputImageType::SizeType roiSize;
  for (unsigned int i = 0; i < ImageDimension; i++)
    {
    roiSize[i] = (unsigned int) fabs(double(pi2[i] - pi1[i]));
    startIndex[i] = (pi1[i]<pi2[i])?pi1[i]:pi2[i];
    }
  InputImageType::RegionType roiRegion( startIndex, roiSize );


  std::cout << "ROI region is " << roiRegion << std::endl;
  if (!roiRegion.Crop(image->GetBufferedRegion()))
    {
    std::cerr << "ROI region has no overlap with the image region of"
              << image->GetBufferedRegion() << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "ROI region is " << roiRegion <<  " : covers voxels = "
    << roiRegion.GetNumberOfPixels() << " : " <<
    image->GetSpacing()[0]*image->GetSpacing()[1]*image->GetSpacing()[2]*roiRegion.GetNumberOfPixels()
   << " mm^3" << std::endl;


  // Write ROI if requested
  if (outputROI)
    {
    typedef itk::RegionOfInterestImageFilter<
      InputImageType, InputImageType > ROIFilterType;

    ROIFilterType::Pointer roiFilter = ROIFilterType::New();
    roiFilter->SetRegionOfInterest( roiRegion );

    typedef itk::ImageFileWriter< InputImageType > ROIWriterType;
    ROIWriterType::Pointer roiWriter = ROIWriterType::New();
    roiWriter->SetFileName( roiImage );
    roiFilter->SetInput( image );
    roiWriter->SetInput( roiFilter->GetOutput() );

    std::cout << "Writing the ROI image as: " <<
      roiImage << std::endl;
    try
      {
      roiWriter->Update();
      }
    catch( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught !" << err << std::endl;
      return EXIT_FAILURE;
      }
    }


  // Run the segmentation filter. Clock ticking...

  std::cout << "\n Running the segmentation filter." << std::endl;
  SegmentationFilterType::Pointer seg = SegmentationFilterType::New();
  seg->SetInput(image);
  seg->SetSeeds(GetSeeds(seedsFiducials,image));
  seg->SetRegionOfInterest(roiRegion);

  //itk::PluginFilterWatcher watchSeg(seg, "Lesion segmentation", CLPProcessInformation);
  // Progress reporting
  //typedef itk::LesionSegmentationCommandLineProgressReporter ProgressReporterType;
  //ProgressReporterType::Pointer progressCommand =
  //  ProgressReporterType::New();
  //seg->AddObserver( itk::ProgressEvent(), progressCommand );
  
  if (sigma[0]>0 && sigma[1] >0 && sigma[2]>0)
    {
    itk::FixedArray< double, 3 > sigVector;
    sigVector[0]=sigma[0];
    sigVector[1]=sigma[1];
    sigVector[2]=sigma[2];
    seg->SetSigma(sigVector);
    }
  seg->SetSigmoidBeta(partSolid ? -500 : -200 );
  seg->Update();


  if (! outputLevelSet.empty())
    {
    std::cout << "Writing the output segmented level set."
      << outputLevelSet <<
      ". The segmentation is an isosurface of this image at a value of -0.5"
      << std::endl;
    OutputWriterType::Pointer writer = OutputWriterType::New();
    writer->SetFileName(outputLevelSet);

    // use full image size for output, if requested
    if (fullSizeOutput) {
      RealImageType::Pointer fullOutputImage = RealImageType::New();
      fullOutputImage->SetRegions(image->GetLargestPossibleRegion());
      fullOutputImage->Allocate();
      fullOutputImage->FillBuffer(-4.0f);
      fullOutputImage->CopyInformation(image);

      // Resample ROI to original size
      ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
      resample->SetInput(seg->GetOutput());
      resample->SetSize(roiSize);
      resample->SetOutputSpacing(image->GetSpacing());
      resample->SetOutputOrigin(seg->GetOutput()->GetOrigin());
      resample->SetTransform(TransformType::New());
      resample->UpdateLargestPossibleRegion();
      resample->SetExtrapolator(ExtrapolatorType::New());
      resample->Update();

      // Paste the image back into the full image
      PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();
      pasteFilter->SetSourceImage ( resample->GetOutput() );
      pasteFilter->SetSourceRegion ( resample->GetOutput()->GetLargestPossibleRegion() );
      pasteFilter->SetDestinationImage ( fullOutputImage );
      pasteFilter->SetDestinationIndex ( startIndex );

      pasteFilter->Update();

      // Make sure our sizes / bounds are set correctly
      pasteFilter->GetOutput()->CopyInformation ( image );
      
      writer->SetInput(pasteFilter->GetOutput());
    } else {    
      writer->SetInput(seg->GetOutput());
    }
    writer->Update();
    }

  return EXIT_SUCCESS;
}
