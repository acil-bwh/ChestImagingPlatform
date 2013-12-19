#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMedianImageFilter.h"
#include "itkCIPPartialLungLabelMapImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "cipConventions.h"
//#include "GeneratePartialLungLabelMapCLP.h"

typedef itk::Image< unsigned short, 3 >                           UShortImageType;
typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< UShortImageType >                   UShortReaderType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< UShortImageType >                   UShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::ImageRegionIteratorWithIndex< UShortImageType >      UShortIteratorType;
typedef itk::PartialLungLabelMapImageFilter< ShortImageType >     PartialLungType;
typedef itk::GDCMImageIO                                          ImageIOType;
typedef itk::GDCMSeriesFileNames                                  NamesGeneratorType;
typedef itk::ImageSeriesReader< ShortImageType >                  SeriesReaderType;

void LowerClipImage( ShortImageType::Pointer, short, short );
void UpperClipImage( ShortImageType::Pointer, short, short );
ShortImageType::Pointer ReadCTFromDirectory( char* );
ShortImageType::Pointer ReadCTFromFile( char*  );

void usage()
{
  std::cerr << "\n";
  std::cerr << "Usage: GeneratePartialLungLabelMap <options> where <options> is one or more " << std::endl;
  std::cerr << "of the following:\n\n";
  std::cerr << "   <-h>     Display (this) usage information\n";
  std::cerr << "   <-ii>    Input CT image file name\n";
  std::cerr << "   <-dir>   Input CT directory\n";
  std::cerr << "   <-o>     Output image file name\n";
  std::cerr << "   <-lcv>   Lower clip value applied to input image before segmentation. This flag\n";
  std::cerr << "            should be followed by two values: the first value is the clip value and\n";
  std::cerr << "            the second value is the replacement value (i.e., everything below the clip\n";
  std::cerr << "            value will be assigned the replacement value)\n";
  std::cerr << "   <-ucv>   Upper clip value applied to input image before segmentation. This flag\n";
  std::cerr << "            should be followed by two values: the first value is the clip value and\n";
  std::cerr << "            the second value is the replacement value (i.e., everything above the clip\n";
  std::cerr << "            value will be assigned the replacement value)\n";
  std::cerr << "   <-cr>    The radius used for morphological closing in physical units (mm). The structuring\n"; 
  std::cerr << "            element is created so that the number of voxels in each direction covers no less\n";
  std::cerr << "            than the specified amount\n";
  std::cerr << "   <-agg>   Set to 1 for aggressive lung splitting.  Set to 0 (default) otherwise\n";
  std::cerr << "   <-lsr>   Radius used to split the left and right lungs (3 by default)\n";
  std::cerr << "   <-ir>    Max airway volume increase rate (default is 2.0). This is passed to the\n";
  std::cerr << "            partial lung label map filter. Decrease this value if you see leakage\n";  
  std::cerr << "   <-min>   Minimum airway volume \n";
  std::cerr << "   <-hf>    Set to 1 if the scan is head first (default) and 0 if feet first\n";
  std::cerr << "   <-hm>    Helper mask file name. This mask is a simple mask (typically binary with 1 as foreground)\n";
  std::cerr << "            such that the lung region is well thresholded and the lungs are separated. The\n";
  std::cerr << "            airways are assumed to be foregournd\n";


  exit(1);
}


int main( int argc, char *argv[] )
{
  typedef itk::MedianImageFilter< ShortImageType, ShortImageType > MedianType;

  bool ok;

  char*    outputLungMaskFileName        = new char[512];  strcpy( outputLungMaskFileName, "q" );
  char*    helperMaskFileName            = new char[512];  strcpy( helperMaskFileName, "q" );
  char*    ctFileName                    = new char[512];  strcpy( ctFileName, "q" );
  char*    ctDir                         = new char[512];  strcpy( ctDir, "q" );
  short    lowerClipValue                = -1024;
  short    lowerReplacementValue         = -1024;
  short    upperClipValue                = 1024;
  short    upperReplacementValue         = 1024;
  double   closingRadius                 = 5.0;
  int      aggressiveLungSplitting       = 0;
  int      lungSplitRadius               = 3;
  int      headFirst                     = 1;
  double   airwayVolumeIncreaseRate      = 2.0;
  double   minAirwayVolume               = 0.0;
  double   maxAirwayVolume               = 50.0;

  while ( argc > 1 )
    {
    ok = false;

    if ((ok == false) && (strcmp(argv[1], "-h") == 0))
      {
      argc--; argv++;
      ok = true;
      usage();      
      }

    if ((ok == false) && (strcmp(argv[1], "-hm") == 0))
      {
      argc--; argv++;
      ok = true;

      helperMaskFileName = argv[1];

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-hf") == 0))
      {
      argc--; argv++;
      ok = true;

      headFirst = atoi( argv[1] );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-min") == 0))
      {
      argc--; argv++;
      ok = true;

      minAirwayVolume = static_cast< double >( atof( argv[1] ) );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-max") == 0))
      {
      argc--; argv++;
      ok = true;

      maxAirwayVolume = static_cast< double >( atof( argv[1] ) );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-agg") == 0))
      {
      argc--; argv++;
      ok = true;

      aggressiveLungSplitting = atoi( argv[1] );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-ir") == 0))
      {
      argc--; argv++;
      ok = true;

      airwayVolumeIncreaseRate = static_cast< double >( atof( argv[1] ) );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-lsr") == 0))
      {
      argc--; argv++;
      ok = true;

      lungSplitRadius = atoi( argv[1] );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-dir") == 0))
      {
      argc--; argv++;
      ok = true;

      ctDir = argv[1];

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-cr") == 0))
      {
      argc--; argv++;
      ok = true;

      closingRadius = static_cast< double >( atof( argv[1] ) );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-ucv") == 0))
      {
      argc--; argv++;
      ok = true;

      upperClipValue        = static_cast< short >( atoi( argv[1] ) );
      argc--; argv++;
      upperReplacementValue = static_cast< short >( atoi( argv[1] ) );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-lcv") == 0))
      {
      argc--; argv++;
      ok = true;

      lowerClipValue        = static_cast< short >( atoi( argv[1] ) );
      argc--; argv++;
      lowerReplacementValue = static_cast< short >( atoi( argv[1] ) );

      argc--; argv++;
      }

    if ((ok == false) && (strcmp(argv[1], "-ii") == 0))
      {
      argc--; argv++;
      ok = true;

      ctFileName = argv[1];

      argc--; argv++;
      }


    if ((ok == false) && (strcmp(argv[1], "-o") == 0))
      {
      argc--; argv++;
      ok = true;

      outputLungMaskFileName = argv[1];

      argc--; argv++;
      }

    }

  //
  // Read the CT image
  //
  ShortImageType::Pointer ctImage = ShortImageType::New();

  if ( strcmp( ctDir, "q") != 0 )
    {
    std::cout << "Reading CT from directory..." << std::endl;
    ctImage = ReadCTFromDirectory( ctDir );
    }
  else if ( strcmp( ctFileName, "q") != 0 )
    {
    std::cout << "Reading CT from file..." << std::endl;
    ctImage = ReadCTFromFile( ctFileName );
    }
  else
    {
    std::cerr << "ERROR: No CT image specified" << std::endl;
    
    return 0;
    }

  ShortImageType::SpacingType spacing = ctImage->GetSpacing();
  
  unsigned long closingNeighborhood[3];
    closingNeighborhood[0] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[0] ) );
    closingNeighborhood[1] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[1] ) );
    closingNeighborhood[2] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[2] ) );

  closingNeighborhood[0] = closingNeighborhood[0]>0 ? closingNeighborhood[0] : 1;
  closingNeighborhood[1] = closingNeighborhood[1]>0 ? closingNeighborhood[1] : 1;
  closingNeighborhood[2] = closingNeighborhood[2]>0 ? closingNeighborhood[2] : 1;

  std::cout << "Clipping low CT image values..." << std::endl;
  LowerClipImage( ctImage, lowerClipValue, lowerReplacementValue );

  std::cout << "Clipping low CT image values (enforcing -1024 lower bound)..." << std::endl;
  LowerClipImage( ctImage, -1024, -1024 );

  std::cout << "Clipping upper CT image values..." << std::endl;
  UpperClipImage( ctImage, upperClipValue, upperReplacementValue );

  {
  ShortImageType::SizeType medianRadius;
    medianRadius[0] = 1;
    medianRadius[1] = 1;
    medianRadius[2] = 1;

  std::cout << "Executing median filter..." << std::endl;
  MedianType::Pointer median = MedianType::New();
    median->SetInput( ctImage );
    median->SetRadius( medianRadius );
    median->Update();

  ShortIteratorType gIt( ctImage, ctImage->GetBufferedRegion() );
  ShortIteratorType mIt( median->GetOutput(), median->GetOutput()->GetBufferedRegion() );

  gIt.GoToBegin();
  mIt.GoToBegin();
  while ( !gIt.IsAtEnd() )
    {
    gIt.Set( mIt.Get() );

    ++gIt;
    ++mIt;
    }
  }

  std::cout << "Executing partial lung filter..." << std::endl;
  PartialLungType::Pointer partialLungFilter = PartialLungType::New();
    partialLungFilter->SetInput( ctImage );
  if ( aggressiveLungSplitting == 1 )
    {
    partialLungFilter->SetAggressiveLeftRightSplitter( true );
    }
  if ( headFirst == 1 )
    {
    partialLungFilter->SetHeadFirst( true );
    }
  else
    {
    partialLungFilter->SetHeadFirst( false );
    }
    partialLungFilter->SetMaxAirwayVolumeIncreaseRate( airwayVolumeIncreaseRate );
    partialLungFilter->SetLeftRightLungSplitRadius( lungSplitRadius );
    partialLungFilter->SetMinAirwayVolume( minAirwayVolume );
    partialLungFilter->SetMaxAirwayVolume( maxAirwayVolume );
    partialLungFilter->SetClosingNeighborhood( closingNeighborhood );
  //
  // Read the helper mask if specified
  //
  if ( strcmp( helperMaskFileName, "q") != 0 )
    {
    std::cout << "Reading helper mask..." << std::endl;
    UShortReaderType::Pointer helperReader = UShortReaderType::New();
      helperReader->SetFileName( helperMaskFileName );
    try
      {
      helperReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading helper mask:";
      std::cerr << excp << std::endl;
      }
    partialLungFilter->SetHelperMask( helperReader->GetOutput() );
    }
    partialLungFilter->Update();

  // Before writing, label by thirds. Here we are only concerned about upper, middle, 
  // and lower, regardless of left right.
  cip::ChestConventions conventions;
  unsigned int totVoxels = 0;

  UShortIteratorType lIt( partialLungFilter->GetOutput(), partialLungFilter->GetOutput()->GetBufferedRegion() );

  lIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
      if ( lIt.Get() != 0 )
	{
	  if ( conventions.GetChestRegionFromValue( lIt.Get() ) > 0 )
	    {
	      totVoxels++;
	    }
	}

      ++lIt;
    }

  unsigned int inc = 0;

  lIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
      if ( lIt.Get() != 0 )
	{
	  unsigned char cipRegion = conventions.GetChestRegionFromValue( lIt.Get() );
	  unsigned char cipType   = conventions.GetChestTypeFromValue( lIt.Get() );
	  if ( cipRegion > 0 )
	    {
	      inc++;

	      if ( double(inc)/double(totVoxels) < 1.0/3.0 )
		{
		  lIt.Set( conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::LOWERTHIRD), cipType ) );
		}
	      else if ( double(inc)/double(totVoxels) < 2.0/3.0 )
		{
		  lIt.Set( conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::MIDDLETHIRD), cipType ) );
		}
	      else 
		{
		  lIt.Set( conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::UPPERTHIRD), cipType ) );
		}
	    }
	}      

      ++lIt;
    }

  std::cout << "Writing lung mask image..." << std::endl;
  UShortWriterType::Pointer maskWriter = UShortWriterType::New(); 
    maskWriter->SetInput( partialLungFilter->GetOutput() );
    maskWriter->SetFileName( outputLungMaskFileName );
    maskWriter->UseCompressionOn();
  try
    {
    maskWriter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while writing lung mask:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return 0;
}


void LowerClipImage( ShortImageType::Pointer image, short clipValue, short replacementValue )
{
  ShortIteratorType iIt( image, image->GetBufferedRegion() );

  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() < clipValue )
      {
      iIt.Set( replacementValue );
      }

    ++iIt;
    }
}


void UpperClipImage( ShortImageType::Pointer image, short clipValue, short replacementValue )
{
  ShortIteratorType iIt( image, image->GetBufferedRegion() );

  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() > clipValue )
      {
      iIt.Set( replacementValue );
      }

    ++iIt;
    }
}


ShortImageType::Pointer ReadCTFromDirectory( char* ctDir )
{
  ImageIOType::Pointer gdcmIO = ImageIOType::New();

  std::cout << "---Getting file names..." << std::endl;
  NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( ctDir );

  const SeriesReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();

  std::cout << "---Reading DICOM image..." << std::endl;
  SeriesReaderType::Pointer dicomReader = SeriesReaderType::New();
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
    }  

  return dicomReader->GetOutput();
}


ShortImageType::Pointer ReadCTFromFile( char* fileName )
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
    }

  return reader->GetOutput();
}
