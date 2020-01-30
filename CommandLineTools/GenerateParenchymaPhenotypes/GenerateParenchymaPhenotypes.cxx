/** \file
 *  \ingroup commandLineTools
 *  \details This program is used to compute regional histograms and   typical
 * parenchyma phenotypes for emphysema assessment and other parenchymal
 * abnormalities.
 *
 *  $Date:  $
 *  $Revision:  $
 *  $Author:ç $
 *
 *  USAGE:
 *
 * ./GenerateParenchymaPhenotypes  [--op <string>] [--oh
 * <string>] [--max <integer>] [--min
 * <integer>] [-l <string>] [-p
 * <string>] -c <string> [--]
 * [--version] [-h]
 *
 * Where:
 *
 * --op <string>
 * Output phenotypes file name
 *
 * --oh <string>
 * Output histogram file name
 *
 * --max <integer>
 *  Value at high end of histogram. Default: 1024
 *
 * --min <integer>
 * Value at low end of histogram. Default: -1024
 *
 * -l <string>,  --ill <string>
 * Input lung lobe label map file name
 *
 * -p <string>,  --ipl <string>
 * Input partial lung label map file name
 *
 * -c <string>,  --ic <string>
 * (required)  Input CT file name
 *
 * --,  --ignore_rest
 * Ignores the rest of the labeled arguments following this flag.
 *
 * --version
 * Displays version information and exits.
 *
 * -h,  --help
 * Displays usage information and exits.
 */


#include <fstream>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "GenerateParenchymaPhenotypesCLP.h"

namespace
{
typedef itk::ImageRegionIterator< cip::CTType >       CTIteratorType;
typedef itk::ImageRegionIterator< cip::LabelMapType > LabelMapIteratorType;

struct PARENCHYMAPHENOTYPES
{
  int    countBelow950;
  int    countBelow925;
  int    countBelow910;
  int    countBelow905;
  int    countBelow900;
  int    countBelow875;
  int    countBelow856;
  int    countAbove0;  
  int    countAbove600;
  int    countAbove250;
  short  tenthPercentileHU;
  short  fifteenthPercentileHU;
  int    totalVoxels;
  double volume;
  double mass;
  double intensityMean;
  double intensitySTD;
  double skewness;
  double kurtosis;
  double mode;
  double median;
};

    
    void InitializeParenchymaPhenotypes( PARENCHYMAPHENOTYPES* phenotypes )
    {
        phenotypes->countBelow950 = 0;
        phenotypes->countBelow925 = 0;
        phenotypes->countBelow910 = 0;
        phenotypes->countBelow905 = 0;
        phenotypes->countBelow900 = 0;
        phenotypes->countBelow875 = 0;
        phenotypes->countBelow856 = 0;
        phenotypes->countAbove0   = 0;
        phenotypes->countAbove600 = 0;
        phenotypes->countAbove250 = 0;
        phenotypes->totalVoxels   = 0;
        phenotypes->volume        = 0.0;
        phenotypes->mass          = 0.0;
        phenotypes->intensityMean = 0.0;
    }
    
    unsigned int GetHistogramNumberOfCounts( std::map< short, unsigned int > histogram, short minBin, short maxBin )
    {
        unsigned int numCounts = 0;
        
        for ( short i = minBin; i<=maxBin; i++ )
        {
            numCounts += histogram[i];
        }
        
        return numCounts;
    }
    
    double GetHistogramSkewness( std::map< short, unsigned int > histogram, double mean, short minBin, short maxBin )
    {
        unsigned int numCounts = GetHistogramNumberOfCounts( histogram, minBin, maxBin );
        
        double numeratorSum   = 0.0;
        double denominatorSum = 0.0;
        
        for ( short i=minBin; i<=maxBin; i++ )
        {
            for ( unsigned int j=0; j<histogram[i]; j++ )
            {
                numeratorSum   += pow( (static_cast< double >( i ) - mean), 3 );
                denominatorSum += pow( (static_cast< double >( i ) - mean), 2 );
            }
        }
        
        double skewness = ((1.0/static_cast<double>(numCounts))*numeratorSum)/pow( (1.0/static_cast<double>(numCounts))*denominatorSum, 1.5 );
        
        return skewness;
    }
    
    double GetHistogramKurtosis( std::map< short, unsigned int > histogram, double mean, short minBin, short maxBin )
    {
        unsigned int numCounts = GetHistogramNumberOfCounts( histogram, minBin, maxBin );
        
        double numeratorSum   = 0.0;
        double denominatorSum = 0.0;
        
        for ( short i=minBin; i<=maxBin; i++ )
        {
            for ( unsigned int j=0; j<histogram[i]; j++ )
            {
                numeratorSum   += pow( (static_cast< double >(i) - mean), 4 );
                denominatorSum += pow( (static_cast< double >(i) - mean), 2 );
            }
        }
        
        double kurtosis = ((1.0/static_cast<double>(numCounts))*numeratorSum)/pow( (1.0/static_cast<double>(numCounts))*denominatorSum, 2 );
        
        return kurtosis;
    }

    
    double GetHistogramSTD( std::map< short, unsigned int > histogram, double mean, short minBin, short maxBin )
    {
        unsigned int counts = 0;
        double histSTD = 0.0;
        
        for ( short i=minBin; i<=maxBin; i++ )
        {
            counts += histogram[i];
            
            histSTD += static_cast< double >( histogram[i] )*std::pow( static_cast< double >(i) - mean, 2 );
        }
        
        histSTD = std::sqrt( histSTD/static_cast< double >( counts ) );
        
        return histSTD;
    }
    
    void ComputeParenchymaPhenotypesSubset( PARENCHYMAPHENOTYPES* phenotypes, std::map< short, unsigned int > histogram, double voxelVolume, short minBin, short maxBin)
    {
        unsigned int tenthPercentileCounter     = 0;
        unsigned int fifteenthPercentileCounter = 0;
        unsigned int medianCounter              = 0;
        unsigned int modeCounter                = 0;
        
        for ( int i=minBin; i<=maxBin; i++ )
        {
            phenotypes->intensityMean += static_cast< double >(i)*static_cast< double >( histogram[i] )/static_cast< double >( phenotypes->totalVoxels );
            phenotypes->mass += static_cast< double >( histogram[i] )*(voxelVolume/1000.0)*(static_cast< double >(i)+1000.0)/1000.0;
            
            if ( histogram[i] > modeCounter )
            {
                modeCounter = histogram[i];
                phenotypes->mode = i;
            }
            
            tenthPercentileCounter += histogram[i];
            if ( static_cast< double >( tenthPercentileCounter )/static_cast< double >( phenotypes->totalVoxels ) <= 0.1 )
            {
                phenotypes->tenthPercentileHU = i;
            }
            
            fifteenthPercentileCounter += histogram[i];
            if ( static_cast< double >( fifteenthPercentileCounter )/static_cast< double >( phenotypes->totalVoxels ) <= 0.15 )
            {
                phenotypes->fifteenthPercentileHU = i;
            }
            
            medianCounter += histogram[i];
            if ( static_cast< double >( medianCounter )/static_cast< double >( phenotypes->totalVoxels ) <= 0.5 )
            {
                phenotypes->median = i;
            }
            
            if ( i<-950 )
            {
                phenotypes->countBelow950 += histogram[i];
            }
            if ( i<-925 )
            {
                phenotypes->countBelow925 += histogram[i];
            }
            if ( i<-910 )
            {
                phenotypes->countBelow910 += histogram[i];
            }
            if ( i<-905 )
            {
                phenotypes->countBelow905 += histogram[i];
            }
            if ( i<-900 )
            {
                phenotypes->countBelow900 += histogram[i];
            }
            if ( i<-875 )
            {
                phenotypes->countBelow875 += histogram[i];
            }
            if ( i<-856 )
            {
                phenotypes->countBelow856 += histogram[i];
            }
            if ( i>0 )
            {
                phenotypes->countAbove0 += histogram[i];
            }
            if ( i>-600 )
            {
                phenotypes->countAbove600 += histogram[i];
            }
            if ( i>-250 )
            {
                phenotypes->countAbove250 += histogram[i];
            }
        }
        
        phenotypes->skewness      = GetHistogramSkewness( histogram, phenotypes->intensityMean, minBin, maxBin );
        phenotypes->kurtosis      = GetHistogramKurtosis( histogram, phenotypes->intensityMean, minBin, maxBin );
        phenotypes->intensitySTD  = GetHistogramSTD( histogram, phenotypes->intensityMean, minBin, maxBin );
    }
    
    
       
    

    
    
    double GetHistogramMean( std::map< short, unsigned int > histogram, short minBin, short maxBin )
    {
        unsigned int numCounts = GetHistogramNumberOfCounts( histogram, minBin, maxBin );
        
        double mean = 0.0;
        
        for ( short i = minBin; i<=maxBin; i++ )
        {
            mean += (static_cast<double>(histogram[i])/static_cast< double >(numCounts))*static_cast< double >(i);
        }
        
        return mean;
    }
    
    


    
    
    void UpdateAllHistogramsAndPhenotypes( cip::CTType::Pointer ctImage, cip::LabelMapType::Pointer labelMap,
                                          PARENCHYMAPHENOTYPES* wholeLungPhenotypes, PARENCHYMAPHENOTYPES* leftLungPhenotypes, PARENCHYMAPHENOTYPES* rightLungPhenotypes, PARENCHYMAPHENOTYPES* lulLungPhenotypes,
                                          PARENCHYMAPHENOTYPES* lllLungPhenotypes, PARENCHYMAPHENOTYPES* rulLungPhenotypes, PARENCHYMAPHENOTYPES* rmlLungPhenotypes, PARENCHYMAPHENOTYPES* rllLungPhenotypes,
                                          PARENCHYMAPHENOTYPES* lutLungPhenotypes, PARENCHYMAPHENOTYPES* lmtLungPhenotypes, PARENCHYMAPHENOTYPES* lltLungPhenotypes, PARENCHYMAPHENOTYPES* rutLungPhenotypes,
                                          PARENCHYMAPHENOTYPES* rmtLungPhenotypes, PARENCHYMAPHENOTYPES* rltLungPhenotypes, PARENCHYMAPHENOTYPES* utLungPhenotypes, PARENCHYMAPHENOTYPES* mtLungPhenotypes,
                                          PARENCHYMAPHENOTYPES* ltLungPhenotypes,
                                          std::map< short, unsigned int >* wholeLungHistogram, std::map< short, unsigned int >* leftLungHistogram, std::map< short, unsigned int >* rightLungHistogram,
                                          std::map< short, unsigned int >* lulLungHistogram, std::map< short, unsigned int >* lllLungHistogram, std::map< short, unsigned int >* rulLungHistogram,
                                          std::map< short, unsigned int >* rmlLungHistogram, std::map< short, unsigned int >* rllLungHistogram, std::map< short, unsigned int >* lutLungHistogram,
                                          std::map< short, unsigned int >* lmtLungHistogram, std::map< short, unsigned int >* lltLungHistogram, std::map< short, unsigned int >* rutLungHistogram,
                                          std::map< short, unsigned int >* rmtLungHistogram, std::map< short, unsigned int >* rltLungHistogram, std::map< short, unsigned int >* utLungHistogram,
                                          std::map< short, unsigned int >* mtLungHistogram, std::map< short, unsigned int >* ltLungHistogram,
                                          double voxelVolume, short minBin, short maxBin )
    {
        cip::ChestConventions conventions;
        
        unsigned char lungRegion;
        
        CTIteratorType cIt( ctImage, ctImage->GetBufferedRegion() );
        LabelMapIteratorType lIt( labelMap, labelMap->GetBufferedRegion() );
        
        cIt.GoToBegin();
        lIt.GoToBegin();
        while ( !cIt.IsAtEnd() )
        {
            if (cIt.Get() < minBin || cIt.Get() > maxBin)
            {
                ++cIt;
                ++lIt;
                continue;
            }
            
            if ( lIt.Get() != 0 )
            {
                lungRegion = conventions.GetChestRegionFromValue( lIt.Get() );
                
                if ( lungRegion != static_cast< unsigned char >( cip::UNDEFINEDREGION ) )
                {
                    (*wholeLungHistogram)[cIt.Get()]++;
                    wholeLungPhenotypes->totalVoxels++;
                    wholeLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTLUNG ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTSUPERIORLOBE ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTINFERIORLOBE ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTLOWERTHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTMIDDLETHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTUPPERTHIRD ) )
                {
                    (*leftLungHistogram)[cIt.Get()]++;
                    leftLungPhenotypes->totalVoxels++;
                    leftLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTLUNG ) ||
                    lungRegion == static_cast< unsigned char >( cip::RIGHTSUPERIORLOBE ) ||
                    lungRegion == static_cast< unsigned char >( cip::RIGHTMIDDLELOBE ) ||
                    lungRegion == static_cast< unsigned char >( cip::RIGHTINFERIORLOBE ) ||
                    lungRegion == static_cast< unsigned char >( cip::RIGHTLOWERTHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::RIGHTMIDDLETHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::RIGHTUPPERTHIRD ) )
                {
                    (*rightLungHistogram)[cIt.Get()]++;
                    rightLungPhenotypes->totalVoxels++;
                    rightLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTSUPERIORLOBE ) )
                {
                    (*lulLungHistogram)[cIt.Get()]++;
                    lulLungPhenotypes->totalVoxels++;
                    lulLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTINFERIORLOBE ) )
                {
                    (*lllLungHistogram)[cIt.Get()]++;
                    lllLungPhenotypes->totalVoxels++;
                    lllLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTSUPERIORLOBE ) )
                {
                    (*rulLungHistogram)[cIt.Get()]++;
                    rulLungPhenotypes->totalVoxels++;
                    rulLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTMIDDLELOBE ) )
                {
                    (*rmlLungHistogram)[cIt.Get()]++;
                    rmlLungPhenotypes->totalVoxels++;
                    rmlLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTINFERIORLOBE ) )
                {
                    (*rllLungHistogram)[cIt.Get()]++;
                    rllLungPhenotypes->totalVoxels++;
                    rllLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTUPPERTHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTUPPERTHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::UPPERTHIRD ) )
                {
                    (*utLungHistogram)[cIt.Get()]++;
                    utLungPhenotypes->totalVoxels++;
                    utLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTMIDDLETHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTMIDDLETHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::MIDDLETHIRD ) )
                {
                    (*mtLungHistogram)[cIt.Get()]++;
                    mtLungPhenotypes->totalVoxels++;
                    mtLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTLOWERTHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::LEFTLOWERTHIRD ) ||
                    lungRegion == static_cast< unsigned char >( cip::LOWERTHIRD ))
                {
                    (*ltLungHistogram)[cIt.Get()]++;
                    ltLungPhenotypes->totalVoxels++;
                    ltLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTUPPERTHIRD ) )
                {
                    (*lutLungHistogram)[cIt.Get()]++;
                    lutLungPhenotypes->totalVoxels++;
                    lutLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTMIDDLETHIRD ) )
                {
                    (*lmtLungHistogram)[cIt.Get()]++;
                    lmtLungPhenotypes->totalVoxels++;
                    lmtLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTLOWERTHIRD ) )
                {
                    (*lltLungHistogram)[cIt.Get()]++;
                    lltLungPhenotypes->totalVoxels++;
                    lltLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTUPPERTHIRD ) )
                {
                    (*rutLungHistogram)[cIt.Get()]++;
                    rutLungPhenotypes->totalVoxels++;
                    rutLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTMIDDLETHIRD ) )
                {
                    (*rmtLungHistogram)[cIt.Get()]++;
                    rmtLungPhenotypes->totalVoxels++;
                    rmtLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTLOWERTHIRD ) )
                {
                    (*rltLungHistogram)[cIt.Get()]++;
                    rltLungPhenotypes->totalVoxels++;
                    rltLungPhenotypes->volume += voxelVolume;
                }
            }
            
            ++cIt;
            ++lIt;
        }
    }
	
	
    void UpdateLobeHistogramsAndPhenotypes( cip::CTType::Pointer ctImage, cip::LabelMapType::Pointer labelMap,
                                           PARENCHYMAPHENOTYPES* lulLungPhenotypes, PARENCHYMAPHENOTYPES* lllLungPhenotypes, PARENCHYMAPHENOTYPES* rulLungPhenotypes, PARENCHYMAPHENOTYPES* rmlLungPhenotypes, PARENCHYMAPHENOTYPES* rllLungPhenotypes,
                                           std::map< short, unsigned int >* lulLungHistogram, std::map< short, unsigned int >* lllLungHistogram, std::map< short, unsigned int >* rulLungHistogram,
                                           std::map< short, unsigned int >* rmlLungHistogram, std::map< short, unsigned int >* rllLungHistogram, double voxelVolume, short minBin, short maxBin )
    {
        cip::ChestConventions conventions;
        
        unsigned char lungRegion;
        
        CTIteratorType cIt( ctImage, ctImage->GetBufferedRegion() );
        LabelMapIteratorType lIt( labelMap, labelMap->GetBufferedRegion() );
		
        cIt.GoToBegin();
        lIt.GoToBegin();
        while ( !cIt.IsAtEnd() )
        {
            
            if (cIt.Get() < minBin || cIt.Get() > maxBin)
            {
                ++cIt;
                ++lIt;
                continue;
            }
            
            if ( lIt.Get() != 0 )
            {
                lungRegion = conventions.GetChestRegionFromValue( lIt.Get() );
                
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTSUPERIORLOBE ) )
                {
                    (*lulLungHistogram)[cIt.Get()]++;
                    lulLungPhenotypes->totalVoxels++;
                    lulLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::LEFTINFERIORLOBE ) )
                {
                    (*lllLungHistogram)[cIt.Get()]++;
                    lllLungPhenotypes->totalVoxels++;
                    lllLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTSUPERIORLOBE ) )
                {
                    (*rulLungHistogram)[cIt.Get()]++;
                    rulLungPhenotypes->totalVoxels++;
                    rulLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTMIDDLELOBE ) )
                {
                    (*rmlLungHistogram)[cIt.Get()]++;
                    rmlLungPhenotypes->totalVoxels++;
                    rmlLungPhenotypes->volume += voxelVolume;
                }
                if ( lungRegion == static_cast< unsigned char >( cip::RIGHTINFERIORLOBE ) )
                {
                    (*rllLungHistogram)[cIt.Get()]++;
                    rllLungPhenotypes->totalVoxels++;
                    rllLungPhenotypes->volume += voxelVolume;
                }
            }
            
            ++cIt;
            ++lIt;
        }
    }
    
} //end namespace

    /*

void InitializeParenchymaPhenotypes( PARENCHYMAPHENOTYPES* );
void ComputeParenchymaPhenotypesSubset( PARENCHYMAPHENOTYPES*, std::map< short, unsigned int >, double, short, short);
double GetHistogramKurtosis( std::map< short, unsigned int >, double, short, short );
double GetHistogramSkewness( std::map< short, unsigned int >, double, short, short );
unsigned int GetHistogramNumberOfCounts( std::map< short, unsigned int >, short, short );
double GetHistogramSTD( std::map< short, unsigned int >, double, short, short );
void UpdateAllHistogramsAndPhenotypes( cip::CTType::Pointer ctImage, cip::LabelMapType::Pointer labelMap,
                                    PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*,
                                    PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*,
                                    PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*,
                                    PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*, PARENCHYMAPHENOTYPES*,
                                    PARENCHYMAPHENOTYPES*,
                                    std::map< short, unsigned int >*, std::map< short, unsigned int >*, std::map< short, unsigned int >*,
                                    std::map< short, unsigned int >*, std::map< short, unsigned int >*, std::map< short, unsigned int >*,
                                    std::map< short, unsigned int >*, std::map< short, unsigned int >*, std::map< short, unsigned int >*,
                                    std::map< short, unsigned int >*, std::map< short, unsigned int >*, std::map< short, unsigned int >*,
                                    std::map< short, unsigned int >*, std::map< short, unsigned int >*, std::map< short, unsigned int >*,
                                    std::map< short, unsigned int >*, std::map< short, unsigned int >*, double, short minBin, short maxBin );
void UpdateLobeHistogramsAndPhenotypes( cip::CTType::Pointer ctImage, cip::LabelMapType::Pointer labelMap,
                                        PARENCHYMAPHENOTYPES* , PARENCHYMAPHENOTYPES* , PARENCHYMAPHENOTYPES* , PARENCHYMAPHENOTYPES* , PARENCHYMAPHENOTYPES* ,
                                        std::map< short, unsigned int >* , std::map< short, unsigned int >* , std::map< short, unsigned int >* ,
                                        std::map< short, unsigned int >* , std::map< short, unsigned int >* , double, short minBin, short maxBin);

*/

int main( int argc, char *argv[] )
{
    

    PARSE_ARGS;

    bool ok;
    
    short minBin = (short) minBinTemp;
    short maxBin = (short) maxBinTemp;

  //
  // Read the CT image
  //
  std::cout << "Reading CT image..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
  ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT image:";
    std::cerr << excp << std::endl;
    return cip::NRRDREADFAILURE;
    }

  cip::CTType::SpacingType spacing = ctReader->GetOutput()->GetSpacing();
  
  double voxelVolume = spacing[0]*spacing[1]*spacing[2];

  //
  // Read the partial lung label map image
  //
  cip::LabelMapReaderType::Pointer partialLungLabelMapReader = cip::LabelMapReaderType::New();
  if ( strcmp( partialLungLabelMapFileName.c_str(), "NA") != 0 )
    {
    std::cout << "Reading label map image..." << std::endl;
      partialLungLabelMapReader->SetFileName( partialLungLabelMapFileName );
    try
      {
      partialLungLabelMapReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading label map image:";
      std::cerr << excp << std::endl;
      return cip::LABELMAPREADFAILURE;
      }
    }

  //
  // Read the lung lobe label map image
  //
  cip::LabelMapReaderType::Pointer lungLobeLabelMapReader = cip::LabelMapReaderType::New();
  if ( strcmp( lungLobeLabelMapFileName.c_str(), "NA") != 0 )
    {
    std::cout << "Reading lung lobe label map image..." << std::endl;
      lungLobeLabelMapReader->SetFileName( lungLobeLabelMapFileName );
    try
      {
      lungLobeLabelMapReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading label map image:";
      std::cerr << excp << std::endl;
      return cip::LABELMAPREADFAILURE; 
      }
    }

  //
  // Define the phenotype and histogram containers
  //
  PARENCHYMAPHENOTYPES wholeLungPhenotypes;   InitializeParenchymaPhenotypes( &wholeLungPhenotypes );
  PARENCHYMAPHENOTYPES leftLungPhenotypes;    InitializeParenchymaPhenotypes( &leftLungPhenotypes );
  PARENCHYMAPHENOTYPES rightLungPhenotypes;   InitializeParenchymaPhenotypes( &rightLungPhenotypes );
  PARENCHYMAPHENOTYPES lulLungPhenotypes;     InitializeParenchymaPhenotypes( &lulLungPhenotypes );
  PARENCHYMAPHENOTYPES lllLungPhenotypes;     InitializeParenchymaPhenotypes( &lllLungPhenotypes );
  PARENCHYMAPHENOTYPES rulLungPhenotypes;     InitializeParenchymaPhenotypes( &rulLungPhenotypes );
  PARENCHYMAPHENOTYPES rmlLungPhenotypes;     InitializeParenchymaPhenotypes( &rmlLungPhenotypes );
  PARENCHYMAPHENOTYPES rllLungPhenotypes;     InitializeParenchymaPhenotypes( &rllLungPhenotypes );
  PARENCHYMAPHENOTYPES lutLungPhenotypes;     InitializeParenchymaPhenotypes( &lutLungPhenotypes );
  PARENCHYMAPHENOTYPES lmtLungPhenotypes;     InitializeParenchymaPhenotypes( &lmtLungPhenotypes );
  PARENCHYMAPHENOTYPES lltLungPhenotypes;     InitializeParenchymaPhenotypes( &lltLungPhenotypes );
  PARENCHYMAPHENOTYPES rutLungPhenotypes;     InitializeParenchymaPhenotypes( &rutLungPhenotypes );
  PARENCHYMAPHENOTYPES rmtLungPhenotypes;     InitializeParenchymaPhenotypes( &rmtLungPhenotypes );
  PARENCHYMAPHENOTYPES rltLungPhenotypes;     InitializeParenchymaPhenotypes( &rltLungPhenotypes );
  PARENCHYMAPHENOTYPES utLungPhenotypes;      InitializeParenchymaPhenotypes( &utLungPhenotypes );
  PARENCHYMAPHENOTYPES mtLungPhenotypes;      InitializeParenchymaPhenotypes( &mtLungPhenotypes );
  PARENCHYMAPHENOTYPES ltLungPhenotypes;      InitializeParenchymaPhenotypes( &ltLungPhenotypes );

  std::map< short, unsigned int > wholeLungHistogram;
  std::map< short, unsigned int > leftLungHistogram;
  std::map< short, unsigned int > rightLungHistogram;
  std::map< short, unsigned int > lulLungHistogram;
  std::map< short, unsigned int > lllLungHistogram;
  std::map< short, unsigned int > rulLungHistogram;
  std::map< short, unsigned int > rmlLungHistogram;
  std::map< short, unsigned int > rllLungHistogram;
  std::map< short, unsigned int > lutLungHistogram;
  std::map< short, unsigned int > lmtLungHistogram;
  std::map< short, unsigned int > lltLungHistogram;
  std::map< short, unsigned int > rutLungHistogram;
  std::map< short, unsigned int > rmtLungHistogram;
  std::map< short, unsigned int > rltLungHistogram;
  std::map< short, unsigned int > utLungHistogram;
  std::map< short, unsigned int > mtLungHistogram;
  std::map< short, unsigned int > ltLungHistogram;

  //
  // Initialize the histograms
  //
  for ( int i = minBin; i<=maxBin; i++ )
    {
    wholeLungHistogram[i] = 0;
    leftLungHistogram[i]  = 0;
    rightLungHistogram[i] = 0;
    lulLungHistogram[i]   = 0;
    lllLungHistogram[i]   = 0;
    rulLungHistogram[i]   = 0;
    rmlLungHistogram[i]   = 0;
    rllLungHistogram[i]   = 0;
    lutLungHistogram[i]   = 0;
    lmtLungHistogram[i]   = 0;
    lltLungHistogram[i]   = 0;
    rutLungHistogram[i]   = 0;
    rmtLungHistogram[i]   = 0;
    rltLungHistogram[i]   = 0;
    utLungHistogram[i]    = 0;
    mtLungHistogram[i]    = 0;
    ltLungHistogram[i]    = 0;
    }

  //
  // Compute the histograms
  //
  if ( strcmp( partialLungLabelMapFileName.c_str(), "NA") != 0 )
    {
    std::cout << "Computing histograms with partial lung label map..." << std::endl;
    UpdateAllHistogramsAndPhenotypes( ctReader->GetOutput(), partialLungLabelMapReader->GetOutput(),
                                   &wholeLungPhenotypes, &leftLungPhenotypes, &rightLungPhenotypes, &lulLungPhenotypes,
                                   &lllLungPhenotypes, &rulLungPhenotypes, &rmlLungPhenotypes, &rllLungPhenotypes,
                                   &lutLungPhenotypes, &lmtLungPhenotypes, &lltLungPhenotypes, &rutLungPhenotypes,
                                   &rmtLungPhenotypes, &rltLungPhenotypes, &utLungPhenotypes, &mtLungPhenotypes,
                                   &ltLungPhenotypes,
                                   &wholeLungHistogram, &leftLungHistogram, &rightLungHistogram,
                                   &lulLungHistogram, &lllLungHistogram, &rulLungHistogram,
                                   &rmlLungHistogram, &rllLungHistogram, &lutLungHistogram,
                                   &lmtLungHistogram, &lltLungHistogram, &rutLungHistogram,
                                   &rmtLungHistogram, &rltLungHistogram, &utLungHistogram,
                                   &mtLungHistogram, &ltLungHistogram,
                                   voxelVolume, minBin, maxBin);
    }

  if ( strcmp( lungLobeLabelMapFileName.c_str(), "NA") != 0 )
    {
    std::cout << "Computing histograms with lung lobe label map..." << std::endl;
    //If partial lung lablemap mask is not provided, we have to compute all the regional metrics.
    if ( strcmp( partialLungLabelMapFileName.c_str(), "NA") == 0 )
      {
      UpdateAllHistogramsAndPhenotypes( ctReader->GetOutput(), lungLobeLabelMapReader->GetOutput(),
                                        &wholeLungPhenotypes, &leftLungPhenotypes, &rightLungPhenotypes, &lulLungPhenotypes,
                                        &lllLungPhenotypes, &rulLungPhenotypes, &rmlLungPhenotypes, &rllLungPhenotypes,
                                        &lutLungPhenotypes, &lmtLungPhenotypes, &lltLungPhenotypes, &rutLungPhenotypes,
                                        &rmtLungPhenotypes, &rltLungPhenotypes, &utLungPhenotypes, &mtLungPhenotypes,
                                        &ltLungPhenotypes,
                                        &wholeLungHistogram, &leftLungHistogram, &rightLungHistogram,
                                        &lulLungHistogram, &lllLungHistogram, &rulLungHistogram,
                                        &rmlLungHistogram, &rllLungHistogram, &lutLungHistogram,
                                        &lmtLungHistogram, &lltLungHistogram, &rutLungHistogram,
                                        &rmtLungHistogram, &rltLungHistogram, &utLungHistogram,
                                        &mtLungHistogram, &ltLungHistogram,
                                        voxelVolume, minBin, maxBin);
      } 
    else 
      {
      // Just compute lobe-based specific metrics. The general metrics were computed above
      UpdateLobeHistogramsAndPhenotypes( ctReader->GetOutput(), lungLobeLabelMapReader->GetOutput(),
                                         &lulLungPhenotypes,&lllLungPhenotypes, &rulLungPhenotypes,
                                         &rmlLungPhenotypes, &rllLungPhenotypes,
                                         &lulLungHistogram, &lllLungHistogram, &rulLungHistogram,
                                         &rmlLungHistogram, &rllLungHistogram,voxelVolume, minBin, maxBin);
      }
    }
  
  //
  // Write histograms to file
  //
  if ( strcmp(histogramFileName.c_str(), "NA") != 0 )
    {
    std::cout << "Writing histogram to file..." << std::endl;
    std::ofstream histogramFile( histogramFileName.c_str() );
    histogramFile << "," << "WHOLELUNG," << "LEFTLUNG," << "RIGHTLUNG," << "LEFTSUPERIORLOBE," << "LEFTINFERIORLOBE,";
    histogramFile << "RIGHTUPPERLOBE," << "RIGHTMIDDLELOBE," << "RIGHTLOWERLOBE," << "LEFTUPPERTHIRD," << "LEFTMIDDLETHIRD,";
    histogramFile << "LEFTLOWERTHIRD," << "RIGHTUPPERTHIRD," << "RIGHTMIDDLETHIRD," << "RIGHTLOWERTHIRD," << "UPPERTHIRD,";
    histogramFile << "MIDDLETHIRD," << "LOWERTHIRD" << std::endl;

    for ( int i=minBin; i<=maxBin; i++ )
      {
      histogramFile << i << ",";
      histogramFile << wholeLungHistogram[i] << ",";
      histogramFile << leftLungHistogram[i]  << ",";
      histogramFile << rightLungHistogram[i] << ",";
      histogramFile << lulLungHistogram[i]   << ",";
      histogramFile << lllLungHistogram[i]   << ",";
      histogramFile << rulLungHistogram[i]   << ",";
      histogramFile << rmlLungHistogram[i]   << ",";
      histogramFile << rllLungHistogram[i]   << ",";
      histogramFile << lutLungHistogram[i]   << ",";
      histogramFile << lmtLungHistogram[i]   << ",";
      histogramFile << lltLungHistogram[i]   << ",";
      histogramFile << rutLungHistogram[i]   << ",";
      histogramFile << rmtLungHistogram[i]   << ",";
      histogramFile << rltLungHistogram[i]   << ",";
      histogramFile << utLungHistogram[i]    << ",";
      histogramFile << mtLungHistogram[i]    << ",";
      histogramFile << ltLungHistogram[i]    << std::endl;
      }
    histogramFile.close();
    }

  //
  // Compute the region phenotypes
  //
  if ( strcmp(phenotypesFileName.c_str(), "NA") != 0 )
  {
    std::cout << "Computing parenchyma phenotypes..." << std::endl;
    ComputeParenchymaPhenotypesSubset( &wholeLungPhenotypes, wholeLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &leftLungPhenotypes,  leftLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &rightLungPhenotypes, rightLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &lulLungPhenotypes,   lulLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &lllLungPhenotypes,   lllLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &rulLungPhenotypes,   rulLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &rmlLungPhenotypes,   rmlLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &rllLungPhenotypes,   rllLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &lutLungPhenotypes,   lutLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &lmtLungPhenotypes,   lmtLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &lltLungPhenotypes,   lltLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &rutLungPhenotypes,   rutLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &rmtLungPhenotypes,   rmtLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &rltLungPhenotypes,   rltLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &utLungPhenotypes,    utLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &mtLungPhenotypes,    mtLungHistogram, voxelVolume, minBin, maxBin );
    ComputeParenchymaPhenotypesSubset( &ltLungPhenotypes,    ltLungHistogram, voxelVolume, minBin, maxBin );

    //
    // Write phenotypes to file
    //
  
    std::cout << "Writing phenotypes to file..." << std::endl;
    std::ofstream phenotypesFile( phenotypesFileName.c_str() );

    phenotypesFile << "wholeLung %Below-950,";
    phenotypesFile << "wholeLung %Below-925,";
    phenotypesFile << "wholeLung %Below-910,";
    phenotypesFile << "wholeLung %Below-905,";
    phenotypesFile << "wholeLung %Below-900,";
    phenotypesFile << "wholeLung %Below-875,";
    phenotypesFile << "wholeLung %Below-856,";
    phenotypesFile << "wholeLung %Above0,";
    phenotypesFile << "wholeLung %Above-600,";
    phenotypesFile << "wholeLung %Above-250,";
    phenotypesFile << "wholeLung tenthPercentileHU,";
    phenotypesFile << "wholeLung fifteenthPercentileHU,";
    phenotypesFile << "wholeLung volume (L),";
    phenotypesFile << "wholeLung mass (g),";
    phenotypesFile << "wholeLung intensityMean,";
    phenotypesFile << "wholeLung intensitySTD,";
    phenotypesFile << "wholeLung skewness,";
    phenotypesFile << "wholeLung kurtosis,";
    phenotypesFile << "wholeLung mode,";
    phenotypesFile << "wholeLung median,";

    phenotypesFile << "leftLung %Below-950,";
    phenotypesFile << "leftLung %Below-925,";
    phenotypesFile << "leftLung %Below-910,";
    phenotypesFile << "leftLung %Below-905,";
    phenotypesFile << "leftLung %Below-900,";
    phenotypesFile << "leftLung %Below-875,";
    phenotypesFile << "leftLung %Below-856,";
    phenotypesFile << "leftLung %Above0,";
    phenotypesFile << "leftLung %Above-600,";
    phenotypesFile << "leftLung %Above-250,";
    phenotypesFile << "leftLung tenthPercentileHU,";
    phenotypesFile << "leftLung fifteenthPercentileHU,";
    phenotypesFile << "leftLung volume (L),";
    phenotypesFile << "leftLung mass (g),";
    phenotypesFile << "leftLung intensityMean,";
    phenotypesFile << "leftLung intensitySTD,";
    phenotypesFile << "leftLung skewness,";
    phenotypesFile << "leftLung kurtosis,";
    phenotypesFile << "leftLung mode,";
    phenotypesFile << "leftLung median,";

    phenotypesFile << "rightLung %Below-950,";
    phenotypesFile << "rightLung %Below-925,";
    phenotypesFile << "rightLung %Below-910,";
    phenotypesFile << "rightLung %Below-905,";
    phenotypesFile << "rightLung %Below-900,";
    phenotypesFile << "rightLung %Below-875,";
    phenotypesFile << "rightLung %Below-856,";
    phenotypesFile << "rightLung %Above0,";
    phenotypesFile << "rightLung %Above-600,";
    phenotypesFile << "rightLung %Above-250,";
    phenotypesFile << "rightLung tenthPercentileHU,";
    phenotypesFile << "rightLung fifteenthPercentileHU,";
    phenotypesFile << "rightLung volume (L),";
    phenotypesFile << "rightLung mass (g),";
    phenotypesFile << "rightLung intensityMean,";
    phenotypesFile << "rightLung intensitySTD,";
    phenotypesFile << "rightLung skewness,";
    phenotypesFile << "rightLung kurtosis,";
    phenotypesFile << "rightLung mode,";
    phenotypesFile << "rightLung median,";

    phenotypesFile << "lulLung %Below-950,";
    phenotypesFile << "lulLung %Below-925,";
    phenotypesFile << "lulLung %Below-910,";
    phenotypesFile << "lulLung %Below-905,";
    phenotypesFile << "lulLung %Below-900,";
    phenotypesFile << "lulLung %Below-875,";
    phenotypesFile << "lulLung %Below-856,";
    phenotypesFile << "lulLung %Above0,";
    phenotypesFile << "lulLung %Above-600,";
    phenotypesFile << "lulLung %Above-250,";
    phenotypesFile << "lulLung tenthPercentileHU,";
    phenotypesFile << "lulLung fifteenthPercentileHU,";
    phenotypesFile << "lulLung volume (L),";
    phenotypesFile << "lulLung mass (g),";
    phenotypesFile << "lulLung intensityMean,";
    phenotypesFile << "lulLung intensitySTD,";
    phenotypesFile << "lulLung skewness,";
    phenotypesFile << "lulLung kurtosis,";
    phenotypesFile << "lulLung mode,";
    phenotypesFile << "lulLung median,";

    phenotypesFile << "lllLung %Below-950,";
    phenotypesFile << "lllLung %Below-925,";
    phenotypesFile << "lllLung %Below-910,";
    phenotypesFile << "lllLung %Below-905,";
    phenotypesFile << "lllLung %Below-900,";
    phenotypesFile << "lllLung %Below-875,";
    phenotypesFile << "lllLung %Below-856,";
    phenotypesFile << "lllLung %Above0,";
    phenotypesFile << "lllLung %Above-600,";
    phenotypesFile << "lllLung %Above-250,";
    phenotypesFile << "lllLung tenthPercentileHU,";
    phenotypesFile << "lllLung fifteenthPercentileHU,";
    phenotypesFile << "lllLung volume (L),";
    phenotypesFile << "lllLung mass (g),";
    phenotypesFile << "lllLung intensityMean,";
    phenotypesFile << "lllLung intensitySTD,";
    phenotypesFile << "lllLung skewness,";
    phenotypesFile << "lllLung kurtosis,";
    phenotypesFile << "lllLung mode,";
    phenotypesFile << "lllLung median,";

    phenotypesFile << "rulLung %Below-950,";
    phenotypesFile << "rulLung %Below-925,";
    phenotypesFile << "rulLung %Below-910,";
    phenotypesFile << "rulLung %Below-905,";
    phenotypesFile << "rulLung %Below-900,";
    phenotypesFile << "rulLung %Below-875,";
    phenotypesFile << "rulLung %Below-856,";
    phenotypesFile << "rulLung %Above0,";
    phenotypesFile << "rulLung %Above-600,";
    phenotypesFile << "rulLung %Above-250,";
    phenotypesFile << "rulLung tenthPercentileHU,";
    phenotypesFile << "rulLung fifteenthPercentileHU,";
    phenotypesFile << "rulLung volume (L),";
    phenotypesFile << "rulLung mass (g),";
    phenotypesFile << "rulLung intensityMean,";
    phenotypesFile << "rulLung intensitySTD,";
    phenotypesFile << "rulLung skewness,";
    phenotypesFile << "rulLung kurtosis,";
    phenotypesFile << "rulLung mode,";
    phenotypesFile << "rulLung median,";

    phenotypesFile << "rmlLung %Below-950,";
    phenotypesFile << "rmlLung %Below-925,";
    phenotypesFile << "rmlLung %Below-910,";
    phenotypesFile << "rmlLung %Below-905,";
    phenotypesFile << "rmlLung %Below-900,";
    phenotypesFile << "rmlLung %Below-875,";
    phenotypesFile << "rmlLung %Below-856,";
    phenotypesFile << "rmlLung %Above0,";
    phenotypesFile << "rmlLung %Above-600,";
    phenotypesFile << "rmlLung %Above-250,";
    phenotypesFile << "rmlLung tenthPercentileHU,";
    phenotypesFile << "rmlLung fifteenthPercentileHU,";
    phenotypesFile << "rmlLung volume (L),";
    phenotypesFile << "rmlLung mass (g),";
    phenotypesFile << "rmlLung intensityMean,";
    phenotypesFile << "rmlLung intensitySTD,";
    phenotypesFile << "rmlLung skewness,";
    phenotypesFile << "rmlLung kurtosis,";
    phenotypesFile << "rmlLung mode,";
    phenotypesFile << "rmlLung median,";

    phenotypesFile << "rllLung %Below-950,";
    phenotypesFile << "rllLung %Below-925,";
    phenotypesFile << "rllLung %Below-910,";
    phenotypesFile << "rllLung %Below-905,";
    phenotypesFile << "rllLung %Below-900,";
    phenotypesFile << "rllLung %Below-875,";
    phenotypesFile << "rllLung %Below-856,";
    phenotypesFile << "rllLung %Above0,";
    phenotypesFile << "rllLung %Above-600,";
    phenotypesFile << "rllLung %Above-250,";
    phenotypesFile << "rllLung tenthPercentileHU,";
    phenotypesFile << "rllLung fifteenthPercentileHU,";
    phenotypesFile << "rllLung volume (L),";
    phenotypesFile << "rllLung mass (g),";
    phenotypesFile << "rllLung intensityMean,";
    phenotypesFile << "rllLung intensitySTD,";
    phenotypesFile << "rllLung skewness,";
    phenotypesFile << "rllLung kurtosis,";
    phenotypesFile << "rllLung mode,";
    phenotypesFile << "rllLung median,";

    phenotypesFile << "lutLung %Below-950,";
    phenotypesFile << "lutLung %Below-925,";
    phenotypesFile << "lutLung %Below-910,";
    phenotypesFile << "lutLung %Below-905,";
    phenotypesFile << "lutLung %Below-900,";
    phenotypesFile << "lutLung %Below-875,";
    phenotypesFile << "lutLung %Below-856,";
    phenotypesFile << "lutLung %Above0,";
    phenotypesFile << "lutLung %Above-600,";
    phenotypesFile << "lutLung %Above-250,";
    phenotypesFile << "lutLung tenthPercentileHU,";
    phenotypesFile << "lutLung fifteenthPercentileHU,";
    phenotypesFile << "lutLung volume (L),";
    phenotypesFile << "lutLung mass (g),";
    phenotypesFile << "lutLung intensityMean,";
    phenotypesFile << "lutLung intensitySTD,";
    phenotypesFile << "lutLung skewness,";
    phenotypesFile << "lutLung kurtosis,";
    phenotypesFile << "lutLung mode,";
    phenotypesFile << "lutLung median,";

    phenotypesFile << "lmtLung %Below-950,";
    phenotypesFile << "lmtLung %Below-925,";
    phenotypesFile << "lmtLung %Below-910,";
    phenotypesFile << "lmtLung %Below-905,";
    phenotypesFile << "lmtLung %Below-900,";
    phenotypesFile << "lmtLung %Below-875,";
    phenotypesFile << "lmtLung %Below-856,";
    phenotypesFile << "lmtLung %Above0,";
    phenotypesFile << "lmtLung %Above-600,";
    phenotypesFile << "lmtLung %Above-250,";
    phenotypesFile << "lmtLung tenthPercentileHU,";
    phenotypesFile << "lmtLung fifteenthPercentileHU,";
    phenotypesFile << "lmtLung volume (L),";
    phenotypesFile << "lmtLung mass (g),";
    phenotypesFile << "lmtLung intensityMean,";
    phenotypesFile << "lmtLung intensitySTD,";
    phenotypesFile << "lmtLung skewness,";
    phenotypesFile << "lmtLung kurtosis,";
    phenotypesFile << "lmtLung mode,";
    phenotypesFile << "lmtLung median,";

    phenotypesFile << "lltLung %Below-950,";
    phenotypesFile << "lltLung %Below-925,";
    phenotypesFile << "lltLung %Below-910,";
    phenotypesFile << "lltLung %Below-905,";
    phenotypesFile << "lltLung %Below-900,";
    phenotypesFile << "lltLung %Below-875,";
    phenotypesFile << "lltLung %Below-856,";
    phenotypesFile << "lltLung %Above0,";
    phenotypesFile << "lltLung %Above-600,";
    phenotypesFile << "lltLung %Above-250,";
    phenotypesFile << "lltLung tenthPercentileHU,";
    phenotypesFile << "lltLung fifteenthPercentileHU,";
    phenotypesFile << "lltLung volume (L),";
    phenotypesFile << "lltLung mass (g),";
    phenotypesFile << "lltLung intensityMean,";
    phenotypesFile << "lltLung intensitySTD,";
    phenotypesFile << "lltLung skewness,";
    phenotypesFile << "lltLung kurtosis,";
    phenotypesFile << "lltLung mode,";
    phenotypesFile << "lltLung median,";

    phenotypesFile << "rutLung %Below-950,";
    phenotypesFile << "rutLung %Below-925,";
    phenotypesFile << "rutLung %Below-910,";
    phenotypesFile << "rutLung %Below-905,";
    phenotypesFile << "rutLung %Below-900,";
    phenotypesFile << "rutLung %Below-875,";
    phenotypesFile << "rutLung %Below-856,";
    phenotypesFile << "rutLung %Above0,";
    phenotypesFile << "rutLung %Above-600,";
    phenotypesFile << "rutLung %Above-250,";
    phenotypesFile << "rutLung tenthPercentileHU,";
    phenotypesFile << "rutLung fifteenthPercentileHU,";
    phenotypesFile << "rutLung volume (L),";
    phenotypesFile << "rutLung mass (g),";
    phenotypesFile << "rutLung intensityMean,";
    phenotypesFile << "rutLung intensitySTD,";
    phenotypesFile << "rutLung skewness,";
    phenotypesFile << "rutLung kurtosis,";
    phenotypesFile << "rutLung mode,";
    phenotypesFile << "rutLung median,";

    phenotypesFile << "rmtLung %Below-950,";
    phenotypesFile << "rmtLung %Below-925,";
    phenotypesFile << "rmtLung %Below-910,";
    phenotypesFile << "rmtLung %Below-905,";
    phenotypesFile << "rmtLung %Below-900,";
    phenotypesFile << "rmtLung %Below-875,";
    phenotypesFile << "rmtLung %Below-856,";
    phenotypesFile << "rmtLung %Above0,";
    phenotypesFile << "rmtLung %Above-600,";
    phenotypesFile << "rmtLung %Above-250,";
    phenotypesFile << "rmtLung tenthPercentileHU,";
    phenotypesFile << "rmtLung fifteenthPercentileHU,";
    phenotypesFile << "rmtLung volume (L),";
    phenotypesFile << "rmtLung mass (g),";
    phenotypesFile << "rmtLung intensityMean,";
    phenotypesFile << "rmtLung intensitySTD,";
    phenotypesFile << "rmtLung skewness,";
    phenotypesFile << "rmtLung kurtosis,";
    phenotypesFile << "rmtLung mode,";
    phenotypesFile << "rmtLung median,";

    phenotypesFile << "rltLung %Below-950,";
    phenotypesFile << "rltLung %Below-925,";
    phenotypesFile << "rltLung %Below-910,";
    phenotypesFile << "rltLung %Below-905,";
    phenotypesFile << "rltLung %Below-900,";
    phenotypesFile << "rltLung %Below-875,";
    phenotypesFile << "rltLung %Below-856,";
    phenotypesFile << "rltLung %Above0,";
    phenotypesFile << "rltLung %Above-600,";
    phenotypesFile << "rltLung %Above-250,";
    phenotypesFile << "rltLung tenthPercentileHU,";
    phenotypesFile << "rltLung fifteenthPercentileHU,";
    phenotypesFile << "rltLung volume (L),";
    phenotypesFile << "rltLung mass (g),";
    phenotypesFile << "rltLung intensityMean,";
    phenotypesFile << "rltLung intensitySTD,";
    phenotypesFile << "rltLung skewness,";
    phenotypesFile << "rltLung kurtosis,";
    phenotypesFile << "rltLung mode,";
    phenotypesFile << "rltLung median,";

    phenotypesFile << "utLung %Below-950,";
    phenotypesFile << "utLung %Below-925,";
    phenotypesFile << "utLung %Below-910,";
    phenotypesFile << "utLung %Below-905,";
    phenotypesFile << "utLung %Below-900,";
    phenotypesFile << "utLung %Below-875,";
    phenotypesFile << "utLung %Below-856,";
    phenotypesFile << "utLung %Above0,";
    phenotypesFile << "utLung %Above-600,";
    phenotypesFile << "utLung %Above-250,";
    phenotypesFile << "utLung tenthPercentileHU,";
    phenotypesFile << "utLung fifteenthPercentileHU,";
    phenotypesFile << "utLung volume (L),";
    phenotypesFile << "utLung mass (g),";
    phenotypesFile << "utLung intensityMean,";
    phenotypesFile << "utLung intensitySTD,";
    phenotypesFile << "utLung skewness,";
    phenotypesFile << "utLung kurtosis,";
    phenotypesFile << "utLung mode,";
    phenotypesFile << "utLung median,";

    phenotypesFile << "mtLung %Below-950,";
    phenotypesFile << "mtLung %Below-925,";
    phenotypesFile << "mtLung %Below-910,";
    phenotypesFile << "mtLung %Below-905,";
    phenotypesFile << "mtLung %Below-900,";
    phenotypesFile << "mtLung %Below-875,";
    phenotypesFile << "mtLung %Below-856,";
    phenotypesFile << "mtLung %Above0,";
    phenotypesFile << "mtLung %Above-600,";
    phenotypesFile << "mtLung %Above-250,";
    phenotypesFile << "mtLung tenthPercentileHU,";
    phenotypesFile << "mtLung fifteenthPercentileHU,";
    phenotypesFile << "mtLung volume (L),";
    phenotypesFile << "mtLung mass (g),";
    phenotypesFile << "mtLung intensityMean,";
    phenotypesFile << "mtLung intensitySTD,";
    phenotypesFile << "mtLung skewness,";
    phenotypesFile << "mtLung kurtosis,";
    phenotypesFile << "mtLung mode,";
    phenotypesFile << "mtLung median,";

    phenotypesFile << "ltLung %Below-950,";
    phenotypesFile << "ltLung %Below-925,";
    phenotypesFile << "ltLung %Below-910,";
    phenotypesFile << "ltLung %Below-905,";
    phenotypesFile << "ltLung %Below-900,";
    phenotypesFile << "ltLung %Below-875,";
    phenotypesFile << "ltLung %Below-856,";
    phenotypesFile << "ltLung %Above0,";
    phenotypesFile << "ltLung %Above-600,";
    phenotypesFile << "ltLung %Above-250,";
    phenotypesFile << "ltLung tenthPercentileHU,";
    phenotypesFile << "ltLung fifteenthPercentileHU,";
    phenotypesFile << "ltLung volume (L),";
    phenotypesFile << "ltLung mass (g),";
    phenotypesFile << "ltLung intensityMean,";
    phenotypesFile << "ltLung intensitySTD,";
    phenotypesFile << "ltLung skewness,";
    phenotypesFile << "ltLung kurtosis,";
    phenotypesFile << "ltLung mode,";
    phenotypesFile << "ltLung median" << std::endl;

    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countBelow950)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countBelow925)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countBelow910)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countBelow905)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countBelow900)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countBelow875)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countBelow856)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countAbove0)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countAbove600)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(wholeLungPhenotypes.countAbove250)/static_cast< double >(wholeLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << wholeLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << wholeLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << wholeLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << wholeLungPhenotypes.mass << ",";
    phenotypesFile << wholeLungPhenotypes.intensityMean << ",";
    phenotypesFile << wholeLungPhenotypes.intensitySTD << ",";
    phenotypesFile << wholeLungPhenotypes.skewness << ",";
    phenotypesFile << wholeLungPhenotypes.kurtosis << ",";
    phenotypesFile << wholeLungPhenotypes.mode << ",";
    phenotypesFile << wholeLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countBelow950)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countBelow925)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countBelow910)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countBelow905)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countBelow900)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countBelow875)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countBelow856)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countAbove0)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countAbove600)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(leftLungPhenotypes.countAbove250)/static_cast< double >(leftLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << leftLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << leftLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << leftLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << leftLungPhenotypes.mass << ",";
    phenotypesFile << leftLungPhenotypes.intensityMean << ",";
    phenotypesFile << leftLungPhenotypes.intensitySTD << ",";
    phenotypesFile << leftLungPhenotypes.skewness << ",";
    phenotypesFile << leftLungPhenotypes.kurtosis << ",";
    phenotypesFile << leftLungPhenotypes.mode << ",";
    phenotypesFile << leftLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countBelow950)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countBelow925)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countBelow910)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countBelow905)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countBelow900)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countBelow875)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countBelow856)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countAbove0)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countAbove600)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rightLungPhenotypes.countAbove250)/static_cast< double >(rightLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << rightLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << rightLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << rightLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << rightLungPhenotypes.mass << ",";
    phenotypesFile << rightLungPhenotypes.intensityMean << ",";
    phenotypesFile << rightLungPhenotypes.intensitySTD << ",";
    phenotypesFile << rightLungPhenotypes.skewness << ",";
    phenotypesFile << rightLungPhenotypes.kurtosis << ",";
    phenotypesFile << rightLungPhenotypes.mode << ",";
    phenotypesFile << rightLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countBelow950)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countBelow925)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countBelow910)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countBelow905)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countBelow900)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countBelow875)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countBelow856)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countAbove0)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countAbove600)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lulLungPhenotypes.countAbove250)/static_cast< double >(lulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << lulLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << lulLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << lulLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << lulLungPhenotypes.mass << ",";
    phenotypesFile << lulLungPhenotypes.intensityMean << ",";
    phenotypesFile << lulLungPhenotypes.intensitySTD << ",";
    phenotypesFile << lulLungPhenotypes.skewness << ",";
    phenotypesFile << lulLungPhenotypes.kurtosis << ",";
    phenotypesFile << lulLungPhenotypes.mode << ",";
    phenotypesFile << lulLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countBelow950)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countBelow925)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countBelow910)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countBelow905)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countBelow900)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countBelow875)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countBelow856)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countAbove0)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countAbove600)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lllLungPhenotypes.countAbove250)/static_cast< double >(lllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << lllLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << lllLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << lllLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << lllLungPhenotypes.mass << ",";
    phenotypesFile << lllLungPhenotypes.intensityMean << ",";
    phenotypesFile << lllLungPhenotypes.intensitySTD << ",";
    phenotypesFile << lllLungPhenotypes.skewness << ",";
    phenotypesFile << lllLungPhenotypes.kurtosis << ",";
    phenotypesFile << lllLungPhenotypes.mode << ",";
    phenotypesFile << lllLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countBelow950)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countBelow925)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countBelow910)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countBelow905)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countBelow900)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countBelow875)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countBelow856)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countAbove0)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countAbove600)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rulLungPhenotypes.countAbove250)/static_cast< double >(rulLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << rulLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << rulLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << rulLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << rulLungPhenotypes.mass << ",";
    phenotypesFile << rulLungPhenotypes.intensityMean << ",";
    phenotypesFile << rulLungPhenotypes.intensitySTD << ",";
    phenotypesFile << rulLungPhenotypes.skewness << ",";
    phenotypesFile << rulLungPhenotypes.kurtosis << ",";
    phenotypesFile << rulLungPhenotypes.mode << ",";
    phenotypesFile << rulLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countBelow950)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countBelow925)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countBelow910)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countBelow905)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countBelow900)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countBelow875)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countBelow856)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countAbove0)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countAbove600)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmlLungPhenotypes.countAbove250)/static_cast< double >(rmlLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << rmlLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << rmlLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << rmlLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << rmlLungPhenotypes.mass << ",";
    phenotypesFile << rmlLungPhenotypes.intensityMean << ",";
    phenotypesFile << rmlLungPhenotypes.intensitySTD << ",";
    phenotypesFile << rmlLungPhenotypes.skewness << ",";
    phenotypesFile << rmlLungPhenotypes.kurtosis << ",";
    phenotypesFile << rmlLungPhenotypes.mode << ",";
    phenotypesFile << rmlLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countBelow950)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countBelow925)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countBelow910)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countBelow905)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countBelow900)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countBelow875)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countBelow856)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countAbove0)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countAbove600)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rllLungPhenotypes.countAbove250)/static_cast< double >(rllLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << rllLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << rllLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << rllLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << rllLungPhenotypes.mass << ",";
    phenotypesFile << rllLungPhenotypes.intensityMean << ",";
    phenotypesFile << rllLungPhenotypes.intensitySTD << ",";
    phenotypesFile << rllLungPhenotypes.skewness << ",";
    phenotypesFile << rllLungPhenotypes.kurtosis << ",";
    phenotypesFile << rllLungPhenotypes.mode << ",";
    phenotypesFile << rllLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countBelow950)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countBelow925)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countBelow910)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countBelow905)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countBelow900)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countBelow875)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countBelow856)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countAbove0)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countAbove600)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lutLungPhenotypes.countAbove250)/static_cast< double >(lutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << lutLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << lutLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << lutLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << lutLungPhenotypes.mass << ",";
    phenotypesFile << lutLungPhenotypes.intensityMean << ",";
    phenotypesFile << lutLungPhenotypes.intensitySTD << ",";
    phenotypesFile << lutLungPhenotypes.skewness << ",";
    phenotypesFile << lutLungPhenotypes.kurtosis << ",";
    phenotypesFile << lutLungPhenotypes.mode << ",";
    phenotypesFile << lutLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countBelow950)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countBelow925)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countBelow910)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countBelow905)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countBelow900)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countBelow875)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countBelow856)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countAbove0)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countAbove600)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lmtLungPhenotypes.countAbove250)/static_cast< double >(lmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << lmtLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << lmtLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << lmtLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << lmtLungPhenotypes.mass << ",";
    phenotypesFile << lmtLungPhenotypes.intensityMean << ",";
    phenotypesFile << lmtLungPhenotypes.intensitySTD << ",";
    phenotypesFile << lmtLungPhenotypes.skewness << ",";
    phenotypesFile << lmtLungPhenotypes.kurtosis << ",";
    phenotypesFile << lmtLungPhenotypes.mode << ",";
    phenotypesFile << lmtLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countBelow950)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countBelow925)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countBelow910)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countBelow905)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countBelow900)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countBelow875)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countBelow856)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countAbove0)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countAbove600)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(lltLungPhenotypes.countAbove250)/static_cast< double >(lltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << lltLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << lltLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << lltLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << lltLungPhenotypes.mass << ",";
    phenotypesFile << lltLungPhenotypes.intensityMean << ",";
    phenotypesFile << lltLungPhenotypes.intensitySTD << ",";
    phenotypesFile << lltLungPhenotypes.skewness << ",";
    phenotypesFile << lltLungPhenotypes.kurtosis << ",";
    phenotypesFile << lltLungPhenotypes.mode << ",";
    phenotypesFile << lltLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countBelow950)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countBelow925)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countBelow910)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countBelow905)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countBelow900)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countBelow875)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countBelow856)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countAbove0)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countAbove600)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rutLungPhenotypes.countAbove250)/static_cast< double >(rutLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << rutLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << rutLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << rutLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << rutLungPhenotypes.mass << ",";
    phenotypesFile << rutLungPhenotypes.intensityMean << ",";
    phenotypesFile << rutLungPhenotypes.intensitySTD << ",";
    phenotypesFile << rutLungPhenotypes.skewness << ",";
    phenotypesFile << rutLungPhenotypes.kurtosis << ",";
    phenotypesFile << rutLungPhenotypes.mode << ",";
    phenotypesFile << rutLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countBelow950)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countBelow925)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countBelow910)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countBelow905)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countBelow900)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countBelow875)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countBelow856)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countAbove0)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countAbove600)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rmtLungPhenotypes.countAbove250)/static_cast< double >(rmtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << rmtLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << rmtLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << rmtLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << rmtLungPhenotypes.mass << ",";
    phenotypesFile << rmtLungPhenotypes.intensityMean << ",";
    phenotypesFile << rmtLungPhenotypes.intensitySTD << ",";
    phenotypesFile << rmtLungPhenotypes.skewness << ",";
    phenotypesFile << rmtLungPhenotypes.kurtosis << ",";
    phenotypesFile << rmtLungPhenotypes.mode << ",";
    phenotypesFile << rmtLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countBelow950)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countBelow925)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countBelow910)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countBelow905)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countBelow900)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countBelow875)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countBelow856)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countAbove0)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countAbove600)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(rltLungPhenotypes.countAbove250)/static_cast< double >(rltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << rltLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << rltLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << rltLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << rltLungPhenotypes.mass << ",";
    phenotypesFile << rltLungPhenotypes.intensityMean << ",";
    phenotypesFile << rltLungPhenotypes.intensitySTD << ",";
    phenotypesFile << rltLungPhenotypes.skewness << ",";
    phenotypesFile << rltLungPhenotypes.kurtosis << ",";
    phenotypesFile << rltLungPhenotypes.mode << ",";
    phenotypesFile << rltLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countBelow950)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countBelow925)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countBelow910)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countBelow905)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countBelow900)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countBelow875)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countBelow856)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countAbove0)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countAbove600)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(utLungPhenotypes.countAbove250)/static_cast< double >(utLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << utLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << utLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << utLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << utLungPhenotypes.mass << ",";
    phenotypesFile << utLungPhenotypes.intensityMean << ",";
    phenotypesFile << utLungPhenotypes.intensitySTD << ",";
    phenotypesFile << utLungPhenotypes.skewness << ",";
    phenotypesFile << utLungPhenotypes.kurtosis << ",";
    phenotypesFile << utLungPhenotypes.mode << ",";
    phenotypesFile << utLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countBelow950)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countBelow925)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countBelow910)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countBelow905)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countBelow900)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countBelow875)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countBelow856)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countAbove0)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countAbove600)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(mtLungPhenotypes.countAbove250)/static_cast< double >(mtLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << mtLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << mtLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << mtLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << mtLungPhenotypes.mass << ",";
    phenotypesFile << mtLungPhenotypes.intensityMean << ",";
    phenotypesFile << mtLungPhenotypes.intensitySTD << ",";
    phenotypesFile << mtLungPhenotypes.skewness << ",";
    phenotypesFile << mtLungPhenotypes.kurtosis << ",";
    phenotypesFile << mtLungPhenotypes.mode << ",";
    phenotypesFile << mtLungPhenotypes.median << ",";

    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countBelow950)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";    
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countBelow925)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countBelow910)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countBelow905)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countBelow900)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countBelow875)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countBelow856)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countAbove0)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countAbove600)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << 100.0*static_cast< double >(ltLungPhenotypes.countAbove250)/static_cast< double >(ltLungPhenotypes.totalVoxels) << ",";
    phenotypesFile << ltLungPhenotypes.tenthPercentileHU << ",";
    phenotypesFile << ltLungPhenotypes.fifteenthPercentileHU << ",";
    phenotypesFile << ltLungPhenotypes.volume/1000000.0 << ",";
    phenotypesFile << ltLungPhenotypes.mass << ",";
    phenotypesFile << ltLungPhenotypes.intensityMean << ",";
    phenotypesFile << ltLungPhenotypes.intensitySTD << ",";
    phenotypesFile << ltLungPhenotypes.skewness << ",";
    phenotypesFile << ltLungPhenotypes.kurtosis << ",";
    phenotypesFile << ltLungPhenotypes.mode << ",";
    phenotypesFile << ltLungPhenotypes.median << ",";

    phenotypesFile.close();
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}
