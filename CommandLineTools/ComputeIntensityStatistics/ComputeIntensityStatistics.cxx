/** \file
 *  \ingroup commandLineTools 
 *  \details This program is used to compute intensity statistics for
 *  chest-region  chest-type pairs. For every pair present in the
 *  input label map, the mean, min, max, median, and standard
 *  deviation are computed. The results are printed to the command
 *  line. 
 *
 *  USAGE: 
 * 
 *  ComputeIntensityStatistics  -c <string> -l <string> 
 *    [--] [--version] [-h]
 * 
 *  Where: 
 * 
 *  -c <string>,  --ctFileName <string>
 *    (required)  Input CT file name
 * 
 *  -l <string>,  --labelMapFileName <string>
 *    (required)  Input label map file name
 * 
 *  --,  --ignore_rest
 *    Ignores the rest of the labeled arguments following this flag.
 * 
 *  --version
 *    Displays version information and exits.
 * 
 *  -h,  --help
 *    Displays usage information and exits.
 *
 *  $Date: 2012-11-05 16:49:21 -0500 (Mon, 05 Nov 2012) $
 *  $Revision: 311 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "ComputeIntensityStatisticsCLP.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"
#include <iostream>
#include <fstream>
#include <limits.h>

namespace
{

   typedef itk::ImageRegionIterator< cip::LabelMapType > LabelMapIteratorType;
   typedef itk::ImageRegionIterator< cip::CTType >       CTIteratorType;

   struct STATS
   {
     double mean;
     double median;
     double std;
     short  min;
     short  max;
     std::list< short > HUs;
   };

}
int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  //
  // Read the label map
  //
  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( labelMapFileName );
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

  //
  // Read the CT image
  //
  std::cout << "Reading CT..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT:";
    std::cerr << excp << std::endl;

    return cip::NRRDREADFAILURE;
    }

  //
  // Get a list of the labels present in the label map 
  //
  std::cout << "Determing structures in label map..." << std::endl;
  std::list< unsigned short > labelsList;

  LabelMapIteratorType lIt( labelMapReader->GetOutput(), labelMapReader->GetOutput()->GetBufferedRegion() );

  while ( !lIt.IsAtEnd() )
    {
    if ( lIt.Get() != 0 )
      {
      labelsList.push_back( lIt.Get() );
      }

    ++lIt;
    }
  labelsList.unique();
  labelsList.sort();
  labelsList.unique();

  //
  // Now collect the CT values for each region-type pair for
  // subsequent stats computation
  //
  std::map< unsigned short, STATS > labelToStatsMap;

  CTIteratorType cIt( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );

  lIt.GoToBegin();
  while ( !cIt.IsAtEnd() )
    {
    if ( lIt.Get() != 0 )
      {
      labelToStatsMap[lIt.Get()].HUs.push_back( cIt.Get() );
      }

    ++lIt;
    ++cIt;
    }

  //
  // Now compute the stats for each label
  //
  std::cout << "Computing statistics..." << std::endl;
  
  //
  // Compute the stats for each label
  // 
  std::map< unsigned short, STATS >::iterator mapIt = labelToStatsMap.begin();

  while ( mapIt != labelToStatsMap.end() )
    {
    //
    // Need to sort in order to compute the median
    //
    mapIt->second.HUs.sort();

    double meanAccum = 0.0;
    mapIt->second.min = SHRT_MAX;
    mapIt->second.max = SHRT_MIN;
    
    std::list< short >::iterator listIt = mapIt->second.HUs.begin();

    unsigned int counter = 0;
    while ( listIt != mapIt->second.HUs.end() )
      {
      if ( *listIt < mapIt->second.min )
        {
        mapIt->second.min = *listIt;
        }
      if ( *listIt > mapIt->second.max )
        {
        mapIt->second.max = *listIt;
        }

      meanAccum += *listIt;

      if ( counter <= mapIt->second.HUs.size()/2 )
        {
        mapIt->second.median = *listIt;
        }

      counter++;
      ++listIt;
      }

    mapIt->second.mean = meanAccum/static_cast< double >( mapIt->second.HUs.size() );

    //
    // Finally, compute the standard deviation
    //
    double stdAccum = 0.0;

    listIt = mapIt->second.HUs.begin();
    while ( listIt != mapIt->second.HUs.end() )
      {
      stdAccum += pow( *listIt - mapIt->second.mean, 2.0 );

      ++listIt;
      }
    mapIt->second.std = sqrt( stdAccum/static_cast< double >( mapIt->second.HUs.size() ) );

    ++mapIt;
    }

  //
  // Now print the results
  //
  cip::ChestConventions conventions;

  mapIt = labelToStatsMap.begin();
  while ( mapIt != labelToStatsMap.end() )
    {
    std::cout << conventions.GetChestRegionNameFromValue( mapIt->first ) << "\t";
    std::cout << conventions.GetChestTypeNameFromValue( mapIt->first ) << ":" << std::endl;
    std::cout << "Mean:\t"   << mapIt->second.mean   << std::endl;
    std::cout << "STD:\t"    << mapIt->second.std    << std::endl;
    std::cout << "Min:\t"    << mapIt->second.min    << std::endl;
    std::cout << "Max:\t"    << mapIt->second.max    << std::endl;
    std::cout << "Median:\t" << mapIt->second.median << std::endl;

    ++mapIt;
    }

  // Print the results to file if the user has specified an output
  // file name
  if (outFileName.compare("NA") != 0)
    {
    std::cout << "Writing results to file..." << std::endl;
    
    std::ofstream file(outFileName.c_str());
        
    // Title
    file<<"File,Region,Type,Mean,STD,Min,Max,Median"<<std::endl;

    // First write the name of label map
    //file << labelMapFileName << ",";

    // First write the header
    mapIt = labelToStatsMap.begin();
    while ( mapIt != labelToStatsMap.end() )
      {
	file << labelMapFileName << ",";
      std::string regionName = conventions.GetChestRegionNameFromValue( mapIt->first );
      std::string typeName   = conventions.GetChestTypeNameFromValue( mapIt->first );
      file << regionName << "," << typeName << "," << mapIt->second.mean << ",";
      file << mapIt->second.std <<",";
      file << mapIt->second.min <<",";
      file << mapIt->second.max <<",";
      file << mapIt->second.median << std::endl;
      ++mapIt;
      }
    
    file.close();
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
