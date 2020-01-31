/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads a CT (DICOM) image, extracts tags of
 *  interest and their values and then prints them file.
 */

#include "cipChestConventions.h"

#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileReader.h"
#include <itksys/SystemTools.hxx>
#include "ReadDicomWriteTagsCLP.h"

namespace
{
  typedef itk::Image< short, 3 >                      Short3DImageType;
  typedef itk::Image< short, 2 >                      Short2DImageType;
  typedef itk::ImageSeriesReader< Short3DImageType >  SeriesReaderType;
  typedef itk::GDCMImageIO                            ImageIOType;
  typedef itk::GDCMSeriesFileNames                    NamesGeneratorType;
  typedef itk::MetaDataDictionary                     DictionaryType;
  typedef itk::MetaDataObject< std::string >          MetaDataStringType;
  typedef itk::ImageFileReader< Short3DImageType >    Image3DReaderType;
  typedef itk::ImageFileReader< Short2DImageType >    Image2DReaderType;
    
  struct TAGS
  {
    std::string patientName;
    std::string patientID;
    std::string studyDate;
    std::string institution;
    std::string ctManufacturer;
    std::string ctModel;
    std::string dateOfLastCalibration;
    std::string convolutionKernel;
    std::string studyDescription;
    std::string modalitiesInStudy;
    std::string imageComments;
    std::string sliceThickness;
    std::string exposureTime;
    std::string xRayTubeCurrent;
    std::string kvp;
    std::string windowCenter;
    std::string windowWidth;
    std::string contrastBolusAgent;
    std::string dataCollectionDiameter;
    std::string reconstructionDiameter;
    std::string distanceSourceToDetector;
    std::string distanceSourceToPatient;
    std::string gantryDetectorTilt;
    std::string tableHeight;
    std::string exposure;
    std::string focalSpots;
    std::string imagePositionPatient;
    std::string sliceLocation;
    std::string pixelSpacing;
    std::string rescaleIntercept;
    std::string rescaleSlope;
    std::string protocolName;
    std::string acquisitionData;
    std::string studyID;
    std::string seriesDescription;
    std::string seriesTime;
    std::string patientBirthDate;
    std::string filterType;
    std::string stationName;
    std::string studyTime;
    std::string acquisitionTime;
    std::string patientPosition;
    std::string  studyInstanceUID;
    std::string  seriesInstanceUID;
    std::string  acquisitionDate;
    std::string  seriesDate;
    std::string  modality;
    int validTags;
  };

  TAGS GetTagValues( std::string  );
  std::string GetTagValue( std::string, const DictionaryType &  );
  
  TAGS GetTagValues( std::string dicomDir )
  {
    TAGS tagValues;
    
    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    
    std::cout << dicomDir << std::endl;
    NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( dicomDir );
    
    std::vector< std::string > filenames = namesGenerator->GetInputFileNames();
        
    if ( filenames.size() == 0 )
      {
        tagValues.validTags = -1;
      }
    else
      {
        tagValues.validTags = 1;
        
        std::cout << "Reading dicom image..." << std::endl;
        Image2DReaderType::Pointer reader = Image2DReaderType::New();
          reader->SetFileName( filenames[0] );
          reader->SetImageIO( gdcmIO );
	try
	  {
          reader->Update();
	  }
        catch ( itk::ExceptionObject &excp )
	  {
          std::cerr << "Exception caught reading image:";
	  std::cerr << excp << std::endl;
	  }
            
        const DictionaryType & dictionary = gdcmIO->GetMetaDataDictionary();
        
        //  Define values for specific entries
        std::string patientNameEntryID             = "0010|0010";
        std::string patientIDEntryID               = "0010|0020";
        std::string studyDateEntryID               = "0008|0020";
        std::string institutionEntryID             = "0008|0080";
        std::string ctManufacturerEntryID          = "0008|0070";
        std::string ctModelEntryID                 = "0008|1090";
        std::string dateOfLastCalibrationEntryID   = "0018|1200";
        std::string convolutionKernelEntryID       = "0018|1210";
        std::string studyDescriptionEntryID        = "0008|1030";
        std::string modalitiesInStudyEntryID       = "0008|0061";
        std::string imageCommentsEntryID           = "0020|4000";
        std::string sliceThicknessEntryID          = "0018|0050";
        std::string exposureTimeEntryID            = "0018|1150";
        std::string xRayTubeCurrentEntryID         = "0018|1151";
        std::string kvpEntryID                     = "0018|0060";
        std::string windowCenterEntryID            = "0028|1050";
        std::string contrastBolusAgentID           = "0018|0010";
        std::string dataCollectionDiameterID       = "0018|0090";
        std::string reconstructionDiameterID       = "0018|1100";
        std::string distanceSourceToDetectorID     = "0018|1110";
        std::string distanceSourceToPatientID      = "0019|1111";
        std::string gantryDetectorTiltID           = "0018|1120";
        std::string tableHeightID                  = "0018|1130";
        std::string exposureID                     = "0018|1152";
        std::string focalSpotsID                   = "0018|1190"; // May have more than one value
        std::string imagePositionPatientID         = "0020|0032"; // Should have three values
        std::string sliceLocationID                = "0020|1041";
        std::string pixelSpacingID                 = "0028|0030";
        std::string rescaleInterceptID             = "0028|1052";
        std::string rescaleSlopeID                 = "0028|1053";
        std::string protocolNameID                 = "0018|1030";
        std::string acquisitionDataID              = "0008|0022";
        std::string studyIDID                      = "0020|0010";
        std::string seriesDescriptionID            = "0008|103e";
        std::string seriesTimeID                   = "0008|0031";
        std::string patientBirthDateID             = "0010|0030";
        std::string filterTypeID                   = "0018|1160";
        std::string stationNameID                  = "0008|1010";
        std::string studyTimeID                    = "0008|0020";
        std::string acquisitionTimeID              = "0008|0032";
        std::string patientPositionID              = "0018|5100";
        std::string studyInstanceUIDID             = "0020|000d";
        std::string seriesInstanceUIDID            = "0020|000e";
        std::string acquisitionDateID              = "0008|0022";
        std::string seriesDateID                   = "0008|0021";
        std::string modalityID                     = "0008|0060";
        
        tagValues.patientName              = GetTagValue( patientNameEntryID, dictionary );
        tagValues.patientID                = GetTagValue( patientIDEntryID, dictionary );
        tagValues.studyDate                = GetTagValue( studyDateEntryID, dictionary );
        tagValues.institution              = GetTagValue( institutionEntryID, dictionary );
        tagValues.ctManufacturer           = GetTagValue( ctManufacturerEntryID, dictionary );
        tagValues.ctModel                  = GetTagValue( ctModelEntryID, dictionary );
        tagValues.dateOfLastCalibration    = GetTagValue( dateOfLastCalibrationEntryID, dictionary );
        tagValues.convolutionKernel        = GetTagValue( convolutionKernelEntryID, dictionary );
        tagValues.studyDescription         = GetTagValue( studyDescriptionEntryID, dictionary );
        tagValues.modalitiesInStudy        = GetTagValue( modalitiesInStudyEntryID, dictionary );
        tagValues.imageComments            = GetTagValue( imageCommentsEntryID, dictionary );
        tagValues.sliceThickness           = GetTagValue( sliceThicknessEntryID, dictionary );
        tagValues.exposureTime             = GetTagValue( exposureTimeEntryID, dictionary );
        tagValues.xRayTubeCurrent          = GetTagValue( xRayTubeCurrentEntryID, dictionary );
        tagValues.kvp                      = GetTagValue( kvpEntryID, dictionary );
        tagValues.windowCenter             = GetTagValue( windowCenterEntryID, dictionary );
        tagValues.contrastBolusAgent       = GetTagValue( contrastBolusAgentID, dictionary );
        tagValues.dataCollectionDiameter   = GetTagValue( dataCollectionDiameterID, dictionary );
        tagValues.reconstructionDiameter   = GetTagValue( reconstructionDiameterID, dictionary );
        tagValues.distanceSourceToDetector = GetTagValue( distanceSourceToDetectorID, dictionary );
        tagValues.distanceSourceToPatient  = GetTagValue( distanceSourceToPatientID, dictionary );
        tagValues.gantryDetectorTilt       = GetTagValue( gantryDetectorTiltID, dictionary );
        tagValues.tableHeight              = GetTagValue( tableHeightID, dictionary );
        tagValues.exposure                 = GetTagValue( exposureID, dictionary );
        tagValues.focalSpots               = GetTagValue( focalSpotsID, dictionary );
        tagValues.imagePositionPatient     = GetTagValue( imagePositionPatientID, dictionary );
        tagValues.sliceLocation            = GetTagValue( sliceLocationID, dictionary );
        tagValues.pixelSpacing             = GetTagValue( pixelSpacingID, dictionary );
        tagValues.rescaleIntercept         = GetTagValue( rescaleInterceptID, dictionary );
        tagValues.rescaleSlope             = GetTagValue( rescaleSlopeID, dictionary );
        tagValues.protocolName             = GetTagValue( protocolNameID, dictionary );
        tagValues.acquisitionData          = GetTagValue( acquisitionDataID, dictionary );
        tagValues.studyID                  = GetTagValue( studyIDID, dictionary );
        tagValues.seriesDescription        = GetTagValue( seriesDescriptionID, dictionary );
        tagValues.seriesTime               = GetTagValue( seriesTimeID, dictionary );
        tagValues.patientBirthDate         = GetTagValue( patientBirthDateID, dictionary );
        tagValues.filterType               = GetTagValue( filterTypeID, dictionary );
        tagValues.stationName              = GetTagValue( stationNameID, dictionary );
        tagValues.studyTime                = GetTagValue( studyTimeID, dictionary );
        tagValues.acquisitionTime          = GetTagValue( acquisitionTimeID, dictionary );
        tagValues.patientPosition          = GetTagValue( patientPositionID, dictionary );
        tagValues.studyInstanceUID         = GetTagValue( studyInstanceUIDID, dictionary );
        tagValues.seriesInstanceUID        = GetTagValue( seriesInstanceUIDID, dictionary );
        tagValues.acquisitionDate          = GetTagValue( acquisitionDateID, dictionary );
        tagValues.seriesDate               = GetTagValue( seriesDateID, dictionary );
        tagValues.modality                 = GetTagValue( modalityID, dictionary );                    
      }
        
    return tagValues;
  }
  
    
  std::string GetTagValue( std::string entryID, const DictionaryType & dictionary )
  {
    std::string tagValue;
    
    DictionaryType::ConstIterator tagItr;
    DictionaryType::ConstIterator end = dictionary.End();
    
    tagItr = dictionary.Find( entryID );
    
    if ( tagItr == end )
      {
        std::cerr << "Tag " << entryID;
        std::cerr << " not found in the DICOM header. Returning blank entry." << std::endl;
        
        return tagValue;
      }
        
    MetaDataStringType::ConstPointer entryValue = dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer());
        
    tagValue = entryValue->GetMetaDataObjectValue();
        
    // Replace commas and new-lines with spaces
    size_t commaLocation = tagValue.find( ',' );
    
    //First check if comma exists
    if (commaLocation <tagValue.length())
      {
        if ( commaLocation != std::string::npos )
          {
            tagValue.replace( commaLocation, 1, " " );
	        }
        
        size_t newlineLocation = tagValue.find( '\n' );
        if ( newlineLocation != std::string::npos )
	        {
            if (newlineLocation<tagValue.length())
	            {
                tagValue.replace( newlineLocation, 1, " " );
	            }
	        }
      }
    
    return tagValue;
  }
    
  
  void WriteTagsToFile( std::string outputFileName, std::vector< std::string > directoryList, std::vector< TAGS > tagsVec )
  {
    std::ofstream csvFile( outputFileName.c_str() );
    
    csvFile << "Directory,patientName,patientID,studyDate,institution,ctManufacturer,ctModel,dateOfLastCalibration,";
    csvFile << "convolutionKernel,studyDescription,modalitiesInStudy,imageComments,sliceThickness,exposureTime,";
    csvFile << "xRayTubeCurrent,kvp,windowCenter,windowWidth,contrastBolusAgent,";
    csvFile << "dataCollectionDiameter,reconstructionDiameter,distanceSourceToDetector,distanceSourceToPatient,";
    csvFile << "gantryDetectorTilt,tableHeight,exposure,focalSpots,imagePositionPatient,sliceLocation,pixelSpacing,";
    csvFile << "rescaleIntercept,rescaleSlope, protocolName, acquisitionData,";
    csvFile << "studyID,seriesDescription,seriesTime,patientBirthDate,filterType,stationName,studyTime,acquisitionTime,";
    csvFile << "patientPosition,studyInstanceUID,seriesInstanceUID,acquisitionDate,seriesDate,modality" << std::endl;
        
    for ( unsigned int i=0; i<directoryList.size(); i++ )
      {
        csvFile << directoryList[i] << ",";
        
        if ( tagsVec[i].validTags == 1 )
	  {
            csvFile << tagsVec[i].patientName<< ",";
            csvFile << tagsVec[i].patientID << ",";
            csvFile << tagsVec[i].studyDate << ",";
            csvFile << tagsVec[i].institution << ",";
            csvFile << tagsVec[i].ctManufacturer << ",";
            csvFile << tagsVec[i].ctModel << ",";
            csvFile << tagsVec[i].dateOfLastCalibration << ",";
            csvFile << tagsVec[i].convolutionKernel << ",";
            csvFile << tagsVec[i].studyDescription << ",";
            csvFile << tagsVec[i].modalitiesInStudy << ",";
            csvFile << tagsVec[i].imageComments << ",";
            csvFile << tagsVec[i].sliceThickness << ",";
            csvFile << tagsVec[i].exposureTime << ",";
            csvFile << tagsVec[i].xRayTubeCurrent << ",";
            csvFile << tagsVec[i].kvp << ",";
            csvFile << tagsVec[i].windowCenter << ",";
            csvFile << tagsVec[i].windowWidth << ",";
            csvFile << tagsVec[i].contrastBolusAgent << ",";
            csvFile << tagsVec[i].dataCollectionDiameter << ",";
            csvFile << tagsVec[i].reconstructionDiameter << ",";
            csvFile << tagsVec[i].distanceSourceToDetector << ",";
            csvFile << tagsVec[i].distanceSourceToPatient << ",";
            csvFile << tagsVec[i].gantryDetectorTilt << ",";
            csvFile << tagsVec[i].tableHeight << ",";
            csvFile << tagsVec[i].exposure << ",";
            csvFile << tagsVec[i].focalSpots << ",";
            csvFile << tagsVec[i].imagePositionPatient << ",";
            csvFile << tagsVec[i].sliceLocation << ",";
            csvFile << tagsVec[i].pixelSpacing << ",";
            csvFile << tagsVec[i].rescaleIntercept << ",";
            csvFile << tagsVec[i].rescaleSlope << ",";
            csvFile << tagsVec[i].protocolName << ",";
            csvFile << tagsVec[i].acquisitionData << ",";
            csvFile << tagsVec[i].studyID << ",";
            csvFile << tagsVec[i].seriesDescription << ",";
            csvFile << tagsVec[i].seriesTime << ",";
            csvFile << tagsVec[i].patientBirthDate << ",";
            csvFile << tagsVec[i].filterType << ",";
            csvFile << tagsVec[i].stationName << ",";
            csvFile << tagsVec[i].studyTime << ",";
            csvFile << tagsVec[i].acquisitionTime << ",";
            csvFile << tagsVec[i].patientPosition << ",";
            csvFile << tagsVec[i].studyInstanceUID<<"," ;
            csvFile << tagsVec[i].seriesInstanceUID << ",";
            csvFile << tagsVec[i].acquisitionDate << ",";
            csvFile << tagsVec[i].seriesDate <<",";
            csvFile << tagsVec[i].modality << std::endl;
	  }
        else
	  {
            csvFile << "NA" << "," << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << "," << "NA" << "," << "NA" << ",";
            csvFile << "NA" << "," << "NA" << ",";
            csvFile << "NA" << "," << "NA" << ",";
            csvFile << "NA" << "," << "NA" << ",";
            csvFile << "NA" << "," << "NA" << "," << "NA" << ",";
            csvFile << "NA" << "," << "NA" << "," << "NA" << "," << std::endl;
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << ",";
            csvFile << "NA" << std::endl;
	  }
      }
    
    csvFile.close();
  }
      
  std::vector< std::string > GetDirectoryList( char* rootDicomDirectory )
  {
    std::vector< std::string > directoryList;
        
    // Write the list of dicom directories (inside the root directory)
    // to a file
    char lsDirectoryCommand[1024];
    sprintf( lsDirectoryCommand, "ls %s > directoryList.txt", rootDicomDirectory );
    system( lsDirectoryCommand );
    
    // Read in the file containing the list of dicom directories and
    // save the entries in a vector
    std::ifstream directoryListFile( "directoryList.txt" );
    
    while ( !directoryListFile.eof() )
      {
        char directory[1024];
        directoryListFile.getline( directory, 1024 );
        
        std::string directoryString( directory );
        
        directoryList.push_back( directoryString );
      }
    
    directoryListFile.close();
    
    // Remove the directory list file that was temporarily written
    system( "rm directoryList.txt" );
    
    return directoryList;
  }
  
} //end namespace

int main( int argc, char* argv[] )
{
  std::vector< std::string > directoryList;
  
  PARSE_ARGS;
  
  for ( unsigned int i=0; i<directoryListArg.size(); i++ )
    {
      directoryList.push_back( directoryListArg[i] );
    }

  // Loop through the directories and get the dicom tags for each
  // dicom dataset
  std::cout << "Getting tags for each dicom dataset..." << std::endl;
  std::vector< TAGS > tagsVec;
  
  for ( unsigned int i=0; i<directoryList.size(); i++ )
    {    
      TAGS tempTags = GetTagValues( directoryList[i] );
      
      tagsVec.push_back( tempTags );
    }
  
  std::cout << "Writing output file..." << std::endl;
  WriteTagsToFile( outputFileName, directoryList, tagsVec );
  
  std::cout << "DONE." << std::endl;
  
  return EXIT_SUCCESS;
}


