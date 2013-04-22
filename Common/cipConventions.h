/**
 *  \file cipConventions
 *  \ingroup common
 *  \brief This file contains CIP-specific enums within the cip
 *  namespace identifying chest regions, types, and exit codes for
 *  executables. Also defined in this file is the ChestConventions
 *  class which provides convenience methods for dealing with the
 *  chest region and type labels.
 *
 *  $Date: 2013-04-02 12:04:01 -0400 (Tue, 02 Apr 2013) $
 *  $Revision: 399 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#ifndef __itkcipConventions_h
#define __itkcipConventions_h

#include <string>
#include <map>
#include <vector>
#include <math.h>
#include <vnl/vnl_math.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

namespace cip {
/**
 *  Define typedefs used throughout the cip
 */
typedef itk::Image< unsigned short, 3 >       LabelMapType;
typedef itk::Image< short, 3 >                CTType;
typedef itk::ImageFileReader< LabelMapType >  LabelMapReaderType;
typedef itk::ImageFileWriter< LabelMapType >  LabelMapWriterType;
typedef itk::ImageFileReader< CTType >        CTReaderType;
typedef itk::ImageFileWriter< CTType >        CTWriterType;

/**
 *  Note that chest regions are inherently hierarchical.  If you add a
 *  region to the enumerated list below, you should also update the
 *  'ChestRegionHierarchyMap' described below.  Additionally, the
 *  ChestRegions should be updated in the constructor. Also need
 *  to update m_NumberOfEnumeratedChestRegions member variable and the
 *  'ChestRegionNames'.
 */
enum ChestRegion { 
  UNDEFINEDREGION,     //0
  WHOLELUNG,           //1
  RIGHTLUNG,           //2
  LEFTLUNG,            //3
  RIGHTSUPERIORLOBE,   //4
  RIGHTMIDDLELOBE,     //5
  RIGHTINFERIORLOBE,   //6
  LEFTSUPERIORLOBE,    //7
  LEFTINFERIORLOBE,    //8
  LEFTUPPERTHIRD,      //9
  LEFTMIDDLETHIRD,     //10
  LEFTLOWERTHIRD,      //11
  RIGHTUPPERTHIRD,     //12
  RIGHTMIDDLETHIRD,    //13
  RIGHTLOWERTHIRD,     //14
  MEDIASTINUM,         //15
  WHOLEHEART,          //16
  AORTA,               //17
  PULMONARYARTERY,     //18
  PULMONARYVEIN,       //19
  UPPERTHIRD,          //20
  MIDDLETHIRD,         //21
  LOWERTHIRD,          //22
  LEFT,                //23
  RIGHT,               //24
};


/**
 *  If you add a type to the enumerated list here, you should also
 *  update the ChestTypes below (in the class constructor). 
 *  Also need to update m_NumberOfEnumeratedChestTypes member variable
 *  and the 'ChestTypeNames' as well as 'ChestTypeColors'
 */
enum ChestType { 
  UNDEFINEDTYPE,                  //0 
  NORMALPARENCHYMA,               //1
  AIRWAY,                         //2
  VESSEL,                         //3
  EMPHYSEMATOUS,                  //4 
  GROUNDGLASS,                    //5
  RETICULAR,                      //6
  NODULAR,                        //7
  OBLIQUEFISSURE,                 //8
  HORIZONTALFISSURE,              //9
  MILDPARASEPTALEMPHYSEMA,        //10
  MODERATEPARASEPTALEMPHYSEMA,    //11
  SEVEREPARASEPTALEMPHYSEMA,      //12
  MILDBULLA,                      //13
  MODERATEBULLA,                  //14
  SEVEREBULLA,                    //15
  MILDCENTRILOBULAREMPHYSEMA,     //16
  MODERATECENTRILOBULAREMPHYSEMA, //17
  SEVERECENTRILOBULAREMPHYSEMA,   //18
  MILDPANLOBULAREMPHYSEMA,        //19
  MODERATEPANLOBULAREMPHYSEMA,    //20
  SEVEREPANLOBULAREMPHYSEMA,      //21
  AIRWAYWALLTHICKENING,           //22
  AIRWAYCYLINDRICALDILATION,      //23
  VARICOSEBRONCHIECTASIS,         //24
  CYSTICBRONCHIECTASIS,           //25
  CENTRILOBULARNODULE,            //26
  MOSAICING,                      //27
  EXPIRATORYMALACIA,              //28
  SABERSHEATH,                    //29
  OUTPOUCHING,                    //30
  MUCOIDMATERIAL,                 //31
  PATCHYGASTRAPPING,              //32
  DIFFUSEGASTRAPPING,             //33
  LINEARSCAR,                     //34
  CYST,                           //35
  ATELECTASIS,                    //36
  HONEYCOMBING,                   //37
  AIRWAYGENERATION0,              //38
  AIRWAYGENERATION1,              //39
  AIRWAYGENERATION2,              //40
  AIRWAYGENERATION3,              //41
  AIRWAYGENERATION4,              //42
  AIRWAYGENERATION5,              //43
  AIRWAYGENERATION6,              //44
  AIRWAYGENERATION7,              //45
  AIRWAYGENERATION8,              //46
  AIRWAYGENERATION9,              //47
  AIRWAYGENERATION10,             //48
  CALCIFICATION,                  //49
  ARTERY,                         //50
  VEIN,                           //51
  PECTORALISMINOR,                //52
  PECTORALISMAJOR,                //53
  ANTERIORSCALENE,                //54
  FISSURE,                        //55
  VESSELGENERATION0,              //56
  VESSELGENERATION1,              //57
  VESSELGENERATION2,              //58
  VESSELGENERATION3,              //59
  VESSELGENERATION4,              //60
  VESSELGENERATION5,              //61
  VESSELGENERATION6,              //62
  VESSELGENERATION7,              //63
  VESSELGENERATION8,              //64
  VESSELGENERATION9,              //65
  VESSELGENERATION10,             //66
  PARASEPTALEMPHYSEMA,            //67
  CENTRILOBULAREMPHYSEMA,         //68
  PANLOBULAREMPHYSEMA,            //69
  SUBCUTANEOUSFAT,                //70
};

enum ReturnCode {
  HELP,
  EXITSUCCESS,
  EXITFAILURE,
  RESAMPLEFAILURE,
  NRRDREADFAILURE,
  NRRDWRITEFAILURE,
  DICOMREADFAILURE,
  ATLASREADFAILURE,
  LABELMAPWRITEFAILURE,
  LABELMAPREADFAILURE,
  ARGUMENTPARSINGERROR,
  ATLASREGISTRATIONFAILURE,
  QUALITYCONTROLIMAGEWRITEFAILURE,
  INSUFFICIENTDATAFAILURE,
  GENERATEDISTANCEMAPFAILURE,
};

} // namespace cip

using namespace cip;
/**
 *  The following class will define the hierarchy among the various
 *  regions defined in 'ChestRegion' above.  If a new region is added
 *  to the enumerated list above, the class below should be updated
 *  as well to reflect the update.  'ChestRegionHierarchyMap' contains
 *  a mapping between all regions in the 'ChestRegion' enumerated list
 *  and the region directly above it in the hierarchy.
 */
class ChestConventions
{
public:
  // TODO: Check proper destructor syntax
  ~ChestConventions(){};
  ChestConventions()
    {
      m_NumberOfEnumeratedChestRegions = 25;
      m_NumberOfEnumeratedChestTypes   = 71;

      typedef std::pair< unsigned char, unsigned char > Region_Pair;

      //
      // For the hierarchical relationships, leftness and rightness
      // are respected before any relationship that transcends
      // leftness or rightness. For example left lower third maps to
      // left lung, not lower third, etc. The exception to this rule
      // is that both left and right lungs are subordinate to
      // WHOLELUNG, not LEFT and RIGHT
      //
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( LEFTSUPERIORLOBE ), 
                                                  static_cast< unsigned char >( LEFTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( LEFTINFERIORLOBE ), 
                                                  static_cast< unsigned char >( LEFTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( RIGHTSUPERIORLOBE ), 
                                                  static_cast< unsigned char >( RIGHTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( RIGHTMIDDLELOBE ), 
                                                  static_cast< unsigned char >( RIGHTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( RIGHTINFERIORLOBE ), 
                                                  static_cast< unsigned char >( RIGHTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( LEFTLUNG ), 
                                                  static_cast< unsigned char >( WHOLELUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( RIGHTLUNG ), 
                                                  static_cast< unsigned char >( WHOLELUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( LEFTUPPERTHIRD ), 
                                                  static_cast< unsigned char >( LEFTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( LEFTMIDDLETHIRD ), 
                                                  static_cast< unsigned char >( LEFTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( LEFTLOWERTHIRD ), 
                                                  static_cast< unsigned char >( LEFTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( RIGHTUPPERTHIRD ), 
                                                  static_cast< unsigned char >( RIGHTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( RIGHTMIDDLETHIRD ), 
                                                  static_cast< unsigned char >( RIGHTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( RIGHTLOWERTHIRD ), 
                                                  static_cast< unsigned char >( RIGHTLUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( WHOLELUNG ), 
                                                  static_cast< unsigned char >( UNDEFINEDREGION ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( MEDIASTINUM ), 
                                                  static_cast< unsigned char >( WHOLEHEART ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( MEDIASTINUM ), 
                                                  static_cast< unsigned char >( AORTA ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( MEDIASTINUM ), 
                                                  static_cast< unsigned char >( PULMONARYARTERY ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( LOWERTHIRD ), 
                                                  static_cast< unsigned char >( WHOLELUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( MIDDLETHIRD ), 
                                                  static_cast< unsigned char >( WHOLELUNG ) ) );
      ChestRegionHierarchyMap.insert( Region_Pair( static_cast< unsigned char >( UPPERTHIRD ), 
                                                  static_cast< unsigned char >( WHOLELUNG ) ) );

      ChestRegions.push_back( static_cast< unsigned char >( UNDEFINEDREGION ) );
      ChestRegions.push_back( static_cast< unsigned char >( WHOLELUNG ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHTLUNG ) );
      ChestRegions.push_back( static_cast< unsigned char >( LEFTLUNG ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHTSUPERIORLOBE ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHTMIDDLELOBE ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHTINFERIORLOBE ) );
      ChestRegions.push_back( static_cast< unsigned char >( LEFTSUPERIORLOBE ) );
      ChestRegions.push_back( static_cast< unsigned char >( LEFTINFERIORLOBE ) );
      ChestRegions.push_back( static_cast< unsigned char >( LEFTUPPERTHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( LEFTMIDDLETHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( LEFTLOWERTHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHTUPPERTHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHTMIDDLETHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHTLOWERTHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( MEDIASTINUM ) );
      ChestRegions.push_back( static_cast< unsigned char >( WHOLEHEART ) );
      ChestRegions.push_back( static_cast< unsigned char >( AORTA ) );
      ChestRegions.push_back( static_cast< unsigned char >( PULMONARYARTERY ) );
      ChestRegions.push_back( static_cast< unsigned char >( PULMONARYVEIN ) );
      ChestRegions.push_back( static_cast< unsigned char >( UPPERTHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( MIDDLETHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( LOWERTHIRD ) );
      ChestRegions.push_back( static_cast< unsigned char >( LEFT ) );
      ChestRegions.push_back( static_cast< unsigned char >( RIGHT ) );

      ChestTypes.push_back( static_cast< unsigned char >( UNDEFINEDTYPE ) );
      ChestTypes.push_back( static_cast< unsigned char >( NORMALPARENCHYMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAY ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSEL ) );
      ChestTypes.push_back( static_cast< unsigned char >( EMPHYSEMATOUS ) );
      ChestTypes.push_back( static_cast< unsigned char >( GROUNDGLASS ) );
      ChestTypes.push_back( static_cast< unsigned char >( RETICULAR ) );
      ChestTypes.push_back( static_cast< unsigned char >( NODULAR ) ); 
      ChestTypes.push_back( static_cast< unsigned char >( OBLIQUEFISSURE ) ); 
      ChestTypes.push_back( static_cast< unsigned char >( HORIZONTALFISSURE ) ); 
      ChestTypes.push_back( static_cast< unsigned char >( MILDPARASEPTALEMPHYSEMA ) ); 
      ChestTypes.push_back( static_cast< unsigned char >( MODERATEPARASEPTALEMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( SEVEREPARASEPTALEMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( MILDBULLA ) );
      ChestTypes.push_back( static_cast< unsigned char >( MODERATEBULLA ) );
      ChestTypes.push_back( static_cast< unsigned char >( SEVEREBULLA ) );
      ChestTypes.push_back( static_cast< unsigned char >( MILDCENTRILOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( MODERATECENTRILOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( SEVERECENTRILOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( MILDPANLOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( MODERATEPANLOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( SEVEREPANLOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYWALLTHICKENING ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYCYLINDRICALDILATION ) );
      ChestTypes.push_back( static_cast< unsigned char >( VARICOSEBRONCHIECTASIS ) );
      ChestTypes.push_back( static_cast< unsigned char >( CYSTICBRONCHIECTASIS ) );
      ChestTypes.push_back( static_cast< unsigned char >( CENTRILOBULARNODULE ) );
      ChestTypes.push_back( static_cast< unsigned char >( MOSAICING ) );
      ChestTypes.push_back( static_cast< unsigned char >( EXPIRATORYMALACIA ) );
      ChestTypes.push_back( static_cast< unsigned char >( SABERSHEATH ) );
      ChestTypes.push_back( static_cast< unsigned char >( OUTPOUCHING ) );
      ChestTypes.push_back( static_cast< unsigned char >( MUCOIDMATERIAL ) );
      ChestTypes.push_back( static_cast< unsigned char >( PATCHYGASTRAPPING ) );
      ChestTypes.push_back( static_cast< unsigned char >( DIFFUSEGASTRAPPING ) );
      ChestTypes.push_back( static_cast< unsigned char >( LINEARSCAR ) );
      ChestTypes.push_back( static_cast< unsigned char >( CYST ) );
      ChestTypes.push_back( static_cast< unsigned char >( ATELECTASIS ) );
      ChestTypes.push_back( static_cast< unsigned char >( HONEYCOMBING ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION0 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION1 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION2 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION3 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION4 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION5 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION6 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION7 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION8 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION9 ) );
      ChestTypes.push_back( static_cast< unsigned char >( AIRWAYGENERATION10 ) );
      ChestTypes.push_back( static_cast< unsigned char >( CALCIFICATION ) );
      ChestTypes.push_back( static_cast< unsigned char >( ARTERY ) );
      ChestTypes.push_back( static_cast< unsigned char >( VEIN ) );
      ChestTypes.push_back( static_cast< unsigned char >( PECTORALISMINOR ) );
      ChestTypes.push_back( static_cast< unsigned char >( PECTORALISMAJOR ) );
      ChestTypes.push_back( static_cast< unsigned char >( ANTERIORSCALENE ) );
      ChestTypes.push_back( static_cast< unsigned char >( FISSURE ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION0 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION1 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION2 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION3 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION4 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION5 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION6 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION7 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION8 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION9 ) );
      ChestTypes.push_back( static_cast< unsigned char >( VESSELGENERATION10 ) );
      ChestTypes.push_back( static_cast< unsigned char >( PARASEPTALEMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( CENTRILOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( PANLOBULAREMPHYSEMA ) );
      ChestTypes.push_back( static_cast< unsigned char >( SUBCUTANEOUSFAT ) );

      ChestRegionNames.push_back( "UNDEFINEDREGION" );
      ChestRegionNames.push_back( "WHOLELUNG" ); 
      ChestRegionNames.push_back( "RIGHTLUNG" ); 
      ChestRegionNames.push_back( "LEFTLUNG" ); 
      ChestRegionNames.push_back( "RIGHTSUPERIORLOBE" );
      ChestRegionNames.push_back( "RIGHTMIDDLELOBE" ); 
      ChestRegionNames.push_back( "RIGHTINFERIORLOBE" );
      ChestRegionNames.push_back( "LEFTSUPERIORLOBE" ); 
      ChestRegionNames.push_back( "LEFTINFERIORLOBE" );
      ChestRegionNames.push_back( "LEFTUPPERTHIRD" ); 
      ChestRegionNames.push_back( "LEFTMIDDLETHIRD" );
      ChestRegionNames.push_back( "LEFTLOWERTHIRD" ); 
      ChestRegionNames.push_back( "RIGHTUPPERTHIRD" );
      ChestRegionNames.push_back( "RIGHTMIDDLETHIRD" ); 
      ChestRegionNames.push_back( "RIGHTLOWERTHIRD" );
      ChestRegionNames.push_back( "MEDIASTINUM" );
      ChestRegionNames.push_back( "WHOLEHEART" );
      ChestRegionNames.push_back( "AORTA" );
      ChestRegionNames.push_back( "PULMONARYARTERY" );
      ChestRegionNames.push_back( "PULMONARYVEIN" );
      ChestRegionNames.push_back( "UPPERTHIRD" );
      ChestRegionNames.push_back( "MIDDLETHIRD" );
      ChestRegionNames.push_back( "LOWERTHIRD" );
      ChestRegionNames.push_back( "LEFT" );
      ChestRegionNames.push_back( "RIGHT" );

      ChestTypeNames.push_back( "UNDEFINEDTYPE" );
      ChestTypeNames.push_back( "NORMALPARENCHYMA" );
      ChestTypeNames.push_back( "AIRWAY" );
      ChestTypeNames.push_back( "VESSEL" );
      ChestTypeNames.push_back( "EMPHYSEMATOUS" );
      ChestTypeNames.push_back( "GROUNDGLASS" );
      ChestTypeNames.push_back( "RETICULAR" );
      ChestTypeNames.push_back( "NODULAR" ); 
      ChestTypeNames.push_back( "OBLIQUEFISSURE" ); 
      ChestTypeNames.push_back( "HORIZONTALFISSURE" ); 
      ChestTypeNames.push_back( "MILDPARASEPTALEMPHYSEMA" ); 
      ChestTypeNames.push_back( "MODERATEPARASEPTALEMPHYSEMA" );
      ChestTypeNames.push_back( "SEVEREPARASEPTALEMPHYSEMA" );
      ChestTypeNames.push_back( "MILDBULLA" );
      ChestTypeNames.push_back( "MODERATEBULLA" );
      ChestTypeNames.push_back( "SEVEREBULLA" );
      ChestTypeNames.push_back( "MILDCENTRILOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "MODERATECENTRILOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "SEVERECENTRILOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "MILDPANLOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "MODERATEPANLOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "SEVEREPANLOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "AIRWAYWALLTHICKENING" );
      ChestTypeNames.push_back( "AIRWAYCYLINDRICALDILATION" );
      ChestTypeNames.push_back( "VARICOSEBRONCHIECTASIS" );
      ChestTypeNames.push_back( "CYSTICBRONCHIECTASIS" );
      ChestTypeNames.push_back( "CENTRILOBULARNODULE" );
      ChestTypeNames.push_back( "MOSAICING" );
      ChestTypeNames.push_back( "EXPIRATORYMALACIA" );
      ChestTypeNames.push_back( "SABERSHEATH" );
      ChestTypeNames.push_back( "OUTPOUCHING" );
      ChestTypeNames.push_back( "MUCOIDMATERIAL" );
      ChestTypeNames.push_back( "PATCHYGASTRAPPING" );
      ChestTypeNames.push_back( "DIFFUSEGASTRAPPING" );
      ChestTypeNames.push_back( "LINEARSCAR" );
      ChestTypeNames.push_back( "CYST" );
      ChestTypeNames.push_back( "ATELECTASIS" );
      ChestTypeNames.push_back( "HONEYCOMBING" );
      ChestTypeNames.push_back( "AIRWAYGENERATION0" );
      ChestTypeNames.push_back( "AIRWAYGENERATION1" );
      ChestTypeNames.push_back( "AIRWAYGENERATION2" );
      ChestTypeNames.push_back( "AIRWAYGENERATION3" );
      ChestTypeNames.push_back( "AIRWAYGENERATION4" );
      ChestTypeNames.push_back( "AIRWAYGENERATION5" );
      ChestTypeNames.push_back( "AIRWAYGENERATION6" );
      ChestTypeNames.push_back( "AIRWAYGENERATION7" );
      ChestTypeNames.push_back( "AIRWAYGENERATION8" );
      ChestTypeNames.push_back( "AIRWAYGENERATION9" );
      ChestTypeNames.push_back( "AIRWAYGENERATION10" );
      ChestTypeNames.push_back( "CALCIFICATION" );
      ChestTypeNames.push_back( "ARTERY" );
      ChestTypeNames.push_back( "VEIN" );
      ChestTypeNames.push_back( "PECTORALISMINOR" );
      ChestTypeNames.push_back( "PECTORALISMAJOR" );
      ChestTypeNames.push_back( "ANTERIORSCALENE" );
      ChestTypeNames.push_back( "FISSURE" );
      ChestTypeNames.push_back( "VESSELGENERATION0" );
      ChestTypeNames.push_back( "VESSELGENERATION1" );
      ChestTypeNames.push_back( "VESSELGENERATION2" );
      ChestTypeNames.push_back( "VESSELGENERATION3" );
      ChestTypeNames.push_back( "VESSELGENERATION4" );
      ChestTypeNames.push_back( "VESSELGENERATION5" );
      ChestTypeNames.push_back( "VESSELGENERATION6" );
      ChestTypeNames.push_back( "VESSELGENERATION7" );
      ChestTypeNames.push_back( "VESSELGENERATION8" );
      ChestTypeNames.push_back( "VESSELGENERATION9" );
      ChestTypeNames.push_back( "VESSELGENERATION10" );
      ChestTypeNames.push_back( "PARASEPTALEMPHYSEMA" );
      ChestTypeNames.push_back( "CENTRILOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "PANLOBULAREMPHYSEMA" );
      ChestTypeNames.push_back( "SUBCUTANEOUSFAT" );

      //
      // Each type is associated with a color. This is generally
      // useful for the interactors for users, e.g. when manually
      // editing/labeling fissures, vessels, and airway partices. If
      // colors are reassigned here, they should be UNIQUE.
      //
      double* c001 = new double[3]; c001[0] = 1.00; c001[1] = 1.00; c001[2] = 1.00; ChestTypeColors.push_back( c001 ); //UNDEFINEDTYPE
      double* c002 = new double[3]; c002[0] = 0.99; c002[1] = 0.99; c002[2] = 0.99; ChestTypeColors.push_back( c002 ); //NORMALPARENCHYMA
      double* c003 = new double[3]; c003[0] = 0.98; c003[1] = 0.98; c003[2] = 0.98; ChestTypeColors.push_back( c003 ); //AIRWAY
      double* c004 = new double[3]; c004[0] = 0.97; c004[1] = 0.97; c004[2] = 0.97; ChestTypeColors.push_back( c004 ); //VESSEL
      double* c005 = new double[3]; c005[0] = 0.96; c005[1] = 0.96; c005[2] = 0.96; ChestTypeColors.push_back( c005 ); //EMPHYSEMATOUS
      double* c006 = new double[3]; c006[0] = 0.95; c006[1] = 0.95; c006[2] = 0.95; ChestTypeColors.push_back( c006 ); //GROUNDGLASS
      double* c007 = new double[3]; c007[0] = 0.94; c007[1] = 0.94; c007[2] = 0.94; ChestTypeColors.push_back( c007 ); //RETICULAR
      double* c008 = new double[3]; c008[0] = 0.93; c008[1] = 0.93; c008[2] = 0.93; ChestTypeColors.push_back( c008 ); //NODULAR
      double* c009 = new double[3]; c009[0] = 0.92; c009[1] = 0.92; c009[2] = 0.92; ChestTypeColors.push_back( c009 ); //OBLIQUEFISSURE
      double* c010 = new double[3]; c010[0] = 0.91; c010[1] = 0.91; c010[2] = 0.91; ChestTypeColors.push_back( c010 ); //HORIZONTALFISSURE
      double* c011 = new double[3]; c011[0] = 0.90; c011[1] = 0.90; c011[2] = 0.90; ChestTypeColors.push_back( c011 ); //MILDPARASEPTALEMPHYSEMA
      double* c012 = new double[3]; c012[0] = 0.89; c012[1] = 0.89; c012[2] = 0.89; ChestTypeColors.push_back( c012 ); //MODERATEPARASEPTALEMPHYSEMA
      double* c013 = new double[3]; c013[0] = 0.88; c013[1] = 0.88; c013[2] = 0.88; ChestTypeColors.push_back( c013 ); //SEVEREPARASEPTALEMPHYSEMA
      double* c014 = new double[3]; c014[0] = 0.87; c014[1] = 0.87; c014[2] = 0.87; ChestTypeColors.push_back( c014 ); //MILDBULLA
      double* c015 = new double[3]; c015[0] = 0.86; c015[1] = 0.86; c015[2] = 0.86; ChestTypeColors.push_back( c015 ); //MODERATEBULLA
      double* c016 = new double[3]; c016[0] = 0.85; c016[1] = 0.85; c016[2] = 0.85; ChestTypeColors.push_back( c016 ); //SEVEREBULLA
      double* c017 = new double[3]; c017[0] = 0.84; c017[1] = 0.84; c017[2] = 0.84; ChestTypeColors.push_back( c017 ); //MILDCENTRILOBULAREMPHYSEMA
      double* c018 = new double[3]; c018[0] = 0.83; c018[1] = 0.83; c018[2] = 0.83; ChestTypeColors.push_back( c018 ); //MODERATECENTRILOBULAREMPHYSEMA
      double* c019 = new double[3]; c019[0] = 0.82; c019[1] = 0.82; c019[2] = 0.82; ChestTypeColors.push_back( c019 ); //SEVERECENTRILOBULAREMPHYSEMA
      double* c020 = new double[3]; c020[0] = 0.81; c020[1] = 0.81; c020[2] = 0.81; ChestTypeColors.push_back( c020 ); //MILDPANLOBULAREMPHYSEMA
      double* c021 = new double[3]; c021[0] = 0.80; c021[1] = 0.70; c021[2] = 0.80; ChestTypeColors.push_back( c021 ); //MODERATEPANLOBULAREMPHYSEMA
      double* c022 = new double[3]; c022[0] = 0.79; c022[1] = 0.79; c022[2] = 0.79; ChestTypeColors.push_back( c022 ); //SEVEREPANLOBULAREMPHYSEMA
      double* c023 = new double[3]; c023[0] = 0.78; c023[1] = 0.78; c023[2] = 0.78; ChestTypeColors.push_back( c023 ); //AIRWAYWALLTHICKENING
      double* c024 = new double[3]; c024[0] = 0.77; c024[1] = 0.77; c024[2] = 0.77; ChestTypeColors.push_back( c024 ); //AIRWAYCYLINDRICALDILATION
      double* c025 = new double[3]; c025[0] = 0.76; c025[1] = 0.76; c025[2] = 0.76; ChestTypeColors.push_back( c025 ); //VARICOSEBRONCHIECTASIS
      double* c026 = new double[3]; c026[0] = 0.75; c026[1] = 0.75; c026[2] = 0.75; ChestTypeColors.push_back( c026 ); //CYSTICBRONCHIECTASIS
      double* c027 = new double[3]; c027[0] = 0.74; c027[1] = 0.74; c027[2] = 0.74; ChestTypeColors.push_back( c027 ); //CENTRILOBULARNODULE
      double* c028 = new double[3]; c028[0] = 0.73; c028[1] = 0.73; c028[2] = 0.73; ChestTypeColors.push_back( c028 ); //MOSAICING
      double* c029 = new double[3]; c029[0] = 0.72; c029[1] = 0.72; c029[2] = 0.72; ChestTypeColors.push_back( c029 ); //EXPIRATORYMALACIA
      double* c030 = new double[3]; c030[0] = 0.71; c030[1] = 0.71; c030[2] = 0.71; ChestTypeColors.push_back( c030 ); //SABERSHEATH
      double* c031 = new double[3]; c031[0] = 0.70; c031[1] = 0.70; c031[2] = 0.70; ChestTypeColors.push_back( c031 ); //OUTPOUCHING
      double* c032 = new double[3]; c032[0] = 0.69; c032[1] = 0.69; c032[2] = 0.69; ChestTypeColors.push_back( c032 ); //MUCOIDMATERIAL
      double* c033 = new double[3]; c033[0] = 0.68; c033[1] = 0.68; c033[2] = 0.68; ChestTypeColors.push_back( c033 ); //PATCHYGASTRAPPING
      double* c034 = new double[3]; c034[0] = 0.67; c034[1] = 0.67; c034[2] = 0.67; ChestTypeColors.push_back( c034 ); //DIFFUSEGASTRAPPING
      double* c035 = new double[3]; c035[0] = 0.66; c035[1] = 0.66; c035[2] = 0.66; ChestTypeColors.push_back( c035 ); //LINEARSCAR
      double* c036 = new double[3]; c036[0] = 0.65; c036[1] = 0.65; c036[2] = 0.65; ChestTypeColors.push_back( c036 ); //CYST
      double* c037 = new double[3]; c037[0] = 0.64; c037[1] = 0.64; c037[2] = 0.64; ChestTypeColors.push_back( c037 ); //ATELECTASIS
      double* c038 = new double[3]; c038[0] = 0.63; c038[1] = 0.63; c038[2] = 0.63; ChestTypeColors.push_back( c038 ); //HONEYCOMBING
      // The airway generation colors are identical to the vessel generation colors except that 0.01 has been
      // added to the red channel value to make these colors unique
      double* c039 = new double[3]; c039[0] = 0.01; c039[1] = 0.00; c039[2] = 0.00; ChestTypeColors.push_back( c039 ); //AIRWAYGENERATION0
      double* c040 = new double[3]; c040[0] = 0.01; c040[1] = 1.00; c040[2] = 0.00; ChestTypeColors.push_back( c040 ); //AIRWAYGENERATION1
      double* c041 = new double[3]; c041[0] = 0.01; c041[1] = 1.00; c041[2] = 1.00; ChestTypeColors.push_back( c041 ); //AIRWAYGENERATION2
      double* c042 = new double[3]; c042[0] = 1.01; c042[1] = 1.00; c042[2] = 0.00; ChestTypeColors.push_back( c042 ); //AIRWAYGENERATION3
      double* c043 = new double[3]; c043[0] = 1.01; c043[1] = 0.00; c043[2] = 1.00; ChestTypeColors.push_back( c043 ); //AIRWAYGENERATION4
      double* c044 = new double[3]; c044[0] = 0.51; c044[1] = 1.00; c044[2] = 0.00; ChestTypeColors.push_back( c044 ); //AIRWAYGENERATION5
      double* c045 = new double[3]; c045[0] = 0.01; c045[1] = 0.50; c045[2] = 1.00; ChestTypeColors.push_back( c045 ); //AIRWAYGENERATION6
      double* c046 = new double[3]; c046[0] = 0.51; c046[1] = 0.00; c046[2] = 0.50; ChestTypeColors.push_back( c046 ); //AIRWAYGENERATION7
      double* c047 = new double[3]; c047[0] = 0.51; c047[1] = 0.50; c047[2] = 0.00; ChestTypeColors.push_back( c047 ); //AIRWAYGENERATION8
      double* c048 = new double[3]; c048[0] = 0.01; c048[1] = 0.50; c048[2] = 0.50; ChestTypeColors.push_back( c048 ); //AIRWAYGENERATION9
      double* c049 = new double[3]; c049[0] = 0.45; c049[1] = 0.44; c049[2] = 0.44; ChestTypeColors.push_back( c049 ); //AIRWAYGENERATION10
      double* c050 = new double[3]; c050[0] = 0.51; c050[1] = 0.51; c050[2] = 0.51; ChestTypeColors.push_back( c050 ); //CALCIFICATION
      double* c051 = new double[3]; c051[0] = 0.40; c051[1] = 0.50; c051[2] = 0.50; ChestTypeColors.push_back( c051 ); //ARTERY
      double* c052 = new double[3]; c052[0] = 0.49; c052[1] = 0.49; c052[2] = 0.49; ChestTypeColors.push_back( c052 ); //VEIN
      double* c053 = new double[3]; c053[0] = 0.48; c053[1] = 0.48; c053[2] = 0.48; ChestTypeColors.push_back( c053 ); //PECTORALISMINOR
      double* c054 = new double[3]; c054[0] = 0.47; c054[1] = 0.47; c054[2] = 0.47; ChestTypeColors.push_back( c054 ); //PECTORALISMAJOR
      double* c055 = new double[3]; c055[0] = 0.46; c055[1] = 0.46; c055[2] = 0.46; ChestTypeColors.push_back( c055 ); //ANTERIORSCALENE
      double* c056 = new double[3]; c056[0] = 0.45; c056[1] = 0.45; c056[2] = 0.45; ChestTypeColors.push_back( c056 ); //FISSURE     
      // The vessel generation colors are identical to the airway generation colors except that the red chanel
      // is 0.01 less than the airway generation red channel. This ensures that the colors are unique
      double* c057 = new double[3]; c057[0] = 0.00; c057[1] = 0.00; c057[2] = 0.00; ChestTypeColors.push_back( c057 ); //VESSELGENERATION0
      double* c058 = new double[3]; c058[0] = 0.00; c058[1] = 1.00; c058[2] = 0.00; ChestTypeColors.push_back( c058 ); //VESSELGENERATION1
      double* c059 = new double[3]; c059[0] = 0.00; c059[1] = 1.00; c059[2] = 1.00; ChestTypeColors.push_back( c059 ); //VESSELGENERATION2
      double* c060 = new double[3]; c060[0] = 1.00; c060[1] = 1.00; c060[2] = 0.00; ChestTypeColors.push_back( c060 ); //VESSELGENERATION3
      double* c061 = new double[3]; c061[0] = 1.00; c061[1] = 0.00; c061[2] = 1.00; ChestTypeColors.push_back( c061 ); //VESSELGENERATION4
      double* c062 = new double[3]; c062[0] = 0.50; c062[1] = 1.00; c062[2] = 0.00; ChestTypeColors.push_back( c062 ); //VESSELGENERATION5
      double* c063 = new double[3]; c063[0] = 0.00; c063[1] = 0.50; c063[2] = 1.00; ChestTypeColors.push_back( c063 ); //VESSELGENERATION6
      double* c064 = new double[3]; c064[0] = 0.50; c064[1] = 0.00; c064[2] = 0.50; ChestTypeColors.push_back( c064 ); //VESSELGENERATION7
      double* c065 = new double[3]; c065[0] = 0.50; c065[1] = 0.50; c065[2] = 0.00; ChestTypeColors.push_back( c065 ); //VESSELGENERATION8
      double* c066 = new double[3]; c066[0] = 0.00; c066[1] = 0.50; c066[2] = 0.50; ChestTypeColors.push_back( c066 ); //VESSELGENERATION9
      double* c067 = new double[3]; c067[0] = 0.44; c067[1] = 0.44; c067[2] = 0.44; ChestTypeColors.push_back( c067 ); //VESSELGENERATION10

      double* c068 = new double[3]; c068[0] = 0.00; c068[1] = 0.68; c068[2] = 0.00; ChestTypeColors.push_back( c068 ); //PARASEPTALEMPHYSEMA
      double* c069 = new double[3]; c069[0] = 0.00; c069[1] = 0.69; c069[2] = 0.69; ChestTypeColors.push_back( c069 ); //CENTRILOBULAREMPHYSEMA
      double* c070 = new double[3]; c070[0] = 0.00; c070[1] = 0.00; c070[2] = 0.70; ChestTypeColors.push_back( c070 ); //PANLOBULAREMPHYSEMA

      double* c071 = new double[3]; c071[0] = 1.00; c071[1] = 0.60; c071[2] = 0.00; ChestTypeColors.push_back( c071 ); //SUBCUTANEOUSFAT
    }
  unsigned char GetNumberOfEnumeratedChestRegions() const
    {
      return m_NumberOfEnumeratedChestRegions;
    };

  unsigned char GetNumberOfEnumeratedChestTypes() const
    {
      return m_NumberOfEnumeratedChestTypes;
    };

  /** This method checks if the chest region 'subordinate' is within
   *  the chest region 'superior'. It assumes that all chest regions are
   *  within the WHOLELUNG lung region. TODO: extend do deal with
   *  chest, not just lung */
  bool CheckSubordinateSuperiorChestRegionRelationship( unsigned char subordinate, unsigned char superior )
    {
      //
      // No matter what the superior and subordinate regions are (even
      // if they are undefined regions), if they are the same then by
      // convention the subordinate is a subset of the superior, so
      // return true
      //
      if ( subordinate == superior )
        {
        return true;
        }

      //
      // The undefined region does not belong to any other
      // region (except the undefined region itself). Similarly,
      // nothing belongs to the undefined region (except the undefined
      // region). So if the above test failed, then we're considering
      // the relationship between a defined region and and undefined
      // region. Therefore return false.
      //
      if ( subordinate == static_cast< unsigned char >( UNDEFINEDREGION ) ||
           superior == static_cast< unsigned char >( UNDEFINEDREGION ) )
        {
        return false;
        }

      if ( superior == static_cast< unsigned char >( WHOLELUNG ) )
        {
        return true;
        }

      unsigned char subordinateTemp = subordinate;

      while ( subordinateTemp != static_cast< unsigned char >( WHOLELUNG ) && 
              subordinateTemp != static_cast< unsigned char >( UNDEFINEDREGION ) )
        {
        if ( ChestRegionHierarchyMap[subordinateTemp] == superior )
          {
          return true;
          }
        else
          {
          subordinateTemp = ChestRegionHierarchyMap[subordinateTemp];
          }
        }

      return false;
    }

  /** Given an unsigned short value, this method will compute the
   *  8-bit region value corresponding to the input */
  unsigned char GetChestRegionFromValue( unsigned short value ) const
    {
      unsigned char regionValue = 0;

      for ( int i=15; i>=0; i-- )
        {
	  int power = static_cast< int >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );
        
        if ( power <= value )
          {
          if ( i < 8 )
            {
            regionValue += power;
            }
      
          value = value % power;
          }
        }
      
      return regionValue;
    };

  /** The 'color' param is assumed to have three components, each in
   *  the interval [0,1]. All chest type colors will be tested until a
   *  color match is found. If no match is found, 'UNDEFINEDTYPYE'
   *  will be returned */  
  unsigned char GetChestTypeFromColor( double* color ) const
    {
      for ( unsigned int i=0; i<m_NumberOfEnumeratedChestTypes; i++ )
        {
        if ( ChestTypeColors[i][0] == color[0] && ChestTypeColors[i][1] == color[1] && 
             ChestTypeColors[i][2] == color[2] )          
          {
          return static_cast< unsigned char >( i );
          }
        }
      return static_cast< unsigned char >( UNDEFINEDTYPE );
    }

  /** Given an unsigned short value, this method will compute the
   *  8-bit type value corresponding to the input */
  unsigned char GetChestTypeFromValue( unsigned short value ) const
    {
      unsigned char typeValue = 0;

      for ( int i=15; i>=0; i-- )
        {
	  int power = static_cast< int >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );
        
        if ( power <= value )
          {
          if ( i >= 8 )
            {
	      typeValue += static_cast< unsigned char >( vcl_pow( static_cast< float >(2), static_cast< float >(i-8) ) );
            }
          
          value = value % power;
          }
        }

      return typeValue;
    };

  /** Given an unsigned char value corresponding to a chest type, this
   *  method will return the string name equivalent. */
  std::string GetChestTypeName( unsigned char whichType ) const
    {
      if ( static_cast< int >( whichType ) > m_NumberOfEnumeratedChestTypes-1 )
        {
        return "UNDEFINEDTYPE";
        }

      return ChestTypeNames[static_cast< int >( whichType )];
    }


  /** Get the chest type color. 'color' param is assumed to be an
   * allocated 3 dimensional double pointer */
  void GetChestTypeColor( unsigned char whichType, double* color ) const
    {
      color[0] = ChestTypeColors[static_cast< int >( whichType )][0];
      color[1] = ChestTypeColors[static_cast< int >( whichType )][1];
      color[2] = ChestTypeColors[static_cast< int >( whichType )][2];
    }

  /** Given an unsigned char value corresponding to a chest region, this
   *  method will return the string name equivalent. */
  std::string GetChestRegionName( unsigned char whichRegion ) const
    {
      if ( static_cast< int >( whichRegion ) > m_NumberOfEnumeratedChestRegions-1 )
        {
        return "UNDEFINEDREGION";
        }

      return ChestRegionNames[static_cast< int >( whichRegion )];
    }

  /** Given an unsigned short value, this method will return the
   *  string name of the corresponding chest region */
  std::string GetChestRegionNameFromValue( unsigned short value ) const
    {
      unsigned char regionValue = 0;

      for ( int i=15; i>=0; i-- )
        {
	  int power = static_cast< int >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );
        
        if ( power <= value )
          {
          if ( i < 8 )
            {
            regionValue += power;
            }
      
          value = value % power;
          }
        }
      
      return ChestRegionNames[static_cast< int >( regionValue )];
    };

  /** Given an unsigned short value, this method will return the
   *  string name of the corresponding chest type */
  std::string GetChestTypeNameFromValue( unsigned short value ) const
    {
      unsigned char typeValue = 0;

      for ( int i=15; i>=0; i-- )
        {
	  int power = static_cast< int >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );
        
        if ( power <= value )
          {
          if ( i >= 8 )
            {
	      typeValue += static_cast< unsigned char >( vcl_pow( static_cast< float >(2), static_cast< float >(i-8) ) );
            }
          
          value = value % power;
          }
        }

      return ChestTypeNames[static_cast< int >( typeValue )];
    };

  unsigned short GetValueFromChestRegionAndType( unsigned char region, unsigned char type ) const
    {
      //
      // Get the binary representation of the region to set
      //
      int regionValue = static_cast< int >( region );

      int regionPlaces[8];
      for ( int i=0; i<8; i++ )
        {
        regionPlaces[i] = 0;
        }

      for ( int i=7; i>=0; i-- )
        {
	  int power = static_cast< int >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );

        if ( power <= regionValue )
          {
          regionPlaces[i] = 1;
          
          regionValue = regionValue % power;
          }
        }
 
      //
      // Get the binary representation of the type to set
      //
      int typeValue = static_cast< int >( type );

      int typePlaces[8];
      for ( int i=0; i<8; i++ )
        {
        typePlaces[i] = 0;
        }

      for ( int i=7; i>=0; i-- )
        {
	  int power = static_cast< int >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );

        if ( power <= typeValue )
          {
          typePlaces[i] = 1;
          
          typeValue = typeValue % power;
          }
        }

      //
      // Compute the new value to assign to the label map voxel 
      //
      unsigned short combinedValue = 0;

      for ( int i=0; i<16; i++ )
        {
        if ( i < 8 )
          {
	    combinedValue += static_cast< unsigned short >( regionPlaces[i] )*static_cast< unsigned short >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );
          }
        else
          {
	    combinedValue += static_cast< unsigned short >( typePlaces[i-8] )*static_cast< unsigned short >( vcl_pow( static_cast< float >(2), static_cast< float >(i) ) );
          }
        }

      return combinedValue;
    };

  /** Given a string identifying one of the enumerated chest regions,
   * this method will return the unsigned char equivalent. If no match
   * is found, the method will retune UNDEFINEDREGION */
  unsigned char GetChestRegionValueFromName( std::string regionString ) const
    {
      for ( int i=0; i<m_NumberOfEnumeratedChestRegions; i++ )
        {
        if ( !regionString.compare( ChestRegionNames[i] ) )
          {
          return ChestRegions[i];
          }
        }
      
      return static_cast< unsigned char >( UNDEFINEDREGION );
    }

  /** Given a string identifying one of the enumerated chest types,
   * this method will return the unsigned char equivalent. If no match
   * is found, the method will retune UNDEFINEDTYPE */
  unsigned char GetChestTypeValueFromName( std::string typeString ) const
    {
      for ( int i=0; i<m_NumberOfEnumeratedChestTypes; i++ )
        {
        if ( !typeString.compare( ChestTypeNames[i] ) )
          {
          return ChestTypes[i];
          }
        }
      
      return static_cast< unsigned char >( UNDEFINEDTYPE );
    }

  /** Get the ith chest region */
  unsigned char GetChestRegion( unsigned int i ) const
    {
      return static_cast< unsigned char >( ChestRegions[i] );
    }

  /** Get the ith chest type */
  unsigned char GetChestType( unsigned int i ) const
    {
      return static_cast< unsigned char >( ChestTypes[i] );
    }


public:
  std::map< unsigned char, unsigned char >  ChestRegionHierarchyMap;
  std::vector< unsigned char >              ChestRegions;
  std::vector< unsigned char >              ChestTypes;
  std::vector< std::string >                ChestRegionNames;
  std::vector< std::string >                ChestTypeNames;
  std::vector< double* >                    ChestTypeColors;

private:
  unsigned char m_NumberOfEnumeratedChestRegions;
  unsigned char m_NumberOfEnumeratedChestTypes;
};

#endif
