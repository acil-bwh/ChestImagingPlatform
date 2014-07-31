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

#ifndef __cipChestConventions_h
#define __cipChestConventions_h

#include <string>
#include <map>
#include <vector>
#include <cmath>
//#include <vnl/vnl_math.h>

#include <iostream>

namespace cip {

/**
 *  Note that chest regions are inherently hierarchical. If you add a
 *  region to the enumerated list below, you should also update the
 *  'ChestRegionHierarchyMap' described below.  Additionally, the
 *  ChestRegions should be updated in the constructor. Also need
 *  to update m_NumberOfEnumeratedChestRegions member variable and the
 *  'ChestRegionNames'. Also update 'ChestRegionColors' appropriately.
 */
enum ChestRegion { 
  UNDEFINEDREGION,        //0
  WHOLELUNG,              //1
  RIGHTLUNG,              //2
  LEFTLUNG,               //3
  RIGHTSUPERIORLOBE,      //4
  RIGHTMIDDLELOBE,        //5
  RIGHTINFERIORLOBE,      //6
  LEFTSUPERIORLOBE,       //7
  LEFTINFERIORLOBE,       //8
  LEFTUPPERTHIRD,         //9
  LEFTMIDDLETHIRD,        //10
  LEFTLOWERTHIRD,         //11
  RIGHTUPPERTHIRD,        //12
  RIGHTMIDDLETHIRD,       //13
  RIGHTLOWERTHIRD,        //14
  MEDIASTINUM,            //15
  WHOLEHEART,             //16
  AORTA,                  //17
  PULMONARYARTERY,        //18
  PULMONARYVEIN,          //19
  UPPERTHIRD,             //20
  MIDDLETHIRD,            //21
  LOWERTHIRD,             //22
  LEFT,                   //23
  RIGHT,                  //24
  LIVER,                  //25
  SPLEEN,                 //26
  ABDOMEN,                //27
  PARAVERTEBRAL,          //28
};


/**
 *  If you add a type to the enumerated list here, you should also
 *  update the ChestTypes below (in the class constructor). 
 *  Also need to update m_NumberOfEnumeratedChestTypes member variable
 *  and the 'ChestTypeNames' as well as 'ChestTypeColors'
 *
 *  Some notes about the types below. Segmental bronchi are considered 
 *  generation 3, sub-segmental are considered generation 4, etc.
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
  TRACHEA,                        //38
  MAINBRONCHUS,                   //39
  UPPERLOBEBRONCHUS,              //40
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
  VISCERALFAT,                    //71
  INTERMEDIATEBRONCHUS,           //72
  LOWERLOBEBRONCHUS,              //73
  SUPERIORDIVISIONBRONCHUS,       //74
  LINGULARBRONCHUS,               //75
  MIDDLELOBEBRONCHUS,             //76
  BRONCHIECTATICAIRWAY,           //77
  NONBRONCHIECTATICAIRWAY,        //78
  AMBIGUOUSBRONCHIECTATICAIRWAY,  //79
  MUSCLE,                         //80
  DIAPHRAGM,                      //81
};

enum ReturnCode {
  EXITSUCCESS,
  HELP,
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
  ~ChestConventions();
  ChestConventions();

  unsigned char GetNumberOfEnumeratedChestRegions() const;
  unsigned char GetNumberOfEnumeratedChestTypes() const;

  /** This method checks if the chest region 'subordinate' is within
   *  the chest region 'superior'. It assumes that all chest regions are
   *  within the WHOLELUNG lung region. TODO: extend do deal with
   *  chest, not just lung */
  bool CheckSubordinateSuperiorChestRegionRelationship( unsigned char subordinate, unsigned char superior );

  /** Given an unsigned short value, this method will compute the
   *  8-bit region value corresponding to the input */
  unsigned char GetChestRegionFromValue( unsigned short value ) const;

  /** The 'color' param is assumed to have three components, each in
   *  the interval [0,1]. All chest type colors will be tested until a
   *  color match is found. If no match is found, 'UNDEFINEDTYPYE'
   *  will be returned */  
  unsigned char GetChestTypeFromColor( double* color ) const;

  /** The 'color' param is assumed to have three components, each in
   *  the interval [0,1]. All chest region colors will be tested until a
   *  color match is found. If no match is found, 'UNDEFINEDTYPYE'
   *  will be returned */  
  unsigned char GetChestRegionFromColor(double* color) const;

  /** Given an unsigned short value, this method will compute the
   *  8-bit type value corresponding to the input */
  unsigned char GetChestTypeFromValue( unsigned short value ) const;
  
  /** A label map voxel value consists of a chest-region designation
   *  and a chest-type designation. For the purposes of representing a
   *  wild card entry (e.g. when using regions and types as keys for 
   *  populating a database), this method is provided. */
  std::string GetChestWildCardName() const;

  /** Given an unsigned char value corresponding to a chest type, this
   *  method will return the string name equivalent. */
  std::string GetChestTypeName( unsigned char whichType ) const;

  /** Get the chest type color. 'color' param is assumed to be an
   * allocated 3 dimensional double pointer */
  void GetChestTypeColor( unsigned char whichType, double* color ) const;

  /** Get the chest region color. 'color' param is assumed to be an
   * allocated 3 dimensional double pointer */
  void GetChestRegionColor(unsigned char whichRegion, double* color) const;

  /** Get the color corresponding to the chest-region chest-pair pair. The
   * color is computed as the average of the two corresponding region and type 
   * colors unless the region or type is undefined, in which case the color of
   * the defined region or type is returned. The 'color' param is assumed to be 
   * an allocated 3 dimensional double pointer */
  void GetColorFromChestRegionChestType(unsigned char whichRegion, unsigned char whichType, double* color) const;

  /** Given an unsigned char value corresponding to a chest region, this
   *  method will return the string name equivalent. */
  std::string GetChestRegionName( unsigned char whichRegion ) const;

  /** Given an unsigned short value, this method will return the
   *  string name of the corresponding chest region */
  std::string GetChestRegionNameFromValue( unsigned short value ) const;

  /** Given an unsigned short value, this method will return the
   *  string name of the corresponding chest type */
  std::string GetChestTypeNameFromValue( unsigned short value ) const;

  unsigned short GetValueFromChestRegionAndType( unsigned char region, unsigned char type ) const;

  /** Given a string identifying one of the enumerated chest regions,
   * this method will return the unsigned char equivalent. If no match
   * is found, the method will retune UNDEFINEDREGION */
  unsigned char GetChestRegionValueFromName( std::string regionString ) const;

  /** Given a string identifying one of the enumerated chest types,
   * this method will return the unsigned char equivalent. If no match
   * is found, the method will retune UNDEFINEDTYPE */
  unsigned char GetChestTypeValueFromName( std::string typeString ) const;

  /** Get the ith chest region */
  unsigned char GetChestRegion( unsigned int i ) const;

  /** Get the ith chest type */
  unsigned char GetChestType( unsigned int i ) const;


public:
  std::map< unsigned char, unsigned char >  ChestRegionHierarchyMap;
  std::vector< unsigned char >              ChestRegions;
  std::vector< unsigned char >              ChestTypes;
  std::vector< std::string >                ChestRegionNames;
  std::vector< std::string >                ChestTypeNames;
  std::vector< double* >                    ChestRegionColors;
  std::vector< double* >                    ChestTypeColors;

private:
  unsigned char m_NumberOfEnumeratedChestRegions;
  unsigned char m_NumberOfEnumeratedChestTypes;
};

} // namespace cip

#endif
