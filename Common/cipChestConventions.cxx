#include "cipChestConventions.h"

cip::ChestConventions::ChestConventions()
{
  m_NumberOfEnumeratedChestRegions = 29;
  m_NumberOfEnumeratedChestTypes   = 82;

  typedef std::pair< unsigned char, unsigned char > Region_Pair;

  // For the hierarchical relationships, leftness and rightness
  // are respected before any relationship that transcends
  // leftness or rightness. For example left lower third maps to
  // left lung, not lower third, etc. The exception to this rule
  // is that both left and right lungs are subordinate to
  // WHOLELUNG, not LEFT and RIGHT
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( LEFTSUPERIORLOBE ), 
					       (unsigned char)( LEFTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( LEFTINFERIORLOBE ), 
					       (unsigned char)( LEFTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( RIGHTSUPERIORLOBE ), 
					       (unsigned char)( RIGHTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( RIGHTMIDDLELOBE ), 
					       (unsigned char)( RIGHTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( RIGHTINFERIORLOBE ), 
					       (unsigned char)( RIGHTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( LEFTLUNG ), 
					       (unsigned char)( WHOLELUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( RIGHTLUNG ), 
					       (unsigned char)( WHOLELUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( LEFTUPPERTHIRD ), 
					       (unsigned char)( LEFTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( LEFTMIDDLETHIRD ), 
					       (unsigned char)( LEFTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( LEFTLOWERTHIRD ), 
					       (unsigned char)( LEFTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( RIGHTUPPERTHIRD ), 
					       (unsigned char)( RIGHTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( RIGHTMIDDLETHIRD ), 
					       (unsigned char)( RIGHTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( RIGHTLOWERTHIRD ), 
					       (unsigned char)( RIGHTLUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( WHOLELUNG ), 
					       (unsigned char)( UNDEFINEDREGION ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( MEDIASTINUM ), 
					       (unsigned char)( WHOLEHEART ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( MEDIASTINUM ), 
					       (unsigned char)( AORTA ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( MEDIASTINUM ), 
					       (unsigned char)( PULMONARYARTERY ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( LOWERTHIRD ), 
					       (unsigned char)( WHOLELUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( MIDDLETHIRD ), 
					       (unsigned char)( WHOLELUNG ) ) );
  ChestRegionHierarchyMap.insert( Region_Pair( (unsigned char)( UPPERTHIRD ), 
					       (unsigned char)( WHOLELUNG ) ) );
  
  ChestRegions.push_back( (unsigned char)( UNDEFINEDREGION ) );
  ChestRegions.push_back( (unsigned char)( WHOLELUNG ) );
  ChestRegions.push_back( (unsigned char)( RIGHTLUNG ) );
  ChestRegions.push_back( (unsigned char)( LEFTLUNG ) );
  ChestRegions.push_back( (unsigned char)( RIGHTSUPERIORLOBE ) );
  ChestRegions.push_back( (unsigned char)( RIGHTMIDDLELOBE ) );
  ChestRegions.push_back( (unsigned char)( RIGHTINFERIORLOBE ) );
  ChestRegions.push_back( (unsigned char)( LEFTSUPERIORLOBE ) );
  ChestRegions.push_back( (unsigned char)( LEFTINFERIORLOBE ) );
  ChestRegions.push_back( (unsigned char)( LEFTUPPERTHIRD ) );
  ChestRegions.push_back( (unsigned char)( LEFTMIDDLETHIRD ) );
  ChestRegions.push_back( (unsigned char)( LEFTLOWERTHIRD ) );
  ChestRegions.push_back( (unsigned char)( RIGHTUPPERTHIRD ) );
  ChestRegions.push_back( (unsigned char)( RIGHTMIDDLETHIRD ) );
  ChestRegions.push_back( (unsigned char)( RIGHTLOWERTHIRD ) );
  ChestRegions.push_back( (unsigned char)( MEDIASTINUM ) );
  ChestRegions.push_back( (unsigned char)( WHOLEHEART ) );
  ChestRegions.push_back( (unsigned char)( AORTA ) );
  ChestRegions.push_back( (unsigned char)( PULMONARYARTERY ) );
  ChestRegions.push_back( (unsigned char)( PULMONARYVEIN ) );
  ChestRegions.push_back( (unsigned char)( UPPERTHIRD ) );
  ChestRegions.push_back( (unsigned char)( MIDDLETHIRD ) );
  ChestRegions.push_back( (unsigned char)( LOWERTHIRD ) );
  ChestRegions.push_back( (unsigned char)( LEFT ) );
  ChestRegions.push_back( (unsigned char)( RIGHT ) );
  ChestRegions.push_back( (unsigned char)( LIVER ) );
  ChestRegions.push_back( (unsigned char)( SPLEEN ) );
  ChestRegions.push_back( (unsigned char)( ABDOMEN ) );
  ChestRegions.push_back( (unsigned char)( PARAVERTEBRAL ) );
  
  ChestTypes.push_back( (unsigned char)( UNDEFINEDTYPE ) );
  ChestTypes.push_back( (unsigned char)( NORMALPARENCHYMA ) );
  ChestTypes.push_back( (unsigned char)( AIRWAY ) );
  ChestTypes.push_back( (unsigned char)( VESSEL ) );
  ChestTypes.push_back( (unsigned char)( EMPHYSEMATOUS ) );
  ChestTypes.push_back( (unsigned char)( GROUNDGLASS ) );
  ChestTypes.push_back( (unsigned char)( RETICULAR ) );
  ChestTypes.push_back( (unsigned char)( NODULAR ) ); 
  ChestTypes.push_back( (unsigned char)( OBLIQUEFISSURE ) ); 
  ChestTypes.push_back( (unsigned char)( HORIZONTALFISSURE ) ); 
  ChestTypes.push_back( (unsigned char)( MILDPARASEPTALEMPHYSEMA ) ); 
  ChestTypes.push_back( (unsigned char)( MODERATEPARASEPTALEMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( SEVEREPARASEPTALEMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( MILDBULLA ) );
  ChestTypes.push_back( (unsigned char)( MODERATEBULLA ) );
  ChestTypes.push_back( (unsigned char)( SEVEREBULLA ) );
  ChestTypes.push_back( (unsigned char)( MILDCENTRILOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( MODERATECENTRILOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( SEVERECENTRILOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( MILDPANLOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( MODERATEPANLOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( SEVEREPANLOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYWALLTHICKENING ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYCYLINDRICALDILATION ) );
  ChestTypes.push_back( (unsigned char)( VARICOSEBRONCHIECTASIS ) );
  ChestTypes.push_back( (unsigned char)( CYSTICBRONCHIECTASIS ) );
  ChestTypes.push_back( (unsigned char)( CENTRILOBULARNODULE ) );
  ChestTypes.push_back( (unsigned char)( MOSAICING ) );
  ChestTypes.push_back( (unsigned char)( EXPIRATORYMALACIA ) );
  ChestTypes.push_back( (unsigned char)( SABERSHEATH ) );
  ChestTypes.push_back( (unsigned char)( OUTPOUCHING ) );
  ChestTypes.push_back( (unsigned char)( MUCOIDMATERIAL ) );
  ChestTypes.push_back( (unsigned char)( PATCHYGASTRAPPING ) );
  ChestTypes.push_back( (unsigned char)( DIFFUSEGASTRAPPING ) );
  ChestTypes.push_back( (unsigned char)( LINEARSCAR ) );
  ChestTypes.push_back( (unsigned char)( CYST ) );
  ChestTypes.push_back( (unsigned char)( ATELECTASIS ) );
  ChestTypes.push_back( (unsigned char)( HONEYCOMBING ) );
  ChestTypes.push_back( (unsigned char)( TRACHEA ) );
  ChestTypes.push_back( (unsigned char)( MAINBRONCHUS ) );
  ChestTypes.push_back( (unsigned char)( UPPERLOBEBRONCHUS ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION3 ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION4 ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION5 ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION6 ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION7 ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION8 ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION9 ) );
  ChestTypes.push_back( (unsigned char)( AIRWAYGENERATION10 ) );
  ChestTypes.push_back( (unsigned char)( CALCIFICATION ) );
  ChestTypes.push_back( (unsigned char)( ARTERY ) );
  ChestTypes.push_back( (unsigned char)( VEIN ) );
  ChestTypes.push_back( (unsigned char)( PECTORALISMINOR ) );
  ChestTypes.push_back( (unsigned char)( PECTORALISMAJOR ) );
  ChestTypes.push_back( (unsigned char)( ANTERIORSCALENE ) );
  ChestTypes.push_back( (unsigned char)( FISSURE ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION0 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION1 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION2 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION3 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION4 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION5 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION6 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION7 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION8 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION9 ) );
  ChestTypes.push_back( (unsigned char)( VESSELGENERATION10 ) );
  ChestTypes.push_back( (unsigned char)( PARASEPTALEMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( CENTRILOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( PANLOBULAREMPHYSEMA ) );
  ChestTypes.push_back( (unsigned char)( SUBCUTANEOUSFAT ) );
  ChestTypes.push_back( (unsigned char)( VISCERALFAT ) );
  ChestTypes.push_back( (unsigned char)( INTERMEDIATEBRONCHUS ) );
  ChestTypes.push_back( (unsigned char)( LOWERLOBEBRONCHUS ) );
  ChestTypes.push_back( (unsigned char)( SUPERIORDIVISIONBRONCHUS ) );
  ChestTypes.push_back( (unsigned char)( LINGULARBRONCHUS ) );
  ChestTypes.push_back( (unsigned char)( MIDDLELOBEBRONCHUS ) );
  ChestTypes.push_back( (unsigned char)( BRONCHIECTATICAIRWAY ) );
  ChestTypes.push_back( (unsigned char)( NONBRONCHIECTATICAIRWAY ) );
  ChestTypes.push_back( (unsigned char)( AMBIGUOUSBRONCHIECTATICAIRWAY ) );
  ChestTypes.push_back( (unsigned char)( MUSCLE ) );
  ChestTypes.push_back( (unsigned char)( DIAPHRAGM ) );
  
  ChestRegionNames.push_back( "UndefinedRegion" );
  ChestRegionNames.push_back( "WholeLung" ); 
  ChestRegionNames.push_back( "RightLung" ); 
  ChestRegionNames.push_back( "LeftLung" ); 
  ChestRegionNames.push_back( "RightSuperiorLobe" );
  ChestRegionNames.push_back( "RightMiddleLobe" ); 
  ChestRegionNames.push_back( "RightInferiorLobe" );
  ChestRegionNames.push_back( "LeftSuperiorLobe" ); 
  ChestRegionNames.push_back( "LeftInferiorLobe" );
  ChestRegionNames.push_back( "LeftUpperThird" ); 
  ChestRegionNames.push_back( "LeftMiddleThird" );
  ChestRegionNames.push_back( "LeftLowerThird" ); 
  ChestRegionNames.push_back( "RightUpperThird" );
  ChestRegionNames.push_back( "RightMiddleThird" ); 
  ChestRegionNames.push_back( "RightLowerThird" );
  ChestRegionNames.push_back( "Mediastinum" );
  ChestRegionNames.push_back( "WholeHeart" );
  ChestRegionNames.push_back( "Aorta" );
  ChestRegionNames.push_back( "PulmonaryArtery" );
  ChestRegionNames.push_back( "PulmonaryVein" );
  ChestRegionNames.push_back( "UpperThird" );
  ChestRegionNames.push_back( "MiddleThird" );
  ChestRegionNames.push_back( "LowerThird" );
  ChestRegionNames.push_back( "Left" );
  ChestRegionNames.push_back( "Right" );
  ChestRegionNames.push_back( "Liver" );
  ChestRegionNames.push_back( "Spleen" );
  ChestRegionNames.push_back( "Abdomen" );
  ChestRegionNames.push_back( "Paravertebral" );
  
  ChestTypeNames.push_back( "UndefinedType" );
  ChestTypeNames.push_back( "NormalParenchyma" );
  ChestTypeNames.push_back( "Airway" );
  ChestTypeNames.push_back( "Vessel" );
  ChestTypeNames.push_back( "Emphysematous" );
  ChestTypeNames.push_back( "GroundGlass" );
  ChestTypeNames.push_back( "Reticular" );
  ChestTypeNames.push_back( "Nodular" ); 
  ChestTypeNames.push_back( "ObliqueFissure" ); 
  ChestTypeNames.push_back( "HorizontalFissure" ); 
  ChestTypeNames.push_back( "MildParaseptalEmphysema" ); 
  ChestTypeNames.push_back( "ModerateParaseptalEmphysema" );
  ChestTypeNames.push_back( "SevereParaseptalEmphysema" );
  ChestTypeNames.push_back( "MildBulla" );
  ChestTypeNames.push_back( "ModerateBulla" );
  ChestTypeNames.push_back( "SevereBulla" );
  ChestTypeNames.push_back( "MildCentrilobularEmphysema" );
  ChestTypeNames.push_back( "ModerateCentrilobularEmphysema" );
  ChestTypeNames.push_back( "SevereCentrilobularEmphysema" );
  ChestTypeNames.push_back( "MildPanlobularEmphysema" );
  ChestTypeNames.push_back( "ModeratePanlobularEmphysema" );
  ChestTypeNames.push_back( "SeverePanlobularEmphysema" );
  ChestTypeNames.push_back( "AirwayWallThickening" );
  ChestTypeNames.push_back( "AirwayCylindricalDilation" );
  ChestTypeNames.push_back( "VaricoseBronchiectasis" );
  ChestTypeNames.push_back( "CysticBronchiectasis" );
  ChestTypeNames.push_back( "CentrilobularNodule" );
  ChestTypeNames.push_back( "Mosaicing" );
  ChestTypeNames.push_back( "ExpiratoryMalacia" );
  ChestTypeNames.push_back( "SaberSheath" );
  ChestTypeNames.push_back( "OutPouching" );
  ChestTypeNames.push_back( "MucoidMaterial" );
  ChestTypeNames.push_back( "PatchyGasTrapping" );
  ChestTypeNames.push_back( "DiffuseGasTrapping" );
  ChestTypeNames.push_back( "LinearScar" );
  ChestTypeNames.push_back( "Cyst" );
  ChestTypeNames.push_back( "Atelectasis" );
  ChestTypeNames.push_back( "Honeycombing" );
  ChestTypeNames.push_back( "Trachea" );
  ChestTypeNames.push_back( "MainBronchus" );
  ChestTypeNames.push_back( "UpperLobeBronchus" );
  ChestTypeNames.push_back( "AirwayGeneration3" );
  ChestTypeNames.push_back( "AirwayGeneration4" );
  ChestTypeNames.push_back( "AirwayGeneration5" );
  ChestTypeNames.push_back( "AirwayGeneration6" );
  ChestTypeNames.push_back( "AirwayGeneration7" );
  ChestTypeNames.push_back( "AirwayGeneration8" );
  ChestTypeNames.push_back( "AirwayGeneration9" );
  ChestTypeNames.push_back( "AirwayGeneration10" );
  ChestTypeNames.push_back( "Calcification" );
  ChestTypeNames.push_back( "Artery" );
  ChestTypeNames.push_back( "Vein" );
  ChestTypeNames.push_back( "PectoralisMinor" );
  ChestTypeNames.push_back( "PectoralisMajor" );
  ChestTypeNames.push_back( "AnteriorScalene" );
  ChestTypeNames.push_back( "Fissure" );
  ChestTypeNames.push_back( "VesselGeneration0" );
  ChestTypeNames.push_back( "VesselGeneration1" );
  ChestTypeNames.push_back( "VesselGeneration2" );
  ChestTypeNames.push_back( "VesselGeneration3" );
  ChestTypeNames.push_back( "VesselGeneration4" );
  ChestTypeNames.push_back( "VesselGeneration5" );
  ChestTypeNames.push_back( "VesselGeneration6" );
  ChestTypeNames.push_back( "VesselGeneration7" );
  ChestTypeNames.push_back( "VesselGeneration8" );
  ChestTypeNames.push_back( "VesselGeneration9" );
  ChestTypeNames.push_back( "VesselGeneration10" );
  ChestTypeNames.push_back( "ParaseptalEmphysema" );
  ChestTypeNames.push_back( "CentrilobularEmphysema" );
  ChestTypeNames.push_back( "PanlobularEmphysema" );
  ChestTypeNames.push_back( "SubcutaneousFat" );
  ChestTypeNames.push_back( "VisceralFat" );
  ChestTypeNames.push_back( "IntermediateBronchus" );
  ChestTypeNames.push_back( "LowerLobeBronchus" );
  ChestTypeNames.push_back( "SuperiorDivisionBronchus" );
  ChestTypeNames.push_back( "LingularBronchus" );
  ChestTypeNames.push_back( "MiddleLobeBronchus" );
  ChestTypeNames.push_back( "BronchiectaticAirway" );
  ChestTypeNames.push_back( "NonBronchiectaticAirway" );
  ChestTypeNames.push_back( "AmbiguousBronchiectaticAirway" );
  ChestTypeNames.push_back( "Muscle" );
  ChestTypeNames.push_back( "Diaphragm" );
  
  BodyCompositionPhenotypeNames.push_back( "AxialCSA" );
  BodyCompositionPhenotypeNames.push_back( "CoronalCSA" );
  BodyCompositionPhenotypeNames.push_back( "SagittalCSA" );
  BodyCompositionPhenotypeNames.push_back( "HUMean" );
  BodyCompositionPhenotypeNames.push_back( "HUStd" );
  BodyCompositionPhenotypeNames.push_back( "HUKurtosis" );
  BodyCompositionPhenotypeNames.push_back( "HUSkewness" );
  BodyCompositionPhenotypeNames.push_back( "HUMode" );
  BodyCompositionPhenotypeNames.push_back( "HUMedian" );
  BodyCompositionPhenotypeNames.push_back( "HUMin" );
  BodyCompositionPhenotypeNames.push_back( "HUMax" );

  ParenchymaPhenotypeNames.push_back( "LAA950" );
  ParenchymaPhenotypeNames.push_back( "LAA910" );
  ParenchymaPhenotypeNames.push_back( "LAA856" );
  ParenchymaPhenotypeNames.push_back( "HAA700" );
  ParenchymaPhenotypeNames.push_back( "HAA600" );
  ParenchymaPhenotypeNames.push_back( "HAA500" );
  ParenchymaPhenotypeNames.push_back( "HAA250" );
  ParenchymaPhenotypeNames.push_back( "Perc10" );
  ParenchymaPhenotypeNames.push_back( "Perc15" );
  ParenchymaPhenotypeNames.push_back( "HUMean" );
  ParenchymaPhenotypeNames.push_back( "HUStd" );
  ParenchymaPhenotypeNames.push_back( "HUKurtosis" );
  ParenchymaPhenotypeNames.push_back( "HUSkewness" );
  ParenchymaPhenotypeNames.push_back( "HUMode" );
  ParenchymaPhenotypeNames.push_back( "HUMedian" );
  ParenchymaPhenotypeNames.push_back( "HUMin" );
  ParenchymaPhenotypeNames.push_back( "HUMax" );
  ParenchymaPhenotypeNames.push_back( "HUMean500" );
  ParenchymaPhenotypeNames.push_back( "HUStd500" );
  ParenchymaPhenotypeNames.push_back( "HUKurtosis500" );
  ParenchymaPhenotypeNames.push_back( "HUSkewness500" );
  ParenchymaPhenotypeNames.push_back( "HUMode500" );
  ParenchymaPhenotypeNames.push_back( "HUMedian500" );
  ParenchymaPhenotypeNames.push_back( "HUMin500" );
  ParenchymaPhenotypeNames.push_back( "HUMax500" );
  ParenchymaPhenotypeNames.push_back( "Volume" );
  ParenchymaPhenotypeNames.push_back( "Mass" );

  //
  // Each type is associated with a color. This is generally
  // useful for the interactors for users, e.g. when manually
  // editing/labeling fissures, vessels, and airway partices. If
  // colors are reassigned here, they should be UNIQUE.
  //
  double* t001 = new double[3]; t001[0] = 1.00; t001[1] = 1.00; t001[2] = 1.00; ChestTypeColors.push_back( t001 ); //UNDEFINEDTYPE
  double* t002 = new double[3]; t002[0] = 0.99; t002[1] = 0.99; t002[2] = 0.99; ChestTypeColors.push_back( t002 ); //NORMALPARENCHYMA
  double* t003 = new double[3]; t003[0] = 0.98; t003[1] = 0.98; t003[2] = 0.98; ChestTypeColors.push_back( t003 ); //AIRWAY
  double* t004 = new double[3]; t004[0] = 0.97; t004[1] = 0.97; t004[2] = 0.97; ChestTypeColors.push_back( t004 ); //VESSEL
  double* t005 = new double[3]; t005[0] = 0.96; t005[1] = 0.96; t005[2] = 0.96; ChestTypeColors.push_back( t005 ); //EMPHYSEMATOUS
  double* t006 = new double[3]; t006[0] = 0.95; t006[1] = 0.95; t006[2] = 0.95; ChestTypeColors.push_back( t006 ); //GROUNDGLASS
  double* t007 = new double[3]; t007[0] = 0.94; t007[1] = 0.94; t007[2] = 0.94; ChestTypeColors.push_back( t007 ); //RETICULAR
  double* t008 = new double[3]; t008[0] = 0.93; t008[1] = 0.93; t008[2] = 0.93; ChestTypeColors.push_back( t008 ); //NODULAR
  double* t009 = new double[3]; t009[0] = 0.92; t009[1] = 0.92; t009[2] = 0.92; ChestTypeColors.push_back( t009 ); //OBLIQUEFISSURE
  double* t010 = new double[3]; t010[0] = 0.91; t010[1] = 0.91; t010[2] = 0.91; ChestTypeColors.push_back( t010 ); //HORIZONTALFISSURE
  double* t011 = new double[3]; t011[0] = 0.90; t011[1] = 0.90; t011[2] = 0.90; ChestTypeColors.push_back( t011 ); //MILDPARASEPTALEMPHYSEMA
  double* t012 = new double[3]; t012[0] = 0.89; t012[1] = 0.89; t012[2] = 0.89; ChestTypeColors.push_back( t012 ); //MODERATEPARASEPTALEMPHYSEMA
  double* t013 = new double[3]; t013[0] = 0.88; t013[1] = 0.88; t013[2] = 0.88; ChestTypeColors.push_back( t013 ); //SEVEREPARASEPTALEMPHYSEMA
  double* t014 = new double[3]; t014[0] = 0.87; t014[1] = 0.87; t014[2] = 0.87; ChestTypeColors.push_back( t014 ); //MILDBULLA
  double* t015 = new double[3]; t015[0] = 0.86; t015[1] = 0.86; t015[2] = 0.86; ChestTypeColors.push_back( t015 ); //MODERATEBULLA
  double* t016 = new double[3]; t016[0] = 0.85; t016[1] = 0.85; t016[2] = 0.85; ChestTypeColors.push_back( t016 ); //SEVEREBULLA
  double* t017 = new double[3]; t017[0] = 0.84; t017[1] = 0.84; t017[2] = 0.84; ChestTypeColors.push_back( t017 ); //MILDCENTRILOBULAREMPHYSEMA
  double* t018 = new double[3]; t018[0] = 0.83; t018[1] = 0.83; t018[2] = 0.83; ChestTypeColors.push_back( t018 ); //MODERATECENTRILOBULAREMPHYSEMA
  double* t019 = new double[3]; t019[0] = 0.82; t019[1] = 0.82; t019[2] = 0.82; ChestTypeColors.push_back( t019 ); //SEVERECENTRILOBULAREMPHYSEMA
  double* t020 = new double[3]; t020[0] = 0.81; t020[1] = 0.81; t020[2] = 0.81; ChestTypeColors.push_back( t020 ); //MILDPANLOBULAREMPHYSEMA
  double* t021 = new double[3]; t021[0] = 0.80; t021[1] = 0.70; t021[2] = 0.80; ChestTypeColors.push_back( t021 ); //MODERATEPANLOBULAREMPHYSEMA
  double* t022 = new double[3]; t022[0] = 0.79; t022[1] = 0.79; t022[2] = 0.79; ChestTypeColors.push_back( t022 ); //SEVEREPANLOBULAREMPHYSEMA
  double* t023 = new double[3]; t023[0] = 0.78; t023[1] = 0.78; t023[2] = 0.78; ChestTypeColors.push_back( t023 ); //AIRWAYWALLTHICKENING
  double* t024 = new double[3]; t024[0] = 0.77; t024[1] = 0.77; t024[2] = 0.77; ChestTypeColors.push_back( t024 ); //AIRWAYCYLINDRICALDILATION
  double* t025 = new double[3]; t025[0] = 0.76; t025[1] = 0.76; t025[2] = 0.76; ChestTypeColors.push_back( t025 ); //VARICOSEBRONCHIECTASIS
  double* t026 = new double[3]; t026[0] = 0.75; t026[1] = 0.75; t026[2] = 0.75; ChestTypeColors.push_back( t026 ); //CYSTICBRONCHIECTASIS
  double* t027 = new double[3]; t027[0] = 0.74; t027[1] = 0.74; t027[2] = 0.74; ChestTypeColors.push_back( t027 ); //CENTRILOBULARNODULE
  double* t028 = new double[3]; t028[0] = 0.73; t028[1] = 0.73; t028[2] = 0.73; ChestTypeColors.push_back( t028 ); //MOSAICING
  double* t029 = new double[3]; t029[0] = 0.72; t029[1] = 0.72; t029[2] = 0.72; ChestTypeColors.push_back( t029 ); //EXPIRATORYMALACIA
  double* t030 = new double[3]; t030[0] = 0.71; t030[1] = 0.71; t030[2] = 0.71; ChestTypeColors.push_back( t030 ); //SABERSHEATH
  double* t031 = new double[3]; t031[0] = 0.70; t031[1] = 0.70; t031[2] = 0.70; ChestTypeColors.push_back( t031 ); //OUTPOUCHING
  double* t032 = new double[3]; t032[0] = 0.69; t032[1] = 0.69; t032[2] = 0.69; ChestTypeColors.push_back( t032 ); //MUCOIDMATERIAL
  double* t033 = new double[3]; t033[0] = 0.68; t033[1] = 0.68; t033[2] = 0.68; ChestTypeColors.push_back( t033 ); //PATCHYGASTRAPPING
  double* t034 = new double[3]; t034[0] = 0.67; t034[1] = 0.67; t034[2] = 0.67; ChestTypeColors.push_back( t034 ); //DIFFUSEGASTRAPPING
  double* t035 = new double[3]; t035[0] = 0.66; t035[1] = 0.66; t035[2] = 0.66; ChestTypeColors.push_back( t035 ); //LINEARSCAR
  double* t036 = new double[3]; t036[0] = 0.65; t036[1] = 0.65; t036[2] = 0.65; ChestTypeColors.push_back( t036 ); //CYST
  double* t037 = new double[3]; t037[0] = 0.64; t037[1] = 0.64; t037[2] = 0.64; ChestTypeColors.push_back( t037 ); //ATELECTASIS
  double* t038 = new double[3]; t038[0] = 0.63; t038[1] = 0.63; t038[2] = 0.63; ChestTypeColors.push_back( t038 ); //HONEYCOMBING
  // The airway generation colors are identical to the vessel generation colors except that 0.01 has been
  // added to the red channel value to make these colors unique
  double* t039 = new double[3]; t039[0] = 0.51; t039[1] = 0.50; t039[2] = 0.50; ChestTypeColors.push_back( t039 ); //TRACHEA
  double* t040 = new double[3]; t040[0] = 0.55; t040[1] = 0.27; t040[2] = 0.07; ChestTypeColors.push_back( t040 ); //MAINBRONCHUS
  double* t041 = new double[3]; t041[0] = 1.00; t041[1] = 0.65; t041[2] = 0.00; ChestTypeColors.push_back( t041 ); //UPPERLOBEBRONCHUS
  double* t042 = new double[3]; t042[0] = 1.00; t042[1] = 1.00; t042[2] = 0.01; ChestTypeColors.push_back( t042 ); //AIRWAYGENERATION3
  double* t043 = new double[3]; t043[0] = 1.00; t043[1] = 0.01; t043[2] = 1.00; ChestTypeColors.push_back( t043 ); //AIRWAYGENERATION4
  double* t044 = new double[3]; t044[0] = 0.51; t044[1] = 1.00; t044[2] = 0.00; ChestTypeColors.push_back( t044 ); //AIRWAYGENERATION5
  double* t045 = new double[3]; t045[0] = 0.01; t045[1] = 0.50; t045[2] = 1.00; ChestTypeColors.push_back( t045 ); //AIRWAYGENERATION6
  double* t046 = new double[3]; t046[0] = 0.51; t046[1] = 0.00; t046[2] = 0.50; ChestTypeColors.push_back( t046 ); //AIRWAYGENERATION7
  double* t047 = new double[3]; t047[0] = 0.51; t047[1] = 0.50; t047[2] = 0.00; ChestTypeColors.push_back( t047 ); //AIRWAYGENERATION8
  double* t048 = new double[3]; t048[0] = 0.01; t048[1] = 0.50; t048[2] = 0.50; ChestTypeColors.push_back( t048 ); //AIRWAYGENERATION9
  double* t049 = new double[3]; t049[0] = 0.45; t049[1] = 0.44; t049[2] = 0.44; ChestTypeColors.push_back( t049 ); //AIRWAYGENERATION10
  double* t050 = new double[3]; t050[0] = 0.51; t050[1] = 0.51; t050[2] = 0.51; ChestTypeColors.push_back( t050 ); //CALCIFICATION
  double* t051 = new double[3]; t051[0] = 0.00; t051[1] = 0.00; t051[2] = 0.99; ChestTypeColors.push_back( t051 ); //ARTERY
  double* t052 = new double[3]; t052[0] = 0.99; t052[1] = 0.00; t052[2] = 0.00; ChestTypeColors.push_back( t052 ); //VEIN
  double* t053 = new double[3]; t053[0] = 0.48; t053[1] = 0.48; t053[2] = 0.48; ChestTypeColors.push_back( t053 ); //PECTORALISMINOR
  double* t054 = new double[3]; t054[0] = 0.47; t054[1] = 0.47; t054[2] = 0.47; ChestTypeColors.push_back( t054 ); //PECTORALISMAJOR
  double* t055 = new double[3]; t055[0] = 0.46; t055[1] = 0.46; t055[2] = 0.46; ChestTypeColors.push_back( t055 ); //ANTERIORSCALENE
  double* t056 = new double[3]; t056[0] = 0.45; t056[1] = 0.45; t056[2] = 0.45; ChestTypeColors.push_back( t056 ); //FISSURE     
  // The vessel generation colors are identical to the airway generation colors except that the red chanel
  // is 0.01 less than the airway generation red channel. This ensures that the colors are unique
  double* t057 = new double[3]; t057[0] = 0.00; t057[1] = 0.00; t057[2] = 0.00; ChestTypeColors.push_back( t057 ); //VESSELGENERATION0
  double* t058 = new double[3]; t058[0] = 0.00; t058[1] = 1.00; t058[2] = 0.00; ChestTypeColors.push_back( t058 ); //VESSELGENERATION1
  double* t059 = new double[3]; t059[0] = 0.00; t059[1] = 1.00; t059[2] = 1.00; ChestTypeColors.push_back( t059 ); //VESSELGENERATION2
  double* t060 = new double[3]; t060[0] = 1.00; t060[1] = 1.00; t060[2] = 0.00; ChestTypeColors.push_back( t060 ); //VESSELGENERATION3
  double* t061 = new double[3]; t061[0] = 1.00; t061[1] = 0.00; t061[2] = 1.00; ChestTypeColors.push_back( t061 ); //VESSELGENERATION4
  double* t062 = new double[3]; t062[0] = 0.50; t062[1] = 1.00; t062[2] = 0.00; ChestTypeColors.push_back( t062 ); //VESSELGENERATION5
  double* t063 = new double[3]; t063[0] = 0.00; t063[1] = 0.50; t063[2] = 1.00; ChestTypeColors.push_back( t063 ); //VESSELGENERATION6
  double* t064 = new double[3]; t064[0] = 0.50; t064[1] = 0.00; t064[2] = 0.50; ChestTypeColors.push_back( t064 ); //VESSELGENERATION7
  double* t065 = new double[3]; t065[0] = 0.50; t065[1] = 0.50; t065[2] = 0.00; ChestTypeColors.push_back( t065 ); //VESSELGENERATION8
  double* t066 = new double[3]; t066[0] = 0.00; t066[1] = 0.50; t066[2] = 0.50; ChestTypeColors.push_back( t066 ); //VESSELGENERATION9
  double* t067 = new double[3]; t067[0] = 0.44; t067[1] = 0.44; t067[2] = 0.44; ChestTypeColors.push_back( t067 ); //VESSELGENERATION10
  
  double* t068 = new double[3]; t068[0] = 0.00; t068[1] = 0.68; t068[2] = 0.00; ChestTypeColors.push_back( t068 ); //PARASEPTALEMPHYSEMA
  double* t069 = new double[3]; t069[0] = 0.00; t069[1] = 0.69; t069[2] = 0.69; ChestTypeColors.push_back( t069 ); //CENTRILOBULAREMPHYSEMA
  double* t070 = new double[3]; t070[0] = 0.00; t070[1] = 0.00; t070[2] = 0.70; ChestTypeColors.push_back( t070 ); //PANLOBULAREMPHYSEMA
  
  double* t071 = new double[3]; t071[0] = 0.59; t071[1] = 0.65; t071[2] = 0.20; ChestTypeColors.push_back( t071 ); //SUBCUTANEOUSFAT
  double* t072 = new double[3]; t072[0] = 0.58; t072[1] = 0.65; t072[2] = 0.20; ChestTypeColors.push_back( t072 ); //VISCERALFAT
  
  double* t073 = new double[3]; t073[0] = 0.85; t073[1] = 0.75; t073[2] = 0.85; ChestTypeColors.push_back( t073 ); //INTERMEDIATEBRONCHUS
  double* t074 = new double[3]; t074[0] = 1.00; t074[1] = 0.02; t074[2] = 0.00; ChestTypeColors.push_back( t074 ); //LOWERLOBEBRONCHUS
  double* t075 = new double[3]; t075[0] = 0.98; t075[1] = 0.50; t075[2] = 0.45; ChestTypeColors.push_back( t075 ); //SUPERIORDIVISIONBRONCHUS
  double* t076 = new double[3]; t076[0] = 0.00; t076[1] = 0.03; t076[2] = 1.00; ChestTypeColors.push_back( t076 ); //LINGULARBRONCHUS
  double* t077 = new double[3]; t077[0] = 0.25; t077[1] = 0.88; t077[2] = 0.82; ChestTypeColors.push_back( t077 ); //MIDDLELOBEBRONCHUS
  
  double* t078 = new double[3]; t078[0] = 0.25; t078[1] = 0.88; t078[2] = 0.81; ChestTypeColors.push_back( t078 ); //BRONCHIECTATICAIRWAY
  double* t079 = new double[3]; t079[0] = 0.25; t079[1] = 0.87; t079[2] = 0.81; ChestTypeColors.push_back( t079 ); //NONBRONCHIECTATICAIRWAY
  double* t080 = new double[3]; t080[0] = 0.25; t080[1] = 0.86; t080[2] = 0.81; ChestTypeColors.push_back( t080 ); //AMBIGUOUSBRONCHIECTATICAIRWAY
  double* t081 = new double[3]; t081[0] = 0.90; t081[1] = 0.00; t081[2] = 0.00; ChestTypeColors.push_back( t081 ); //MUSCLE
  double* t082 = new double[3]; t082[0] = 0.91; t082[1] = 0.00; t082[2] = 0.00; ChestTypeColors.push_back( t082 ); //DIAPHRAGM
  
  //
  // Each region is associated with a color. This is generally
  // useful creating overlay images for quick segmentation inspectio, 
  // e.g. If colors are reassigned here, they should be UNIQUE. 
  //
  double* r001 = new double[3]; r001[0] = 0.00; r001[1] = 0.00; r001[2] = 0.00; ChestRegionColors.push_back( r001 ); //UNDEFINEDREGION      
  double* r002 = new double[3]; r002[0] = 0.42; r002[1] = 0.38; r002[2] = 0.75; ChestRegionColors.push_back( r002 ); //WHOLELUNG
  double* r003 = new double[3]; r003[0] = 0.26; r003[1] = 0.64; r003[2] = 0.10; ChestRegionColors.push_back( r003 ); //RIGHTLUNG
  double* r004 = new double[3]; r004[0] = 0.80; r004[1] = 0.11; r004[2] = 0.36; ChestRegionColors.push_back( r004 ); //LEFTLUNG
  double* r005 = new double[3]; r005[0] = 0.04; r005[1] = 0.00; r005[2] = 0.00; ChestRegionColors.push_back( r005 ); //RIGHTSUPERIORLOBE
  double* r006 = new double[3]; r006[0] = 0.05; r006[1] = 0.00; r006[2] = 0.00; ChestRegionColors.push_back( r006 ); //RIGHTMIDDLELOBE
  double* r007 = new double[3]; r007[0] = 0.06; r007[1] = 0.00; r007[2] = 0.00; ChestRegionColors.push_back( r007 ); //RIGHTINFERIORLOBE
  double* r008 = new double[3]; r008[0] = 0.07; r008[1] = 0.00; r008[2] = 0.00; ChestRegionColors.push_back( r008 ); //LEFTSUPERIORLOBE
  double* r009 = new double[3]; r009[0] = 0.08; r009[1] = 0.00; r009[2] = 0.00; ChestRegionColors.push_back( r009 ); //LEFTINFERIORLOBE
  double* r010 = new double[3]; r010[0] = 0.95; r010[1] = 0.03; r010[2] = 0.03; ChestRegionColors.push_back( r010 ); //LEFTUPPERTHIRD
  double* r011 = new double[3]; r011[0] = 0.95; r011[1] = 0.89; r011[2] = 0.03; ChestRegionColors.push_back( r011 ); //LEFTMIDDLETHIRD
  double* r012 = new double[3]; r012[0] = 0.03; r012[1] = 0.34; r012[2] = 0.95; ChestRegionColors.push_back( r012 ); //LEFTLOWERTHIRD
  double* r013 = new double[3]; r013[0] = 0.06; r013[1] = 0.91; r013[2] = 0.91; ChestRegionColors.push_back( r013 ); //RIGHTUPPERTHIRD
  double* r014 = new double[3]; r014[0] = 1.00; r014[1] = 0.00; r014[2] = 0.91; ChestRegionColors.push_back( r014 ); //RIGHTMIDDLETHIRD
  double* r015 = new double[3]; r015[0] = 0.34; r015[1] = 0.41; r015[2] = 0.09; ChestRegionColors.push_back( r015 ); //RIGHTLOWERTHIRD
  double* r016 = new double[3]; r016[0] = 0.00; r016[1] = 0.06; r016[2] = 0.00; ChestRegionColors.push_back( r016 ); //MEDIASTINUM
  double* r017 = new double[3]; r017[0] = 0.00; r017[1] = 0.07; r017[2] = 0.00; ChestRegionColors.push_back( r017 ); //WHOLEHEART
  double* r018 = new double[3]; r018[0] = 0.00; r018[1] = 0.08; r018[2] = 0.00; ChestRegionColors.push_back( r018 ); //AORTA
  double* r019 = new double[3]; r019[0] = 0.00; r019[1] = 0.09; r019[2] = 0.00; ChestRegionColors.push_back( r019 ); //PULMONARYARTERY
  double* r020 = new double[3]; r020[0] = 0.00; r020[1] = 0.00; r020[2] = 0.01; ChestRegionColors.push_back( r020 ); //PULMONARYVEIN
  double* r021 = new double[3]; r021[0] = 0.00; r021[1] = 0.00; r021[2] = 0.02; ChestRegionColors.push_back( r021 ); //UPPERTHIRD
  double* r022 = new double[3]; r022[0] = 0.00; r022[1] = 0.00; r022[2] = 0.03; ChestRegionColors.push_back( r022 ); //MIDDLETHIRD
  double* r023 = new double[3]; r023[0] = 0.00; r023[1] = 0.00; r023[2] = 0.04; ChestRegionColors.push_back( r023 ); //LOWERTHIRD
  double* r024 = new double[3]; r024[0] = 0.34; r024[1] = 0.33; r024[2] = 0.80; ChestRegionColors.push_back( r024 ); //LEFT
  double* r025 = new double[3]; r025[0] = 0.74; r025[1] = 0.34; r025[2] = 0.14; ChestRegionColors.push_back( r025 ); //RIGHT
  double* r026 = new double[3]; r026[0] = 0.66; r026[1] = 0.36; r026[2] = 0.40; ChestRegionColors.push_back( r026 ); //LIVER
  double* r027 = new double[3]; r027[0] = 1.00; r027[1] = 1.00; r027[2] = 0.01; ChestRegionColors.push_back( r027 ); //SPLEEN
  double* r028 = new double[3]; r028[0] = 1.00; r028[1] = 0.50; r028[2] = 0.01; ChestRegionColors.push_back( r028 ); //ABDOMEN
  double* r029 = new double[3]; r029[0] = 1.00; r029[1] = 0.51; r029[2] = 0.01; ChestRegionColors.push_back( r029 ); //PARAVERTEBRAL  
}

cip::ChestConventions::~ChestConventions()
{
  for ( unsigned int i=0; i<this->ChestRegionColors.size(); i++ )
    {
      delete[] this->ChestRegionColors[i];
    }

  for ( unsigned int i=0; i<this->ChestTypeColors.size(); i++ )
    {
      delete[] this->ChestTypeColors[i];
    }
}

unsigned char cip::ChestConventions::GetNumberOfEnumeratedChestRegions() const
{
  return m_NumberOfEnumeratedChestRegions;
}

unsigned char cip::ChestConventions::GetNumberOfEnumeratedChestTypes() const
{
  return m_NumberOfEnumeratedChestTypes;
}

/** This method checks if the chest region 'subordinate' is within
 *  the chest region 'superior'. It assumes that all chest regions are
 *  within the WHOLELUNG lung region. TODO: extend do deal with
 *  chest, not just lung */
bool cip::ChestConventions::CheckSubordinateSuperiorChestRegionRelationship( unsigned char subordinate, unsigned char superior )
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
  if ( subordinate == (unsigned char)( UNDEFINEDREGION ) ||
       superior == (unsigned char)( UNDEFINEDREGION ) )
    {
      return false;
    }
  
  if ( superior == (unsigned char)( WHOLELUNG ) )
    {
      return true;
    }
  
  unsigned char subordinateTemp = subordinate;
  
  while ( subordinateTemp != (unsigned char)( WHOLELUNG ) && 
	  subordinateTemp != (unsigned char)( UNDEFINEDREGION ) )
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

std::string cip::ChestConventions::GetChestWildCardName() const
{
  return std::string("WildCard");
}

/** Given an unsigned short value, this method will compute the
 *  8-bit region value corresponding to the input */
unsigned char cip::ChestConventions::GetChestRegionFromValue( unsigned short value ) const
{
  return value - ((value >> 8) << 8);
}

/** The 'color' param is assumed to have three components, each in
 *  the interval [0,1]. All chest type colors will be tested until a
 *  color match is found. If no match is found, 'UNDEFINEDTYPYE'
 *  will be returned */  
unsigned char cip::ChestConventions::GetChestTypeFromColor( double* color ) const
{
  for ( unsigned int i=0; i<m_NumberOfEnumeratedChestTypes; i++ )
    {
      if ( ChestTypeColors[i][0] == color[0] && ChestTypeColors[i][1] == color[1] && 
  	   ChestTypeColors[i][2] == color[2] )          
  	{
          return (unsigned char)( i );
  	}
    }
  return (unsigned char)( UNDEFINEDTYPE );
}

/** The 'color' param is assumed to have three components, each in
 *  the interval [0,1]. All chest region colors will be tested until a
 *  color match is found. If no match is found, 'UNDEFINEDTYPYE'
 *  will be returned */  
unsigned char cip::ChestConventions::GetChestRegionFromColor(double* color) const
{
  for (unsigned int i=0; i<m_NumberOfEnumeratedChestRegions; i++)
    {
      if (ChestRegionColors[i][0] == color[0] && ChestRegionColors[i][1] == color[1] && 
  	  ChestRegionColors[i][2] == color[2] )          
  	{
  	  return (unsigned char)(i);
  	}
    }
  return (unsigned char)(UNDEFINEDTYPE);
}

/** Given an unsigned short value, this method will compute the
 *  8-bit type value corresponding to the input */
unsigned char cip::ChestConventions::GetChestTypeFromValue( unsigned short value ) const
{
  return (value >> 8);
}

/** Given an unsigned char value corresponding to a chest type, this
 *  method will return the string name equivalent. */
std::string cip::ChestConventions::GetChestTypeName( unsigned char whichType ) const
{
  if ( int( whichType ) > m_NumberOfEnumeratedChestTypes-1 )
    {
      return "UNDEFINEDTYPE";
    }
  
  return ChestTypeNames[int( whichType )];
}


/** Get the chest type color. 'color' param is assumed to be an
 * allocated 3 dimensional double pointer */
void cip::ChestConventions::GetChestTypeColor( unsigned char whichType, double* color ) const
{
  color[0] = ChestTypeColors[int(whichType)][0];
  color[1] = ChestTypeColors[int(whichType)][1];
  color[2] = ChestTypeColors[int(whichType)][2];
}

/** Get the chest region color. 'color' param is assumed to be an
 * allocated 3 dimensional double pointer */
void cip::ChestConventions::GetChestRegionColor(unsigned char whichRegion, double* color) const
{
  color[0] = ChestRegionColors[int(whichRegion)][0];
  color[1] = ChestRegionColors[int(whichRegion)][1];
  color[2] = ChestRegionColors[int(whichRegion)][2];
}

/** Get the color corresponding to the chest-region chest-pair pair. The
 * color is computed as the average of the two corresponding region and type 
 * colors unless the region or type is undefined, in which case the color of
 * the defined region or type is returned. The 'color' param is assumed to be 
 * an allocated 3 dimensional double pointer */
void cip::ChestConventions::GetColorFromChestRegionChestType(unsigned char whichRegion, unsigned char whichType, double* color) const
{
  if (whichRegion == (unsigned char)(cip::UNDEFINEDREGION))
    {
      color[0] = ChestTypeColors[int(whichType)][0];
      color[1] = ChestTypeColors[int(whichType)][1];
      color[2] = ChestTypeColors[int(whichType)][2];
    }
  else if (whichType == (unsigned char)(cip::UNDEFINEDTYPE))
    {
      color[0] = ChestRegionColors[int(whichRegion)][0];
      color[1] = ChestRegionColors[int(whichRegion)][1];
      color[2] = ChestRegionColors[int(whichRegion)][2];
    }
  else
    {
      color[0] = (ChestRegionColors[int(whichRegion)][0] + ChestTypeColors[int(whichType)][0])/2.0;
      color[1] = (ChestRegionColors[int(whichRegion)][1] + ChestTypeColors[int(whichType)][1])/2.0;
      color[2] = (ChestRegionColors[int(whichRegion)][2] + ChestTypeColors[int(whichType)][2])/2.0;
    }
}

/** Given an unsigned char value corresponding to a chest region, this
 *  method will return the string name equivalent. */
std::string cip::ChestConventions::GetChestRegionName( unsigned char whichRegion ) const
{
  if ( int( whichRegion ) > m_NumberOfEnumeratedChestRegions-1 )
    {
      return "UNDEFINEDREGION";
    }
  
  return ChestRegionNames[int( whichRegion )];
}

/** Given an unsigned short value, this method will return the
 *  string name of the corresponding chest region */
std::string cip::ChestConventions::GetChestRegionNameFromValue( unsigned short value ) const
{
  unsigned char regionValue = 0;
  
  regionValue = this->GetChestRegionFromValue(value);
  
  return this->GetChestRegionName(regionValue);
};

/** Given an unsigned short value, this method will return the
 *  string name of the corresponding chest type */
std::string cip::ChestConventions::GetChestTypeNameFromValue( unsigned short value ) const
{
  unsigned char typeValue = 0;
      
  typeValue = this->GetChestTypeFromValue(value);
  return this->GetChestTypeName(typeValue);
}

unsigned short cip::ChestConventions::GetValueFromChestRegionAndType( unsigned char region, unsigned char type ) const
{
  unsigned short regionValue = (unsigned short) region;
  unsigned short tmp = (unsigned short) type;
  unsigned short regionType = (tmp << 8);
  unsigned short combinedValue = regionValue + regionType;
  return combinedValue;
}

/** Given a string identifying one of the enumerated chest regions,
 * this method will return the unsigned char equivalent. If no match
 * is found, the method will retune UNDEFINEDREGION */
unsigned char cip::ChestConventions::GetChestRegionValueFromName( std::string regionString ) const
{
  for ( int i=0; i<m_NumberOfEnumeratedChestRegions; i++ )
    {
      if ( !regionString.compare( ChestRegionNames[i] ) )
	{
          return ChestRegions[i];
	}
    }
  
  return (unsigned char)( UNDEFINEDREGION );
}

/** Given a string identifying one of the enumerated chest types,
 * this method will return the unsigned char equivalent. If no match
 * is found, the method will retune UNDEFINEDTYPE */
unsigned char cip::ChestConventions::GetChestTypeValueFromName( std::string typeString ) const
{
  for ( int i=0; i<m_NumberOfEnumeratedChestTypes; i++ )
    {
      if ( !typeString.compare( ChestTypeNames[i] ) )
	{
          return ChestTypes[i];
	}
    }
  
  return (unsigned char)( UNDEFINEDTYPE );
}

/** Get the ith chest region */
unsigned char cip::ChestConventions::GetChestRegion( unsigned int i ) const
{
  return (unsigned char)( ChestRegions[i] );
}

/** Get the ith chest type */
unsigned char cip::ChestConventions::GetChestType( unsigned int i ) const
{
  return (unsigned char)( ChestTypes[i] );
}

/** Returns true if the passed string name is among the allowed body composition 
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsBodyCompositionPhenotypeName( std::string pheno ) const
{  
  for ( int i=0; i<this->BodyCompositionPhenotypeNames.size(); i++ )
    {
      if ( !this->BodyCompositionPhenotypeNames[i].compare( pheno ) )
	{
	  return true;
	}
    }
  
  return false;
}

/** Returns true if the passed string name is among the allowed parenchyma
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsParenchymaPhenotypeName( std::string pheno ) const
{  
  for ( int i=0; i<this->ParenchymaPhenotypeNames.size(); i++ )
    {
      if ( !this->ParenchymaPhenotypeNames[i].compare( pheno ) )
	{
	  return true;
	}
    }
  
  return false;
}

/** Returns true if the passed string name is among the allowed 
 *  phenotype names and returns false otherwise */
bool cip::ChestConventions::IsPhenotypeName( std::string pheno ) const
{  
  for ( int i=0; i<this->ParenchymaPhenotypeNames.size(); i++ )
    {
      if ( !this->ParenchymaPhenotypeNames[i].compare( pheno ) )
	{
	  return true;
	}
    }

  for ( int i=0; i<this->BodyCompositionPhenotypeNames.size(); i++ )
    {
      if ( !this->BodyCompositionPhenotypeNames[i].compare( pheno ) )
	{
	  return true;
	}
    }

  for ( int i=0; i<this->HistogramPhenotypeNames.size(); i++ )
    {
      if ( !this->HistogramPhenotypeNames[i].compare( pheno ) )
	{
	  return true;
	}
    }
  
  return false;
}
