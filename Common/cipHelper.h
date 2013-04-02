//
//  \file cipHelper
//  \ingroup common
//  \brief This class is intended to contain a collection of functions that are routinely 
//	used in other programs.
//
//  $Date$
//  $Revision$
//  $Author$
//

#ifndef __cipHelper_h
#define __cipHelper_h

#include "cipConventions.h"
 
//
// Function that downsamples a label map. Takes in as input a value for the downsampling amount and 
// a pointer to a LabelMapType, and returns a pointer to a downsampled LabelMapType.
//
static cip::LabelMapType::Pointer DownsampleLabelMap( short samplingAmount, cip::LabelMapType::Pointer inputLabelMap );

//
// Function that upsamples a label map. Takes in as input a value for the upsampling
// amount and a pointer to a LabelMapType, and returns a pointer to a upsampled LabelMapType.
//
static cip::LabelMapType::Pointer UpsampleLabelMap( short samplingAmount, cip::LabelMapType::Pointer inputLabelMap );

//
// Function that downsamples a CT. Takes in as input a value for the downsampling amount and 
// a pointer to a CTType, and returns a pointer to a downsampled CTType. 
//
static cip::CTType::Pointer DownsampleCT( short samplingAmount, cip::CTType::Pointer inputCT );

//
// Function that upsamples a label CT. Takes in as input a value for the upsampling
// amount and a pointer to a CTType, and returns a pointer to a upsampled CTType. 
//
static cip::CTType::Pointer UpsampleCT( short samplingAmount, cip::CTType::Pointer inputCT );

#endif
