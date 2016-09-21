/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeAirwayWall.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkComputeAirwayWall.h"

#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "vtkImageReformatAlongRay.h"
#include "vtkPointData.h"
#include "vtkMath.h"
#include "vtkCellArray.h"
#include "vtkPoints.h"
#include "vtkGeneralizedPhaseCongruency.h"
#include "vtkCollection.h"
#include "vtkImageExtractComponents.h"
#include "vtkDataArrayCollection.h"
#include "vtkSmoothLines.h"
#include "vtkCardinalSpline.h"

#include "vtkPolyDataWriter.h"

#include <math.h>

vtkStandardNewMacro(vtkComputeAirwayWall);

//----------------------------------------------------------------------------
vtkComputeAirwayWall::vtkComputeAirwayWall()
{
this->WallThreshold = 300;
this->GradientThreshold = 100;
this->PCThreshold = 0.6; //threshold for PC response
this->Method = 0;

// Params for Phase Congruency
this->NumberOfScales = 4;
this->Bandwidth = 1.4635;
this->MultiplicativeFactor = 1.5;
this->MinimumWavelength = 4;
this->Weights = vtkDoubleArray::New();
this->UseWeights = 0;

// Params for Ray configuration
this->RMin = 0;
this->RMax = 12.7;
this->Delta = 0.5;
this->Scale = 3;

this->NumberOfThetaSamples = 128;

// Params for Sector Statistics
this->ThetaMin =0;
this->ThetaMax = 2*vtkMath::Pi();
this->ActivateSector = 0;

// Params for Densitometric Airway Phenotype
// Alpha: how many wall thickness we want to include in the radius.
this->Alpha = 3;
this->T = 7.5;

    
// Outlier detection paramas
this->StdFactor = 2.0;
    
this->StatsMean = vtkDoubleArray::New();
this->StatsStd = vtkDoubleArray::New();
this->StatsMin = vtkDoubleArray::New();
this->StatsMax = vtkDoubleArray::New();
this->InnerContour = vtkPolyData::New();
this->OuterContour = vtkPolyData::New();

this->NumberOfQuantities = 21;
}

//----------------------------------------------------------------------------
vtkComputeAirwayWall::~vtkComputeAirwayWall()
{
this->Weights->Delete();
this->StatsMean->Delete();
this->StatsStd->Delete();
this->StatsMax->Delete();
this->StatsMin->Delete();
this->InnerContour->Delete();
this->OuterContour->Delete();
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteInformation()
int vtkComputeAirwayWall::RequestInformation (
  vtkInformation       *  vtkNotUsed(request),
  vtkInformationVector ** vtkNotUsed(inputVector),
  vtkInformationVector *  vtkNotUsed(outputVector))
{
  //this->GetOutput()->SetScalarType(VTK_INT);
  // This filter does not produce an image.
  // Num of output components = 0
  //this->GetOutput()->SetNumberOfScalarComponents(3);

  // Make sure the number of rays is even to have antipodal rays
  if (this->NumberOfThetaSamples % 2 != 0)
    this->NumberOfThetaSamples++;

  return 1;
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - introduced to replace ExecuteData()
void vtkComputeAirwayWall::ExecuteDataWithInformation(vtkDataObject *out,
  vtkInformation* outInfo)
{
  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());

  // Make sure the Input has been set.
  if ( input == NULL )
    {
    vtkErrorMacro(<< "ExecuteData: Input is not set.");
    return;
    }

  // Too many filters have floating point exceptions to execute
  // with empty input/ no request.
  if (this->UpdateExtentIsEmpty(outInfo, out))
    {
    return;
    }

 // Check number of image components: each component is one kernel
 int numKernels = input->GetNumberOfScalarComponents();

 if (this->Method == 3 && numKernels<=1) {
   vtkErrorMacro(<< "Phase congruency with multiple kernels requires a multicomponent input (one component per kernel");
   return;
  }

 // Loop reformating rays
 // Go into method to extract maximum PC point.
 vtkDoubleArray *samples = vtkDoubleArray::New();
 vtkDoubleArray *signal;
 //double dth = this->GetThetaSampling();
 vtkCollection *rayCollection = vtkCollection::New();
 vtkCollection *extractCollection = vtkCollection::New();
 vtkDataArrayCollection *signalCollection=vtkDataArrayCollection::New();

 // Center information
 int dims[3];
 input->GetDimensions(dims);
 double center[3];
 double delta = this->Delta;
 center[0] = dims[0]/2-0.5;
 center[1] = dims[1]/2-0.5;
 center[2] = 0.0;

 // Intensity information
 double range[2];
 input->GetScalarRange(range);

 //This logic should be outside
 /*
 double scale =1;
 //Define scale based on method

 switch(this->Method) {
  case 0:
    scale = 3;
    break;
  case 1:
    scale = 3;
    break;
  case 2:
    scale = 1;
    break;
  case 3:
    scale = 1;
    break;
 }
 */

 vtkImageReformatAlongRay *ray;
 vtkImageExtractComponents *extract;

 for (int i= 0; i<numKernels;i++)
   {
   ray = vtkImageReformatAlongRay::New();
   rayCollection->AddItem(ray);
   extract = vtkImageExtractComponents::New();
   extract->SetInputData(input);
   extract->SetComponents(i);
   extract->Update();
   ray->SetInputData(extract->GetOutput());
   //Set ray extraction filter
   ray->SetCenter(center);
   ray->SetRMin(this->RMin);
   ray->SetRMax(this->RMax);
   ray->SetScale(this->Scale);
   ray->SetDelta(delta);

   extractCollection->AddItem(extract);
   }

 //ray->SetInput(this->GetInput());

 // Objects to store points and cell data
 vtkPoints *ip = vtkPoints::New();
 vtkPoints *op = vtkPoints::New();
 vtkCellArray *ic = vtkCellArray::New();
 vtkCellArray *oc = vtkCellArray::New();

 double dth = 2*vtkMath::Pi()/this->NumberOfThetaSamples;

 //Full statistics variables
 // Mean and Std
 double meanRi =0;
 double stdRi =0;
 double meanRo =0;
 double stdRo =0;
 double meanWth =0;
 double stdWth=0;
 double meanI =0;
 double stdI =0;
 double Ai =0;
 double Ae =0;
 double WAp = 0; // Wall Area percentage
 double sqrtWA = 0;
 double Pi = 0;
 double meanPeakI = 0;
 double stdPeakI = 0;
 double meanInnerI = 0;
 double stdInnerI = 0;
 double meanOuterI = 0;
 double stdOuterI = 0;
 double meanVesselI = 0;
 double stdVesselI = 0;
 double meanRLInnerDiam = 0;
 double stdRLInnerDiam =0;
 double meanAPInnerDiam = 0;
 double stdAPInnerDiam = 0;
 double meanRLOuterDiam = 0;
 double stdRLOuterDiam =0;
 double meanAPOuterDiam = 0;
 double stdAPOuterDiam = 0;
 //Peak Lumen attenuation
 double meanLA = 0;
 double stdLA = 0;
 //Parenchyma atten.
 double meanPA = 0;
 double stdPA = 0;
 //Airway Phenotype based on energy conservation
 double meanEnergy = 0;
 double stdEnergy = 0;
 double meanPower = 0;
 double stdPower = 0;

 int Isamples=0;
 int Wsamples=0;

 // Min - Max
 double minRi = this->StatsMin->GetDataTypeMax();
 double maxRi = this->StatsMax->GetDataTypeMin();
 double minRo =this->StatsMin->GetDataTypeMax();
 double maxRo = this->StatsMax->GetDataTypeMin();
 double minWth =this->StatsMin->GetDataTypeMax();
 double maxWth =this->StatsMax->GetDataTypeMin();
 double minWI =0;
 double maxWI =0;
 double minPeakI =0;
 double maxPeakI =0;
 double maxInnerI = 0;
 double minInnerI = 0;
 double maxOuterI = 0;
 double minOuterI = 0;
 double maxVesselI = 0;
 double minVesselI = 0;
 double maxRLInnerDiam = 0;
 double minRLInnerDiam = 0;
 double maxAPInnerDiam = 0;
 double minAPInnerDiam =0;
 double maxRLOuterDiam = 0;
 double minRLOuterDiam = 0;
 double maxAPOuterDiam = 0;
 double minAPOuterDiam =0;
 double minLA = 5000;
 double maxLA = -5000;
 double minPA = 5000;
 double maxPA = -5000;
 double minEnergy = 5000;
 double maxEnergy = -5000;
 double minPower = 5000;
 double maxPower = -5000;

 //Sector statistics variables
 double meanRiS =0;
 double stdRiS =0;
 double meanRoS =0;
 double stdRoS =0;
 double meanWthS =0;
 double stdWthS=0;
 double meanIS =0;
 double stdIS =0;
 double AiS = 0;
 double AeS = 0;
 double WApS = 0;
 double sqrtWAS = 0;
 double PiS =0;
 double meanPeakIS = 0;
 double stdPeakIS = 0;
 double meanInnerIS = 0;
 double stdInnerIS = 0;
 double meanOuterIS = 0;
 double stdOuterIS = 0;
 double meanVesselIS = 0;
 double stdVesselIS = 0;
 double meanRLInnerDiamS = 0;
 double stdRLInnerDiamS =0;
 double meanAPInnerDiamS = 0;
 double stdAPInnerDiamS = 0;
 double meanRLOuterDiamS = 0;
 double stdRLOuterDiamS =0;
 double meanAPOuterDiamS = 0;
 double stdAPOuterDiamS = 0;
 double meanLAS = 0;
 double stdLAS = 0;
 //Parenchyma atten,
 double meanPAS = 0;
 double stdPAS = 0;
 //Airway Phenotype based on energy conservation
 double meanEnergyS = 0;
 double stdEnergyS = 0;
 double meanPowerS = 0;
 double stdPowerS = 0;

 int IsamplesS=0;
 int WsamplesS=0;

 // Min - Max
 double minRiS =this->StatsMin->GetDataTypeMax();
 double maxRiS =this->StatsMax->GetDataTypeMin();
 double minRoS =this->StatsMin->GetDataTypeMax();
 double maxRoS = this->StatsMax->GetDataTypeMin();
 double minWthS =this->StatsMin->GetDataTypeMax();
 double maxWthS =this->StatsMax->GetDataTypeMin();
 double minWIS =0;
 double maxWIS =0;
 double minPeakIS =0;
 double maxPeakIS =0;
 double maxInnerIS = 0;
 double minInnerIS = 0;
 double maxOuterIS = 0;
 double minOuterIS = 0;
 double maxVesselIS = 0;
 double minVesselIS = 0;
 double maxRLInnerDiamS = 0;
 double minRLInnerDiamS = 0;
 double maxAPInnerDiamS = 0;
 double minAPInnerDiamS =0;
 double maxRLOuterDiamS = 0;
 double minRLOuterDiamS = 0;
 double maxAPOuterDiamS = 0;
 double minAPOuterDiamS =0;
 double minLAS = 5000;
 double maxLAS = -5000;
 double minPAS = 5000;
 double maxPAS = -5000;
 double minEnergyS = 5000;
 double maxEnergyS = -5000;
 double minPowerS = 5000;
 double maxPowerS = -5000;

 double sp[3];
 double loc1,loc2,tmp;
 double tmpMax, tmpMin;

 double tmpWIMax = -5000;
 double tmpPeakMax = -5000;
 double tmpInnerMax = -5000;
 double tmpOuterMax = -5000;
 double tmpWIMin = 5000;
 double tmpPeakMin = 5000;
 double tmpInnerMin = 5000;
 double tmpOuterMin = 5000;
 double tmpLA = 5000;
 double tmpPA = 0;
 double tmpEnergy = 0;
 double tmpPower = 0;

 // Use this tmp variable to avoid overflowding.
 double tmpExpectedWI2 = 1000*1000;

 double tmpWIMaxS = -5000;
 double tmpPeakMaxS = -5000;
 double tmpInnerMaxS = -5000;
 double tmpOuterMaxS = -5000;
 double tmpWIMinS = 5000;
 double tmpPeakMinS = 5000;
 double tmpInnerMinS = 5000;
 double tmpOuterMinS = 5000;

 // boolean variables
 int wrapping,condition;
 if (this->ThetaMax>2*vtkMath::Pi())
   wrapping = 1;
 else
   wrapping = 0;

vtkDoubleArray *radiusInner = vtkDoubleArray::New();
vtkDoubleArray *radiusOuter = vtkDoubleArray::New();
vtkDoubleArray *angleInner = vtkDoubleArray::New();
vtkDoubleArray *angleOuter = vtkDoubleArray::New();

vtkDoubleArray *lumenA = vtkDoubleArray::New();
int idx=0;
 for (double th =0 ; th < 2*vtkMath::Pi()-dth/2; th +=dth,idx++) {
    signalCollection->RemoveAllItems();
    for (int i=0; i<numKernels; i++) {
      ray = static_cast<vtkImageReformatAlongRay*> (rayCollection->GetItemAsObject(i));
      ray->SetTheta(th);
      ray->Update();
      signal = (vtkDoubleArray *)ray->GetOutput()->GetPointData()->GetScalars();
      ray->GetOutput()->GetSpacing(sp);
      signalCollection->AddItem(signal);
    }

    switch(this->Method) {
       case 0:
          this->FWHM(signal,samples);
          break;
       case 1:
          this->ZeroCrossing(signal,samples);
          break;
       case 2:
          this->PhaseCongruency(signal,samples);
          break;
       case 3:
          this->PhaseCongruencyMultipleKernels(signalCollection,samples,sp[0]);
          break;
    }
    loc1 = samples->GetValue(0);
    loc2 = samples->GetValue(1);

    if (loc1>loc2 && loc2!= -1) {
      cout<<"WARNING: Inner radius (loc1="<<loc1<<") is greater than outer radius (loc2="<<loc2<<")."<<endl;
      loc1=-1;
      loc2=-1;
    }
    //if (loc1>loc2)
    //  {
    //  loc1 = radiusInner->GetValue(idx-1)/delta;

    //if (th ==0)
    //  cout<<"Loc1: "<<loc1<<" "<<"Loc2: "<<loc2<<endl;

    //Take only into account good rays
    if (loc1 >0 && loc2 >0 ) {
        tmp = loc1*sp[0];
        meanRi +=tmp;
        stdRi += tmp*tmp;
        if (tmp>maxRi)
           maxRi = tmp;
        if (tmp<minRi)
           minRi = tmp;

        tmp = loc2*sp[0];
        meanRo +=tmp;
        stdRo +=tmp*tmp;
        if (tmp>maxRo)
          maxRo = tmp;
        if (tmp<minRo)
          minRo = tmp;

        tmp = (loc2-loc1)*sp[0];
        meanWth +=tmp;
        stdWth +=tmp*tmp;
        if (tmp > maxWth)
           maxWth = tmp;
        if (tmp < minWth)
           minWth = tmp;

        tmpMax = -5000;
        tmpMin = 5000;
        for(int k= (int) loc1; k< (int) loc2; k++) {
            tmp = signal->GetComponent(k,0);
            meanI += tmp;
            stdI += tmp*tmp;
            if (tmp>tmpMax)
              tmpMax = tmp;
            if (tmp<tmpMin)
              tmpMin = tmp;
            Isamples++;
        }
        // Peak Intensity
        meanPeakI +=tmpMax;
        stdPeakI +=tmpMax*tmpMax;

        // Min-Max
        // Wall Intensity
        if (tmpMax > tmpWIMax)
          tmpWIMax = tmpMax;
        if (tmpMin < tmpWIMin)
          tmpWIMin = tmpMin;
        // Peak Intensity
        if (tmpMax> tmpPeakMax)
          tmpPeakMax = tmpMax;
        if (tmpMax< tmpPeakMin)
          tmpPeakMin = tmpMax;
        // Inner and Outer: Mean and Min-Max
        tmp = signal->GetComponent((int) loc1,0);
        meanInnerI += tmp;
        stdInnerI += tmp*tmp;
        if (tmp > tmpInnerMax)
          tmpInnerMax = tmp;
        if (tmp < tmpInnerMin)
          tmpInnerMin = tmp;
        tmp = signal->GetComponent((int) loc2,0);
        meanOuterI += tmp;
        stdOuterI += tmp*tmp;
        if (tmp > tmpOuterMax)
          tmpOuterMax = tmp;
        if (tmp < tmpOuterMin)
          tmpOuterMin = tmp;

        maxInnerI = tmpInnerMax;
        minInnerI = tmpInnerMin;
        maxOuterI = tmpOuterMax;
        minOuterI = tmpOuterMin;

        Ai += pow(loc1*sp[0],2)*sin(dth)*0.5;
        Ae += pow(loc2*sp[0],2)*sin(dth)*0.5;
        Pi += sin(dth)*loc1*sp[0];

        if (th == 0 || (th > vtkMath::Pi()-dth/2 && th< vtkMath::Pi()+dth/2))
          {
          meanRLInnerDiam = loc1*sp[0] + meanRLInnerDiam;
          meanRLOuterDiam = loc2*sp[0] + meanRLOuterDiam;
          }

        if ((th > vtkMath::Pi()/2-dth/2 && th< vtkMath::Pi()/2+dth/2) || (th > 3*vtkMath::Pi()/2-dth/2 && th< 3*vtkMath::Pi()/2+dth/2))
          {
          meanAPInnerDiam = loc1*sp[0] + meanAPInnerDiam;
          meanAPOuterDiam = loc2*sp[0] + meanAPOuterDiam;
          }

         // Lumen attenuation: minimal attenuation inside the lumen
         // Find min value for this ray between 0 and 1/4th of the wall location
         // Insert values in the lumen Array: The stats should be computed outside the loop
         tmpLA = 10000;
         for (int k=0; k< (int) (loc1/4.0); k++) {
            if (signal->GetComponent(k,0)<tmpLA)
              tmpLA = signal->GetComponent(k,0);
         }
         lumenA->InsertNextValue(tmpLA);

        // Parenchymal attenuation: mean attenuation in the parenchyma
        // The parenchyma region is defined as the mean attenuation between
        // the first zero and lumen+alpha*(wall thickness)
        int gradsign = 1;
        if (signal->GetComponent((int) loc2,1) >= 0)
          gradsign = 1;
        else
          gradsign = -1;
       int zeroLoc = -1;
        for (int k=(int) loc2; k<signal->GetNumberOfTuples();k++)
          {
          if(signal->GetComponent(k,1) >= 0)
            {
            if (gradsign == -1)
              {
               zeroLoc = k;
               break;
              }
            }
          else {
            if (gradsign == 1)
             {
             zeroLoc = k;
             break;
             }
          }
          }
        // Compute mean parenchyma attenuation
        double Tpa = (loc1 + this->Alpha * (loc2-loc1));
        tmpPA = 0;
        int tmpSamples = 0;
        if ((int)Tpa >= signal->GetNumberOfTuples())
          {
          Tpa = signal->GetNumberOfTuples()-1;
          }
        if (zeroLoc > loc2)
          {
          for (int k = (int) zeroLoc ; k<(int) (Tpa); k++)
            {
            tmpPA += signal->GetComponent(k,0);
            tmpSamples++;
            }
          if (tmpSamples == 0)
            {
           tmpPA = signal->GetComponent((int) (Tpa),0);
           tmpSamples++;
            }
          }
        else
          {
          tmpPA = signal->GetComponent((int) (Tpa),0);
          tmpSamples++;
          }
        tmpPA=tmpPA/tmpSamples;
        meanPA += tmpPA;
        stdPA += tmpPA*tmpPA;
        if (tmpPA > maxPA)
          maxPA = tmpPA;
        if (tmpPA < minPA)
          minPA = tmpPA;

        // Compute energy metric:
        // E = IR * lumenA + thickness * wall attenuation + PA * (T- IR - WT)
        tmpEnergy = loc1*sp[0]*tmpLA + (loc2-loc1)*sp[0]*tmpMax + tmpPA*(this->T - loc1*sp[0] - (loc2-loc1)*sp[0]);
        meanEnergy += tmpEnergy;
        stdEnergy += tmpEnergy*tmpEnergy;
        if (tmpEnergy<minEnergy)
          minEnergy = tmpEnergy;
        if (tmpEnergy>maxEnergy)
          maxEnergy = tmpEnergy;

        // Compute power metric:
        // P = E/T
        tmpPower = tmpEnergy/(this->T);
        meanPower += tmpPower;
        stdPower += tmpPower*tmpPower;
        if (tmpPower<minPower)
          minPower = tmpPower;
        if (tmpPower>maxPower)
          maxPower = tmpPower;

        Wsamples++;

        //Check sector statistics
        if (this->ActivateSector) {
          condition = ((wrapping && (th>= this->ThetaMin || th<= ( this->ThetaMax - 2*vtkMath::Pi() ) )) ||
                       ((!wrapping) && (th>= this->ThetaMin && th<= this->ThetaMax)) );

        if (condition) {
          tmp = loc1*sp[0];
          meanRiS +=tmp;
          stdRiS += tmp*tmp;
          if (tmp>maxRiS)
           maxRiS = tmp;
          if (tmp<minRiS)
           minRiS = tmp;

          tmp = loc2*sp[0];
          meanRoS +=tmp;
          stdRoS +=tmp*tmp;
          if (tmp>maxRoS)
           maxRoS = tmp;
          if (tmp<minRoS)
           minRoS = tmp;

          tmp = (loc2-loc1)*sp[0];
          meanWthS +=tmp;
          stdWthS +=tmp*tmp;
          if (tmp>maxWthS)
           maxWthS = tmp;
          if (tmp<minWthS)
           minWthS = tmp;

          tmpMax = -5000;
          tmpMin = 5000;
          for(int k= (int) loc1; k< (int) loc2; k++) {
            tmp = signal->GetComponent(k,0);
            meanIS += tmp;
            stdIS += tmp*tmp;
            if (tmp>tmpMax)
              tmpMax = tmp;
            if (tmp<tmpMin)
              tmpMin = tmp;
            IsamplesS++;
          }
          // Peak Intensity
          meanPeakIS +=tmpMax;
          stdPeakIS +=tmpMax*tmpMax;

          // Min-Max
          // Wall Intensity
          if (tmpMax > tmpWIMaxS)
            tmpWIMaxS = tmpMax;
          if (tmpMin < tmpWIMinS)
            tmpWIMinS = tmpMin;
          // Peak Intensity
          if (tmpMax> tmpPeakMaxS)
            tmpPeakMaxS = tmpMax;
          if (tmpMax< tmpPeakMinS)
            tmpPeakMinS = tmpMax;

          // Inner and Outer: Mean and Min-Max
          tmp = signal->GetComponent((int) loc1,0);
          meanInnerIS += tmp;
          stdInnerIS += tmp*tmp;
          if (tmp > tmpInnerMaxS)
            tmpInnerMaxS = tmp;
          if (tmp < tmpInnerMinS)
            tmpInnerMinS = tmp;
          tmp = signal->GetComponent((int) loc2,0);
          meanOuterIS += tmp;
          stdOuterIS += tmp*tmp;
          if (tmp > tmpOuterMaxS)
            tmpOuterMaxS = tmp;
          if (tmp < tmpOuterMinS)
            tmpOuterMinS = tmp;

        AiS += pow(loc1*sp[0],2)*sin(dth)*0.5;
        AeS += pow(loc2*sp[0],2)*sin(dth)*0.5;
        PiS += sin(dth)*loc1*sp[0];

         // Lumen attenuation: it should be the same than the full airway
         // because the sector only affects the outer wall.

        // Compute mean parenchyma attenuation
        meanPAS += tmpPA;
        stdPAS += tmpPA*tmpPA;
        if (tmpPA > maxPAS)
          maxPAS = tmpPA;
        if (tmpPA < minPAS)
          minPAS = tmpPA;
        // Compute energy metric
        meanEnergyS += tmpEnergy;
        stdEnergyS += tmpEnergy*tmpEnergy;
        if (tmpEnergy<minEnergyS)
          minEnergyS = tmpEnergy;
        if (tmpEnergy>maxEnergyS)
          maxEnergyS = tmpEnergy;

        // Compute power metric:
        // P = E/(T)
        meanPowerS += tmpPower;
        stdPowerS += tmpPower*tmpPower;
        if (tmpPower<minPowerS)
          minPowerS = tmpPower;
        if (tmpPower>maxPowerS)
          maxPowerS = tmpPower;

        WsamplesS++;
        }
        }
     }
    else {
      // Add to the lumen array a value that is close
      lumenA->InsertNextValue(10000);
    }
    //Add points to final contour
//    cout<<th<<" "<<loc1<<" "<<loc2<<endl;
     if (loc1 > 0) {
       radiusInner->InsertNextValue(loc1*delta);
       angleInner->InsertNextValue(th);
       //ip->InsertNextPoint(center[0]+loc1*delta*cos(th),center[1]+loc1*delta*sin(th),0);
     } else {
       radiusInner->InsertNextValue(-1);
       angleInner->InsertNextValue(th);
     }
     if (loc2 > 0) {
       radiusOuter->InsertNextValue(loc2*delta);
       angleOuter->InsertNextValue(th);
       //op->InsertNextPoint(center[0]+loc2*delta*cos(th),center[1]+loc2*delta*sin(th),0);
     } else {
       radiusOuter->InsertNextValue(-1);
       angleOuter->InsertNextValue(th);
     }
 }

 //Remove outlier
 this->RemoveOutliers(radiusInner);
 this->RemoveOutliers(radiusOuter);

// Interpolate points that have not been computed
/*
vtkCardinalSpline *is = vtkCardinalSpline::New();
vtkCardinalSpline *os = vtkCardinalSpline::New();
idx = 0;
for (double th =-dth ; th <= 2*vtkMath::Pi(); th +=dth) {
 //Add an extra point at each end to deal with wrapping
  if (th==-dth)
     {
     if (radiusInner->GetValue(radiusInner->GetNumberOfTuples()-1) > 0)
       is->AddPoint(th,radiusInner->GetValue(radiusInner->GetNumberOfTuples()-1));
     if (radiusOuter->GetValue(radiusOuter->GetNumberOfTuples()-1) > 0)
       os->AddPoint(th,radiusOuter->GetValue(radiusOuter->GetNumberOfTuples()-1));
     }
  else if (th==2*vtkMath::Pi())
    {
     if (radiusInner->GetValue(0) > 0)
       is->AddPoint(th,radiusInner->GetValue(0));
     if (radiusOuter->GetValue(0) > 0)
       os->AddPoint(th,radiusOuter->GetValue(0));
    }
  else
    {
    if (radiusInner->GetValue(idx) > 0)
      is->AddPoint(th,radiusInner->GetValue(idx));
    if (radiusOuter->GetValue(idx) > 0)
      os->AddPoint(th,radiusOuter->GetValue(idx));
    idx++;
    }
}

// Reinterpolate results
idx = 0;
for (double th =0 ; th < 2*vtkMath::Pi(); th +=dth) {
  if (radiusInner->GetValue(idx) <=0) {
    cout<<"Theta inner="<<th<<" Value="<<is->Evaluate(th)<<endl;
    radiusInner->SetValue(idx,is->Evaluate(th));
  }
  if (radiusOuter->GetValue(idx) <=0) {
    cout<<"Th outer="<<th<<" Value="<<is->Evaluate(th)<<endl;
    radiusOuter->SetValue(idx,os->Evaluate(th));
  }
  idx++;
}
is->Delete();
os->Delete();
*/

// Configure output polylines:
// Fit spline to reparam
// Smooth line
// Save data

//vtkCardinalSpline *splineInner = vtkCardinalSpline::New();
//for (int i=0; i<radiusInner->GetNumberOfTuples();i++)
//  {
//  splineInner->AddPoint(angleInner->GetComponent(i,0),radiusInner->GetComponent(i,0));
//  }
//vtkCardinalSpline *splineOuter = vtkCardinalSpline::New();
//for (int i=0; i<radiusOuter->GetNumberOfTuples();i++)
//  {
//  splineOuter->AddPoint(angleOuter->GetComponent(i,0),radiusOuter->GetComponent(i,0));
//  }
// Reinterpolate
//radiusInner->Reset();
//angleInner->Reset();
//radiusOuter->Reset();
//angleOuter->Reset();
//for (double th =0 ; th < 2*vtkMath::Pi(); th +=dth) {
//  radiusInner->InsertNextValue(splineInner->Evaluate(th));
//  angleInner->InsertNextValue(th);
//  radiusOuter->InsertNextValue(splineOuter->Evaluate(th));
//  angleOuter->InsertNextValue(th);
//}
//splineInner->Delete();
//splineOuter->Delete();

// Smooth
vtkSmoothLines *sFilter = vtkSmoothLines::New();
sFilter->SetNumberOfIterations(10);
vtkDoubleArray *radiusInnerFilt = vtkDoubleArray::New();
vtkDoubleArray *radiusOuterFilt = vtkDoubleArray::New();
//sFilter->SolveHeatEquation(radiusInner,radiusInnerFilt);
//sFilter->SolveHeatEquation(radiusOuter,radiusOuterFilt);

//Save into points
//ip->SetNumberOfPoints(radiusInner->GetNumberOfTuples());
//op->SetNumberOfPoints(radiusOuter->GetNumberOfTuples());
int newcell = 1;
int cellcount = 0;
int pointIdx=0;
for (int i=0; i<radiusInner->GetNumberOfTuples();i++)
  {
  if (radiusInner->GetValue(i)>0) {
    ip->InsertNextPoint(center[0]+radiusInner->GetValue(i)*cos(angleInner->GetValue(i)),center[1]+radiusInner->GetValue(i)*sin(angleInner->GetValue(i)),0);
    if (newcell) {
      ic->InsertNextCell(radiusInner->GetNumberOfTuples());
      cellcount=0;
      newcell = 0;
    }
    ic->InsertCellPoint(pointIdx);
    cellcount++;
    pointIdx++;
  } else {
    // Finish this cell
     if (newcell == 0 ) {
      ic->UpdateCellCount(cellcount);
      newcell = 1;
     }
  }
  }
 //Finish the last cell
if (newcell == 0)
  ic->UpdateCellCount(cellcount);

pointIdx = 0;
cellcount = 0;
newcell = 1;
for (int i=0; i<radiusOuter->GetNumberOfTuples();i++)
  {
  if (radiusOuter->GetValue(i) > 0) {
    op->InsertNextPoint(center[0]+radiusOuter->GetValue(i)*cos(angleOuter->GetValue(i)),center[1]+radiusOuter->GetValue(i)*sin(angleOuter->GetValue(i)),0);
    if (newcell) {
      oc->InsertNextCell(radiusOuter->GetNumberOfTuples());
      cellcount=0;
      newcell = 0;
    }
    oc->InsertCellPoint(pointIdx);
    cellcount++;
    pointIdx++;
  } else {
    // Finish this cell
     if (newcell == 0) {
      oc->UpdateCellCount(cellcount);
      newcell = 1;
     }
  }
  }
//Finish the last cell
if (newcell == 0)
  oc->UpdateCellCount(cellcount);

//Delete smooth objects
radiusInnerFilt->Delete();
radiusOuterFilt->Delete();
radiusInner->Delete();
radiusOuter->Delete();
angleInner->Delete();
angleOuter->Delete();
sFilter->Delete();

/*
if (ip->GetNumberOfPoints()>0)
 ip->InsertNextPoint(ip->GetPoint(0));
if (op->GetNumberOfPoints()>0)
 op->InsertNextPoint(op->GetPoint(0));

*/
 //Add cell information
 /*
 ic->InsertNextCell(ip->GetNumberOfPoints());
 for (int i=0; i<ip->GetNumberOfPoints(); i++ ) {
   ic->InsertCellPoint(i);
 }

 oc->InsertNextCell(op->GetNumberOfPoints());
 for (int i=0; i<op->GetNumberOfPoints(); i++ ) {
   oc->InsertCellPoint(i);
 }
 // Impose circular symmetry by setting last point sample the first one
 ic->InsertCellPoint(0);
 oc->InsertCellPoint(0);
 */

 // Set output polydata
 this->InnerContour->SetPoints(ip);
 this->InnerContour->SetLines(ic);
 ip->Delete();
 ic->Delete();

 /*
 vtkSmoothLines *sf;
 sf = vtkSmoothLines::New();
 sf->SetInput(this->InnerContour);
 sf->SetNumberOfIterations(10);
 sf->Update();
 this->InnerContour->DeepCopy(sf->GetOutput());
 sf->Delete();
*/

 this->OuterContour->SetPoints(op);
 this->OuterContour->SetLines(oc);
 op->Delete();
 oc->Delete();

 /*
 vtkPolyDataWriter * ww=vtkPolyDataWriter::New();
 ww->SetInput(this->InnerContour);
 ww->SetFileName("innercontour.vtk");
 ww->Write();
 ww->Delete();
 vtkPolyDataWriter *ww2=vtkPolyDataWriter::New();
 ww2->SetInput(this->OuterContour);
 ww2->SetFileName("outercontour.vtk");
 ww2->Write();
ww2->Delete();
*/

 /*
 sf = vtkSmoothLines::New();
 sf->SetInput(this->OuterContour);
 sf->SetNumberOfIterations(10);
 sf->Update();
 this->OuterContour->DeepCopy(sf->GetOutput());
 sf->Delete();
*/

 meanRi = meanRi/Wsamples;
 meanRo = meanRo/Wsamples;
 meanWth = meanWth/Wsamples;
 stdRi = sqrt(stdRi/Wsamples -meanRi*meanRi);
 stdRo = sqrt(stdRo/Wsamples -meanRo*meanRo);
 stdWth = sqrt(stdWth/Wsamples -meanWth*meanWth);

 meanI = meanI/Isamples;
 stdI = sqrt(stdI/Isamples - meanI*meanI);
 //Inner and Outer mean boundary Intensity
 meanInnerI = meanInnerI/Wsamples;
 stdInnerI = sqrt(stdInnerI/Wsamples - meanInnerI*meanInnerI);
 meanOuterI = meanOuterI/Wsamples;
 stdOuterI = sqrt(stdOuterI/Wsamples - meanOuterI*meanOuterI);

 WAp = 100*(Ae-Ai)/Ae;

 sqrtWA = sqrt(Ae-Ai);

 //Peak for intensity
 meanPeakI = meanPeakI/Wsamples;
 stdPeakI = sqrt(stdPeakI/Wsamples - meanPeakI*meanPeakI);

  maxWI = tmpWIMax;
  minWI = tmpWIMin;
  maxPeakI = tmpPeakMax;
  minPeakI = tmpPeakMin;

  maxInnerI = tmpInnerMax;
  minInnerI = tmpInnerMin;
  maxOuterI = tmpOuterMax;
  minOuterI = tmpOuterMin;

// Vessel intensity
  meanVesselI = range[1];
  stdVesselI = 0.0;
  maxVesselI = range[1];
  minVesselI = range[1];

// RL and AP Diameters (this values are the same for sector)
  minRLInnerDiam = meanRLInnerDiam;
  maxRLInnerDiam = meanRLInnerDiam;
  minRLOuterDiam = meanRLOuterDiam;
  maxRLOuterDiam = meanRLOuterDiam;
  stdRLInnerDiam = 0;
  stdRLOuterDiam = 0;
  stdAPInnerDiam = 0;
  stdAPOuterDiam = 0;

  minAPInnerDiam = meanAPInnerDiam;
  maxAPInnerDiam = meanAPInnerDiam;
  minAPOuterDiam = meanAPOuterDiam;
  maxAPOuterDiam = meanAPOuterDiam;

// Lumen attenuation
int gap = this->NumberOfThetaSamples/2;
double lumenA1,lumenA2,lumenMin;
int lumenSamples=0;
meanLA = 0;
stdLA = 0;

for (int k=0; k< lumenA->GetNumberOfTuples()/2;k++)
  {
  lumenA1 = lumenA->GetValue(k);
  lumenA2 = lumenA->GetValue(k+gap);
  if (lumenA1<lumenA2)
    lumenMin = lumenA1;
  else
    lumenMin = lumenA2;
  if (lumenMin != 10000)
    {
    meanLA += lumenMin;
    stdLA += lumenMin*lumenMin;
    if (lumenMin<minLA)
      minLA = lumenMin;
    if (lumenMin>maxLA)
      maxLA = lumenMin;
    lumenSamples++;
    }
}

meanLA = meanLA/lumenSamples;
stdLA = sqrt(stdLA/lumenSamples - meanLA*meanLA);

// Parenchyma attenuation
meanPA = meanPA/Wsamples;
stdPA = sqrt(stdPA/Wsamples - meanPA*meanPA);

// Energy
meanEnergy = meanEnergy/Wsamples;
stdEnergy = sqrt(stdEnergy/Wsamples - meanEnergy*meanEnergy);

// Power
meanPower = meanPower/Wsamples;
stdPower = sqrt(stdPower/Wsamples - meanPower*meanPower);

 if (this->ActivateSector) {
   meanRiS = meanRiS/WsamplesS;
   meanRoS = meanRoS/WsamplesS;
   meanWthS = meanWthS/WsamplesS;
   stdRiS = sqrt(stdRiS/WsamplesS -meanRiS*meanRiS);
   stdRoS = sqrt(stdRoS/WsamplesS -meanRoS*meanRoS);
   stdWthS = sqrt(stdWthS/WsamplesS -meanWthS*meanWthS);

   meanIS = meanIS/IsamplesS;
   stdIS = sqrt(stdIS/IsamplesS - meanIS*meanIS);
   //Inner and Outer mean boundary Intensity
   meanInnerIS = meanInnerIS/WsamplesS;
   stdInnerIS = sqrt(stdInnerIS/WsamplesS - meanInnerIS*meanInnerIS);
   meanOuterIS = meanOuterI/WsamplesS;
   stdOuterIS = sqrt(stdOuterIS/WsamplesS - meanOuterIS*meanOuterIS);

   //Adjust sector area and perimiter values to account for that fact
   // that those values are only given over a sector.
   double factorSector;
   factorSector = (2*vtkMath::Pi())/(this->ThetaMax-this->ThetaMin);

   AeS = AeS*factorSector;
   AiS = AiS*factorSector;
   PiS = PiS*factorSector;
   WApS = 100*(AeS-AiS)/AeS;
   sqrtWAS = sqrt(AeS-AiS);

   //Peak for intensity
   meanPeakIS = meanPeakIS/WsamplesS;
   stdPeakIS = sqrt(stdPeakIS/WsamplesS - meanPeakIS*meanPeakIS);

   maxWIS = tmpWIMaxS;
   minWIS = tmpWIMinS;
   maxPeakIS = tmpPeakMaxS;
   minPeakIS = tmpPeakMinS;

   maxInnerIS = tmpInnerMaxS;
   minInnerIS = tmpInnerMinS;
   maxOuterIS = tmpOuterMaxS;
   minOuterIS = tmpOuterMinS;

   // Vessel intensity
   meanVesselIS = range[1];
   stdVesselIS = 0.0;
   maxVesselIS = range[1];
   minVesselIS = range[1];

 // RL and AP Diameters (this values are the same for sector)
  meanRLInnerDiamS = meanRLInnerDiam;
  meanRLOuterDiamS = meanRLOuterDiam;
  meanAPInnerDiamS = meanAPInnerDiam;
  meanAPOuterDiamS = meanAPOuterDiam;
  stdRLInnerDiamS = 0;
  stdRLOuterDiamS = 0;
  stdAPInnerDiamS = 0;
  stdAPOuterDiamS = 0;
  minRLInnerDiamS = meanRLInnerDiam;
  maxRLInnerDiamS = meanRLInnerDiam;
  minRLOuterDiamS = meanRLOuterDiam;
  maxRLOuterDiamS = meanRLOuterDiam;
  minAPInnerDiamS = meanAPInnerDiam;
  maxAPInnerDiamS = meanAPInnerDiam;
  minAPOuterDiamS = meanAPOuterDiam;
  maxAPOuterDiamS = meanAPOuterDiam;

   // Lumen attenuation
   meanLAS = meanLA;
   stdLAS = stdLA;
   // Parenchyma attenuation
   meanPAS = meanPAS/WsamplesS;
   stdPAS = sqrt(stdPAS/WsamplesS - meanPAS*meanPAS);
   // Energy
   meanEnergyS = meanEnergyS/WsamplesS;
   stdEnergyS = sqrt(stdEnergyS/WsamplesS - meanEnergyS*meanEnergyS);
   // Power
   meanPowerS = meanPowerS/WsamplesS;
   stdPowerS = sqrt(stdPowerS/WsamplesS - meanPowerS*meanPowerS);
}

 this->StatsMean->Initialize();
 this->StatsMean->SetNumberOfComponents(2);
 this->StatsMean->SetNumberOfTuples(this->NumberOfQuantities);
 this->StatsStd->Initialize();
 this->StatsStd->SetNumberOfComponents(2);
 this->StatsStd->SetNumberOfTuples(this->NumberOfQuantities);

 this->StatsMean->SetComponent(0,0,meanRi);
 this->StatsStd->SetComponent(0,0,stdRi);
 this->StatsMean->SetComponent(1,0,meanRo);
 this->StatsStd->SetComponent(1,0,stdRo);
 this->StatsMean->SetComponent(2,0,meanWth);
 this->StatsStd->SetComponent(2,0,stdWth);
 this->StatsMean->SetComponent(3,0,meanI);
 this->StatsStd->SetComponent(3,0,stdI);
 this->StatsMean->SetComponent(4,0,WAp);
 this->StatsStd->SetComponent(4,0,0.0);
 this->StatsMean->SetComponent(5,0,Pi);
 this->StatsStd->SetComponent(5,0,0.0);
 this->StatsMean->SetComponent(6,0,sqrtWA);
 this->StatsStd->SetComponent(6,0,0.0);
 this->StatsMean->SetComponent(7,0,Ai);
 this->StatsStd->SetComponent(7,0,0.0);
 this->StatsMean->SetComponent(8,0,Ae);
 this->StatsStd->SetComponent(8,0,0.0);
 this->StatsMean->SetComponent(9,0,meanPeakI);
 this->StatsStd->SetComponent(9,0,stdPeakI);
 this->StatsMean->SetComponent(10,0,meanInnerI);
 this->StatsStd->SetComponent(10,0,stdInnerI);
 this->StatsMean->SetComponent(11,0,meanOuterI);
 this->StatsStd->SetComponent(11,0,stdOuterI);
 this->StatsMean->SetComponent(12,0,meanVesselI);
 this->StatsStd->SetComponent(12,0,stdVesselI);
 this->StatsMean->SetComponent(13,0,meanRLInnerDiam);
 this->StatsStd->SetComponent(13,0,stdRLInnerDiam);
 this->StatsMean->SetComponent(14,0,meanRLOuterDiam);
 this->StatsStd->SetComponent(14,0,stdRLOuterDiam);
 this->StatsMean->SetComponent(15,0,meanAPInnerDiam);
 this->StatsStd->SetComponent(15,0,stdAPInnerDiam);
 this->StatsMean->SetComponent(16,0,meanAPOuterDiam);
 this->StatsStd->SetComponent(16,0,stdAPOuterDiam);
 this->StatsMean->SetComponent(17,0,meanLA);
 this->StatsStd->SetComponent(17,0,stdLA);
 this->StatsMean->SetComponent(18,0,meanPA);
 this->StatsStd->SetComponent(18,0,stdPA);
 this->StatsMean->SetComponent(19,0,meanEnergy);
 this->StatsStd->SetComponent(19,0,stdEnergy);
 this->StatsMean->SetComponent(20,0,meanPower);
 this->StatsStd->SetComponent(20,0,stdPower);

 this->StatsMean->SetComponent(0,1,meanRiS);
 this->StatsStd->SetComponent(0,1,stdRiS);
 this->StatsMean->SetComponent(1,1,meanRoS);
 this->StatsStd->SetComponent(1,1,stdRoS);
 this->StatsMean->SetComponent(2,1,meanWthS);
 this->StatsStd->SetComponent(2,1,stdWthS);
 this->StatsMean->SetComponent(3,1,meanIS);
 this->StatsStd->SetComponent(3,1,stdIS);
 this->StatsMean->SetComponent(4,1,WApS);
 this->StatsStd->SetComponent(4,1,0.0);
 this->StatsMean->SetComponent(5,1,PiS);
 this->StatsStd->SetComponent(5,1,0.0);
 this->StatsMean->SetComponent(6,1,sqrtWAS);
 this->StatsStd->SetComponent(6,1,0.0);
 this->StatsMean->SetComponent(7,1,AiS);
 this->StatsStd->SetComponent(7,1,0.0);
 this->StatsMean->SetComponent(8,1,AeS);
 this->StatsStd->SetComponent(8,1,0.0);
 this->StatsMean->SetComponent(9,1,meanPeakIS);
 this->StatsStd->SetComponent(9,1,stdPeakIS);
 this->StatsMean->SetComponent(10,1,meanInnerIS);
 this->StatsStd->SetComponent(10,1,stdInnerIS);
 this->StatsMean->SetComponent(11,1,meanOuterIS);
 this->StatsStd->SetComponent(11,1,stdOuterIS);
 this->StatsMean->SetComponent(12,1,meanVesselIS);
 this->StatsStd->SetComponent(12,1,stdVesselIS);
 this->StatsMean->SetComponent(13,1,meanRLInnerDiam);
 this->StatsStd->SetComponent(13,1,stdRLInnerDiam);
 this->StatsMean->SetComponent(14,1,meanRLOuterDiam);
 this->StatsStd->SetComponent(14,1,stdRLOuterDiam);
 this->StatsMean->SetComponent(15,1,meanAPInnerDiam);
 this->StatsStd->SetComponent(15,1,stdAPInnerDiam);
 this->StatsMean->SetComponent(16,1,meanAPOuterDiam);
 this->StatsStd->SetComponent(16,1,stdAPOuterDiam);
 this->StatsMean->SetComponent(17,1,meanLAS);
 this->StatsStd->SetComponent(17,1,stdLAS);
 this->StatsMean->SetComponent(18,1,meanPAS);
 this->StatsStd->SetComponent(18,1,stdPAS);
 this->StatsMean->SetComponent(19,1,meanEnergyS);
 this->StatsStd->SetComponent(19,1,stdEnergyS);
 this->StatsMean->SetComponent(20,1,meanPowerS);
 this->StatsStd->SetComponent(20,1,stdPowerS);

 this->StatsMin->Initialize();
 this->StatsMin->SetNumberOfComponents(2);
 this->StatsMin->SetNumberOfTuples(this->NumberOfQuantities);
 this->StatsMax->Initialize();
 this->StatsMax->SetNumberOfComponents(2);
 this->StatsMax->SetNumberOfTuples(this->NumberOfQuantities);
    
 this->StatsMin->SetComponent(0,0,minRi);
 this->StatsMax->SetComponent(0,0,maxRi);
 this->StatsMin->SetComponent(1,0,minRo);
 this->StatsMax->SetComponent(1,0,maxRo);
 this->StatsMin->SetComponent(2,0,minWth);
 this->StatsMax->SetComponent(2,0,maxWth);
 this->StatsMin->SetComponent(3,0,minWI);
 this->StatsMax->SetComponent(3,0,maxWI);
 this->StatsMin->SetComponent(4,0,WAp);
 this->StatsMax->SetComponent(4,0,WAp);
 this->StatsMin->SetComponent(5,0,Pi);
 this->StatsMax->SetComponent(5,0,Pi);
 this->StatsMin->SetComponent(6,0,sqrtWA);
 this->StatsMax->SetComponent(6,0,sqrtWA);
 this->StatsMin->SetComponent(7,0,Ai);
 this->StatsMax->SetComponent(7,0,Ai);
 this->StatsMin->SetComponent(8,0,Ae);
 this->StatsMax->SetComponent(8,0,Ae);
 this->StatsMin->SetComponent(9,0,minPeakI);
 this->StatsMax->SetComponent(9,0,maxPeakI);
 this->StatsMin->SetComponent(10,0,minInnerI);
 this->StatsMax->SetComponent(10,0,maxInnerI);
 this->StatsMin->SetComponent(11,0,minOuterI);
 this->StatsMax->SetComponent(11,0,maxOuterI);
 this->StatsMin->SetComponent(12,0,minVesselI);
 this->StatsMax->SetComponent(12,0,maxVesselI);
 this->StatsMin->SetComponent(13,0,minRLInnerDiam);
 this->StatsMax->SetComponent(13,0,maxRLInnerDiam);
 this->StatsMin->SetComponent(14,0,minRLOuterDiam);
 this->StatsMax->SetComponent(14,0,maxRLOuterDiam);
 this->StatsMin->SetComponent(15,0,minAPInnerDiam);
 this->StatsMax->SetComponent(15,0,maxAPInnerDiam);
 this->StatsMin->SetComponent(16,0,minAPOuterDiam);
 this->StatsMax->SetComponent(16,0,maxAPOuterDiam);
 this->StatsMin->SetComponent(17,0,minLA);
 this->StatsMax->SetComponent(17,0,maxLA);
 this->StatsMin->SetComponent(18,0,minPA);
 this->StatsMax->SetComponent(18,0,maxPA);
 this->StatsMin->SetComponent(19,0,minEnergy);
 this->StatsMax->SetComponent(19,0,maxEnergy);
 this->StatsMin->SetComponent(20,0,minPower);
 this->StatsMax->SetComponent(20,0,maxPower);

 if (this->ActivateSector == 0)
   {
   for (int k=0;k<this->NumberOfQuantities;k++)
     {
     this->StatsMin->SetComponent(k,1,0.0);
     this->StatsMax->SetComponent(k,1,0.0);

     }
   }
 else
   {
 this->StatsMin->SetComponent(0,1,minRiS);
 this->StatsMax->SetComponent(0,1,maxRiS);
 this->StatsMin->SetComponent(1,1,minRoS);
 this->StatsMax->SetComponent(1,1,maxRoS);
 this->StatsMin->SetComponent(2,1,minWthS);
 this->StatsMax->SetComponent(2,1,maxWthS);
 this->StatsMin->SetComponent(3,1,minWIS);
 this->StatsMax->SetComponent(3,1,maxWIS);
 this->StatsMin->SetComponent(4,1,WApS);
 this->StatsMax->SetComponent(4,1,WApS);
 this->StatsMin->SetComponent(5,1,PiS);
 this->StatsMax->SetComponent(5,1,PiS);
 this->StatsMin->SetComponent(6,1,sqrtWAS);
 this->StatsMax->SetComponent(6,1,sqrtWAS);
 this->StatsMin->SetComponent(7,1,AiS);
 this->StatsMax->SetComponent(7,1,AiS);
 this->StatsMin->SetComponent(8,1,AeS);
 this->StatsMax->SetComponent(8,1,AeS);
 this->StatsMin->SetComponent(9,1,minPeakIS);
 this->StatsMax->SetComponent(9,1,maxPeakIS);
 this->StatsMin->SetComponent(10,1,minInnerIS);
 this->StatsMax->SetComponent(10,1,maxInnerIS);
 this->StatsMin->SetComponent(11,1,minOuterIS);
 this->StatsMax->SetComponent(11,1,maxOuterIS);
 this->StatsMin->SetComponent(12,1,minVesselIS);
 this->StatsMax->SetComponent(12,1,maxVesselIS);
 this->StatsMin->SetComponent(13,1,minRLInnerDiamS);
 this->StatsMax->SetComponent(13,1,maxRLInnerDiamS);
 this->StatsMin->SetComponent(14,1,minRLOuterDiamS);
 this->StatsMax->SetComponent(14,1,maxRLOuterDiamS);
 this->StatsMin->SetComponent(15,1,minAPInnerDiamS);
 this->StatsMax->SetComponent(15,1,maxAPInnerDiamS);
 this->StatsMin->SetComponent(16,1,minAPOuterDiamS);
 this->StatsMax->SetComponent(16,1,maxAPOuterDiamS);
 this->StatsMin->SetComponent(17,1,minLAS);
 this->StatsMax->SetComponent(17,1,maxLAS);
 this->StatsMin->SetComponent(18,1,minPAS);
 this->StatsMax->SetComponent(18,1,maxPAS);
 this->StatsMin->SetComponent(19,1,minEnergyS);
 this->StatsMax->SetComponent(19,1,maxEnergyS);
 this->StatsMin->SetComponent(20,1,minPowerS);
 this->StatsMax->SetComponent(20,1,maxPowerS);
 }
 //remove objects
 samples->Delete();
 for (int i=0; i<numKernels; i++) {
   ray = static_cast<vtkImageReformatAlongRay*> (rayCollection->GetItemAsObject(i));
   //ray->Delete();
   extract = static_cast<vtkImageExtractComponents*> (extractCollection->GetItemAsObject(i));
   //extract->Delete();
 }

signalCollection->Delete();
rayCollection->Delete();
extractCollection->Delete();
lumenA->Delete();
}

void vtkComputeAirwayWall::RemoveOutliers(vtkDoubleArray *r) {
  double mean=0;
  double std=0;
  double e2=0;
  int tt = 0;
  for (int k=0; k < r->GetNumberOfTuples(); k++) {
    if ( r->GetValue(k) > 0) {
      mean += r->GetValue(k);
      e2 += r->GetValue(k) * r->GetValue(k);
      tt++;
    }
  }
  mean = mean/tt;
  e2 = e2/tt;
  std = sqrt(e2-mean*mean);

  //Compute a mean and std that is robust to outliers
  // We use (r-mean)+- 2sigma
  double meanr = 0;
  double e2r =0;
  double stdr = 0;
  tt = 0;
  for (int k=0;k<r->GetNumberOfTuples(); k++) {
    if (r->GetValue(k)>0) {
      if (fabs((r->GetValue(k)-mean)) < 2*std) {
	meanr += r->GetValue(k);
	e2r += r->GetValue(k) * r->GetValue(k);
        tt++;
      }
    }
  }

  meanr = meanr/tt;
  e2r = e2r/tt;

  stdr = sqrt(e2r-meanr*meanr);
  //cout<<"Robust mean = "<<meanr<<" Robust std = "<<stdr<<endl;

  //Set points to -1 that fall beyond the criteria
  for (int k=0; k<r->GetNumberOfTuples(); k++) {
    if (fabs(r->GetValue(k)-meanr) >= this->StdFactor*stdr) {
      r->SetValue(k,-1);
    }
  }
}

void vtkComputeAirwayWall::FWHM(vtkDoubleArray *ray,vtkDoubleArray *values) {
double rmin,rmax;
vtkDoubleArray *c = vtkDoubleArray::New();
vtkDoubleArray *cp = vtkDoubleArray::New();
vtkDoubleArray *cpp = vtkDoubleArray::New();

int ntuples = ray->GetNumberOfTuples();
c->SetNumberOfTuples(ntuples);
cp->SetNumberOfTuples(ntuples);
cpp->SetNumberOfTuples(ntuples);

for (int k=0; k<ntuples;k++) {
    c->SetValue(k,ray->GetComponent(k,0));
    cp->SetValue(k,ray->GetComponent(k,1));
    cpp->SetValue(k,ray->GetComponent(k,2));
}

this->FWHM(c,cp,cpp,rmin,rmax);

values->Initialize();
//values->SetNumberOfComponents(1);
values->SetNumberOfValues(2);
values->SetValue(0,rmin);
values->SetValue(1,rmax);

c->Delete();
cp->Delete();
cpp->Delete();
}

void vtkComputeAirwayWall::FWHM(vtkDoubleArray *c,vtkDoubleArray *cp, vtkDoubleArray *cpp, double &rmin, double &rmax) {
vtkDoubleArray *gzeros = vtkDoubleArray::New();
//vtkDoubleArray *hzeros = vtkDoubleArray::New();

this->FindZeros(cp,cpp,NULL,gzeros);
vtkDebugMacro("FWHM: Num zeros: "<<gzeros->GetNumberOfTuples());
//this->FindZeros(cpp,NULL,NULL,hzeros);
//vtkDebugMacro("FWHM: Num zeros: "<<hzeros->GetNumberOfTuples());

int nzeros = gzeros->GetNumberOfTuples();
double loc,loc1,loc2;
double val,val1,val2;
double valg;
int ntuples = c->GetNumberOfTuples();

if (nzeros<1)
  {
  rmin=-1;
  rmax=-1;
  return;
  }

//Auto adjust wall threhold if the current one is lower that a mean of the estimated luminal samples
int wallTh;
int meanLuminalI = 0;
int lumenloc = 15;
//Find mean intensity in lumen based on first zero
if (gzeros->GetValue(0)<lumenloc & gzeros->GetValue(0)>=1)
  lumenloc = (int) gzeros->GetValue(0);

for (int k=0;k< (int) lumenloc;k++)
  {
  if (k>=ntuples-1)
    break;
  meanLuminalI += (int) (c->GetValue(k));
  }
meanLuminalI = (int) (meanLuminalI/lumenloc);

if(this->WallThreshold <= meanLuminalI )
  {
  wallTh = this->WallThreshold + meanLuminalI;
  //cout<<"Mean luminal I: "<<meanLuminalI<<endl;
  //cout<<"Th floor: "<<this->WallThreshold<<endl;
  //cout<<"Wall threhosld readjusted to: "<<wallTh<<endl;
  }
else
  {
  wallTh = this->WallThreshold;
  }

int tmp;
for (int k=0; k<nzeros; k++) {
  loc=gzeros->GetValue(k);
  tmp = (int) (loc);
  //Check loc is in the allowed range
  if (int(loc)>=ntuples-1 || int(loc)<0)
    continue;
  //Wall Candidate
  val = c->GetValue((int) loc);
  valg = cp->GetValue((int) loc);
  //cout<<"Loc: "<<loc<<" Zero value: "<<val<<endl;
  if (val>wallTh ) {
       //Wall center point (loc) has to be a maxima (cpp =< 0). We let inflection points pass.
        if (cpp->GetValue((int) loc) > 0) {
	  continue;
	}
	// Get valley locations at both size of the wall maxima.
	if (k==0)
  {
	  loc1=1;
  }
	else
    {
    loc1 = gzeros->GetValue(k-1);
    }
  if (k>=nzeros-1)
    {
    loc2 = ntuples-1;
    }
  else
    {
    loc2 = gzeros->GetValue(k+1);
    }

  //Check loc1 is in the allowed range
  if (int(loc1) >=ntuples || int(loc1) <0) {
	  //Loc1 is out of range
	  break;
	} else {
    val1 = c->GetValue((int) loc1);
    rmin=this->FindValue(c,(int) loc1,(val+val1)/2);
    valg = cp->GetValue((int) rmin);
    // Check that the inner wall location gradient is above the threshold.
    if (fabs(valg)<this->GradientThreshold)
	    {
	      //cout<<" Gradient= "<<fabs(valg)<<" at rmin="<<rmin<<endl;
          continue;
	    }
            //cout<<"Find rmin: "<<rmin<<endl;
    }
	//Check loc2 is in the allowed range
  if (int(loc2) >=ntuples || int(loc2) <0) {
	  // Loc2 is out of range but loc1 was assigned, set rmax to -1 and let it finish.
    rmax = -1;
	} else {
            val2 = c->GetValue((int) loc2);
            rmax=this->FindValue(c,(int) loc,(val+val2)/2);
            //cout<<"Find rmax: "<<rmax<<endl;
	}
    gzeros->Delete();
    //hzeros->Delete();
    return;
    }
}

//We did not find a wall
rmin = -1;
rmax = -1;
gzeros->Delete();
//hzeros->Delete();
}

void vtkComputeAirwayWall::ZeroCrossing(vtkDoubleArray *ray,vtkDoubleArray *values) {
double rmin,rmax;
vtkDoubleArray *c = vtkDoubleArray::New();
vtkDoubleArray *cp = vtkDoubleArray::New();
vtkDoubleArray *cpp = vtkDoubleArray::New();

int ntuples = ray->GetNumberOfTuples();
c->SetNumberOfValues(ntuples);
cp->SetNumberOfValues(ntuples);
cpp->SetNumberOfValues(ntuples);

for (int k=0; k<ntuples;k++) {
    c->SetValue(k,ray->GetComponent(k,0));
    cp->SetValue(k,ray->GetComponent(k,1));
    cpp->SetValue(k,ray->GetComponent(k,2));
}

this->ZeroCrossing(c,cp,cpp,rmin,rmax);

values->Initialize();
//values->SetNumberOfComponents(1);
values->SetNumberOfValues(2);
values->SetValue(0,rmin);
values->SetValue(1,rmax);

c->Delete();
cp->Delete();
cpp->Delete();
}

void vtkComputeAirwayWall::ZeroCrossing(vtkDoubleArray *c,vtkDoubleArray *cp, vtkDoubleArray *cpp, double &rmin, double &rmax) {
vtkDoubleArray *gzeros = vtkDoubleArray::New();
vtkDoubleArray *hzeros = vtkDoubleArray::New();

this->FindZeros(cp,cpp,NULL,gzeros);
this->FindZeros(cpp,NULL,NULL,hzeros);

int ngzeros = gzeros->GetNumberOfTuples();
int nhzeros = hzeros->GetNumberOfTuples();
double loc,loc1,loc2;
double val,valg;
rmin=-1;
rmax=-1;
int nc = c->GetNumberOfTuples();
if (nc<3)
 {
 return;
 }

//Auto adjust wall threhold if the current one is lower that a mean of the first samples
int wallTh;
if(this->WallThreshold < (c->GetValue(0) + c->GetValue(1))/2)
  wallTh = this->WallThreshold + (int) ((c->GetValue(0) + c->GetValue(1))/2);
else
  wallTh = this->WallThreshold;

for (int k=0; k<ngzeros; k++) {
  loc=gzeros->GetValue(k);
  //Check loc is in the allowed range
  if ((int)(loc) >= nc-1 || (int)(loc) < 0)
    continue;

   //Wall center point (loc) has to be a maxima (cpp <= 0). We let inflection points pass.
   if (cpp->GetValue((int) loc) > 0) {
     continue;
   }
  //Wall Candidate
  val = c->GetValue((int) loc);
  if (val>wallTh) {
    for (int j=0;j<nhzeros-1;j++) {
        loc1=hzeros->GetValue(j);
        loc2=hzeros->GetValue(j+1);
        valg = cp->GetValue((int) loc1);
	//Zero crossing is beyond zero gradient
        if (loc1> loc) {
          break;
        }

        // Check zero crossing is between gradient zero and
        // that inner wall location gradient is above the threshold
        if(loc1<=loc && loc2>=loc && fabs(valg)>=this->GradientThreshold) {
            rmin=loc1;
            rmax=loc2;
            gzeros->Delete();
            hzeros->Delete();
            return;
        }
    }
  }
}

gzeros->Delete();
hzeros->Delete();
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - removed inputSignal->SetWholeExtent(0,ntuples-1,0,0,0,0);
//   reference: http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Removal_of_SetWholeExtent
// - removed inputSignal->UpdateInformation();
//   reference: http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Removal_of_Update

void vtkComputeAirwayWall::PhaseCongruency(vtkDoubleArray *ray,vtkDoubleArray *values) {
// Derivatives of phase congruency
vtkDoubleArray *pc1 = vtkDoubleArray::New();
vtkDoubleArray *pc2 = vtkDoubleArray::New();
vtkDoubleArray *c = vtkDoubleArray::New();
vtkDoubleArray *cp = vtkDoubleArray::New();

int ntuples = ray->GetNumberOfTuples();
pc1->SetNumberOfValues(ntuples);
pc2->SetNumberOfValues(ntuples);
c->SetNumberOfValues(ntuples);
cp->SetNumberOfValues(ntuples);

for (int k=0; k<ntuples;k++) {
  c->SetValue(k,ray->GetComponent(k,0));
  cp->SetValue(k,ray->GetComponent(k,1));
}

vtkImageData *inputSignal = vtkImageData::New();
inputSignal->SetDimensions(ntuples,1,1);
// FIX_ME_VTK6
//inputSignal->SetWholeExtent(0,ntuples-1,0,0,0,0);
inputSignal->SetExtent(0,ntuples-1,0,0,0,0);
inputSignal->SetSpacing(1,1,1);
//inputSignal->SetNumberOfScalarComponents(1);
//inputSignal->SetScalarTypeToDouble();
inputSignal->GetPointData()->SetScalars(c);
//inputSignal->UpdateInformation();

vtkGeneralizedPhaseCongruency *pcFilter = vtkGeneralizedPhaseCongruency::New();
pcFilter->SetInputData(inputSignal);
pcFilter->SetNumberOfScales(this->NumberOfScales);
pcFilter->SetBandwidth(this->Bandwidth);
pcFilter->SetMultiplicativeFactor(this->MultiplicativeFactor);
pcFilter->SetMinimumWavelength(this->MinimumWavelength);
pcFilter->SetUseWeights(this->UseWeights);
pcFilter->SetWeights(this->Weights);
pcFilter->Update();

// Vector with the Phase Congruency results
vtkDoubleArray *pcV = (vtkDoubleArray *) (pcFilter->GetOutput()->GetPointData()->GetScalars());

for (int k=0; k<ntuples;k++) {
  pc1->SetValue(k,pcV->GetComponent(k,1));
  pc2->SetValue(k,pcV->GetComponent(k,2));
}

this->PhaseCongruency(c, cp, pc1, pc2, values);

pcFilter->Delete();
inputSignal->Delete();
c->Delete();
cp->Delete();
pc1->Delete();
pc2->Delete();
}

void vtkComputeAirwayWall::PhaseCongruency(vtkDoubleArray *c, vtkDoubleArray *cp,vtkDoubleArray *pcV, vtkDoubleArray *values) {
vtkDoubleArray *pc1 = vtkDoubleArray::New();
vtkDoubleArray *pc2 = vtkDoubleArray::New();

int ntuples = pcV->GetNumberOfTuples();
pc1->SetNumberOfValues(ntuples);
pc2->SetNumberOfValues(ntuples);

for (int k=0; k<ntuples;k++) {
  pc1->SetValue(k,pcV->GetComponent(k,1));
  pc2->SetValue(k,pcV->GetComponent(k,2));
}

this->PhaseCongruency(c,cp,pc1,pc2,values);

pc1->Delete();
pc2->Delete();
}

void vtkComputeAirwayWall::PhaseCongruency(vtkDoubleArray *c, vtkDoubleArray *cp, vtkDoubleArray *pc1, vtkDoubleArray *pc2, vtkDoubleArray *values)
{
double rmin,rmax;

int ntuples = pc1->GetNumberOfTuples();
int nc = c->GetNumberOfTuples();

vtkDoubleArray *dpc1 = vtkDoubleArray::New();
vtkDoubleArray *dpc2 = vtkDoubleArray::New();

dpc1->SetNumberOfValues(ntuples);
dpc2->SetNumberOfValues(ntuples);

//Finite differences using five-point method
for (int k=2;k<ntuples-2;k++) {
   dpc1->SetValue(k,(-pc1->GetValue(k+2)+8*pc1->GetValue(k+1)-8*pc1->GetValue(k-1)+pc1->GetValue(k-2))/12);
   dpc2->SetValue(k,(-pc2->GetValue(k+2)+8*pc2->GetValue(k+1)-8*pc2->GetValue(k-1)+pc2->GetValue(k-2))/12);
}
//For points close to the boundaries use central finite differences
int boundaries[2];
boundaries[0]=1;
boundaries[1]=ntuples-2;
for (int bb=0;bb<2;bb++) {
  int k = boundaries[bb];
  dpc1->SetValue(k,(pc1->GetValue(k+1)-pc1->GetValue(k-1))*0.5);
  dpc2->SetValue(k,(pc2->GetValue(k+1)-pc2->GetValue(k-1))*0.5);
}
dpc1->SetValue(0,dpc1->GetValue(1));
dpc2->SetValue(0,dpc2->GetValue(1));
dpc1->SetValue(ntuples-1,dpc1->GetValue(ntuples-2));
dpc2->SetValue(ntuples-1,dpc2->GetValue(ntuples-2));

vtkDoubleArray *pczeros = vtkDoubleArray::New();
int npczeros;
double loc=0;
double val, valg, pcval;
rmin=-1;
rmax=-1;
if (nc<3)
 {
 return;
 }

this->FindZeros(dpc1,NULL,NULL,pczeros);
npczeros = pczeros->GetNumberOfTuples();

//Auto adjust wall threhold if the current one is lower that a mean of the estimated luminal samples
double wallTh;
int meanLuminalI=0;
int zeroLoc=-1;
//Find first location where pc correspondig to pczero is greater than zero (or PCThreshold to avoid noisy areas)
for (int k=0; k< npczeros;k++) {
  loc = pczeros->GetValue(k);
  if (loc<nc)
    {
    if (pc1->GetValue((int) loc) > this->PCThreshold)
      {
      zeroLoc = k;
      break;
      }
    }
}

if (zeroLoc >= 0)
{
  for (int k=0; k< (int) (pczeros->GetValue(zeroLoc));k++)
    {
      if (k>=nc-1)
        break;
      meanLuminalI += (int) (c->GetValue(k));
    }
  meanLuminalI = (int) (meanLuminalI/pczeros->GetValue(zeroLoc));
} else {
  meanLuminalI = (int) (c->GetValue(0) + c->GetValue(1))/2;
}

// For PC, wallTh is used to probe the pixel value at the boundary
//point, no like FWHM and ZeroCrossing.
// We define a more relax threshold (100% of the original value).
if(this->WallThreshold < meanLuminalI)
  {
  wallTh = meanLuminalI;
  }
else
  {
  //Threalhold at airway wall is taken as FWHM value
  wallTh = (this->WallThreshold + (meanLuminalI))*0.5;
  }

double dc_offset=0;
if (wallTh < 0 )
  dc_offset = 1000;

//Gradient threshold
double wallGradTh = this->GradientThreshold;

// Find Inner Wall
for (int k=0; k<npczeros; k++) {
  loc=pczeros->GetValue(k);

  //Check loc is in the allowed range
  if ( loc >=nc-1 || loc >=ntuples-1 || loc<2)
    continue;

  //Wall Candidate
  val = c->GetValue((int) loc);
  valg = cp->GetValue((int) loc);
  pcval = pc1->GetValue((int) loc);
  //We check if the relative wall intensity is greater than -10% * pcval (so
  // the threshold is modulated by pcval such as if pcval=1 we allow a 10% negative variability in the
  // intensity threshold) and that pcval is greater than the threshold
  if ((val - wallTh)/(wallTh+dc_offset) > -0.10*pcval && (fabs(valg) - wallGradTh)/wallGradTh > -0.10*pcval && pcval > this->PCThreshold) {
    rmin = loc;
    break;
  }
}

pczeros->Delete();
pczeros = vtkDoubleArray::New();

// Find Outer Wall
this->FindZeros(dpc2,NULL,NULL,pczeros);
npczeros = pczeros->GetNumberOfTuples();

for (int k=0; k<npczeros; k++) {
  loc=pczeros->GetValue(k);
  //Wall Candidate
  val = c->GetValue((int) loc);
  pcval = pc2->GetValue((int) loc);
  if ((val - wallTh)/(wallTh+dc_offset) > -0.20*pcval && pcval > this->PCThreshold && loc>rmin) {
    rmax = loc;
    break;
  }
}

values->Initialize();
//values->SetNumberOfComponents(1);
values->SetNumberOfValues(2);
values->SetValue(0,rmin);
values->SetValue(1,rmax);

pczeros->Delete();
dpc1->Delete();
dpc2->Delete();
}

void vtkComputeAirwayWall::PhaseCongruencyMultipleKernels(vtkDataArrayCollection *signalCollection,vtkDoubleArray *values,double sp) {
double rmin,rmax;

int numKernels = signalCollection->GetNumberOfItems();

vtkDoubleArray *upcrossing = vtkDoubleArray::New();
vtkDoubleArray *downcrossing = vtkDoubleArray::New();

rmin=-1;
rmax=-1;

//Auto adjust wall threhold if the current one is lower that a mean of the first samples
vtkDoubleArray *c = (vtkDoubleArray *) (signalCollection->GetItemAsObject(0));
int wallTh;
if(this->WallThreshold < (c->GetComponent(0,0) + c->GetComponent(1,0))/2)
  wallTh = this->WallThreshold + (int) ((c->GetComponent(0,0) + c->GetComponent(1,0))/2);
else
  wallTh = this->WallThreshold;

vtkDoubleArray *ci;
vtkDoubleArray *cj;
double f1,f2,df1,df2,ddf1,ddf2;
double A,B,C,res;
double zc[2];
//Find crossing by doing every pair combination
for (int i=0; i<numKernels-1; i++) {
  for (int j=i+1;j<numKernels;j++) {
    ci = static_cast<vtkDoubleArray *> (signalCollection->GetItemAsObject(i));
    cj = static_cast<vtkDoubleArray *> (signalCollection->GetItemAsObject(j));

    for (int k=1;k<ci->GetNumberOfTuples()-1;k++) {
       f1=ci->GetComponent(k,0);
       df1=ci->GetComponent(k,1);
       ddf1=ci->GetComponent(k,2);
       f2=cj->GetComponent(k,0);
       df2=cj->GetComponent(k,1);
       ddf2=cj->GetComponent(k,2);
       df1 = df1*sp;
       //ddf1 = 0;
       df2 = df2*sp;
       //ddf2 = 0;
       ddf2 = 0;
       ddf1 = 0;

       //Build quadratic equation
       A = 0.5*(ddf1-ddf2);
       B = df1-df2-ddf1*k+ddf2*k;
       C = -(f2-f1+df1*k-df2*k-0.5*ddf1*k*k+0.5*ddf2*k*k);
       res = B*B-4*A*C;
       if (res<0)
         break;
       // Linear case
       if (A == 0) {
         zc[0] = -C/B;
         zc[1] = -C/B;
       } else {
         zc[0] = (-B + sqrt(res))/(2*A);
         zc[1] = (-B - sqrt(res))/(2*A);
       }
       //cout<<"zc[0]: "<<zc[0]<<" zc[1]: "<<zc[1]<<endl;

       for (int sol=0 ; sol<2; sol++) {
         if ((zc[sol]<(k+1) && zc[sol]>(k-1)) && (f1+f2)/2 > wallTh)
           {
           //Insert values
           if ((df1+df2)/2>0)
             upcrossing->InsertNextValue(zc[sol]);
           else
             downcrossing->InsertNextValue(zc[sol]);
           }
        }
    }
  } //loop retreiving cj kernel
} //loop retreiving ci kernel

// Find Inner Wall: median of upcrossing for now
//std::list<double> medianinner;
//for (int i=0;i<upcrossing->GetNumberOfTuples();i++) {
//  medianinner.push_back(upcrossing->GetValue(i));
//}
//medianinner.sort();
rmin=-1;
double tmp;

//Compute mean of values that gathered around 10 units of the first value
if (upcrossing->GetNumberOfTuples() > 0)
  rmin = upcrossing->GetValue(0);
for (int i=1;i<upcrossing->GetNumberOfTuples();i++) {
  tmp = (rmin + upcrossing->GetValue(i)/i)*i/(i+1);
  if (fabs(tmp - upcrossing->GetValue(i))<10)
    rmin = tmp;
}
/*
if (upcrossing->GetNumberOfTuples() > 0)
  rmin /=upcrossing->GetNumberOfTuples();
*/

// Find Outer Wall: median of downcrossing for now
rmax=-1;
/*
for (int i=0;i<downcrossing->GetNumberOfTuples();i++) {
  rmax += downcrossing->GetValue(i);
}
if (downcrossing->GetNumberOfTuples() > 0)
  rmax /=downcrossing->GetNumberOfTuples();
*/
for (int i=0;i<downcrossing->GetNumberOfTuples();i++) {
  if (downcrossing->GetValue(i) > rmin) {
    rmax = downcrossing->GetValue(i);
    break;
  }
}

//Compute mean of values that gathered around 10 units of the first value
for (int i=1;i<downcrossing->GetNumberOfTuples();i++) {
  if (downcrossing->GetValue(i) > rmin) {
    tmp = (rmax + downcrossing->GetValue(i)/i)*i/(i+1);
    if (fabs(tmp - downcrossing->GetValue(i)) < 10)
      rmax=tmp;
  }
}

values->Initialize();
//values->SetNumberOfComponents(1);
values->SetNumberOfValues(2);
values->SetValue(0,rmin);
values->SetValue(1,rmax);

upcrossing->Delete();
downcrossing->Delete();
}

double vtkComputeAirwayWall::FindValue(vtkDoubleArray *c, int loc, double target) {
int ntuples = c->GetNumberOfTuples();
double val0 = c->GetValue(loc);
int up;
if (val0<target) {
 up = 1;
} else {
 up =0;
}
for (int k=loc; k< ntuples; k++) {
  val0=c->GetValue(k);
  if (up == 1 && val0 > target) {
    return (k+(k-1))/2;
  }
  if (up == 0 && val0 < target) {
    return (k+(k-1))/2;
  }
}
// If we do not find the value return -1.
return -1;
}

//Find the zeros of a signal. The signal is given in a vtkDoubleArray and the zeros are return as 0-based coordinates.
// If derivaties of the signal are available, these are passed in cp and cpp.
void vtkComputeAirwayWall::FindZeros(vtkDoubleArray *c, vtkDoubleArray *cp, vtkDoubleArray *cpp,vtkDoubleArray *zeros) {
if (zeros == NULL)
    return;
zeros->Initialize();
int np = c->GetNumberOfTuples();

int derivatives =0;
if (cp != NULL) {
   derivatives++;
   if (cpp != NULL)
     derivatives++;
}
int initIdx =0;
int err;
double val0,val1,zero;
int k=0;
while (initIdx < np-1) {
   val0 = c->GetValue(initIdx);

   //Check if we are in a zero
   if (val0 ==0)
     {
     zeros->InsertNextValue(initIdx);
     initIdx++;
     continue;
     }

   for (k =initIdx+1; k<np; k++) {
      val1 = c->GetValue(k);
      //cout<<"k: "<<k<<" Val1: "<<val1<<endl;
      //Check for a change of sign
      if (val0*val1 <0) {
        val0 = c->GetValue(k-1);
         switch (derivatives) {
             case 0:
                err =this->FindZeroLocation(val0,val1,1,zero);
                break;
            case 1:
                err =this->FindZeroLocation(val0,cp->GetValue(k-1),
                                        val1,cp->GetValue(k),1,zero);
                //err = this->FindZeroLocation(val0,val1,1,zero);
                break;
            case 2:
                err =this->FindZeroLocation(val0,cp->GetValue(k-1),cpp->GetValue(k-1),
                                        val1,cp->GetValue(k),cpp->GetValue(k),1,zero);
                break;
          }
         if(err == 1) {
           zeros->InsertNextValue(k-1+zero);
         }
         //Break to while loop starting from initIdx
         // val1 becomes val0
         initIdx= k;
         break;
      }
    }
   if (k>=np)
    break;
}
}

// Finds the coordinate of the zero crossing based on the bracket points around the zero.
// First and second order derivative are known.
// fm1 = f(x_1): point left to the zero
// f1 = f(x_1): point rigth to the zero
// Function return zero is problem has been found finding zero location.
int  vtkComputeAirwayWall::FindZeroLocation(double fm1, double fm1p, double fm1pp, double f1, double f1p, double f1pp, double delta, double & zero) {
double a = 0.5*fm1pp - 0.5*f1pp;
double b = fm1p - f1p + delta * f1pp;
double c = fm1 - f1 + delta*f1p - delta*delta*0.5*f1pp;

if (b*b < 4 *a * c) {
  // zero is not a real number
  return 0;
}

double x1 = (-b + sqrt(b*b - 4*a*c))/(2*a);
double x2 = (-b - sqrt(b*b - 4*a*c))/(2*a);

if (x1< 0 && x2>0)
  zero = x2;
else if (x2<0 && x1>0)
  zero = x1;
else
  return 0;

return 1;
}
// Finds the coordinate of the zero crossing based on the bracket points around the zero.
// First order derivative are known.
// fm1 = f(x_1): point left to the zero
// f1 = f(x_1): point rigth to the zero
// Function return zero is problem has been found finding zero location.
int  vtkComputeAirwayWall::FindZeroLocation(double fm1, double fm1p, double f1, double f1p, double delta, double & zero) {
//zero = (f1 - delta*f1p - fm1)/(fm1p - f1p);

if (fm1p < 1e-15 && f1p > 1e-15)
  {
  zero = delta*(1-f1/f1p);
  }
else if (fm1p > 1e-15 && f1p < 1e-15)
  {
  zero = -delta*fm1/fm1p;
  }
else if (fm1p > 1e-15 && f1p > 1e-15)
  {
  //Better formula based on two Newton-Raphson estimate for left point and right point
  zero = 0.5* ( -delta*fm1/fm1p + delta*(1-f1/f1p));
  }
else
  {
  vtkComputeAirwayWall::FindZeroLocation(fm1, f1,delta,zero);
  }
if (zero < 0)
  return 0;
else
  return 1;
}

int  vtkComputeAirwayWall::FindZeroLocation(double fm1, double f1, double delta, double & zero) {
//zero = delta*fabs(fm1)/(fabs(fm1)+fabs(f1));
//Same results but without fabs
zero = -delta * fm1/(f1-fm1);
return 1;
}

void vtkComputeAirwayWall::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

