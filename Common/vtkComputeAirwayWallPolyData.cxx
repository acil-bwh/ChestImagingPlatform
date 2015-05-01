/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkComputeAirwayWallPolyData.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkComputeAirwayWallPolyData.h"

#include "vtkCellArray.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkPolyData.h"
#include "vtkMath.h"
#include "vtkImageResliceWithPlane.h"
#include "vtkStructuredPointsWriter.h"
#include "vtkEllipseFitting.h"
#include "vtkImageMapToColors.h"
#include "vtkLookupTable.h"
#include "vtkPNGWriter.h"

#ifdef WIN32
#define round(x) floor((x)+0.5)
#endif

vtkStandardNewMacro(vtkComputeAirwayWallPolyData);

vtkComputeAirwayWallPolyData::vtkComputeAirwayWallPolyData()
{
  this->AxisMode = VTK_HESSIAN;
  this->Reformat = 1;
  this->WallSolver = vtkComputeAirwayWall::New();
  this->AxisArray = vtkDoubleArray::New();
  this->SelfTuneModelSmooth[0]=0.0017;
  this->SelfTuneModelSmooth[1]=454.0426;
  this->SelfTuneModelSmooth[2]=3.0291;
  this->SelfTuneModelSharp[0]=0.0020;
  this->SelfTuneModelSharp[1]=493.6570;
  this->SelfTuneModelSharp[2]=2.9639;
  
  this->SegmentPercentage = 0.5;
  
  this->Reconstruction = VTK_SMOOTH;
  this->Image = NULL;
  
  this->AirwayImagePrefix= NULL;
  this->SaveAirwayImage=0;
}

vtkComputeAirwayWallPolyData::~vtkComputeAirwayWallPolyData()
{
  this->WallSolver->Delete();
  this->AxisArray->Delete();
  if (this->Image)
    this->Image->Delete();
}

/*
void vtkComputeAirwayWallPolyData::ExecuteInformation()
{

    if (this->GetInput() == NULL || this->GetImage() == NULL)
    {
    vtkErrorMacro("Missing input or image");
    return;
    }

  // Copy whole extent ...
  this->vtkSource::ExecuteInformation();

  

 //Create point Data for each stats
 vtkDoubleArray *mean = vtkDoubleArray::New();
 vtkDoubleArray *std = vtkDoubleArray::New();
 vtkDoubleArray *min = vtkDoubleArray::New();
 vtkDoubleArray *max = vtkDoubleArray::New();
 
 int nc = this->WallSolver->GetNumberOfQuantities();
 int np = this->GetInput()->GetNumberOfPoints();
 
 mean->SetName("Mean");
 mean->SetNumberOfComponents(nc);
 mean->SetNumberOfTuples(np);
 std->SetName("Std");
 std->SetNumberOfComponents(nc);
 std->SetNumberOfTuples(np);
 min->SetName("Min");
 min->SetNumberOfComponents(nc);
 min->SetNumberOfTuples(np);
 max->SetName("Max");
 max->SetNumberOfComponents(nc);
 max->SetNumberOfTuples(np);
 
 this->GetOutput()->GetPointData()->AddArray(mean);
 this->GetOutput()->GetPointData()->AddArray(std);
 this->GetOutput()->GetPointData()->AddArray(min);
 this->GetOutput()->GetPointData()->AddArray(max);
 
 mean->Delete();
 std->Delete();
 min->Delete();
 max->Delete();

 if (this->GetOutput()->GetPointData()->GetScalars("Mean") != NULL)
 {
   cout<<"Allocated"<<endl;
 }
 
 
 //Allocate Cell Arrays
 if (this->GetInput()->GetLines())
 {
   vtkCellArray *inLines =this->GetInput()->GetLines();
   int nl = inLines->GetNumberOfCells();
   
   vtkDoubleArray *mean = vtkDoubleArray::New();
   vtkDoubleArray *std = vtkDoubleArray::New();
   vtkDoubleArray *min = vtkDoubleArray::New();
   vtkDoubleArray *max = vtkDoubleArray::New();
  
   mean->SetName("Mean");
   mean->SetNumberOfComponents(nc);
   mean->SetNumberOfTuples(nl);
   std->SetName("Std");
   std->SetNumberOfComponents(nc);
   std->SetNumberOfTuples(nl);
   min->SetName("Min");
   min->SetNumberOfComponents(nc);
   min->SetNumberOfTuples(nl);
   max->SetName("Max");
   max->SetNumberOfComponents(nc);
   max->SetNumberOfTuples(nl);
   
   this->GetOutput()->GetCellData()->AddArray(mean);
   this->GetOutput()->GetCellData()->AddArray(std);
   this->GetOutput()->GetCellData()->AddArray(min);
   this->GetOutput()->GetCellData()->AddArray(max);
 
   mean->Delete();
   std->Delete();
   min->Delete();
   max->Delete();
 }
 

}
*/


void vtkComputeAirwayWallPolyData::ComputeAirwayAxisFromLines() {
  
  vtkIdType *pts = 0;
  vtkIdType npts = 0;
  double d[3];
  vtkPolyData *input = vtkPolyData::SafeDownCast(this->GetInput());
  
  if (input->GetLines() == NULL) {
    vtkErrorMacro("No Lines to compute airway axis. Either use hessian or do not reformat airways");
    return;
  }
  
  vtkCellArray *inLines = input->GetLines();
  int numLines = inLines->GetNumberOfCells();
  
  //Allocate axis DataArray
  this->AxisArray->SetNumberOfComponents(3);
  this->AxisArray->SetNumberOfTuples(input->GetNumberOfPoints());
  inLines->InitTraversal();
  for (int id=0; inLines->GetNextCell(npts,pts); id++)
    {
      for (int kk=0; kk<npts; kk++)
      {
	for (int c=0;c<3;c++) {
	  if (kk == 0)
	  {
	    d[c]=input->GetPoint(pts[kk+1])[c]-input->GetPoint(pts[kk])[c];
	  }  
	  else if (kk == npts-1) {
	    d[c]=input->GetPoint(pts[kk])[c]-input->GetPoint(pts[kk-1])[c];
	  } else {
	    d[c]=input->GetPoint(pts[kk+1])[c]-input->GetPoint(pts[kk-1])[c];
	  }   
	}
	vtkMath::Normalize(d);
	d[0]=-d[0];
	d[1]=-d[1];
	for (int c=0;c<3;c++) {
	  this->AxisArray->SetComponent(pts[kk],c,d[c]);
	}
      }
	
      }
}

// ---------------------------------------------------------------------------
// VTK6 migration note:
// Introduced to replace Execute().
int vtkComputeAirwayWallPolyData::RequestData(vtkInformation *request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkPolyData *input= vtkPolyData::SafeDownCast(this->GetInput());
  vtkPolyData *output = this->GetOutput();
  vtkImageData *im = this->GetImage();
  double orig[3];
  double sp[3], p[3],ijk[3];
  double x[3],y[3],z[3];
  im->GetOrigin(orig);
  im->GetSpacing(sp);
  
  output->DeepCopy(input);
  
  cout<<"Spacing: "<<sp[0]<<" "<<sp[1]<<" "<<sp[2]<<endl;
  cout<<"Origin: "<<orig[0]<<" "<<orig[1]<<" "<<orig[2]<<endl;
  
  double resolution = 0.1;

  //Create helper objects
  vtkImageResliceWithPlane *reslicer = vtkImageResliceWithPlane::New();
  // Set up options
  if (this->GetReformat()){
    reslicer->InPlaneOff();
  } else {
    reslicer->InPlaneOn();
  }
  reslicer->SetInputData(this->GetImage());
  reslicer->SetInterpolationModeToCubic();
  reslicer->ComputeCenterOff();
  reslicer->SetDimensions(256,256,1);
  reslicer->SetSpacing(resolution,resolution,resolution);

  switch(this->GetAxisMode()) {
    case VTK_HESSIAN:
      reslicer->ComputeAxesOn();
      break;
    case VTK_POLYDATA:
      if (input->GetLines() == NULL) {
        reslicer->ComputeAxesOn();
        this->SetAxisMode(VTK_HESSIAN);
      } else {
        this->ComputeAirwayAxisFromLines();
        reslicer->ComputeAxesOff();
      }
      break;
    case VTK_VECTOR:
      if (input->GetPointData()->GetVectors() == NULL) 
       {
        reslicer->ComputeAxesOn();
        this->SetAxisMode(VTK_HESSIAN);
       } else {
        reslicer->ComputeAxesOff();
	cout<<"Using vectors"<<endl;
       }
      break;
   }


  // Allocate data
  //Create point Data for each stats
  vtkDoubleArray *mean;
  vtkDoubleArray *std;
  vtkDoubleArray *min;
  vtkDoubleArray *max;
  vtkDoubleArray *ellipse;
  if (output->GetFieldData()->GetArray("Mean") == NULL)
  {
    mean = vtkDoubleArray::New();
    mean->SetName("Mean");
    this->GetOutput()->GetFieldData()->AddArray(mean);
    mean->Delete();
    
  }
  if (output->GetFieldData()->GetArray("Std") == NULL)
  {
    std = vtkDoubleArray::New();
    std->SetName("Std");
    this->GetOutput()->GetFieldData()->AddArray(std);
    std->Delete();
  }
  if (output->GetFieldData()->GetArray("Min") == NULL)
  {
    min = vtkDoubleArray::New();
    min->SetName("Min");
    this->GetOutput()->GetFieldData()->AddArray(min);
    min->Delete();

  }
  if (output->GetFieldData()->GetArray("Max") == NULL)
  {
    max = vtkDoubleArray::New();
    max->SetName("Max");
    this->GetOutput()->GetFieldData()->AddArray(max);
    max->Delete();
  }
  if (output->GetFieldData()->GetArray("Ellipse") == NULL)
  {
    ellipse = vtkDoubleArray::New();
    ellipse->SetName("Ellipse");
    this->GetOutput()->GetFieldData()->AddArray(ellipse);
    ellipse->Delete();
  }
  
  // Pointer to data
  mean = static_cast<vtkDoubleArray*> (output->GetFieldData()->GetArray("Mean"));
  std = static_cast<vtkDoubleArray*> (output->GetFieldData()->GetArray("Std"));
  min = static_cast<vtkDoubleArray*> (output->GetFieldData()->GetArray("Min"));
  max = static_cast<vtkDoubleArray*> (output->GetFieldData()->GetArray("Max"));
  ellipse = static_cast<vtkDoubleArray*> (output->GetFieldData()->GetArray("Ellipse"));

  int nc = this->WallSolver->GetNumberOfQuantities();
  int np = input->GetNumberOfPoints();
 
    
  if (mean == NULL) {
     cout<<"Problem"<<endl;
  }

  
  mean->SetNumberOfComponents(nc);
  mean->SetNumberOfTuples(np);
  std->SetNumberOfComponents(nc);
  std->SetNumberOfTuples(np);
  min->SetNumberOfComponents(nc);
  min->SetNumberOfTuples(np);
  max->SetNumberOfComponents(nc);
  max->SetNumberOfTuples(np);
  ellipse->SetNumberOfComponents(6);
  ellipse->SetNumberOfTuples(np);

  
  vtkEllipseFitting *eifit = vtkEllipseFitting::New();
  vtkEllipseFitting *eofit = vtkEllipseFitting::New();
  
  // Loop through each point
  int npts = input->GetNumberOfPoints();
  for (vtkIdType k=0; k<npts; k++) {
    //for (vtkIdType k=0; k<500; k++) {
    input->GetPoints()->GetPoint(k,p);

    //cout<<"Processing point "<<k<<" out of "<<npts<<endl;
  
   
  //reslicer->SetCenter(0.5+(p[0]+orig[0])/sp[0],511-((p[1]+orig[1])/sp[1])+0.5,(p[2]-orig[2])/sp[2]);
  ijk[0]=(p[0]-orig[0])/sp[0] ;
  ijk[1]=511-((p[1]-orig[1])/sp[1]);
  ijk[2]=(p[2]-orig[2])/sp[2];
  //cout<<"Ijk: "<<ijk[0]<<" "<<ijk[1]<<" "<<ijk[2]<<endl;
  reslicer->SetCenter(ijk[0],ijk[1],ijk[2]);
  //cout<<"Original point: "<<k<<" "<<p[0]<<" "<<p[1]<<" "<<p[2]<<endl;
  //cout<<"Proing point: "<<k<<" "<<ijk[0]<<" "<<ijk[1]<<" "<<ijk[2]<<endl;
   
   switch(this->GetAxisMode()) {
     case VTK_HESSIAN:
       reslicer->ComputeAxesOn();
       break;
     case VTK_POLYDATA:
       z[0]=this->AxisArray->GetComponent(k,0);
       z[1]=this->AxisArray->GetComponent(k,1);
       z[2]=this->AxisArray->GetComponent(k,2);
       //cout<<"Tangent: "<<z[0]<<" "<<z[1]<<" "<<z[2]<<endl;
       vtkMath::Perpendiculars(z,x,y,0);
       reslicer->SetXAxis(x);
       reslicer->SetYAxis(y);
       reslicer->SetZAxis(z);
       break;
     case VTK_VECTOR:
       z[0]=input->GetPointData()->GetVectors()->GetComponent(k,0);
       z[1]=input->GetPointData()->GetVectors()->GetComponent(k,1);
       z[2]=input->GetPointData()->GetVectors()->GetComponent(k,2);
       //cout<<"Tangent: "<<z[0]<<" "<<z[1]<<" "<<z[2]<<endl;

       vtkMath::Perpendiculars(z,x,y,0);
       reslicer->SetXAxis(x);
       reslicer->SetYAxis(y);
       reslicer->SetZAxis(z);
       break;
   }

   //cout<<"Before reslice"<<endl;
   reslicer->Update();
   //cout<<"After reslice"<<endl;
   
   /*
   if (k<300)
   {
     vtkStructuredPointsWriter *w = vtkStructuredPointsWriter::New();
     char file[256];
     sprintf(file,"/var/tmp/kk-%d.vtk",k);
     w->SetFileName(file);
     w->SetInput(reslicer->GetOutput());
     w->Write();
   }
   */
   
   this->WallSolver->SetInputData(reslicer->GetOutput());
   //Maybe we have to update the threshold depending on the center value.
   if (this->WallSolver->GetMethod()==2) {
     // Use self tune phase congruency
     vtkComputeAirwayWall *tmp = vtkComputeAirwayWall::New();
     this->SetWallSolver(this->WallSolver,tmp);
     tmp->SetInputData(reslicer->GetOutput());
     tmp->ActivateSectorOff();
     tmp->SetBandwidth(1.577154);
     tmp->SetNumberOfScales(12);
     tmp->SetMultiplicativeFactor(1.27);
     tmp->SetMinimumWavelength(2);
     tmp->UseWeightsOn();
     vtkDoubleArray *weights = vtkDoubleArray::New();
     weights->SetNumberOfTuples(12);
     double tt[12]={1.249966,0.000000,0.000000,0.734692,0.291580,0.048616,0.718651,0.000000,0.620357,0.212188,0.000000,1.094157};
     for (int i=0;i<12;i++) { 
       weights->SetValue(i,tt[i]);
     }
     tmp->SetWeights(weights);
     tmp->Update();
     double wt = tmp->GetStatsMean()->GetComponent(4,0);
     tmp->Delete();
     weights->Delete();
     double ml;
     double *factors;
     switch (this->Reconstruction) {
       case VTK_SMOOTH:
	 factors = this->SelfTuneModelSmooth;
	 break;
       case VTK_SHARP:
	 factors = this->SelfTuneModelSharp;
	 break;
     }
     ml = exp(factors[0]*pow(log(wt*factors[1]),factors[2]));
     this->WallSolver->SetMultiplicativeFactor(ml);
   }
   
   //cout<<"Update solver"<<endl;
   this->WallSolver->Update();
   //cout<<"Done solver"<<endl;
   
   
   // Fit ellipse model to obtain those parameters ->Move this to compute airway wall
   
   //cout<<"Ellipse fitting 1: "<<this->WallSolver->GetInnerContour()->GetNumberOfPoints()<<endl;
   eifit->SetInputData(this->WallSolver->GetInnerContour());
   eifit->Update();
   //cout<<"Ellipse fitting 2: "<<this->WallSolver->GetOuterContour()->GetNumberOfPoints()<<endl;
   eofit->SetInputData(this->WallSolver->GetOuterContour());
   eofit->Update();
   //cout<<"Done ellipse fitting"<<endl;
   
   // Collect results and assign them to polydata
 
   for (int c = 0; c<this->WallSolver->GetNumberOfQuantities();c++) {
     mean->SetComponent(k,c,this->WallSolver->GetStatsMean()->GetComponent(2*c,0));
     std->SetComponent(k,c,this->WallSolver->GetStatsMean()->GetComponent((2*c)+1,0));
     min->SetComponent(k,c,this->WallSolver->GetStatsMinMax()->GetComponent(2*c,0));
     max->SetComponent(k,c,this->WallSolver->GetStatsMinMax()->GetComponent((2*c)+1,0));
   }
   
   ellipse->SetComponent(k,0,eifit->GetMinorAxisLength()*resolution);
   ellipse->SetComponent(k,1,eifit->GetMajorAxisLength()*resolution);
   ellipse->SetComponent(k,2,eifit->GetAngle());
   ellipse->SetComponent(k,3,eofit->GetMinorAxisLength()*resolution);
   ellipse->SetComponent(k,4,eofit->GetMajorAxisLength()*resolution);
   ellipse->SetComponent(k,5,eofit->GetAngle());
   
   if (this->SaveAirwayImage) {
     char fileName[10*256];
     vtkPNGWriter *writer = vtkPNGWriter::New();
     vtkImageData *airwayImage = vtkImageData::New();
     this->CreateAirwayImage(reslicer->GetOutput(),eifit,eofit,airwayImage);
     writer->SetInputData(airwayImage);
     sprintf(fileName,"%s%03lld.png",this->AirwayImagePrefix,k);
     writer->SetFileName(fileName);
     writer->Write();
     airwayImage->Delete();
     writer->Delete();
  }
  
  }
  
  eifit->Delete();
  eofit->Delete();
  //Compute stats for each line if lines are available
  if (input->GetLines()) {
    this->ComputeCellData();
  }
  
  return 1;
}

void vtkComputeAirwayWallPolyData::ComputeCellData()
{
  vtkPolyData *input= vtkPolyData::SafeDownCast(this->GetInput());
  vtkPolyData *output = this->GetOutput();
  
  vtkCellArray *inLines = input->GetLines();
  int numLines = inLines->GetNumberOfCells();
  vtkIdType *pts = 0;
  vtkIdType npts = 0;
  
   int nl = inLines->GetNumberOfCells();
   int nc = this->WallSolver->GetNumberOfQuantities();

   vtkDoubleArray *mean = vtkDoubleArray::New();
   vtkDoubleArray *std = vtkDoubleArray::New();
   vtkDoubleArray *min = vtkDoubleArray::New();
   vtkDoubleArray *max = vtkDoubleArray::New();
  
   mean->SetName("Mean");
   mean->SetNumberOfComponents(nc);
   mean->SetNumberOfTuples(nl);
   std->SetName("Std");
   std->SetNumberOfComponents(nc);
   std->SetNumberOfTuples(nl);
   min->SetName("Min");
   min->SetNumberOfComponents(nc);
   min->SetNumberOfTuples(nl);
   max->SetName("Max");
   max->SetNumberOfComponents(nc);
   max->SetNumberOfTuples(nl);
   
   this->GetOutput()->GetCellData()->AddArray(mean);
   this->GetOutput()->GetCellData()->AddArray(std);
   this->GetOutput()->GetCellData()->AddArray(min);
   this->GetOutput()->GetCellData()->AddArray(max);
 
   mean->Delete();
   std->Delete();
   min->Delete();
   max->Delete();
  
  vtkDataArray *meanp = output->GetPointData()->GetScalars("Mean");
  vtkDataArray *stdp = output->GetPointData()->GetScalars("Std");
  vtkDataArray *minp = output->GetPointData()->GetScalars("Min");
  vtkDataArray *maxp = output->GetPointData()->GetScalars("Max");
  
  vtkDataArray *meanc = output->GetCellData()->GetScalars("Mean");
  vtkDataArray *stdc = output->GetCellData()->GetScalars("Std");
  vtkDataArray *minc = output->GetCellData()->GetScalars("Min");
  vtkDataArray *maxc = output->GetCellData()->GetScalars("Max"); 
  
  
  inLines->InitTraversal();
  for (int id=0; inLines->GetNextCell(npts,pts); id++)
  {
    //Compute segment length and define initial and end array id.
    double d[3],p1[3],p2[3];
    double length = 0;
    for (int k=0; k<npts-1;k++) {
      input->GetPoints()->GetPoint(pts[k],p1);
      input->GetPoints()->GetPoint(pts[k+1],p2);
      d[0]=p2[0]-p1[0];
      d[1]=p2[1]-p1[1];
      d[2]=p2[2]-p1[2];
      length +=vtkMath::Norm(d);
    }
    
    //Compute statistics for this cell from point data
    double cumlength = 0;
    double mean,std=0;
    double min=+100000;
    double max=-100000;
    int nq = this->WallSolver->GetNumberOfQuantities();
    int count=0;

    for (int q=0; q<nq; q++)
    {
      meanc->SetComponent(id,q,0);
      stdc->SetComponent(id,q,0);
      minc->SetComponent(id,q,+100000);
      maxc->SetComponent(id,q,-100000);
    }
    
    for (int k=0 ; k<npts-1;k++) {
      input->GetPoints()->GetPoint(pts[k],p1);
      input->GetPoints()->GetPoint(pts[k+1],p2);
      d[0]=p2[0]-p1[0];
      d[1]=p2[1]-p1[1];
      d[2]=p2[2]-p1[2];
      cumlength +=vtkMath::Norm(d);
      if (cumlength > length*(1-this->SegmentPercentage)/2 & cumlength < length*(1+this->SegmentPercentage)/2) {
	// Loop through all the quantities
	count++;
	for (int q=0; q<nq; q++) {
	  meanc->SetComponent(id,q,meanc->GetComponent(id,q)+meanp->GetComponent(pts[k],q));
	  stdc->SetComponent(id,q,stdc->GetComponent(id,q)+stdp->GetComponent(pts[k],q));
	  if (minp->GetComponent(pts[k],q) < minc->GetComponent(id,q))
	    minc->SetComponent(id,q,minp->GetComponent(pts[k],q));
	  if (maxp->GetComponent(pts[k],q) > maxc->GetComponent(id,q))
	    maxc->SetComponent(id,q,maxp->GetComponent(pts[k],q));
	}
      }
    }
    
    if (count > 0) {
    for (int q=0; q<nq; q++) {
      meanc->SetComponent(id,q,meanc->GetComponent(id,q)/count);
      stdc->SetComponent(id,q,stdc->GetComponent(id,q)/count);
    }
    }
    
  }
  
}

    
    
    

void vtkComputeAirwayWallPolyData::SetWallSolver(vtkComputeAirwayWall *ref, vtkComputeAirwayWall *out) {
  
  out->SetMethod(ref->GetMethod());
  out->SetWallThreshold(ref->GetWallThreshold());
  out->SetNumberOfScales(ref->GetNumberOfScales());
  out->SetBandwidth(ref->GetBandwidth());
  out->SetMinimumWavelength(ref->GetMinimumWavelength());
  out->SetMultiplicativeFactor(ref->GetMultiplicativeFactor());
  out->SetUseWeights(ref->GetUseWeights());
  out->SetWeights(ref->GetWeights());
  out->SetThetaMax(ref->GetThetaMax());
  out->SetThetaMin(ref->GetThetaMin());
  out->SetRMin(ref->GetRMin());
  out->SetRMax(ref->GetRMax());
  out->SetDelta(ref->GetDelta());
  out->SetScale(ref->GetScale());
  out->SetNumberOfThetaSamples(ref->GetNumberOfThetaSamples());
  out->SetAlpha(out->GetAlpha());
  out->SetT(out->GetT());
  out->SetActivateSector(ref->GetActivateSector());
  
}


void vtkComputeAirwayWallPolyData::CreateAirwayImage(vtkImageData *resliceCT,vtkEllipseFitting *eifit,vtkEllipseFitting *eofit,vtkImageData *airwayImage)
{
  vtkImageMapToColors *rgbFilter = vtkImageMapToColors::New();
  vtkLookupTable *lut = vtkLookupTable::New();
  
  rgbFilter->SetInputData(resliceCT);
  rgbFilter->SetOutputFormatToRGB();
  
  lut->SetSaturationRange(0,0);
  lut->SetHueRange(0,0);
  lut->SetValueRange(0,1);
  lut->SetTableRange(-150,1500);
  lut->Build();
  rgbFilter->SetLookupTable(lut);
  
  rgbFilter->Update();
  
  //Set Image voxels based on ellipse information
  vtkImageData *rgbImage=rgbFilter->GetOutput();
  double sp[3];
  rgbImage->GetSpacing(sp);
  int npoints=128;
  
  vtkEllipseFitting *arr[2];
  arr[0]=eifit;
  arr[1]=eofit;
  vtkEllipseFitting *eFit;
  
  float centerX = (eifit->GetCenter()[0] + eofit->GetCenter()[0])/2.0;
  float centerY = (eifit->GetCenter()[1] + eofit->GetCenter()[1])/2.0;
  
  int colorChannel[2];
  colorChannel[0]=0;
  colorChannel[1]=1;
  for (int ii=0;ii<2;ii++) {
    
    //eFit = static_cast <vtkEllipseFitting > (arr->GetNextItemAsObject());
    eFit =arr[ii];
    int rx,ry;
    for (int k=0;k<npoints;k++) {
      float t = -3.14159 + 2.0 * 3.14159 * k/(npoints -1.0);
      float angle = eFit->GetAngle();
      float px = centerX + eFit->GetMajorAxisLength() *cos(t) * cos(angle) - 
                           eFit->GetMinorAxisLength() * sin(t) * sin(angle);
      float py = centerY + eFit->GetMajorAxisLength() *cos(t) * sin(angle) + 
                           eFit->GetMinorAxisLength() * sin(t) * cos(angle);
			   
      //Set Image Value with antialiasing
      //rx= floor(px);
      //ry= floor(py);
      //rgbImage->SetScalarComponentFromFloat(rx,ry,0,colorChannel[ii],255*(1-(rx-px))*(1-(ry-py)));
      //So on and so forth...
      // Simple NN
      for (int cc=0;cc<rgbImage->GetNumberOfScalarComponents();cc++)
	rgbImage->SetScalarComponentFromFloat(round(px),round(py),0,cc,0);
      
      rgbImage->SetScalarComponentFromFloat(round(px),round(py),0,colorChannel[ii],255);
      
    }
  }

  airwayImage->DeepCopy(rgbImage);
  
  lut->Delete();
  rgbFilter->Delete();
  
}
  
  
void vtkComputeAirwayWallPolyData::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Reforma: " << this->Reformat << "\n";
  os << indent << "Axis Mode: " << this->AxisMode << "\n";
}
