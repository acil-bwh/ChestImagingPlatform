/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTubularScalePolyDataFilter.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkTubularScalePolyDataFilter.h"

#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkNRRDExport.h"

vtkStandardNewMacro(vtkTubularScalePolyDataFilter);

//---------------------------------------------------------------------------
// Construct object with initial Tolerance of 0.0
vtkTubularScalePolyDataFilter::vtkTubularScalePolyDataFilter()
{

  this->ImageData = NULL;
  this->TubularType = VTK_VALLEY;
  this->InitialScale = 1.0;
  this->FinalScale = 12.0;
  this->StepScale = 0.25;
  
}

//--------------------------------------------------------------------------
vtkTubularScalePolyDataFilter::~vtkTubularScalePolyDataFilter()
{

  this->SetImageData(NULL);
}

//--------------------------------------------------------------------------
// VTK6 migration note:
// Introduced to replace Execute().
int vtkTubularScalePolyDataFilter::RequestData(vtkInformation *request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{  
  vtkPolyData *input = vtkPolyData::SafeDownCast(this->GetInput()); //always defined on entry into Execute
  if ( !input )
    {
    vtkErrorMacro("Input not defined");
    return 0;
    }
  vtkPoints   *inPts = input->GetPoints();
  vtkIdType   numPts = input->GetNumberOfPoints();

  vtkDebugMacro(<<"Beginning PolyData clean");
  if ( (numPts<1) || (inPts == NULL ) )
    {
    vtkDebugMacro(<<"No data to Operate On!");
    return 0;
    }

  if ( this->GetImageData() == NULL) 
  {
    vtkDebugMacro(<<"No image data to Operate On!");
    return 0;
  }
  
  vtkImageData * inData = this->GetImageData();
  double sp[3], org[3];
  inData->GetSpacing(sp);
  inData->GetOrigin(org);
  cout<<"org: "<<org[0]<<" "<<org[1]<<" "<<org[2]<<endl;
  
  vtkPoints *newPts = vtkPoints::New();
  vtkDoubleArray *scaleArray = vtkDoubleArray::New();
  scaleArray->SetNumberOfComponents(1);
  scaleArray->SetNumberOfTuples(input->GetNumberOfPoints());

  // we'll be needing these

  vtkCellArray *inLines  = input->GetLines();

  vtkPolyData  *output   = this->GetOutput();
  output->DeepCopy(input);
  
  //Export vtkImageData to Nrrd
  vtkNRRDExport * nrrdexport= vtkNRRDExport::New();
  nrrdexport->SetInputData(inData);
  Nrrd *nin = nrrdexport->GetNRRDPointer();

  //Create array of context for each scale
  double finalScale = this->GetFinalScale();
  double initScale = this->GetInitialScale();
  double scaleStep = this->GetStepScale();
  
  // Creating context and assigning volume to context.
  int E = 0;
  gageContext *gtx = gageContextNew();
  gageParmSet(gtx, gageParmRenormalize, AIR_TRUE); // slows things down if true
  gagePerVolume *pvl; 
  if (!E) E |= !(pvl = gagePerVolumeNew(gtx, nin, gageKindScl));
  if (!E) E |= gagePerVolumeAttach(gtx, pvl);
  if (E) {
    //vtkErrorMacro("Error Setting Gage Context... Leaving Execute");
    //Delete local objects
    cout<<"Error Setting Gage Context... Leaving Execute"<<endl;
    gageContextNix(gtx);
    nrrdexport->Delete();
    return 0;
   }
  
  vtkTubularScaleSelection *helper = vtkTubularScaleSelection::New();
  helper->SetTubularType(this->GetTubularType());
  
  // Loop through each line
  vtkIdType npts;
  vtkIdType *pts;
  double xyzin[3];
  int ijk[3];
  double pcoords[3];
  double coord[3];
  double scale;
  
  double xout[3];
  inLines->InitTraversal();
  cout<<" Num cells: :"<<inLines->GetNumberOfCells()<<endl;
  for (int i = 0; i<inLines->GetNumberOfCells();i++) {
    //Get list point in cell
    inLines->GetNextCell(npts,pts);
    
    for (int j=0;j<npts;j++)
      {
      inPts->GetPoint(pts[j],xyzin);
    
      int flag =inData->ComputeStructuredCoordinates(xyzin,ijk,pcoords);
      for (int k=0;k<3;k++) {
        coord[k]=ijk[k]+pcoords[k];
        coord[k]=(xyzin[k]-org[k])/sp[k];
      }
      //cout<<"flag: "<<flag<<" pid: "<<pts[j]<<" "<<xyzin[0]<<" "<<xyzin[1]<<" "<<xyzin[2]<<endl;
      //cout<<"ijk: "<<ijk[0]<<" "<<ijk[1]<<" "<<ijk[2]<<endl;
      scale = helper->ScaleSelection(gtx,pvl,coord,initScale,finalScale,scaleStep);
      //cout<<"coords: "<<coord[0]<<" "<<coord[1]<<" "<<coord[2]<<" Scale: "<<scale<<endl;

      scaleArray->SetValue(pts[j],scale);
      }  
    }
 
  output->GetPointData()->SetScalars(scaleArray);
  gageContextNix(gtx);
  nrrdexport->Delete();
  scaleArray->Delete();
     
  return 1;
}    

//--------------------------------------------------------------------------
void vtkTubularScalePolyDataFilter::PrintSelf(ostream& os, vtkIndent indent) 
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "running class: "<<this->GetClassName()<<endl;

}


