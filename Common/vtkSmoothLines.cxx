/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkSmoothLines.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkSmoothLines.h"

#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"

vtkStandardNewMacro(vtkSmoothLines);

//---------------------------------------------------------------------------
// Construct object with initial Tolerance of 0.0
vtkSmoothLines::vtkSmoothLines()
{
  this->Beta = 0.05;
  this->NumberOfIterations = 20;
  this->Delta=0.1;
}

//--------------------------------------------------------------------------
vtkSmoothLines::~vtkSmoothLines()
{
}

//--------------------------------------------------------------------------
// VTK6 migration note:
// - replaced Execute()
// - changed this->GetInput() to vtkPolyData::GetData(inInfoVec[0])
// - changed this->GetOutput() to vtkPolyData::GetData(outInfoVec)
//--------------------------------------------------------------------------
int vtkSmoothLines::RequestData(vtkInformation* vtkNotUsed(request), 
  vtkInformationVector** inInfoVec, 
  vtkInformationVector* outInfoVec)
{  
  vtkPolyData *input = vtkPolyData::GetData(inInfoVec[0]); //always defined on entry into Execute
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


  vtkPoints *newPts = vtkPoints::New();
  vtkDoubleArray *comp = vtkDoubleArray::New();
  vtkDoubleArray *smooth[3];
  smooth[0] = vtkDoubleArray::New();
  smooth[1] = vtkDoubleArray::New();
  smooth[2] = vtkDoubleArray::New();

  // we'll be needing these

  vtkCellArray *inLines  = input->GetLines();

  vtkPolyData  *output   = vtkPolyData::GetData(outInfoVec);
  output->DeepCopy(input);
  
  // Loop through each line
  vtkIdType npts;
  vtkIdType *pts;
  double xin[3];
  double xout[3];
  inLines->InitTraversal();
  for (int i = 0; i<inLines->GetNumberOfCells();i++) {
    //Get list point in cell
    inLines->GetNextCell(npts,pts);
    
    //Get each components of point list and filter
   
   if (npts<3)
     continue;
   
   comp->Reset();
   comp->SetNumberOfTuples(npts);
   for (int k=0; k<3;k++) { 
      for (int j=0;j<npts;j++)
        {
        inPts->GetPoint(pts[j],xin);
        comp->SetValue(j,xin[k]); 
        }
	//Call smoothing methods
	this->SolveHeatEquation(comp,smooth[k]);
	  
    }
    //Set the result   
   for (int j=0;j<npts;j++)
     {
      xout[0]=smooth[0]->GetValue(j);
      xout[1]=smooth[1]->GetValue(j);
      xout[2]=smooth[2]->GetValue(j);
      output->GetPoints()->SetPoint(pts[j], xout);
     }	
     
   }
 
  smooth[0]->Delete();
  smooth[1]->Delete();
  smooth[2]->Delete();
  comp->Delete();
     
  return 1;
}    

void vtkSmoothLines::SolveHeatEquation(vtkDoubleArray *in, vtkDoubleArray *out)
{

 int np = in->GetNumberOfTuples();
 
 //Allocate output array
 out->Reset();
 out->SetNumberOfTuples(np);     
 
 //First and last point are fixed 
 out->SetValue(0,in->GetValue(0));
 out->SetValue(np-1,in->GetValue(np-1));
 
 vtkDoubleArray *iterk = vtkDoubleArray::New();
 vtkDoubleArray *iterkp1,*tmp;
 
 iterk->DeepCopy(in);
 
 double update,val;
 double meanupdate;
 iterkp1 = out;
 for (int iter =0; iter < this->NumberOfIterations;iter++) 
   { 
   //First and last point are fixed 
   iterkp1->SetValue(0,in->GetValue(0));
   iterkp1->SetValue(np-1,in->GetValue(np-1));
   meanupdate = 0.0;
   for (int c=1; c<np-1;c++)
     {
     update = iterk->GetValue(c-1) -2 * iterk->GetValue(c) + iterk->GetValue(c+1) + 
              this->Beta * (in->GetValue(c) - iterk->GetValue(c));
     val = iterk->GetValue(c) + this->Delta * update; 
     meanupdate += fabs(update);
     iterkp1->SetValue(c,val);
     }
     meanupdate=meanupdate/np;
   tmp=iterk;  
   iterk = iterkp1;
   iterkp1 = tmp;
   }
  
 
 //Make sure the result is in out
  if (iterk != out)
    out->DeepCopy(iterk);
  
  iterk->Delete();
  
}    
  
  
//--------------------------------------------------------------------------
void vtkSmoothLines::PrintSelf(ostream& os, vtkIndent indent) 
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "running class: "<<this->GetClassName()<<endl;

}


