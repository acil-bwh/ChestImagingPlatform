#include "vtkMaskBoundingBox.h"
#include <vtkExecutive.h>
#include <vtkObjectFactory.h>
#include <vtkImageToImageStencil.h>
#include "vtkImageData.h"

vtkStandardNewMacro(vtkMaskBoundingBox);

//----------------------------------------------------------------------------
vtkMaskBoundingBox::vtkMaskBoundingBox()
{
  // Objects that should be set by the user, but can be self-created
  this->Label = 0;
  for (int i=0;i<6;i++) 
    {
    this->BoundingBox[i]=0;
    }
  this->Stencil = vtkImageStencilData::New();
  this->SetNumberOfInputPorts(1);
}

//----------------------------------------------------------------------------
vtkMaskBoundingBox::~vtkMaskBoundingBox()
{
  this->Stencil->Delete();
}

//----------------------------------------------------------------------------
// VTK6 migration note:
// - changed name from SetInput(...)
// - changed SetNthInput(0, input) to SetInputDataInternal(0, input)
void vtkMaskBoundingBox::SetInputData(vtkImageData *input)
{
  this->SetInputDataInternal(0, input);
}

//----------------------------------------------------------------------------
vtkImageData * vtkMaskBoundingBox::GetInput()
{
  if (this->GetNumberOfInputConnections(0) < 1)
    {
    return NULL;
    }

  return vtkImageData::SafeDownCast(this->GetExecutive()->GetInputData(0, 0));
}

//----------------------------------------------------------------------------
void vtkMaskBoundingBox::Compute()
{
  vtkImageData *input = this->GetInput();
  
  vtkImageToImageStencil *stencil = vtkImageToImageStencil::New();
  stencil->SetInputData(input);
  stencil->ThresholdByUpper(this->GetLabel());
  stencil->Update();
  
  int r1,r2,xmin,xmax,yidx,zidx,iter;
  iter = 0;
  int *ext = input->GetExtent();

  xmin =ext[0];
  xmax = ext[1];
  
  
  this->Stencil->DeepCopy(stencil->GetOutput());
  
  //Set origin and spacing to match the input
  //st->SetSpacing(input->GetSpacing());
  //st->SetOrigin(input->GetOrigin());
  
  
  int val;
  //Init Bounding Box variable
  for (int i=0;i<3;i++) 
    {
    this->BoundingBox[2*i]=VTK_INT_MAX;
    this->BoundingBox[2*i+1]=VTK_INT_MIN;
    } 
  
  cout<<"Starting looping..."<<endl;
  for (zidx = ext[4]; zidx <= ext[5]; zidx++)
    {
    for (yidx = ext[2]; yidx <= ext[3]; yidx++)
      {
      iter=0;
      while(1)
        {
	      val=this->Stencil->GetNextExtent(r1,r2,xmin,xmax,yidx,zidx,iter);
	      if(val == 0)
	        {
	        break;
	        }
        else
          {  
	          //cout<<"r1: "<<r1<<"  r2: "<<r2<<" iter: "<<iter<<endl;
	    
    	    if (r1<this->BoundingBox[0])
    	       this->BoundingBox[0]=r1;
    	    if (r2>this->BoundingBox[1])
    	       this->BoundingBox[1]=r2;
       
    	    if (yidx<this->BoundingBox[2])
    	      this->BoundingBox[2]=yidx;
    	    if (yidx>this->BoundingBox[3])
    	      this->BoundingBox[3]=yidx;
           
    	    if (zidx<this->BoundingBox[4])
    	      this->BoundingBox[4] = zidx;
    	    if (zidx>this->BoundingBox[5])
    	      this->BoundingBox[5] = zidx;        
          }
	      }
	    }
    }
      
   for (int i =0 ;i<6;i++)
     {
     cout<<"BB "<<i<<": "<<this->BoundingBox[i]<<endl;
     }
 }

void vtkMaskBoundingBox::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Label Value: " << this->Label << "\n";
}

