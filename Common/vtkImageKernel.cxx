#include "vtkImageKernel.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkImageKernel);

vtkImageKernel::vtkImageKernel()
{
  
  // default kernel in fourier domain
  
  this->SetKernelDomainToFourier();
  
  // and has the zero frequency at the origin
  
  this->SetZeroFrequencyLocationToOrigin();
  
  // default is 1 scalar component
  
  this->ComplexKernelOff();
  
  this->SetKernelDirection(0,0,0);
  
}

vtkImageKernel::~vtkImageKernel(){

}
//----------------------------------------------------------------------------
void vtkImageKernel::ShallowCopy(vtkDataObject *dataObject)
{
  vtkImageKernel *imageKernel = vtkImageKernel::SafeDownCast(dataObject);

  if ( imageKernel != NULL )
    {
    this->InternalImageDataCopy(imageKernel);
    }

  // Do superclass
  this->vtkImageData::ShallowCopy(dataObject);
}

//----------------------------------------------------------------------------
void vtkImageKernel::DeepCopy(vtkDataObject *dataObject)
{
  vtkImageKernel *imageKernel = vtkImageKernel::SafeDownCast(dataObject);

  if ( imageKernel != NULL )
    {
    this->InternalImageDataCopy(imageKernel);
    }

  // Do superclass
  this->vtkImageData::DeepCopy(dataObject);
}

//----------------------------------------------------------------------------
// This copies all the local variables (but not objects).
void vtkImageKernel::InternalImageDataCopy(vtkImageKernel *src)
{

  this->KernelDomain = src->KernelDomain;
  this->ZeroFrequencyLocation = src->ZeroFrequencyLocation;
  this->ComplexKernel = src->ComplexKernel;
  for(int i=0; i<3 ;i++)
   this->KernelDirection[i] = src->KernelDirection[i];
  
}


void vtkImageKernel::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkImageData::PrintSelf(os,indent);
  
  os << indent << "Kernel Domain: "<<(this->KernelDomain ? "Spatial domain\n" : "Frequency domain\n");
  
  os << indent << "Frequency Location: "<<(this->ZeroFrequencyLocation ? "Zero Frequency at origin\n" : "Zero Frequency at center.\n");
}
