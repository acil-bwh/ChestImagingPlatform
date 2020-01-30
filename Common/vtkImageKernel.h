// .NAME vtkImageKernel - A subclass of ImageData.
// .SECTION Description
// vtkImageKernel is a subclass of ImageData to define kernels either in the
// spacial domain or frequency domain to be used by vtkConvolveFilter.

#ifndef __vtkImageKernel_h
#define __vtkImageKernel_h

#include "vtkImageData.h"
#include "vtkCIPCommonConfigure.h"

#define VTK_KERNEL_FOURIER_DOMAIN 0
#define VTK_KERNEL_SPATIAL_DOMAIN 1
#define VTK_KERNEL_ZERO_FREQUENCY_CENTER 0
#define VTK_KERNEL_ZERO_FREQUENCY_ORIGIN 1

class VTK_CIP_COMMON_EXPORT vtkImageKernel : public vtkImageData
{
public:
  static vtkImageKernel *New();
  vtkTypeMacro(vtkImageKernel, vtkImageData);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Create a similar type object
  //vtkDataObject *MakeObject() {return vtkImageKernel::New();}

  // Description:
  // To simplify filter superclasses,
  int GetDataObjectType() override {return VTK_IMAGE_DATA;}

  // Description:
  // Whether the kernel is located in the
  // Spatial (type 1) or Fourier (type 0) domains.
  // Support for this depends on the subclass.
  vtkSetMacro(KernelDomain,int);
  vtkGetMacro(KernelDomain,int);
  void SetKernelDomainToFourier()
    {this->SetKernelDomain(VTK_KERNEL_FOURIER_DOMAIN);}
  void SetKernelDomainToSpatial()
    {this->SetKernelDomain(VTK_KERNEL_SPATIAL_DOMAIN);}

  // Description:
  // Direction of the kernel, assuming there is a
  // principal direction
  vtkSetVector3Macro(KernelDirection,float);
  vtkGetVector3Macro(KernelDirection,float);

  //I think it's not a good idea to do this here.
  // Description:
  // Transform the kernel to Fourier or Spatial
  //void ChangeDomainToFourier()
  //void ChangeDomainToSpatial()

  // Description:
  // Whether the kernel (if in the Fourier domain)
  // has the center frequency at the origin or in the center
  // Note the origin is the "vtk origin" which is really the
  // corner of the image or volume.
  vtkSetMacro(ZeroFrequencyLocation,int);
  vtkGetMacro(ZeroFrequencyLocation,int);
  void SetZeroFrequencyLocationToCenter()
    {this->SetZeroFrequencyLocation(VTK_KERNEL_ZERO_FREQUENCY_CENTER);};
  void SetZeroFrequencyLocationToOrigin()
    {this->SetZeroFrequencyLocation(VTK_KERNEL_ZERO_FREQUENCY_ORIGIN);};

  // Description:
  // Whether to a kernel with real and imaginary parts.
  // In vtk real and imaginary are represented as two scalar components
  // in each voxel.  Subclasses can be told to output 2 components
  // (even if for example the kernel is all real) using this flag.
  // Also, subclasses should set this in the constructor if they
  // will always output complex kernels.
  // This class will hold 2 scalar components if this flag is set.
  //
  // VTK6 migration note:
  // - SetNumberOfScalarComponents(int n) is not available
  //   since this property has moved to vtkInformation
  //   instead static SetNumberOfScalarComponents(int, vtkInformation*) is
  //   provided but it seems the use of this method here is not appropriate or
  //   necessary
  void ComplexKernelOn()
    {/*this->SetNumberOfScalarComponents(2);*/ this->ComplexKernel=1;};
  void ComplexKernelOff()
    {/*this->SetNumberOfScalarComponents(1);*/ this->ComplexKernel=0;};
  void SetComplexKernel(int complex)
    {complex ? this->ComplexKernelOn() : this->ComplexKernelOff();};
  vtkGetMacro(ComplexKernel,int);

  // Description:
  // Shallow and Deep copy.
  void ShallowCopy(vtkDataObject *src) override;
  void DeepCopy(vtkDataObject *src) override;

protected:
  vtkImageKernel();
  ~vtkImageKernel();
  int KernelDomain;
  int ZeroFrequencyLocation;
  int ComplexKernel;
  float KernelDirection[3];

private:
  void InternalImageDataCopy(vtkImageKernel *src);
private:
  vtkImageKernel (const vtkImageKernel&);
  void operator=(const vtkImageKernel&);
};

#endif

