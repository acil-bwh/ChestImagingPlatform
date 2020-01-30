// .NAME vtkImageKernelSource - Create a kernel
// .SECTION Description
// vtkImageKernelSource produces images which are convolution kernels

#ifndef __vtkImageKernelSource_h
#define __vtkImageKernelSource_h

#include "vtkImageAlgorithm.h"
#include "vtkImageKernel.h"
#include "vtkMultiThreader.h"
#include "vtkCIPCommonConfigure.h"

// VTK6 migration note:
// - replaced superclass vtkImageSource with vtkImageAlgorithm

class VTK_CIP_COMMON_EXPORT vtkImageKernelSource : public vtkImageAlgorithm
{
public:
  static vtkImageKernelSource *New();
  vtkTypeMacro(vtkImageKernelSource, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Get the output of this source.
  void SetOutput(vtkImageKernel *output);
  vtkImageKernel *GetOutput();
  vtkImageKernel *GetOutput(int idx);

  // Description:
  // Set/Get the extent of the whole output kernel.
  // For a 20x20x1 kernel this would be
  // SetWholeExtent(0,19,0,19,0,1)
  void SetWholeExtent(int xMinx, int xMax, int yMin, int yMax,
		      int zMin, int zMax);
  void SetWholeExtent(int ext[6])
    {this->SetWholeExtent(ext[0],ext[1],ext[2],ext[3],ext[4],ext[5]);}

  // Description:
  // Voxel Dimensions in the vtkImageKernel that this kernel
  // will be convolved with. This is used in subclasses
  // to design kernels that take into account the voxel shape.
  vtkSetVector3Macro(VoxelSpacing, double);
  vtkGetVector3Macro(VoxelSpacing, double);

#define VTK_KERNEL_OUTPUT_FOURIER_DOMAIN 0
#define VTK_KERNEL_OUTPUT_SPATIAL_DOMAIN 1
  // Description:
  // Whether the output kernel is located in the
  // Spatial or Fourier domains.
  // Support for this depends on the subclass.
  void SetOutputDomain(int _arg)
    {if(this->OutputDomain != _arg) {
     this->OutputDomain = _arg;
     this->GetOutput()->SetKernelDomain(_arg);
     this->Modified();
     }
     }
  vtkGetMacro(OutputDomain,int);
  void SetOutputDomainToFourier()
    {this->SetOutputDomain(VTK_KERNEL_OUTPUT_FOURIER_DOMAIN);}
  void SetOutputDomainToSpatial()
    {this->SetOutputDomain(VTK_KERNEL_OUTPUT_SPATIAL_DOMAIN);}

#define VTK_KERNEL_ZERO_FREQUENCY_CENTER 0
#define VTK_KERNEL_ZERO_FREQUENCY_ORIGIN 1
  // Description:
  // Whether the output kernel (if in the Fourier domain)
  // has the center frequency at the origin or in the center
  // Support for this depends on the subclass.
  // Note the origin is the "vtk origin" which is really the
  // corner of the image or volume.
  void SetZeroFrequencyLocation(int _arg)
    {if(this->ZeroFrequencyLocation != _arg) {
     this->ZeroFrequencyLocation = _arg;
     this->GetOutput()->SetZeroFrequencyLocation(_arg);
     this->Modified();
     }
     }
  vtkGetMacro(ZeroFrequencyLocation,int);
  void SetZeroFrequencyLocationToCenter()
    {this->SetZeroFrequencyLocation(VTK_KERNEL_ZERO_FREQUENCY_CENTER);}
  void SetZeroFrequencyLocationToOrigin()
    {this->SetZeroFrequencyLocation(VTK_KERNEL_ZERO_FREQUENCY_ORIGIN);}

  // Description:
  // Whether to output a kernel with real and imaginary parts.
  // In vtk real and imaginary are represented as two scalar components
  // in each voxel.  Subclasses can be told to output 2 components
  // (even if for example the kernel is all real) using this flag.
  // Also, subclasses should set this in the constructor if they
  // will always output complex kernels.
  // This class will output 2 scalar components if this flag is set.
  void SetComplexOutput(int _arg)
     {if(this->ComplexOutput != _arg) {
      this->ComplexOutput = _arg;
      this->GetOutput()->SetComplexKernel(_arg);
      this->Modified();
      }
      }
  vtkGetMacro(ComplexOutput,int);
  vtkBooleanMacro(ComplexOutput,int);

  // Description:
  // Putting this here until I merge graphics and imaging streaming.
  virtual int SplitExtent(int splitExt[6], int startExt[6],
			                    int num, int total);

  // Description:
  // If the subclass does not define an Execute method, then the task
  // will be broken up, multiple threads will be spawned, and each thread
  // will call this method. It is public so that the thread functions
  // can call this method.
  virtual void ThreadedExecute(vtkImageData *inData,
                               vtkImageData *outData,
                               int extent[6], int threadId);

  void ThreadedFourierCenterExecute(vtkImageData *inData, vtkImageData *outData,
                                    int outExt[6], int id);

  // Description:
  // Get/Set the number of threads to create when rendering
  vtkSetClampMacro( NumberOfThreads, int, 1, VTK_MAX_THREADS );
  vtkGetMacro( NumberOfThreads, int );

  // Description:
  // Private methods kept public for template execute functions.
  // Permute Increment and Extent. Used when Fourier kernels
  // has to be center in the corner of the image
  void ComputeInputIndex(int outIndx[3],int mid[3],
                         int inIndx[3]);

protected:
  vtkImageKernelSource();
  ~vtkImageKernelSource();

  vtkMultiThreader *Threader;
  int NumberOfThreads;

  int WholeExtent[6];
  double VoxelSpacing[3];

  int OutputDomain;
  int ZeroFrequencyLocation;
  int ComplexOutput;

  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;
                                 
  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;
private:
  vtkImageKernelSource(const vtkImageKernelSource&);
  void operator=(const vtkImageKernelSource&);
};

#endif
