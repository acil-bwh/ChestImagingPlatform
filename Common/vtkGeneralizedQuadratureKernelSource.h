// .NAME vtkGeneralizedQuadratureKernelSource - Create a quadrature kernel
// .SECTION Description
// vtkGeneralizedQuadratureKernelSource produces
// images which are quadrature convolution kernels.
// These images are created in the fourier domain with
// the DC component in the center of the volume.
// In future they could be shifted to have the DC at the origin
// before output.

#ifndef __vtkGeneralizedQuadratureKernelSource_h
#define __vtkGeneralizedQuadratureKernelSource_h

#include "vtkImageKernelSource.h"
#include "vtkCIPCommonConfigure.h"

class VTK_CIP_COMMON_EXPORT vtkGeneralizedQuadratureKernelSource : public vtkImageKernelSource
{
public:
  static vtkGeneralizedQuadratureKernelSource *New();
  vtkTypeMacro(vtkGeneralizedQuadratureKernelSource, vtkImageKernelSource);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // Description:
  // Center frequency where the filter gives the
  // highest response in the Fourier domain.
  // This is in radians and the default is pi/4.
  vtkSetMacro(CenterFrequency, double);
  vtkGetMacro(CenterFrequency, double);

  // Description:
  // Relative bandwidth describes the width around the
  // center frequency where the filter will respond.
  // The value is in octaves and 2 is the default.
  vtkSetMacro(RelativeBandwidth, double);
  vtkGetMacro(RelativeBandwidth, double);

  // Description:
  // This is the direction of the filter in the
  // Fourier domain.
  vtkSetVector3Macro(FilterDirection, double);
  vtkGetVector3Macro(FilterDirection, double);

  // Description:
  // Controls angular filter shape by multiplying
  // the cosine exponent
  vtkSetMacro(AngularExponent, double);
  vtkGetMacro(AngularExponent, double);

  // Description:
  // Set a cosine shape Window funciton to mask the filter response
  // Window = prod_i cos(pi/2 |u_i/pi|^N)^K  where u_i is the freq coordinates.
  vtkSetMacro(WindowFunction,int);
  vtkGetMacro(WindowFunction,int);
  vtkBooleanMacro(WindowFunction, int);

  // Description:
  // Controls the N exponent of the window function
  vtkSetMacro(WindowNExponent, double);
  vtkGetMacro(WindowNExponent, double);

  // Description:
  // Controls the K exponent of the window function
  vtkSetMacro(WindowKExponent, double);
  vtkGetMacro(WindowKExponent, double);

  // Description:
  // Get the Dimensionality of the signal domain. Generalized Quadrature filter
  // dimensionality is given by Dimensionality+1;
  vtkGetMacro(Dimensionality,int);

  // Description:
  // We have not implemented this feature of the superclass
  void SetOutputDomainToSpatial()
    {vtkErrorMacro("This filter only outputs kernels in the fourier domain.");
    this->SetOutputDomain(VTK_KERNEL_OUTPUT_FOURIER_DOMAIN);}

  void ThreadedExecute(vtkImageData *inData,
                       vtkImageData *outData,
                       int outExt[6], int id) override;
protected:
  vtkGeneralizedQuadratureKernelSource();
  ~vtkGeneralizedQuadratureKernelSource() {};
  vtkGeneralizedQuadratureKernelSource(const vtkGeneralizedQuadratureKernelSource&) {};
  void operator=(const vtkGeneralizedQuadratureKernelSource&) {};

  double CenterFrequency;
  double RelativeBandwidth;
  double FilterDirection[3];
  double AngularExponent;
  int WindowFunction;
  double WindowKExponent;
  double WindowNExponent;

  int Dimensionality;

  void CalculateFourierCoordinates(double *max_x, double *max_y,
				   double *max_z, double *k_dx,
				   double *k_dy,  double *k_dz);

  // This is a convenience method that is implemented in many subclasses
  // instead of RequestData.  It is called by RequestData.
  virtual void ExecuteDataWithInformation(vtkDataObject *output,
                                          vtkInformation* outInfo) override;
                                         
  virtual int RequestInformation(vtkInformation *, vtkInformationVector**,
                                 vtkInformationVector *) override;
};

#endif
