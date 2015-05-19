#include "cipChestConventions.h"
#include "vtkSmartPointer.h"
#include "vtkNRRDReaderCIP.h"
#include "vtkImageChangeInformation.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkImageMathematics.h"
#include "vtkComputeAirwayWall.h"
#include "vtkComputeAirwayWallPolyData.h"
#include "itkImageToVTKImageFilter.h"
#include <vtksys/SystemTools.hxx>
#include "cipHelper.h"

#include "ComputeAirwayWallFromParticlesCLP.h"

namespace
{
  typedef itk::ImageToVTKImageFilter< cip::CTType > ConnectorType;
}

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
  
  std::cout << "Reading CT..." << std::endl;
  cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
    reader->SetFileName( inCT );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading mask:";
    std::cerr << excp << std::endl;

    return cip::NRRDREADFAILURE;
    }

  ConnectorType::Pointer connector = ConnectorType::New();
    connector->SetInput( reader->GetOutput() );
    connector->Update();

  // Note: since we converted to using the ITK-VTK glue to handle reading the image,
  // we are now using the 'connector' to communicate with VTK. 'ras' here no longer
  // serves the purpose it previously did.
  vtkSmartPointer< vtkMatrix4x4 > ras = vtkSmartPointer< vtkMatrix4x4 >::New();
  
  //Particles are in LPS (b/c we assume that images come from DICOM).
  //NRRD reader assumes RAS, so we have to transform the origin for the pipeline to work
  vtkSmartPointer< vtkImageChangeInformation > ctOrigin = vtkSmartPointer< vtkImageChangeInformation >::New();
    ctOrigin->SetInputData( connector->GetOutput() );
    ctOrigin->SetOutputOrigin(-1.0*ras->GetElement(0,3),-1.0*ras->GetElement(1,3),ras->GetElement(2,3));
    ctOrigin->Update();
  
  std::cout << "Reading particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > airwayParticlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    airwayParticlesReader->SetFileName( inAirwayParticles.c_str() );
    airwayParticlesReader->Update();

  vtkPolyData* particles=airwayParticlesReader->GetOutput();
  
  //Transform CT data to be positive
  vtkSmartPointer< vtkImageMathematics > transformedCT = vtkSmartPointer< vtkImageMathematics >::New();
    transformedCT->SetInput1Data(ctOrigin->GetOutput());
    transformedCT->SetOperationToAddConstant();
    transformedCT->SetConstantC(1024);
    transformedCT->Update();
  
  //Check to see if we have the proper point data field
  if (inPlane == false && particles->GetPointData()->GetArray("hevec2") == NULL)
    {
      std::cerr<<"Particle data does not have local frame information. This vtk file might not be a particle dataset"<<std::endl;
      return cip::EXITFAILURE;
    } 
  else 
    {
      particles->GetPointData()->SetActiveVectors("hevec2");
    }
  
  // Solver filter
  vtkSmartPointer< vtkComputeAirwayWall > solver  = vtkSmartPointer< vtkComputeAirwayWall >::New();
  
  if (method == "FWHM")
    {
      solver->SetMethod(0);
    } 
  else if (method == "ZC")
    {
      solver->SetMethod(1);
    } 
  else if (method == "PC")
    {
      solver->SetMethod(2);
    }
  
  solver->SetNumberOfThetaSamples(numberOfRays);

  // Airway wall computing filter
  vtkSmartPointer< vtkComputeAirwayWallPolyData > filter = 
    vtkSmartPointer< vtkComputeAirwayWallPolyData >::New();
    filter->SetInputData(particles);
    filter->SetImage(transformedCT->GetOutput());
    filter->SetWallSolver(solver);
  
  // Setting up thresholds
  solver->SetGradientThreshold(gradientThreshold);
  solver->SetWallThreshold(wallThreshold);
  solver->SetPCThreshold(pcThreshold);
  
  if ( largeAirways == true)
    {
      solver->SetRMax(25);
      filter->SetResolution(0.2);
    }
    
  if (inPlane == true)
    {
      filter->ReformatOff();
    } 
  else 
    {
      filter->ReformatOn();
      filter->SetAxisModeToVector();
    }
      
  if (saveAirwayImages == true)
    {
      filter->SaveAirwayImageOn();
      std::vector<std::string> components;
      components.push_back(outputDirectory + "/");
      components.push_back(saveAirwayPrefix);
      filter->SetAirwayImagePrefix(vtksys::SystemTools::JoinPath(components).c_str());
    } 
  else
    {
      filter->SaveAirwayImageOff();
    }
    
  filter->Update();
  
  std::cout << "Writing particles file..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetFileName( outAirwayParticles.c_str() );
    particlesWriter->SetInputConnection( filter->GetOutputPort() );
    particlesWriter->SetFileTypeToBinary();
    particlesWriter->Write();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}
