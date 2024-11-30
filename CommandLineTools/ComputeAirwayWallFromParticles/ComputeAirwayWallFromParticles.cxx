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
    
  
  //Extract direction information
  cip::CTType::DirectionType d=reader->GetOutput()->GetDirection();
  vtkSmartPointer< vtkMatrix4x4 > lps = vtkSmartPointer< vtkMatrix4x4 >::New();
  for (int i=0; i<3; i++)
        for (int k=0; k<3; k++)
            lps->SetElement(i,k, d(i,k));
    
  // Add translation to the user matrix
  cip::CTType::PointType origin=reader->GetOutput()->GetOrigin();
  
  for (int i=0; i<3; i++)
    {
        lps->SetElement(i,3, origin[i]);
    }

  
  //Particles are in LPS (b/c we assume that images come from DICOM).
  //Make sure that the origin in the vtkImageData is properly set up.
    vtkSmartPointer< vtkImageChangeInformation > ctOrigin = vtkSmartPointer< vtkImageChangeInformation >::New();
    ctOrigin->SetInputData( connector->GetOutput() );
    ctOrigin->SetOutputOrigin(1.0*lps->GetElement(0,3),1.0*lps->GetElement(1,3),lps->GetElement(2,3));
    ctOrigin->Update();
  
    //std::cout<<"Origin ITK reader: "<<reader->GetOutput()->GetOrigin()[0]<<" "<<
    //reader->GetOutput()->GetOrigin()[1]<<" "<<reader->GetOutput()->GetOrigin()[2]<<std::endl;

    //std::cout<<"Origing tt: "<<ctOrigin->GetOutput()->GetOrigin()[0]<<" "<<
    //ctOrigin->GetOutput()->GetOrigin()[1]<<" "<<ctOrigin->GetOutput()->GetOrigin()[2]<<std::endl;
  
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
    
  solver->SetDelta(0.5);
  solver->SetNumberOfThetaSamples(numberOfRays);
    
  // Outlier rejection parameter
  solver->SetStdFactor(stdFactor);
    

  // Airway wall computing filter
  vtkSmartPointer< vtkComputeAirwayWallPolyData > filter = 
    vtkSmartPointer< vtkComputeAirwayWallPolyData >::New();
    filter->SetInputData(particles);
    filter->SetImage(transformedCT->GetOutput());
    filter->SetWallSolver(solver);
    if ( fineCentering == true )
    {
      filter->FineCenteringOn();
    } else {
      filter->FineCenteringOff();
    }
    if ( centroidCentering == true)
    {
      filter->CentroidCenteringOn();
    } else {
      filter->CentroidCenteringOff();
    }
    
  // Setting up thresholds
  solver->SetGradientThreshold(gradientThreshold);
  solver->SetWallThreshold(wallThreshold);
  solver->SetPCThreshold(pcThreshold);
  
  if ( largeAirways == true )
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
