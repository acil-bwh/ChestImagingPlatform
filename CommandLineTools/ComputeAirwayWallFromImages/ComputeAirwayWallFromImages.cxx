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
#include "vtkExtractVOI.h"
#include "vtkEllipseFitting.h"
#include "vtkImageResliceWithPlane.h"
#include <vtksys/SystemTools.hxx>
#include "cipHelper.h"

#include "ComputeAirwayWallFromImagesCLP.h"

namespace
{
  typedef itk::ImageToVTKImageFilter< cip::CTType > ConnectorType;
}

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
  
  std::cout << "Reading CT..." << std::endl;
  cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
    reader->SetFileName( inImage );
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
  
  //Transform CT data to be positive
  vtkSmartPointer< vtkImageMathematics > transformedCT = vtkSmartPointer< vtkImageMathematics >::New();
    transformedCT->SetInput1Data(ctOrigin->GetOutput());
    transformedCT->SetOperationToAddConstant();
    transformedCT->SetConstantC(1024);
    transformedCT->Update();
  
  
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
  
  
  // Setting up thresholds
  solver->SetGradientThreshold(gradientThreshold);
  solver->SetWallThreshold(wallThreshold);
  solver->SetPCThreshold(pcThreshold);

  // Airway wall computing helper
  vtkSmartPointer< vtkComputeAirwayWallPolyData > filter = 
    vtkSmartPointer< vtkComputeAirwayWallPolyData >::New();
    filter->SetImage(transformedCT->GetOutput());
    filter->SetWallSolver(solver);
    filter->ReformatOff();

  if ( largeAirways == true )
    {
      solver->SetRMax(25);
      filter->SetResolution(0.2);
    }
  
  // Loop through image array extracting slices
  int dims[3];
  transformedCT->GetOutput()->GetDimensions(dims);
  
  vtkDoubleArray *mean = vtkDoubleArray::New();
  vtkDoubleArray *std = vtkDoubleArray::New();
  vtkDoubleArray *min = vtkDoubleArray::New();
  vtkDoubleArray *max = vtkDoubleArray::New();
  vtkDoubleArray *ellipse = vtkDoubleArray::New();
  
  int nc = solver->GetNumberOfQuantities();
  int np = dims[2];
  
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

  
  for (int zz=0; zz<dims[2]; zz++)
    {
      vtkSmartPointer< vtkExtractVOI > slice_filter =
          vtkSmartPointer< vtkExtractVOI >::New();
      
      std::cout<<"Processing slize number: "<<zz<<std::endl;
      
      slice_filter->SetInputData(transformedCT->GetOutput());
      slice_filter->SetVOI(0,dims[0]-1,0, dims[1]-1,zz,zz);
      slice_filter->Update();
      
      float resolution = filter->GetResolution();
      
      vtkImageResliceWithPlane *reslicer = vtkImageResliceWithPlane::New();
      // Set up options
      reslicer->InPlaneOn();
      reslicer->SetInputData(slice_filter->GetOutput());
      reslicer->SetInterpolationModeToCubic();
      reslicer->ComputeCenterOff();
      reslicer->SetCenter((dims[0]-1)/2.0,(dims[1]-1)/2.0,zz);
      reslicer->SetDimensions(256,256,1);
      reslicer->SetSpacing(resolution,resolution,resolution);
      reslicer->Update();
      
      solver->SetInputData(reslicer->GetOutput());
      
      vtkSmartPointer< vtkEllipseFitting > eifit =
              vtkSmartPointer< vtkEllipseFitting >::New();
      vtkSmartPointer< vtkEllipseFitting > eofit =
              vtkSmartPointer< vtkEllipseFitting >::New();
      
      filter->ComputeWallFromSolver(solver,eifit,eofit);
      
      // Get metrics
      for (int c = 0; c < solver->GetNumberOfQuantities();c++) {
        mean->SetComponent(zz,c,solver->GetStatsMean()->GetComponent(c,0));
        std->SetComponent(zz,c,solver->GetStatsStd()->GetComponent(c,0));
        min->SetComponent(zz,c,solver->GetStatsMin()->GetComponent(c,0));
        max->SetComponent(zz,c,solver->GetStatsMax()->GetComponent(c,0));
      }
      
      ellipse->SetComponent(zz,0,eifit->GetMinorAxisLength()*resolution);
      ellipse->SetComponent(zz,1,eifit->GetMajorAxisLength()*resolution);
      ellipse->SetComponent(zz,2,eifit->GetAngle());
      ellipse->SetComponent(zz,3,eofit->GetMinorAxisLength()*resolution);
      ellipse->SetComponent(zz,4,eofit->GetMajorAxisLength()*resolution);
      ellipse->SetComponent(zz,5,eofit->GetAngle());
      
      if (saveAirwayImages == true)
        {
          std::vector<std::string> components;
          components.push_back(outputDirectory + "/");
          components.push_back(saveAirwayPrefix);
          const char *saveAirwayImagePrefix=vtksys::SystemTools::JoinPath(components).c_str();
          char fileName[10*256];
          sprintf(fileName,"%s_airwayWallImage%s-%05d.png",saveAirwayImagePrefix,method.c_str(),zz);
          filter->SaveQualityControlImage(fileName,reslicer->GetOutput(),eifit,eofit);
        }

    }
  
  std::cout << "Writing csv file..." << std::endl;
  std::cout << "Writing phenotypes to file..." << std::endl;
  std::ofstream phenotypesFile( outMetricsCSV.c_str() );
  phenotypesFile << "z,";
  phenotypesFile << "method,";
  phenotypesFile << "meanInnerRadius,";
  phenotypesFile << "meanOuterRadius,";
  phenotypesFile << "meanWallThickness,";
  phenotypesFile << "meanInnerRadiusEllipseFitting,";
  phenotypesFile << "meanOuterRadiusEllipseFitting,";
  phenotypesFile << "meanWallThicknessEllipseFitting,";
  phenotypesFile << "stdInnerRadius,";
  phenotypesFile << "stdOuterRadius,";
  phenotypesFile << "stdWallThickness" << std::endl;
  
  for (int zz=0; zz<dims[2]; zz++)
  {
    
    phenotypesFile << zz << ",";
    phenotypesFile << method << ",";
    for (int c=0; c<3;c++)
    {
      phenotypesFile << mean->GetComponent(zz,c) << ",";
    }
    
    float ai = ellipse->GetComponent(zz,1);
    float bi = ellipse->GetComponent(zz,0);
    float ao = ellipse->GetComponent(zz,4);
    float bo = ellipse->GetComponent(zz,3);
    
    phenotypesFile << std::sqrt(ai*bi) << ",";
    phenotypesFile << std::sqrt(ao*bo) << ",";
    phenotypesFile << (ao*bo-ai*bi)/(std::sqrt(ai*bi)+std::sqrt(ao*bo)) << ",";
    
    for (int c=0; c<3;c++)
    {
      phenotypesFile << std->GetComponent(zz,c) << ",";
    }
    phenotypesFile << std::endl;
    
  }
  
  phenotypesFile.close();
  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}
