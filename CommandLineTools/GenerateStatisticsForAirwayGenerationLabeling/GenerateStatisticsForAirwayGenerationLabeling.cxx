/** \file
 *  \ingroup commandLineTools 
 *  \details This program computes statistics needed as inputs to the 
 *  'LabelAirwayParticlesByGeneration' program. It computes these statistics
 *  over (possibly) multiple, labeled input airway particles datasets. The
 *  user must specify information needed to construct the minimum spanning
 *  tree (which encodes topology over the particles). This information should
 *  be the same that is used for the 'LabelAirwayParticlesByGeneration'
 *  program.
 *
 *  USAGE: 
 *
 *  GenerateStatisticsForAirwayGenerationLabeling  [-d <double>] -i
 *                                         <string> ...  [--] [--version]
 *                                         [-h]
 * 
 *  Where:
 * 
 *  -d <double>,  --distThresh <double>
 *    Particle distance threshold for constructing minimum spanning tree.
 *    Particles further apart than this distance will not have an edge
 *    placed between them in the weighted graph passed to the min spanning
 *    tree algorithm
 * 
 *  -i <string>,  --input <string>  (accepted multiple times)
 *    (required)  Input particles file names
 * 
 *  --,  --ignore_rest
 *    Ignores the rest of the labeled arguments following this flag.
 * 
 *  --version
 *    Displays version information and exits.
 * 
 *  -h,  --help
 *    Displays usage information and exits. 
 *
 *  $Author: jross $
 *  $Date: 2013-03-25 13:23:52 -0400 (Mon, 25 Mar 2013) $
 *  $Revision: 383 $ 
 *
 */


//#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkPolyData.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkMutableUndirectedGraph.h"
#include "vtkDoubleArray.h"
#include "vtkBoostKruskalMinimumSpanningTree.h"
#include "vtkExtractSelectedGraph.h"
#include "vtkMutableUndirectedGraph.h"
#include "vtkDataSetAttributes.h"
#include "vtkEdgeListIterator.h"
#include <cfloat>
#include <math.h>
#include <fstream>
#include "GenerateStatisticsForAirwayGenerationLabelingCLP.h"

// namespace
// {
//   struct TRANSITION
//   {
//     unsigned char from;
//     unsigned char to;
//     std::vector<double> scaleDifferences;
//     std::vector<double> angles;
//   };

//   struct EMISSION
//   {
//     double distance;
//     double scaleDiff;
//     double angle;
//   };

//   /*vtkSmartPointer<vtkMutableUndirectedGraph> GetMinimumSpanningTree(vtkSmartPointer<vtkPolyData>, double, double);
//     bool GetEdgeWeight(unsigned int, unsigned int, vtkSmartPointer<vtkPolyData>, double*, double, double);
//     void UpdateEmissionProbabilityData(vtkSmartPointer<vtkPolyData>, vtkSmartPointer<vtkPolyData>,
//     double, std::map<unsigned char, std::vector<EMISSION> >*);
//     void PrintEmissionProbabilityData(std::map<unsigned char, std::vector<EMISSION> >);
//     void WriteEmissionProbabilityData(std::map<unsigned char, std::vector<EMISSION> >, std::string);
//     void WriteTransitionProbabilityScaleAngleData(std::vector<TRANSITION>, std::string);
//     void WriteTransitionProbabilities(std::vector<TRANSITION>, std::string);
//     bool IsTransitionPermitted( unsigned char, unsigned char );
//     void WriteTransitionDataISBI( std::vector<TRANSITION> transitionVec );
//     void WriteEmissionDataISBI( std::map<unsigned char, std::vector<EMISSION> > emissionProbData );
//   */
         
//   bool GetEdgeWeight(unsigned int particleID1, unsigned int particleID2, vtkSmartPointer<vtkPolyData> particles,
// 		     double* weight, double particleDistanceThreshold, double edgeWeightAngleSigma)
//   {
//     //
//     // Determine the vector connecting the two particles
//     //
//     double point1[3];
//     point1[0] = particles->GetPoint(particleID1)[0];
//     point1[1] = particles->GetPoint(particleID1)[1];
//     point1[2] = particles->GetPoint(particleID1)[2];
        
//     double point2[3];
//     point2[0] = particles->GetPoint(particleID2)[0];
//     point2[1] = particles->GetPoint(particleID2)[1];
//     point2[2] = particles->GetPoint(particleID2)[2];
        
//     cip::VectorType connectingVec(3);
//     connectingVec[0] = point1[0] - point2[0];
//     connectingVec[1] = point1[1] - point2[1];
//     connectingVec[2] = point1[2] - point2[2];
        
//     double connectorMagnitude = cip::GetVectorMagnitude(connectingVec);
        
//     if (connectorMagnitude> particleDistanceThreshold)
//       {
// 	return false;
//       }
        
//     cip::VectorType particle1Hevec2(3);
//       particle1Hevec2[0] = particles->GetPointData()->GetArray("hevec2")->GetTuple(particleID1)[0];
//       particle1Hevec2[1] = particles->GetPointData()->GetArray("hevec2")->GetTuple(particleID1)[1];
//       particle1Hevec2[2] = particles->GetPointData()->GetArray("hevec2")->GetTuple(particleID1)[2];
        
//     cip::VectorType particle2Hevec2(3);
//       particle2Hevec2[0] = particles->GetPointData()->GetArray("hevec2")->GetTuple(particleID2)[0];
//       particle2Hevec2[1] = particles->GetPointData()->GetArray("hevec2")->GetTuple(particleID2)[1];
//       particle2Hevec2[2] = particles->GetPointData()->GetArray("hevec2")->GetTuple(particleID2)[2];
        
//     double angle1 =  cip::GetAngleBetweenVectors(particle1Hevec2, connectingVec, true);
//     double angle2 =  cip::GetAngleBetweenVectors(particle2Hevec2, connectingVec, true);
        
//     if (angle1 <angle2)
//       {
// 	*weight = connectorMagnitude*(1.0 + 1.1*exp(-pow((90.0 - angle1)/edgeWeightAngleSigma, 2)));
//       }
//     else
//       {
// 	*weight = connectorMagnitude*(1.0 + 1.1*exp(-pow((90.0 - angle2)/edgeWeightAngleSigma, 2)));
//       }
        
//     return true;
//   }
    
//   vtkSmartPointer<vtkMutableUndirectedGraph> GetMinimumSpanningTree(vtkSmartPointer<vtkPolyData> particles, double particleDistanceThreshold,
// 								    double edgeWeightAngleSigma)
//   {
//     unsigned int numberParticles = particles->GetNumberOfPoints();
        
//     // Now create the weighted graph that will be passed to the minimum
//     // spanning tree filter
//     std::map<unsigned int, unsigned int> particleIDToNodeIDMap;
//     std::map<unsigned int, unsigned int> nodeIDToParticleIDMap;
        
//     vtkSmartPointer<vtkMutableUndirectedGraph> weightedGraph =
//       vtkSmartPointer<vtkMutableUndirectedGraph>::New();
        
//     for (unsigned int i=0; i<numberParticles; i++)
//       {
// 	vtkIdType nodeID = weightedGraph->AddVertex();
            
// 	particleIDToNodeIDMap[i]      = nodeID;
// 	nodeIDToParticleIDMap[nodeID] = i;
//       }
        
//     vtkSmartPointer<vtkDoubleArray> edgeWeights = vtkSmartPointer<vtkDoubleArray>::New();
//     edgeWeights->SetNumberOfComponents(1);
//     edgeWeights->SetName("Weights");
        
//     for (unsigned int i=0; i<numberParticles; i++)
//       {
// 	for (unsigned int j=i+1; j<numberParticles; j++)
// 	  {
// 	    double weight;
                
// 	    if (GetEdgeWeight(i, j, particles, &weight, particleDistanceThreshold, edgeWeightAngleSigma))
// 	      {
// 		weightedGraph->AddEdge(particleIDToNodeIDMap[i], particleIDToNodeIDMap[j]);
// 		edgeWeights->InsertNextValue(weight);
// 	      }
// 	  }
//       }
        
//     weightedGraph->GetEdgeData()->AddArray(edgeWeights);
//     weightedGraph->SetPoints(particles->GetPoints());
        
//     vtkSmartPointer<vtkBoostKruskalMinimumSpanningTree> minimumSpanningTreeFilter =
//       vtkSmartPointer<vtkBoostKruskalMinimumSpanningTree>::New();
//     minimumSpanningTreeFilter->SetInput(weightedGraph);
//     minimumSpanningTreeFilter->SetEdgeWeightArrayName("Weights");
//     minimumSpanningTreeFilter->Update();
        
//     vtkSmartPointer<vtkExtractSelectedGraph> extractSelection = vtkSmartPointer<vtkExtractSelectedGraph>::New();
//     extractSelection->SetInput(0, weightedGraph);
//     extractSelection->SetInput(1, minimumSpanningTreeFilter->GetOutput()) ;
//     extractSelection->Update();
        
//     return vtkMutableUndirectedGraph::SafeDownCast(extractSelection->GetOutput());
//   }

//   void UpdateEmissionProbabilityData(vtkSmartPointer<vtkPolyData> refParticles, vtkSmartPointer<vtkPolyData> particles,
// 				     double epsilon, std::map<unsigned char, std::vector<EMISSION> >* emissionProbData)
//   {
//     unsigned int numRefParticles = refParticles->GetNumberOfPoints();
//     unsigned int numParticles = particles->GetNumberOfPoints();
        
//     for (unsigned int i=0; i<numRefParticles; i++)
//       {
// 	float refParticleType = refParticles->GetPointData()->GetArray("ChestType")->GetTuple(i)[0];
            
// 	for (unsigned int j=0; j<numParticles; j++)
// 	  {
// 	    float particleType = particles->GetPointData()->GetArray("ChestType")->GetTuple(j)[0];
// 	    if (refParticleType == particleType)
// 	      {
// 		cip::VectorType vec(3);
// 		  vec[0] = refParticles->GetPoint(i)[0] - particles->GetPoint(j)[0];
// 		  vec[1] = refParticles->GetPoint(i)[1] - particles->GetPoint(j)[1];
// 		  vec[2] = refParticles->GetPoint(i)[2] - particles->GetPoint(j)[2];
                    
// 		double dist = cip::GetVectorMagnitude(vec);
// 		if (dist <= epsilon)
// 		  {
// 		    float scale1 = refParticles->GetPointData()->GetArray("scale")->GetTuple(i)[0];
// 		    float scale2 = particles->GetPointData()->GetArray("scale")->GetTuple(j)[0];
                        
// 		    cip::VectorType direction1(3);
// 		      direction1[0] = refParticles->GetPointData()->GetArray("hevec2")->GetTuple(i)[0];
// 		      direction1[1] = refParticles->GetPointData()->GetArray("hevec2")->GetTuple(i)[1];
// 		      direction1[2] = refParticles->GetPointData()->GetArray("hevec2")->GetTuple(i)[2];
                        
// 		    cip::VectorType direction2(3);
// 		      direction2[0] = particles->GetPointData()->GetArray("hevec2")->GetTuple(j)[0];
// 		      direction2[1] = particles->GetPointData()->GetArray("hevec2")->GetTuple(j)[1];
// 		      direction2[2] = particles->GetPointData()->GetArray("hevec2")->GetTuple(j)[2];
                        
// 		    EMISSION emissionData;
// 		    emissionData.distance = dist;
// 		    emissionData.angle = cip::GetAngleBetweenVectors(direction1, direction2, true);
                        
// 		    // std::cout << "**:\t" << emissionData.angle << "\t" << direction1[0] << "," << direction1[1] << "," << direction1[2] << "\t";
// 		    // std::cout << direction2[0] << "," << direction2[1] << "," << direction2[2] << std::endl;
// 		    emissionData.scaleDiff = scale1 - scale2;
                        
// 		    (*emissionProbData)[(unsigned char)(refParticleType)].push_back(emissionData);
// 		  }
// 	      }
// 	  }
//       }
//   }
    
    
//   void PrintEmissionProbabilityData(std::map<unsigned char, std::vector<EMISSION> > emissionProbData)
//   {
//     cip::ChestConventions conventions;
        
//     std::map<unsigned char, std::vector<EMISSION> >::iterator mapIt = emissionProbData.begin();
        
//     while (mapIt != emissionProbData.end())
//       {
// 	unsigned int numEntries = (mapIt->second).size();
            
// 	if (numEntries == 0)
// 	  {
// 	    std::cout << "Warning: Emission probability epsilon ball is too small for ";
// 	    std::cout << conventions.GetChestTypeName(mapIt->first) << std::endl;
// 	  }
            
// 	double distanceAccum  = 0.0;
// 	double scaleDiffAccum = 0.0;
// 	double angleAccum     = 0.0;
            
// 	// Compute means
// 	for (unsigned int i=0; i<numEntries; i++)
// 	  {
// 	    distanceAccum  += (mapIt->second)[i].distance;
// 	    scaleDiffAccum += (mapIt->second)[i].scaleDiff;
// 	    angleAccum     += (mapIt->second)[i].angle;
// 	  }
            
// 	double distanceMean  = distanceAccum/float(numEntries);
// 	double scaleDiffMean = scaleDiffAccum/float(numEntries);
// 	double angleMean     = angleAccum/float(numEntries);
            
// 	std::cout << "angleAccum:\t" << angleAccum << std::endl;
// 	std::cout << "numEntries:\t" << numEntries << std::endl;
            
// 	// Compute the standard deviations
// 	distanceAccum  = 0.0;
// 	scaleDiffAccum = 0.0;
// 	angleAccum     = 0.0;
            
// 	for (unsigned int i=0; i<numEntries; i++)
// 	  {
// 	    distanceAccum  += pow((mapIt->second)[i].distance - distanceMean, 2.0);
// 	    scaleDiffAccum += pow((mapIt->second)[i].scaleDiff - scaleDiffMean, 2.0);
// 	    angleAccum     += pow((mapIt->second)[i].angle - angleMean, 2.0);
// 	  }
            
// 	double distanceSTD  = sqrt(distanceAccum/float(numEntries));
// 	double scaleDiffSTD = sqrt(scaleDiffAccum/float(numEntries));
// 	double angleSTD     = sqrt(angleAccum/float(numEntries));
            
// 	std::cout << "------------------------------------------" << std::endl;
// 	std::cout << "Emission probability stats for " << conventions.GetChestTypeName(mapIt->first) << ":" << std::endl;
// 	std::cout << "Distance:\t"    << distanceMean  << "\t" << " +/- " << distanceSTD << std::endl;
// 	std::cout << "Scale Diff:\t"  << scaleDiffMean << "\t" << " +/- " << scaleDiffSTD << std::endl;
// 	std::cout << "Angle:\t"       << angleMean     << "\t" << " +/- " << angleSTD << std::endl;
// 	std::cout << "Num Samples:\t" << numEntries    << std::endl;
            
// 	++mapIt;
//       }
//   }
    
    
//   void WriteEmissionProbabilityData(std::map<unsigned char, std::vector<EMISSION> > emissionProbData, std::string fileName)
//   {
//     std::ofstream file(fileName.c_str());
        
//     file << "ChestType, scaleDiffMean, scaleDiffSTD, distanceMean, distanceSTD, angleMean, angleSTD, numSamples" << std::endl;
        
//     cip::ChestConventions conventions;
        
//     std::map<unsigned char, std::vector<EMISSION> >::iterator mapIt = emissionProbData.begin();
        
//     while (mapIt != emissionProbData.end())
//       {
// 	unsigned int numEntries = (mapIt->second).size();
            
// 	if (numEntries == 0)
// 	  {
// 	    std::cout << "Warning: Emission probability epsilon ball is too small for ";
// 	    std::cout << conventions.GetChestTypeName(mapIt->first) << std::endl;
// 	  }
            
// 	double distanceAccum  = 0.0;
// 	double scaleDiffAccum = 0.0;
// 	double angleAccum     = 0.0;
            
// 	// Compute means
// 	for (unsigned int i=0; i<numEntries; i++)
// 	  {
// 	    distanceAccum  += (mapIt->second)[i].distance;
// 	    scaleDiffAccum += (mapIt->second)[i].scaleDiff;
// 	    angleAccum += (mapIt->second)[i].angle;
// 	  }
            
// 	double distanceMean  = distanceAccum/float(numEntries);
// 	double scaleDiffMean = scaleDiffAccum/float(numEntries);
// 	double angleMean     = angleAccum/float(numEntries);
            
// 	// Compute the standard deviations
// 	distanceAccum  = 0.0;
// 	scaleDiffAccum = 0.0;
// 	angleAccum     = 0.0;
            
// 	for (unsigned int i=0; i<numEntries; i++)
// 	  {
// 	    if ((mapIt->second)[i].distance == (mapIt->second)[i].distance)
// 	      {
// 		distanceAccum  += pow((mapIt->second)[i].distance - distanceMean, 2.0);
// 	      }
// 	    if ((mapIt->second)[i].scaleDiff == (mapIt->second)[i].scaleDiff)
// 	      {
// 		scaleDiffAccum += pow((mapIt->second)[i].scaleDiff - scaleDiffMean, 2.0);
// 	      }
// 	    if ((mapIt->second)[i].angle == (mapIt->second)[i].angle)
// 	      {
// 		angleAccum += pow((mapIt->second)[i].angle - angleMean, 2.0);
// 	      }
// 	  }
            
// 	double distanceSTD  = sqrt(distanceAccum/float(numEntries));
// 	double scaleDiffSTD = sqrt(scaleDiffAccum/float(numEntries));
// 	double angleSTD     = sqrt(angleAccum/float(numEntries));
            
// 	file << conventions.GetChestTypeName(mapIt->first) << ",";
// 	file << scaleDiffMean << ",";
// 	file << scaleDiffSTD  << ",";
// 	file << distanceMean  << ",";
// 	file << distanceSTD   << ",";
// 	file << angleMean     << ",";
// 	file << angleSTD      << ",";
// 	file << numEntries    << std::endl;
            
// 	++mapIt;
//       }
        
//     file.close();
//   }
    
    
//   void WriteTransitionProbabilityScaleAngleData(std::vector<TRANSITION> transitionVec, std::string fileName)
//   {
//     std::ofstream file(fileName.c_str());
        
//     file << "fromChestType, toChestType, scaleDiffMean, scaleDiffSTD, angleMean, angleSTD, numSamples" << std::endl;
        
//     cip::ChestConventions conventions;
//     for (unsigned int i=0; i<transitionVec.size(); i++)
//       {
// 	unsigned int N = transitionVec[i].angles.size();
            
// 	double scaleAccum = 0.0;
// 	double angleAccum = 0.0;
// 	for (unsigned int j=0; j<N; j++)
// 	  {
// 	    scaleAccum += transitionVec[i].scaleDifferences[j];
// 	    angleAccum += transitionVec[i].angles[j];
// 	  }
            
// 	double scaleDiffMean = scaleAccum/static_cast<double>(N);
// 	double angleMean     = angleAccum/static_cast<double>(N);
            
// 	scaleAccum = 0.0;
// 	angleAccum = 0.0;
// 	for (unsigned int j=0; j<N; j++)
// 	  {
// 	    scaleAccum += pow(transitionVec[i].scaleDifferences[j] - scaleDiffMean, 2.0);
// 	    angleAccum += pow(transitionVec[i].angles[j] - angleMean, 2.0);
// 	  }
            
// 	double scaleDiffSTD = sqrt(scaleAccum/static_cast<double>(N));
// 	double angleSTD     = sqrt(angleAccum/static_cast<double>(N));
            
// 	file << conventions.GetChestTypeName(static_cast<unsigned char>(transitionVec[i].from)) << ",";
// 	file << conventions.GetChestTypeName(static_cast<unsigned char>(transitionVec[i].to))   << ",";
// 	file << scaleDiffMean << ",";
// 	file << scaleDiffSTD << ",";
// 	file << angleMean << ",";
// 	file << angleSTD << ",";
// 	file << N << std::endl;
//       }
//   }
    
    
//   void WriteTransitionProbabilities(std::vector<TRANSITION> transitionVec, std::string fileName)
//   {
//     std::ofstream file(fileName.c_str());
        
//     //
//     // Initialized the probability matrix. Note that we will support
//     // up to airway generation 10.
//     //
//     std::vector<std::vector<double> > probMat;
//     std::vector<double> rowSum;
        
//     for (unsigned int i=0; i<=10; i++)
//       {
// 	std::vector<double> tmp;
// 	for (unsigned int j=0; j<=10; j++)
// 	  {
// 	    tmp.push_back(0.0);
// 	  }
// 	probMat.push_back(tmp);
// 	rowSum.push_back(0.0);
//       }
        
//     for (unsigned int i=0; i<transitionVec.size(); i++)
//       {
// 	unsigned int N = transitionVec[i].angles.size();
            
// 	int row = transitionVec[i].from - 38;
// 	int col = transitionVec[i].to - 38;
            
// 	if (row>=0 && row <= 10 && col>=0 && col <=10)
// 	  {
// 	    probMat[row][col] = float(N);
// 	    rowSum[row] += float(N);
// 	  }
//       }
        
//     // Now normalize and write to file
//     for (unsigned int i=0; i<=10; i++)
//       {
// 	for (unsigned int j=0; j<=10; j++)
// 	  {
// 	    if (rowSum[i] > 0.0)
// 	      {
// 		probMat[i][j] = probMat[i][j]/rowSum[i];
// 	      }
// 	    file << probMat[i][j] << ",";
// 	  }
// 	file << std::endl;
//       }
        
//     file.close();
//   }
    
//   void WriteEmissionDataISBI( std::map<unsigned char, std::vector<EMISSION> > emissionProbData )
//   {
//     cip::ChestConventions conventions;
        
//     std::map<unsigned char, std::vector<EMISSION> >::iterator mapIt;
        
//     mapIt = emissionProbData.begin();
//     while ( mapIt != emissionProbData.end() )
//       {
// 	std::string fileName = "/Users/jross/Documents/ConferencesAndJournals/ISBI/2014/AirwayGenerationLabeling/Emissions/";
// 	fileName.append( conventions.GetChestTypeName(mapIt->first) );
// 	fileName.append(".csv");
// 	std::ofstream file( fileName.c_str() );
            
// 	for ( unsigned int i=0; i<mapIt->second.size(); i++ )
// 	  {
// 	    file << mapIt->second[i].distance << ",";
// 	    file << mapIt->second[i].scaleDiff << ",";
// 	    file << mapIt->second[i].angle << std::endl;
// 	  }
            
// 	file.close();
            
// 	mapIt++;
//       }
//   }
    
//   // This function was written for the ISBI 2014 effort. It takes all the accumulated transition data
//   // and writes to file. Each transition is written to its own file: the first column is the scale
//   // difference, and the second column is the angle. These data files are meant to be read and analyzed
//   // externally in order to create suitable parametric representations of the probability densities
//   void WriteTransitionDataISBI( std::vector<TRANSITION> transitionVec )
//   {
//     cip::ChestConventions conventions;
//     for (unsigned int i=0; i<transitionVec.size(); i++)
//       {
// 	std::cout << "---------------------------------" << std::endl;
// 	unsigned int N = transitionVec[i].angles.size();
// 	std::cout << N << std::endl;
            
// 	std::string transitionFileName = "/Users/jross/Documents/ConferencesAndJournals/ISBI/2014/AirwayGenerationLabeling/Transitions/";
// 	transitionFileName.append(conventions.GetChestTypeName(static_cast<unsigned char>(transitionVec[i].from)));
// 	transitionFileName.append("_to_");
// 	transitionFileName.append(conventions.GetChestTypeName(static_cast<unsigned char>(transitionVec[i].to)));
// 	transitionFileName.append(".csv");
// 	std::cout << transitionFileName << std::endl;
            
// 	std::ofstream file( transitionFileName.c_str() );
// 	file << N << std::endl;
// 	for (unsigned int j=0; j<N; j++)
// 	  {
// 	    file << transitionVec[i].scaleDifferences[j] << ",";
// 	    file << transitionVec[i].angles[j] << std::endl;
// 	  }
            
// 	file.close();
//       }
//   }
    
//   bool IsTransitionPermitted( unsigned char fromState, unsigned char toState )
//   {
//     if ( fromState == toState )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION5) &&
// 	      toState == (unsigned char)(cip::AIRWAYGENERATION4) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION4) &&
// 	      toState == (unsigned char)(cip::AIRWAYGENERATION3) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION3) &&
// 	      toState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION3) &&
// 	      toState == (unsigned char)(cip::INTERMEDIATEBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION3) &&
// 	      toState == (unsigned char)(cip::MIDDLELOBEBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION3) &&
// 	      toState == (unsigned char)(cip::LOWERLOBEBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION3) &&
// 	      toState == (unsigned char)(cip::SUPERIORDIVISIONBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::AIRWAYGENERATION3) &&
// 	      toState == (unsigned char)(cip::LINGULARBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::UPPERLOBEBRONCHUS) &&
// 	      toState == (unsigned char)(cip::MAINBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::INTERMEDIATEBRONCHUS) &&
// 	      toState == (unsigned char)(cip::MAINBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::MIDDLELOBEBRONCHUS) &&
// 	      toState == (unsigned char)(cip::LOWERLOBEBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::LOWERLOBEBRONCHUS) &&
// 	      toState == (unsigned char)(cip::INTERMEDIATEBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::LOWERLOBEBRONCHUS) &&
// 	      toState == (unsigned char)(cip::MAINBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::MAINBRONCHUS) &&
// 	      toState == (unsigned char)(cip::TRACHEA) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::SUPERIORDIVISIONBRONCHUS) &&
// 	      toState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
//       {
// 	return true;
//       }
//     else if ( fromState == (unsigned char)(cip::LINGULARBRONCHUS) &&
// 	      toState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
//       {
// 	return true;
//       }
        
//     return false;
//   }
// } //end namespace


int main(int argc, char *argv[])
{
  //
  // Begin by defining the arguments to be passed
  //
  std::vector<std::string>   inParticlesFileNames;
  /*  std::string                refParticlesFileName           = "NA";
      std::string                emissionProbsFileName          = "NA";
      std::string                transitionProbsFileName        = "NA";
      std::string                normTransProbsMeanVarFileName  = "NA";
      std::string                transProbsFileName             = "NA";*/
  //  double                     particleDistanceThreshold      = 20.0;
  double                     edgeWeightAngleSigma           = 1.0;
  //  double                     emissionProbDistThresh         = 20.0;

  //
  // Argument descriptions for user help
  //
  /*
  //std::string inParticlesFileNamesDesc  = "Input particles file names";
  //  std::string particleDistanceThresholdDesc = "Particle distance threshold for constructing \
  minimum spanning tree. Particles further apart than this distance will not have an \
  edge placed between them in the weighted graph passed to the min spanning tree \
  algorithm";
  // std::string refParticlesFileNameDesc = "Specify a (labeled) reference particle dataset \
  to compute statistics for emission probabilities. For each particle in this \
  dataset, every other particle in the files specified with the -i flag will be \
  considered. If the two particles have the same generation label and are within \
  the distance specified by the --ed flag, then the scale difference, \
  angle, and distance between the particles will be computed and used to compute the \
  class conditional probabilities for that generation. This is an optional argument.\
  Note that if it is specified, the same file should not also appear as an input \
  specified with the -i flag.";
  //  std::string emissionProbDistThreshDesc = "The radius of the epsilon ball used when \
  considering if a particle should be considered for computing the class-conditional \
  emission probabilities. Only necessary if a reference particle dataset is specified \
  with the --ed flag.";
  // std::string emissionProbsFileNameDesc = "csv file in which to write the computed \
  emission probability statistics.";
  //  std::string normTransProbsMeanVarFileNameDesc = "csv file in which to write the \
  computed transition probability scale and angle statics";
  //  std::string transProbsFileNameDesc = "csv file in which to write the transition \
  probabilities. The output will be an 11x11 matrix. The rows indicate the 'from' \
  generation and the columns represent the 'to' generation. The probabilities are \
  computed simply by counting the number of times a given transition occurs and \
  then normalizing.";

  //
  // Parse the input arguments
  //
  // try
  // {
  // TCLAP::CmdLine cl(programDesc, ' ', "$Revision: 383 $");

  //TCLAP::MultiArg<std::string>  inParticlesFileNamesArg("i", "input", inParticlesFileNamesDesc, true, "string", cl);
  //TCLAP::ValueArg<std::string>  refParticlesFileNameArg ("r", "ref", refParticlesFileNameDesc, false, refParticlesFileName, "string", cl);
  //TCLAP::ValueArg<std::string>  emissionProbsFileNameArg ("e", "", emissionProbsFileNameDesc, false, emissionProbsFileName, "string", cl);
  //  TCLAP::ValueArg<std::string>  normTransProbsMeanVarFileNameArg("", "ntp", normTransProbsMeanVarFileNameDesc, false, normTransProbsMeanVarFileName, "string", cl);
  
  //     TCLAP::ValueArg<std::string>  transProbsFileNameArg("", "tp", transProbsFileNameDesc, false, normTransProbsMeanVarFileName, "string", cl);
        
  //TCLAP::ValueArg<double> particleDistanceThresholdArg ("d", "distThresh", particleDistanceThresholdDesc, false, particleDistanceThreshold, "double", cl);
  //TCLAP::ValueArg<double> emissionProbDistThreshArg ("", "ed", emissionProbDistThreshDesc, false, emissionProbDistThresh, "double", cl);

  // cl.parse(argc, argv);

  //particleDistanceThreshold     = particleDistanceThresholdArg.getValue();
  //refParticlesFileName          = refParticlesFileNameArg.getValue();
  //emissionProbDistThresh        = emissionProbDistThreshArg.getValue();
  //emissionProbsFileName         = emissionProbsFileNameArg.getValue();
  //normTransProbsMeanVarFileName = normTransProbsMeanVarFileNameArg.getValue();
  //transProbsFileName            = transProbsFileNameArg.getValue();
  */
  PARSE_ARGS;
    
//   for (unsigned int i=0; i<inParticlesFileNamesArg.size(); i++)
//     {
//       inParticlesFileNames.push_back(inParticlesFileNamesArg[i]);
//     }
//   //}

// /*catch (TCLAP::ArgException excp)
//   {
//   std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
//   return cip::ARGUMENTPARSINGERROR;
//   }
// */

// std::vector<TRANSITION> transitionVec;

// // Read the reference particle dataset if specified
// vtkSmartPointer<vtkPolyDataReader> refReader = vtkSmartPointer<vtkPolyDataReader>::New();
// if (refParticlesFileName.compare("NA") != 0)
//   {
//     std::cout << "Reading reference airway particles..." << std::endl;  
//     refReader->SetFileName(refParticlesFileName.c_str());
//     refReader->Update();
//   }

// // This container will collect all the data needed to compute statistics for 
// // the emission probabilities
// std::map<unsigned char, std::vector<EMISSION> > emissionProbData;

// // Read the particles to which generation labels are to be assigned
// for (unsigned int i=0; i<inParticlesFileNames.size(); i++)
//   {
//     std::cout << "Reading airway particles..." << std::endl;
//     vtkSmartPointer<vtkPolyDataReader> particlesReader = vtkSmartPointer<vtkPolyDataReader>::New();
//     particlesReader->SetFileName(inParticlesFileNames[i].c_str());
//     particlesReader->Update();

//     // If a reference particle dataset has been specified, update the data
//     // needed to compute the conditional probabilities
//     if (refParticlesFileName.compare("NA") != 0)
//       {
// 	std::cout << "Updating emission probability data..." << std::endl;
// 	UpdateEmissionProbabilityData(refReader->GetOutput(), particlesReader->GetOutput(), 
// 				      emissionProbDistThresh, &emissionProbData);
//       }

    // // Construct the minimum spanning tree 
    // std::cout << "Getting minimum spanning tree..." << std::endl;
    // vtkSmartPointer<vtkMutableUndirectedGraph> minimumSpanningTree = 
    // 	GetMinimumSpanningTree(particlesReader->GetOutput(), particleDistanceThreshold, edgeWeightAngleSigma);

    // // Create a mapping between min spanning tree node IDs and the 
    // // particle IDs
    // std::map<unsigned int, unsigned int> nodeIDToParticleIDMap;
    // std::map<unsigned int, unsigned int> particleIDToNodeIDMap;
    // for (unsigned int j=0; j<minimumSpanningTree->GetPoints()->GetNumberOfPoints(); j++)
    // 	{
    // 	  for (unsigned int k=0; k<particlesReader->GetOutput()->GetNumberOfPoints(); k++)
    // 	    {
    // 	      if (particlesReader->GetOutput()->GetPoint(k)[0] == minimumSpanningTree->GetPoint(j)[0] &&
    // 		   particlesReader->GetOutput()->GetPoint(k)[1] == minimumSpanningTree->GetPoint(j)[1] &&
    // 		   particlesReader->GetOutput()->GetPoint(k)[2] == minimumSpanningTree->GetPoint(j)[2])
    // 		{
    // 		  nodeIDToParticleIDMap[j] = k;
    // 		  particleIDToNodeIDMap[k] = j;

    // 		  break;
    // 		}
    // 	    }
    // 	}

    // // Iteratve over the edges in the minimum spanning tree and 
    // // gather the scale differences and angles between states
    // vtkSmartPointer<vtkEdgeListIterator> edgeIt = vtkSmartPointer<vtkEdgeListIterator>::New();
    // minimumSpanningTree->GetEdges(edgeIt);

    // while (edgeIt->HasNext())
    // 	{
    // 	  vtkEdgeType edge = edgeIt->Next();
	  
    // 	  unsigned int p1ID = nodeIDToParticleIDMap[edge.Source];
    // 	  unsigned int p2ID = nodeIDToParticleIDMap[edge.Target];

    // 	  float state1 = particlesReader->GetOutput()->GetPointData()->GetArray("ChestType")->GetTuple(p1ID)[0];
    // 	  float state2 = particlesReader->GetOutput()->GetPointData()->GetArray("ChestType")->GetTuple(p2ID)[0];

    // 	  float scale1 = particlesReader->GetOutput()->GetPointData()->GetArray("scale")->GetTuple(p1ID)[0];
    // 	  float scale2 = particlesReader->GetOutput()->GetPointData()->GetArray("scale")->GetTuple(p2ID)[0];

    // 	  double direction1[3];
    // 	    direction1[0] = particlesReader->GetOutput()->GetPointData()->GetArray("hevec2")->GetTuple(p1ID)[0];
    // 	    direction1[1] = particlesReader->GetOutput()->GetPointData()->GetArray("hevec2")->GetTuple(p1ID)[1];
    // 	    direction1[2] = particlesReader->GetOutput()->GetPointData()->GetArray("hevec2")->GetTuple(p1ID)[2];

    // 	  double direction2[3];
    // 	    direction2[0] = particlesReader->GetOutput()->GetPointData()->GetArray("hevec2")->GetTuple(p2ID)[0];
    // 	    direction2[1] = particlesReader->GetOutput()->GetPointData()->GetArray("hevec2")->GetTuple(p2ID)[1];
    // 	    direction2[2] = particlesReader->GetOutput()->GetPointData()->GetArray("hevec2")->GetTuple(p2ID)[2];

    // 	  double scaleDifference;
    // 	  unsigned char fromState, toState;
    // 	  if ( IsTransitionPermitted( (unsigned char)(state1), (unsigned char)(state2) ) )
    // 	    {
    // 	      scaleDifference = scale2 - scale1;
    // 	      fromState = static_cast<unsigned char>(state1);
    // 	      toState   = static_cast<unsigned char>(state2);
    // 	    }
    // 	  else
    // 	    {
    // 	      scaleDifference = scale1 - scale2;
    // 	      fromState = static_cast<unsigned char>(state2);
    // 	      toState   = static_cast<unsigned char>(state1);
    // 	    }

    // 	  double angle = cip::GetAngleBetweenVectors(direction1, direction2, true);

    // 	  // Now add this data to our current set of transition
    // 	  // data containers
    // 	  bool found = false;
    // 	  for (unsigned int e=0; e<transitionVec.size(); e++)
    // 	    {
    // 	      if (transitionVec[e].from == fromState && transitionVec[e].to == toState)
    // 		{
    // 		  found = true;
    // 		  transitionVec[e].angles.push_back(angle);
    // 		  transitionVec[e].scaleDifferences.push_back(scaleDifference);
    // 		  break;
    // 		}
    // 	    }
    // 	  if (!found)
    // 	    {
    // 	      TRANSITION trans;
    // 	      trans.from = fromState;
    // 	      trans.to   = toState;
    // 	      trans.angles.push_back(angle);
    // 	      trans.scaleDifferences.push_back(scaleDifference);
    // 	      transitionVec.push_back(trans);
    // 	    }
    // 	}
//   }      

// std::cout << "Writing emission data..." << std::endl;
// WriteEmissionDataISBI( emissionProbData );

// std::cout << "Writing transition data..." << std::endl;
// WriteTransitionDataISBI( transitionVec );

// Now compute and print the transition statistics
// cip::ChestConventions conventions;
// for (unsigned int i=0; i<transitionVec.size(); i++)
//   {
//     std::cout << "---------------------------------" << std::endl;
//     unsigned int N = transitionVec[i].angles.size();
      
//     double scaleAccum = 0.0;
//     double angleAccum = 0.0;
//     for (unsigned int j=0; j<N; j++)
// 	{
// 	  scaleAccum += transitionVec[i].scaleDifferences[j];
// 	  angleAccum += transitionVec[i].angles[j];
// 	}

//     double scaleDiffMean = scaleAccum/static_cast<double>(N);
//     double angleMean     = angleAccum/static_cast<double>(N);

//     scaleAccum = 0.0;
//     angleAccum = 0.0;
//     for (unsigned int j=0; j<N; j++)
// 	{
// 	  scaleAccum += pow(transitionVec[i].scaleDifferences[j] - scaleDiffMean, 2.0);
// 	  angleAccum += pow(transitionVec[i].angles[j] - angleMean, 2.0);
// 	}
      
//     double scaleDiffSTD = sqrt(scaleAccum/static_cast<double>(N));
//     double angleSTD     = sqrt(angleAccum/static_cast<double>(N));

//     std::cout << "From:\t" << conventions.GetChestTypeName(static_cast<unsigned char>(transitionVec[i].from)) << "\t";
//     std::cout << "To:\t"   << conventions.GetChestTypeName(static_cast<unsigned char>(transitionVec[i].to))   << std::endl;
//     std::cout << "Scale Difference Mean:\t" << scaleDiffMean << " +/- " << scaleDiffSTD << std::endl;
//     std::cout << "Angles Mean:\t" << angleMean << " +/- " << angleSTD << std::endl;
//     std::cout << "Number of Occurrences:\t" << N << std::endl;
//   }

// // Print emission probability stats if needed
// if (refParticlesFileName.compare("NA") != 0)
//   {
//     PrintEmissionProbabilityData(emissionProbData);
//   }

// // Write emission probability stats to file if needed
// if (emissionProbsFileName.compare("NA") != 0)
//   {
//     std::cout << "Writing emission probability stats..." << std::endl;
//     WriteEmissionProbabilityData(emissionProbData, emissionProbsFileName);
//   }

// // Write transition probability scale and angle stats 
// // to file if needed
// if (normTransProbsMeanVarFileName.compare("NA") != 0)
//   {
//     std::cout << "Writing transition probability scale and angle stats..." << std::endl;
//     WriteTransitionProbabilityScaleAngleData(transitionVec, normTransProbsMeanVarFileName);
//   }

// // Write the transition probabilities to file if needed
// if (transProbsFileName.compare("NA") != 0)
//   {
//     std::cout << "Writing transition probabilities..." << std::endl;
//     WriteTransitionProbabilities(transitionVec, transProbsFileName);
//   }

std::cout << "DONE." << std::endl;

return cip::EXITSUCCESS;
}



// AIRWAYGENERATION5
// AIRWAYGENERATION4
// AIRWAYGENERATION3
// UPPERLOBEBRONCHUS
// INTERMEDIATEBRONCHUS
// MIDDLELOBEBRONCHUS
// LOWERLOBEBRONCHUS
// MAINBRONCHUS
// SUPERIORDIVISIONBRONCHUS
// LINGULARBRONCHUS
// TRACHEA

