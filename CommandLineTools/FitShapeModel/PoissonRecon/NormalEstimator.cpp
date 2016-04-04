#include "NormalEstimator.h"
#include <vcg/complex/complex.h>

#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/flag.h>
#include <vcg/complex/algorithms/create/ball_pivoting.h>
#include <vcg/complex/algorithms/pointcloud_normal.h>

// input output
#include <wrap/io_trimesh/import_asc.h>
#include <wrap/io_trimesh/export_obj.h>
#include <vector>
#include <float.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace vcg;
using namespace std;

class MyFace;
class MyVertex;

struct MyUsedTypes : public UsedTypes< Use<MyVertex>::AsVertexType, Use<MyFace>::AsFaceType > {};

class MyVertex  : public Vertex< MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::BitFlags, vertex::Mark >{};
class MyFace    : public Face  < MyUsedTypes, face::VertexRef, face::Normal3f, face::BitFlags > {};
class MyMesh    : public vcg::tri::TriMesh< vector<MyVertex>, vector<MyFace> > {};

NormalEstimator::NormalEstimator( const std::vector< RealType >& points )
: _points(points)
{
}

void NormalEstimator::run( std::vector< RealType >& normals )
{
  normals.clear();
  
  MyMesh m;

  double min_org[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  double max_org[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

  unsigned int numPoints = _points.size() / 3;
  for (unsigned int i = 0; i < numPoints; i++)
  {
    MyMesh::VertexIterator vi = Allocator<MyMesh>::AddVertices( m, 1 );
    for (int k = 0; k < 3; k++)
    {
      vi->P()[k] = _points[3*i+k];
      min_org[k] = MIN( min_org[k], vi->P()[k] );
      max_org[k] = MAX( max_org[k], vi->P()[k] );
    }
  }
  
  // estimate normal using k nearest points (vcglib)
  vcg::tri::PointCloudNormal<MyMesh>::Param param;
  vcg::tri::PointCloudNormal<MyMesh>::Compute( m, param );

  // check bounding box to determine the right normal orientation
  double min_adj[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  double max_adj[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  
  MyMesh::VertexIterator vi = m.vert.begin();
  while (vi != m.vert.end())
  {
    for (int k = 0; k < 3; k++)
    {
      min_adj[k] = MIN( min_adj[k], vi->P()[k] + vi->N()[k] );
      max_adj[k] = MAX( max_adj[k], vi->P()[k] + vi->N()[k] );
    }
    vi++;
  }
  
  double width_org[3];
  double width_adj[3];
  
  for (int k = 0; k < 3; k++)
  {
    width_org[k] = max_org[k] - min_org[k];
    width_adj[k] = max_adj[k] - min_adj[k];
    //printf("%g %g\n", width_org[k], width_adj[k]);
  }
  
  bool flip_normal = (width_adj[0] < width_org[0] &&
                      width_adj[1] < width_org[1] &&
                      width_adj[2] < width_org[2]);
  
  double f = 1.0;
  if (flip_normal)
  {
    printf("Flipping normals...\n");
    f = -1.0;
  }
  
  vi = m.vert.begin();
  while (vi != m.vert.end())
  {
    for (int k = 0; k < 3; k++)
    {
      normals.push_back( f * vi->N()[k] );
    }
    vi++;
  }
}
