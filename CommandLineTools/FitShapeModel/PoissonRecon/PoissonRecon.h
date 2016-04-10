#ifndef _POISSONRECON_H_
#define _POISSONRECON_H_

#include "Ply.h"

class PoissonRecon
{
public:
  typedef CoredVectorMeshData< PlyVertex< float > > MeshData;
  class VolumeData { // output volume
  public:
    VolumeData() : data( 0 ) {}
    ~VolumeData() { if (data) delete data; }
    unsigned int res;
    float* data;
    float scale;
    double center[3];
  };
  PoissonRecon() {};
  ~PoissonRecon() {};
  int run( int argc, char* argv[] ); // file-based input / output
  MeshData& createIsoSurfaceAndMesh( const std::vector< float >& points, 
                                     VolumeData& volume ); // in-memory input / output
private:
  MeshData _mesh;
};

#endif