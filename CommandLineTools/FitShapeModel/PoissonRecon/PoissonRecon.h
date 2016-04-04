#ifndef _POISSONRECON_H_
#define _POISSONRECON_H_

#include "Ply.h"

class PoissonRecon
{
public:
  typedef CoredVectorMeshData< PlyVertex< float > > MeshData;
  PoissonRecon() {};
  ~PoissonRecon() {};
  int run( int argc, char* argv[] ); // file-based input / output
  MeshData& createMesh( const std::vector< float >& points ); // in-memory input / output
private:
  MeshData _mesh;
};

#endif