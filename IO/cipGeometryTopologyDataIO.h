/**
 *  \class cipGeometryTopologyDataIO
 *  \ingroup io
 *  \brief This class handles the input and output of geometry/topology
 *  data structures. Entities in these structures include, e.g., points
 *  and bounding boxes.
 */

#ifndef __cipGeometryTopologyDataIO_h
#define __cipGeometryTopologyDataIO_h

#include "cipGeometryTopologyData.h"
#include <string>
#include <libxml/xmlwriter.h>
#include <libxml/parser.h>

namespace cip
{
  class GeometryTopologyDataIO
  {
  public:
    ~GeometryTopologyDataIO();
    GeometryTopologyDataIO();
    
    void SetFileName( std::string );
    
    void Read();
    void Write() const;
    
    /** The returned shape model intentionally left non-const because we
     *  may with to modify the mode weights of the shape model. Doing
     *  this will allow an easy read-modify-write flow */
    cip::GeometryTopologyData* GetOutput();
    
    /** Set the shape model to write to file */
    void SetInput( cip::GeometryTopologyData& );
    
  private:
    void ParsePoint( xmlNodePtr );
    void ParseBoundingBox( xmlNodePtr );

    cip::GeometryTopologyData* m_GeometryTopologyData;
    
    std::string m_FileName;
  };
  
} // end namespace cip

#endif // __cipGeometryTopologyDataIO_h
