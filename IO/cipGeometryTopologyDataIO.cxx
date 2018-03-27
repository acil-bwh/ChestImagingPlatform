#include "cipGeometryTopologyDataIO.h"
#include "cipExceptionObject.h"
#include "cipChestConventions.h"

#include <string>
#include <sstream>
#include <cstdlib>

using namespace cip;

GeometryTopologyDataIO::GeometryTopologyDataIO()
{
	this->m_FileName = "NA";
	this->m_GeometryTopologyData = new cip::GeometryTopologyData();
}

GeometryTopologyDataIO::~GeometryTopologyDataIO()
{
	delete this->m_GeometryTopologyData;
}

void GeometryTopologyDataIO::SetFileName( std::string fileName )
{
	this->m_FileName = fileName;
}

void GeometryTopologyDataIO::SetInput( cip::GeometryTopologyData& geometryTopologyData )
{
	*this->m_GeometryTopologyData = geometryTopologyData;
}

void GeometryTopologyDataIO::Write() const
{
	xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
	xmlNodePtr root_node = xmlNewNode(NULL, BAD_CAST "GeometryTopologyData");
	xmlDocSetRootElement(doc, root_node);

  // GENERAL INFO
  xmlNewChild( root_node, NULL, BAD_CAST "CoordinateSystem",
          BAD_CAST this->m_GeometryTopologyData->CoordinateSystem.c_str() );

  if (this->m_GeometryTopologyData->LPS_to_IJK_TransformationMatrix.size() > 0) {
    xmlNodePtr node = xmlNewChild(root_node, NULL, BAD_CAST "LPStoIJKTransformationMatrix", NULL );
    for ( unsigned int j=0; j<this->m_GeometryTopologyData->LPS_to_IJK_TransformationMatrix.size(); j++ )
    {
      std::stringstream coordinateStream;
      coordinateStream << this->m_GeometryTopologyData->LPS_to_IJK_TransformationMatrix[j];
      std::string coordinateString = coordinateStream.str();
      xmlNewChild( node, NULL, BAD_CAST "value", BAD_CAST coordinateString.c_str() );
    }
  }

	if (this->m_GeometryTopologyData->m_Spacing.size() > 0) {
		xmlNodePtr node = xmlNewChild(root_node, NULL, BAD_CAST "Spacing", NULL );
		for ( unsigned int j=0; j<this->m_GeometryTopologyData->m_Spacing.size(); j++ )
		{
			std::stringstream coordinateStream;
			coordinateStream << this->m_GeometryTopologyData->m_Spacing[j];
			std::string coordinateString = coordinateStream.str();
			xmlNewChild( node, NULL, BAD_CAST "value", BAD_CAST coordinateString.c_str() );
		}
	}
	if (this->m_GeometryTopologyData->m_Origin.size() > 0) {
		xmlNodePtr node = xmlNewChild(root_node, NULL, BAD_CAST "Origin", NULL );
		for ( unsigned int j=0; j<this->m_GeometryTopologyData->m_Origin.size(); j++ )
		{
			std::stringstream coordinateStream;
			coordinateStream << this->m_GeometryTopologyData->m_Origin[j];
			std::string coordinateString = coordinateStream.str();
			xmlNewChild( node, NULL, BAD_CAST "value", BAD_CAST coordinateString.c_str() );
		}
	}
	if (this->m_GeometryTopologyData->m_Dimensions.size() > 0) {
		xmlNodePtr node = xmlNewChild(root_node, NULL, BAD_CAST "Dimensions", NULL );
		for ( unsigned int j=0; j<this->m_GeometryTopologyData->m_Dimensions.size(); j++ )
		{
			std::stringstream coordinateStream;
			coordinateStream << this->m_GeometryTopologyData->m_Dimensions[j];
			std::string coordinateString = coordinateStream.str();
			xmlNewChild( node, NULL, BAD_CAST "value", BAD_CAST coordinateString.c_str() );
		}
	}

  // POINTS
	for ( unsigned int i=0; i<this->m_GeometryTopologyData->GetNumberOfPoints(); i++ )
	{
		xmlNodePtr point_node = xmlNewChild(root_node, NULL, BAD_CAST "Point", NULL );

    std::stringstream idStream;
    idStream << int(this->m_GeometryTopologyData->GetPointId(i));
    std::string idString = idStream.str();
    xmlNewChild( point_node, NULL, BAD_CAST "Id", BAD_CAST idString.c_str() );

		std::stringstream chestRegionStream;
		chestRegionStream << int(this->m_GeometryTopologyData->GetPointChestRegion(i));
		std::string chestRegionString = chestRegionStream.str();
		xmlNewChild( point_node, NULL, BAD_CAST "ChestRegion", BAD_CAST chestRegionString.c_str() );

		std::stringstream chestTypeStream;
		chestTypeStream << int(this->m_GeometryTopologyData->GetPointChestType(i));
		std::string chestTypeString = chestTypeStream.str();
		xmlNewChild( point_node, NULL, BAD_CAST "ChestType", BAD_CAST chestTypeString.c_str() );

		std::stringstream imageFeatureStream;
		imageFeatureStream << int(this->m_GeometryTopologyData->GetPointImageFeature(i));
		std::string imageFeatureString = imageFeatureStream.str();
		xmlNewChild( point_node, NULL, BAD_CAST "ImageFeature", BAD_CAST imageFeatureString.c_str() );

    if (this->m_GeometryTopologyData->GetPointDescription(i).size() > 0) {
      xmlNewChild(point_node, NULL, BAD_CAST "Description",
                  BAD_CAST this->m_GeometryTopologyData->GetPointDescription(i).c_str());
    }

    if (this->m_GeometryTopologyData->GetPoint(i).timestamp.size() > 0) {
      xmlNewChild(point_node, NULL, BAD_CAST "Timestamp",
                  BAD_CAST this->m_GeometryTopologyData->GetPoint(i).timestamp.c_str());
    }
    if (this->m_GeometryTopologyData->GetPoint(i).userName.size() > 0) {
      xmlNewChild(point_node, NULL, BAD_CAST "UserName",
                  BAD_CAST this->m_GeometryTopologyData->GetPoint(i).userName.c_str());
    }

    if (this->m_GeometryTopologyData->GetPoint(i).machineName.size() > 0) {
      xmlNewChild(point_node, NULL, BAD_CAST "MachineName",
                  BAD_CAST this->m_GeometryTopologyData->GetPoint(i).machineName.c_str());
    }

    xmlNodePtr coordinate_node = xmlNewChild(point_node, NULL, BAD_CAST "Coordinate", NULL );

    cip::GeometryTopologyData::CoordinateType coordinate =
            this->m_GeometryTopologyData->GetPointCoordinate(i);

    for ( unsigned int j=0; j<coordinate.size(); j++ )
    {
      std::stringstream coordinateStream;
      coordinateStream << coordinate[j];
      std::string coordinateString = coordinateStream.str();
      xmlNewChild( coordinate_node, NULL, BAD_CAST "value", BAD_CAST coordinateString.c_str() );
    }
	}

  // BOUNDING BOXES
	for ( unsigned int i=0; i<this->m_GeometryTopologyData->GetNumberOfBoundingBoxes(); i++ )
	{
		xmlNodePtr bb_node = xmlNewChild(root_node, NULL, BAD_CAST "BoundingBox", NULL );

    std::stringstream idStream;
    idStream << int(this->m_GeometryTopologyData->GetBoundingBoxId(i));
    std::string idString = idStream.str();
    xmlNewChild( bb_node, NULL, BAD_CAST "Id", BAD_CAST idString.c_str() );

		std::stringstream chestRegionStream;
		chestRegionStream << int(this->m_GeometryTopologyData->GetBoundingBoxChestRegion(i));
		std::string chestRegionString = chestRegionStream.str();
		xmlNewChild( bb_node, NULL, BAD_CAST "ChestRegion", BAD_CAST chestRegionString.c_str() );

		std::stringstream chestTypeStream;
		chestTypeStream << int(this->m_GeometryTopologyData->GetBoundingBoxChestType(i));
		std::string chestTypeString = chestTypeStream.str();
		xmlNewChild( bb_node, NULL, BAD_CAST "ChestType", BAD_CAST chestTypeString.c_str() );

		std::stringstream imageFeatureStream;
		imageFeatureStream << int(this->m_GeometryTopologyData->GetBoundingBoxImageFeature(i));
		std::string imageFeatureString = imageFeatureStream.str();
		xmlNewChild( bb_node, NULL, BAD_CAST "ImageFeature", BAD_CAST imageFeatureString.c_str() );

    if (this->m_GeometryTopologyData->GetBoundingBoxDescription(i).size() > 0) {
      xmlNewChild(bb_node, NULL, BAD_CAST "Description",
              BAD_CAST this->m_GeometryTopologyData->GetBoundingBoxDescription(i).c_str());
    }

    if (this->m_GeometryTopologyData->GetBoundingBox(i).timestamp.size() > 0) {
      xmlNewChild(bb_node, NULL, BAD_CAST "Timestamp",
              BAD_CAST this->m_GeometryTopologyData->GetBoundingBox(i).timestamp.c_str());
    }
    if (this->m_GeometryTopologyData->GetBoundingBox(i).userName.size() > 0) {
      xmlNewChild(bb_node, NULL, BAD_CAST "UserName",
              BAD_CAST this->m_GeometryTopologyData->GetBoundingBox(i).userName.c_str());
    }

    if (this->m_GeometryTopologyData->GetBoundingBox(i).machineName.size() > 0) {
      xmlNewChild(bb_node, NULL, BAD_CAST "MachineName",
              BAD_CAST this->m_GeometryTopologyData->GetBoundingBox(i).machineName.c_str());
    }

		xmlNodePtr start_node = xmlNewChild(bb_node, NULL, BAD_CAST "Start", NULL );

		cip::GeometryTopologyData::StartType start =
						this->m_GeometryTopologyData->GetBoundingBoxStart(i);

		for ( unsigned int j=0; j<start.size(); j++ )
		{
			std::stringstream startStream;
			startStream << start[j];
			std::string startString = startStream.str();

			xmlNewChild( start_node, NULL, BAD_CAST "value", BAD_CAST startString.c_str() );
		}

		xmlNodePtr size_node = xmlNewChild(bb_node, NULL, BAD_CAST "Size", NULL );

		cip::GeometryTopologyData::SizeType size =
						this->m_GeometryTopologyData->GetBoundingBoxSize(i);

		for ( unsigned int j=0; j<size.size(); j++ )
		{
			std::stringstream sizeStream;
			sizeStream << size[j];
			std::string sizeString = sizeStream.str();

			xmlNewChild( size_node, NULL, BAD_CAST "value", BAD_CAST sizeString.c_str() );
		}
	}

	xmlSaveFormatFileEnc( this->m_FileName.c_str(), doc, "UTF-8", 1 );
	xmlFreeDoc(doc);
}

void GeometryTopologyDataIO::Read()
{
	xmlDocPtr  doc;
	xmlNodePtr cur;
	doc = xmlParseFile( this->m_FileName.c_str() );

	if ( doc == NULL )
	{
		throw cip::ExceptionObject( __FILE__, __LINE__,
																"cip::GeometryTopologyDataIO::Read",
																"Could not parse file" );
	}
	else
	{
		cur = xmlDocGetRootElement( doc );

		if (cur == NULL)
		{
			throw cip::ExceptionObject( __FILE__, __LINE__,
																	"cip::GeometryTopologyDataIO::Read",
																	"Empty document" );
		}
		else
		{
			if (xmlStrcmp(cur->name, (const xmlChar *) "GeometryTopologyData"))
			{
				throw cip::ExceptionObject( __FILE__, __LINE__,
																		"cip::GeometryTopologyDataIO::Read",
																		"Wrong document type. Root node is not GeometryTopologyData" );
			}
			else
			{
				cur = cur->xmlChildrenNode;
				while ( cur != NULL )
				{
          if( !xmlStrcmp(cur->name, (const xmlChar *)"CoordinateSystem") )
          {
            this->m_GeometryTopologyData->CoordinateSystem = std::string((const char*)xmlNodeGetContent(cur));
          }
          else if ( !xmlStrcmp(cur->name, (const xmlChar *)"LPStoIJKTransformationMatrix") )
          {
            xmlNodePtr co = cur->xmlChildrenNode;
            while ( co != NULL )
            {
              if ( !xmlStrcmp(co->name, (const xmlChar *)"value") )
              {
                this->m_GeometryTopologyData->LPS_to_IJK_TransformationMatrix.push_back(std::atof((const char*)xmlNodeGetContent(co)));
              }
              co = co->next;
            }
          }
					else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Spacing") )
					{
						xmlNodePtr co = cur->xmlChildrenNode;
						while ( co != NULL )
						{
							if ( !xmlStrcmp(co->name, (const xmlChar *)"value") )
							{
								this->m_GeometryTopologyData->m_Spacing.push_back(std::atof((const char*)xmlNodeGetContent(co)));
							}
							co = co->next;
						}
					}
					else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Origin") )
					{
						xmlNodePtr co = cur->xmlChildrenNode;
						while ( co != NULL )
						{
							if ( !xmlStrcmp(co->name, (const xmlChar *)"value") )
							{
								this->m_GeometryTopologyData->m_Origin.push_back(std::atof((const char*)xmlNodeGetContent(co)));
							}
							co = co->next;
						}
					}
					else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Dimensions") )
					{
						xmlNodePtr co = cur->xmlChildrenNode;
						while ( co != NULL )
						{
							if ( !xmlStrcmp(co->name, (const xmlChar *)"value") )
							{
								this->m_GeometryTopologyData->m_Dimensions.push_back(std::atof((const char*)xmlNodeGetContent(co)));
							}
							co = co->next;
						}
					}
					else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Point") )
					{
						this->ParsePoint( cur );
					}
					else if ( !xmlStrcmp(cur->name, (const xmlChar *)"BoundingBox") )
					{
						this->ParseBoundingBox( cur );
					}

					cur = cur->next;
				}
			}
		}
	}
	xmlFreeDoc( doc );
  this->m_GeometryTopologyData->UpdateSeed();
}

void GeometryTopologyDataIO::ParsePoint( xmlNodePtr ptNode )
{
  unsigned int id;
	unsigned char cipRegion;
	unsigned char cipType;
	unsigned char cipImageFeature;
	cip::GeometryTopologyData::CoordinateType coordinate;
	std::string description;
  std::string userName;
  std::string machineName;
  std::string timestamp;

	xmlNodePtr cur = ptNode->xmlChildrenNode;
	while ( cur != NULL )
	{
    if ( !xmlStrcmp(cur->name, (const xmlChar *)"Id") )
    {
      id = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
    }
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"ChestRegion") )
		{
			cipRegion = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"ChestType") )
		{
			cipType = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"ImageFeature") )
		{
			cipImageFeature = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Coordinate") )
		{
			xmlNodePtr co = cur->xmlChildrenNode;
			while ( co != NULL )
			{
				if ( !xmlStrcmp(co->name, (const xmlChar *)"value") )
				{
					coordinate.push_back(std::atof((const char*)xmlNodeGetContent(co)));
				}
				co = co->next;
			}
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Description") )
		{
			description = std::string((const char*)xmlNodeGetContent(cur));
		}
    else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Timestamp") )
    {
      timestamp = std::string((const char*)xmlNodeGetContent(cur));
    }
    else if ( !xmlStrcmp(cur->name, (const xmlChar *)"UserName") )
    {
      userName = std::string((const char*)xmlNodeGetContent(cur));
    }
    else if ( !xmlStrcmp(cur->name, (const xmlChar *)"MachineName") )
    {
      machineName = std::string((const char*)xmlNodeGetContent(cur));
    }

		cur = cur->next;
	}

	cip::GeometryTopologyData::POINT* p = this->m_GeometryTopologyData->InsertPoint( id, cipRegion, cipType, cipImageFeature,
                                                                                   coordinate, description,
                                                                                   timestamp, userName, machineName);
}

void GeometryTopologyDataIO::ParseBoundingBox( xmlNodePtr bbNode )
{
	unsigned int id;
	unsigned char cipRegion;
	unsigned char cipType;
	unsigned char cipImageFeature;
	cip::GeometryTopologyData::StartType start;
	cip::GeometryTopologyData::StartType size;
	std::string description;
	std::string userName;
	std::string machineName;
	std::string timestamp;

	xmlNodePtr cur = bbNode->xmlChildrenNode;
	while ( cur != NULL )
	{
		if ( !xmlStrcmp(cur->name, (const xmlChar *)"Id") )
		{
			id = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"ChestRegion") )
		{
			cipRegion = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"ChestType") )
		{
			cipType = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"ImageFeature") )
		{
			cipImageFeature = (unsigned char)std::atoi((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Start") )
		{
			xmlNodePtr st = cur->xmlChildrenNode;
			while ( st != NULL )
			{
				if ( !xmlStrcmp(st->name, (const xmlChar *)"value") )
				{
					start.push_back(std::atof((const char*)xmlNodeGetContent(st)));
				}
				st = st->next;
			}
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Size") )
		{
			xmlNodePtr sz = cur->xmlChildrenNode;
			while ( sz != NULL )
			{
				if ( !xmlStrcmp(sz->name, (const xmlChar *)"value") )
				{
					size.push_back(std::atof((const char*)xmlNodeGetContent(sz)));
				}
				sz = sz->next;
			}
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Description") )
		{
			description = std::string((const char*)xmlNodeGetContent(cur));
		}
    else if ( !xmlStrcmp(cur->name, (const xmlChar *)"Timestamp") )
    {
      timestamp = std::string((const char*)xmlNodeGetContent(cur));
    }
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"UserName") )
		{
			userName = std::string((const char*)xmlNodeGetContent(cur));
		}
		else if ( !xmlStrcmp(cur->name, (const xmlChar *)"MachineName") )
		{
			machineName = std::string((const char*)xmlNodeGetContent(cur));
		}
		cur = cur->next;
	}

	cip::GeometryTopologyData::BOUNDINGBOX* bb = this->m_GeometryTopologyData->InsertBoundingBox( id,
                                                                                               cipRegion, cipType, cipImageFeature,
                                                                                               start, size, description,
                                                                                               timestamp, userName, machineName);
}

cip::GeometryTopologyData* GeometryTopologyDataIO::GetOutput()
{
	return this->m_GeometryTopologyData;
}
