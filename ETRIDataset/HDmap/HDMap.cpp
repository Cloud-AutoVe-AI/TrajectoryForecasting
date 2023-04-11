#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HDMap.h"


CHDMap::CHDMap()
{
	
}


CHDMap::~CHDMap()
{

}



bool CHDMap::LoadHDMap()
{
	const int bufSize = 1024 * 128;

	FILE* fp;
	char dbFile[256], buf[bufSize];
	char delim[] = { " " };


	int count = 0;

	/////////////////////////////////////////////////////////////////////
	// load poi
	strcpy(dbFile, "../../txt/etridb_plus_LAYER_POI.txt");
	

	if ((fp = fopen(dbFile, "r")) == NULL)
	{
		printf(">>>>> ERROR: POI Layer Open\n");
	}
	else
	{
		while (fgets(buf, bufSize, fp))
		{
			// ID field
			char *key = strtok(buf, delim);
			int ID = atoi(key);

			// linkID field
			key = strtok(NULL, delim);
			int linkID = atoi(key);

			// Name field
			key = strtok(NULL, delim);

			// x field
			key = strtok(NULL, delim);
			double x = atof(key);

			// y field
			key = strtok(NULL, delim);
			double y = atof(key);


			count++;
		}

		fclose(fp);

		printf(">>> HDMap(POI) Load Success (%d)\n", count);
	}





	/////////////////////////////////////////////////////////////////////
	// load roadmark
	count = 0;

	strcpy(dbFile, "../../txt/etridb_plus_LAYER_ROADMARK.txt");
	

	if ((fp = fopen(dbFile, "r")) == NULL)
	{
		printf(">>>>> ERROR: ROADMARK Layer Open\n");
	}
	else
	{
		while (fgets(buf, bufSize, fp))
		{
			// ID field
			char *key = strtok(buf, delim);
			int ID = atoi(key);

			// Type
			key = strtok(NULL, delim);
			ROADMARK_TYPE type = (ROADMARK_TYPE)atoi(key);

			// SubType
			key = strtok(NULL, delim);
			ROADMARK_SUBTYPE subType = (ROADMARK_SUBTYPE)atoi(key);


			// # of stop
			key = strtok(NULL, delim);
			int numStop = atoi(key);
			for (int i = 0; i < numStop; i++)
			{
				key = strtok(NULL, delim);

				int stopLineID = atoi(key);
			}

			// polygon geometry
			key = strtok(NULL, delim);
			int numPts = atoi(key);
			
			for (int i = 0; i < numPts; i++)
			{
				key = strtok(NULL, delim);
				double x = atof(key);

				key = strtok(NULL, delim);
				double y = atof(key);

				
			}
			
			count++;
		}

		fclose(fp);

		printf(">>> HDMap(ROADMARK) Load Success (%d)\n", count);
	}



	/////////////////////////////////////////////////////////////////////
	// load road_light
	count = 0;
	strcpy(dbFile, "../../txt/etridb_plus_LAYER_ROADLIGHT.txt");

	if ((fp = fopen(dbFile, "r")) == NULL)
	{
		printf(">>>>> ERROR: _LAYER_ROADLIGHT Layer Open\n");
	}
	else
	{
		while (fgets(buf, bufSize, fp))
		{
			// ID field
			char *key = strtok(buf, delim);
			int ID = atoi(key);

			// LaneID field
			key = strtok(NULL, delim);
			int laneID = atoi(key);

			// Type field
			key = strtok(NULL, delim);
			ROADLIGHT_TYPE type = (ROADLIGHT_TYPE)atoi(key);

			// SubType field
			key = strtok(NULL, delim);
			ROADLIGHT_SUBTYPE subType = (ROADLIGHT_SUBTYPE)atoi(key);

			// Div field
			key = strtok(NULL, delim);
			ROADLIGHT_DIV div = (ROADLIGHT_DIV)atoi(key);

			// Stop ID
			key = strtok(NULL, delim);
			int numStop = atoi(key);
			for (int i = 0; i < numStop; i++)
			{
				key = strtok(NULL, delim);

				int stopLineID = atoi(key);
			}

			// Geometry

			key = strtok(NULL, delim);
			int numPts = atoi(key);

			for (int i = 0; i < numPts; i++)
			{
				key = strtok(NULL, delim);
				double x = atof(key);

				key = strtok(NULL, delim);
				double y = atof(key);

			}

			count++;
		}

		fclose(fp);

		printf(">>> HDMap(ROADLIGHT) Load Success (%d)\n", count);
	}


	/////////////////////////////////////////////////////////////////////
	// load laneside
	count = 0;
	strcpy(dbFile, "../../txt/etridb_plus_LAYER_LANESIDE.txt");

	if ((fp = fopen(dbFile, "r")) == NULL)
	{
		printf(">>>>> ERROR: _LAYER_LANESIDE Layer Open\n");
	}
	else
	{
		while (fgets(buf, bufSize, fp))
		{
			// ID field
			char *key = strtok(buf, delim);
			int ID = atoi(key);

			// MID field
			key = strtok(NULL, delim);
			int mID = atoi(key);

			// LaneID field
			key = strtok(NULL, delim);
			int laneID = atoi(key);

			// Type field
			key = strtok(NULL, delim);
			LANESIDE_TYPE type = (LANESIDE_TYPE)atoi(key);

			// color field
			key = strtok(NULL, delim);
			LANESIDE_COLOR color = (LANESIDE_COLOR)atoi(key);

			// Geometry
			key = strtok(NULL, delim);
			int numPts = atoi(key);

			for (int i = 0; i < numPts; i++)
			{
				key = strtok(NULL, delim);
			
				double x = atof(key);

				key = strtok(NULL, delim);
				double y = atof(key);
			}


			count++;
		}

		fclose(fp);

		printf(">>> HDMap(LANESIDE) Load Success (%d)\n", count);
	}


	/////////////////////////////////////////////////////////////////////
	// load LN_LINK
	count = 0;
	strcpy(dbFile, "../../txt/etridb_plus_LAYER_LN_LINK.txt");

	if ((fp = fopen(dbFile, "r")) == NULL)
	{
		printf(">>>>> ERROR: LN_LINK Layer Open\n");
	}
	else
	{	
		while (fgets(buf, bufSize, fp))
		{
			// ID field
			char *key = strtok(buf, delim);
			int linkID = atoi(key);

			// MID field
			key = strtok(NULL, delim);
			int mID = atoi(key);
			// LID
			key = strtok(NULL, delim);
			int lID = atoi(key);
			// RID
			key = strtok(NULL, delim);
			int rID = atoi(key);


			// inMID field
			key = strtok(NULL, delim);
			int inMID = atoi(key);
			// inLID
			key = strtok(NULL, delim);
			int inLID = atoi(key);
			// inRID
			key = strtok(NULL, delim);
			int inRID = atoi(key);


			// outMID field
			key = strtok(NULL, delim);
			int outMID = atoi(key);
			// outLID
			key = strtok(NULL, delim);
			int outLID = atoi(key);
			// outRID
			key = strtok(NULL, delim);
			int outRID = atoi(key);


			// junction
			key = strtok(NULL, delim);
			int junction = atoi(key);
			// type
			key = strtok(NULL, delim);
			LANE_TYPE type = (LANE_TYPE)atoi(key);
			// subtype
			key = strtok(NULL, delim);
			LANE_SUB_TYPE subType = (LANE_SUB_TYPE)atoi(key);
			// twoway
			key = strtok(NULL, delim);
			unsigned char twoway = atoi(key);
			// RLID
			key = strtok(NULL, delim);
			int RLID = atoi(key);


			// LLINK
			key = strtok(NULL, delim);
			int lLinkID = atoi(key);
			// RLINK
			key = strtok(NULL, delim);
			int rLinkID = atoi(key);
			// SNODE
			key = strtok(NULL, delim);
			int sNodeID = atoi(key);
			// ENODE
			key = strtok(NULL, delim);
			int eNodeID = atoi(key);
			// Speed
			key = strtok(NULL, delim);
			int speed = atoi(key);


			// Geometry
			key = strtok(NULL, delim);
			int numPts = atoi(key);

			for (int i = 0; i < numPts; i++)
			{
				key = strtok(NULL, delim);
				double x = atof(key);

				key = strtok(NULL, delim);
				double y = atof(key);

			}

			count++;
		}

		fclose(fp);

		printf(">>> HDMap(LN_LINK) Load Success (%d)\n", count);
	}


	/////////////////////////////////////////////////////////////////////
	// load LN_NODE
	count = 0;
	strcpy(dbFile, "../../txt/etridb_plus_LAYER_LN_NODE.txt");

	if ((fp = fopen(dbFile, "r")) == NULL)
	{
		printf(">>>>> ERROR: _LAYER_LN_NODE Layer Open\n");
	}
	else
	{
		while (fgets(buf, bufSize, fp))
		{
			// ID field
			char *key = strtok(buf, delim);
			int nodeID = atoi(key);

			// connected link
			key = strtok(NULL, delim);
			int conCount = atoi(key);
			for (int i = 0; i < conCount; i++)
			{
				key = strtok(NULL, delim);
				int linkID = atoi(key);
			}

			// x field
			key = strtok(NULL, delim);
			double x = atof(key);

			// y field
			key = strtok(NULL, delim);
			double y = atof(key);

			count++;
		}

		fclose(fp);

		printf(">>> HDMap(LN_NODE) Load Success (%d)\n", count);
	}


	return true;
}

