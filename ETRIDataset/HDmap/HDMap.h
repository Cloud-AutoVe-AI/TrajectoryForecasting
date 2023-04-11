/* HDMap.h *//*---------------------------------------------------*//*
//  Class Name  : CHDMap
//  Creator     : Kyoung-Wook Min
//  Create Date : 2020.07.31
//  Revisor		:
//  Revised Date:
//  Platform	: Windows
//  Language	: C++
//  Description : HDMap Parser
*//*----------------------------------------------------------------------*/



#pragma once

// 로드마크 종별(POLYGON)
enum ROADMARK_TYPE
{
	RM_CROSSWALK = 1,
	RM_SPEEDBUMP = 2,
	RM_ARROW = 3,
	RM_NUMERIC = 4,
	RM_SHAPE = 6,
	RM_CHAR = 5,
	RM_STOPLINE = 7,
	RM_BUSSTOP = 8,
	RM_VIRTUAL_STOPLINE = 9
};


enum ROADMARK_SUBTYPE
{
	RM_ARROW_S = 1,
	RM_ARROW_L = 2,
	RM_ARROW_R = 3,
	RM_ARROW_SL = 4,
	RM_ARROW_SR = 5,
	RM_ARROW_U = 6,
	RM_ARROW_US = 7,
	RM_ARROW_UL = 8,
	RM_ARROW_LR = 9,
	RM_ARROW_FORBID_L = 10,
	RM_ARROW_FORBID_R = 11,
	RM_ARROW_FORBID_S = 12,
	RM_ARROW_FORBID_U = 13,
	RM_STOPLINE_UNSIGNED_INTERSECTION = 14
};

enum ROADLIGHT_TYPE
{
	RL_HOR = 1,
	RL_VIR = 2
};

enum ROADLIGHT_SUBTYPE
{
	RL_2 = 1,
	RL_3 = 2,
	RL_4 = 3,
	RL_5 = 4
};

enum ROADLIGHT_DIV
{
	GEN_RL = 1,	// 일반신호등
	BUS_RL = 2  // 버스전용 신호등
};

enum LANESIDE_TYPE
{

	LS_SOLID = 1,
	LS_DOT = 2,
	LS_DOUBLE = 3,
	LS_BOUNDARY = 4,
	LS_VIRTUAL = 5
};

enum LANESIDE_COLOR
{
	LS_WHITE = 0,
	LS_YELLOW = 1,
	LS_BLUE = 2
};

enum LANE_TYPE
{
	LANE_TYPE_NONE = 0,
	GEN_S = 1, // 일반 직진
	JUN_S = 2, // 교차로 직진
	JUN_L = 3, // 교차로 좌
	JUN_R = 4, // 교차로 우
	JUN_U = 5, // 교차로 유
	POCKET_L = 6,	// 좌 포켓 차로
	POCKET_R = 7,	// 우 포켓 차로
	JUN_UNPROTECTED_L = 8

	//BUS_ONLY = 8
};

enum LANE_SUB_TYPE
{
	GEN = 1, // 일반
	BUS_ONLY = 2, // 버스전용
	HIGHPASS = 3, // 하이패스
	TURNAL = 4 // 터널

};


class CHDMap
{
	
public:
	CHDMap();
	~CHDMap();

	bool LoadHDMap();

};