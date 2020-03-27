#pragma once
#include <string>
#include <vector>
using namespace std;
struct JDXX//存放文件中的数据
{
double x, y, l, r;//交点坐标的x，y值，缓和曲线长度l，曲线半径r
};
struct QXYS//存放曲线信息
{
double r, l, a;//曲线半径、缓和曲线长度、转角
double p, q, b;//内移距离p，切线增长q，缓和曲线对应中心角
double Th, Lh, Eh, Ly, Dh;//切线长TH,曲线长Lh,曲线外距Eh,圆曲线长Ly，切曲差Dh
string ZX;
};
struct LH//存放里程信息
{
double QD, ZH, HY, QZ, YH, HZ, ZD;
};
struct ZB//存放点的坐标信息
{
double ZHX, ZHY, HYX, HYY, HZX, HZY, YHX, YHY;//最后一个为线段方位角，其余为对应点的X，Y坐标
};
struct AG {
double D, FWJ;//线段的长度与与X轴的夹角值
};
class Data {
public:
void getdata();//获取文本信息 1
void js_angle();//计算转角与方位角 1
void js_jsys();//计算Th, Lh, Eh, Ly, Dh， p, q, b 1
void js_lh();//计算里程 1
void js_zuobiao();//计算坐标 1
void xianshi();//显示计算结果
};



