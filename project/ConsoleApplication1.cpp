
// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "Data.h"
#include <iostream>
using namespace std;
int main()
{
Data a;
a.getdata();//获取文本信息 
a.js_angle();//计算转角与方位角 
a.js_jsys();//计算Th, Lh, Eh, Ly, Dh， p, q, b 
a.js_lh();//计算里程 
a.js_zuobiao();//计算坐标 
a.xianshi();//输出
return 0;