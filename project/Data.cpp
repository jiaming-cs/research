#include "Data.h"
#include<fstream>
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;
int n;//记录交点个数
vector <JDXX> JD;
vector <QXYS> QX;
vector <LH> L;
vector<ZB> Z;
vector<AG> A;
void Data::getdata(){
ifstream infile;
vector<double>t;
infile.open("JD.txt",ios::in); //文件路径
if (!infile)
{
cout << "文件不存在" << endl;
return;
}
if (infile.is_open())
{
double temp;
while (infile >> temp) {
t.push_back(temp);
}
infile.close();
n = t[0];
vector<JDXX> J;//缓存存放数组
J.resize(n); int a = 0;
for (unsigned int i = 1; i < t.size(); i = i + 4) //将数据进行分类,4个一组
{
J[a].x = t[i];
J[a].y = t[i + 1];
J[a].r = t[i + 2];
J[a].l = t[i + 3];
JD.push_back(J[a]);
cout << JD[a].x << endl;
cout << JD[a].y << endl;
cout << JD[a].r << endl;
cout << JD[a].l << endl;
a++;
} 
}A.resize(n - 1);
cout << "文件获取成功" << endl;
}
void Data::js_angle()//计算一缓和曲线转角，以及各交点连线的方位角。
{
vector<double>D1, D2, DX,Angle,fwj;
QX.resize(n - 2);
for (int i = 0; i < n-2; i++) {
D1.push_back(sqrt(pow(JD[i].x - JD[i + 1].x, 2) + pow(JD[i].y - JD[i + 1].y, 2)));
D2.push_back(sqrt(pow(JD[i+1].x - JD[i + 2].x, 2) + pow(JD[i+1].y - JD[i + 2].y, 2)));
DX.push_back(((JD[i + 1].x - JD[i].x) * (JD[i + 2].x - JD[i + 1].x)) + ((JD[i + 1].y - JD[i].y) * (JD[i + 2].y - JD[i + 1].y)));
Angle.push_back(acos(DX[i] / (D1[i] * D2[i]))*180 / acos(-1));
QX[i].a = Angle[i];
QX[i].r = JD[i + 1].r;
QX[i].l = JD[i + 1].l;
} for (int i = 0; i < n - 1; i++) {
D1.push_back(sqrt(pow(JD[i].x - JD[i + 1].x, 2) + pow(JD[i].y - JD[i + 1].y, 2)));
A[i].D = D1[i];//计算各交点连线长度
}
for (int i = 0; i < n - 1; i++) {
fwj.push_back(atan((JD[i + 1].y - JD[i].y) / (JD[i + 1].x - JD[i].x)));
A[i].FWJ = fwj[i];//计算各连线与X轴夹角
}
cout << "角度计算成功" << endl;
}
void Data::js_jsys() //计算缓和曲线各要素值
{
QX.resize(n - 1);
double pi = acos(-1);
for (int i = 0; i < n-2; i++) {
QX[i].p=pow(QX[i].l, 2) / (24 * QX[i].r);
QX[i].q= QX[i].l / 2 - pow(QX[i].l, 3) / (240 * pow(QX[i].r, 2));
QX[i].b = QX[i].l * 180 / (2 * QX[i].r * pi);
QX[i].Th = (QX[i].r + QX[i].p) * tan(QX[i].a * pi / 360) + QX[i].q;
QX[i].Lh = QX[i].r * pi * (QX[i].a - 2 * QX[i].b) / 180 + 2 * QX[i].l;
QX[i].Eh = (QX[i].r + QX[i].p) / cos(QX[i].a * pi / 360 )- QX[i].r;
QX[i].Ly = QX[i].r * pi * (QX[i].a - 2 * QX[i].b) / 180;
QX[i].Dh = 2 * QX[i].Th - QX[i].Lh;
}
QX[n - 2].Th = 0;//为了下一步计算需要而设置
cout << "要素计算成功" << endl;
}
void Data::js_lh() //计算曲线个点里程
{
L.resize(n );
string a;
cout << "输入起点里程DKXXX+YYY;";
cin >> a;
int a1 = a.find("K");
int b = a.find("+");
string c = a.substr(a1+1, b - a1-1);
string cc = a.substr(b+1, a.size() - b-1);
double a2 = atof(c.c_str());
double a3 = atof(cc.c_str());
L[0].QD = a2 * 1000 + a3; 
L[0].ZH = L[0].QD + A[0].D - QX[0].Th;
for (int i = 0; i < n - 2; i++) {
L[i].HY = L[i].ZH + QX[i].l;
L[i].QZ = L[i].HY + QX[i].Ly / 2;
L[i].YH = L[i].QZ + QX[i].Ly / 2;
L[i].HZ = L[i].YH + QX[i].l;
L[i + 1].QD = L[i].QD + A[i].D;
L[i + 1].ZH = L[i + 1].QD + A[i + 1].D - QX[i].Th - QX[i + 1].Th;
}
L[n - 2].ZD = L[n - 1].QD;//设置终点里程
cout << "里程计算成功" << endl;
}
void Data::js_zuobiao() //计算曲线个点坐标
{
Z.resize(n - 1);
vector<double>x0, y0;
if (JD[ 1].x - JD[0].x > 0) //设置第一个ZH点坐标
{
Z[0].ZHX = JD[0].x + (A[0].D - QX[0].Th) * cos(A[0].FWJ);
Z[0].ZHY = JD[0].y + (A[0].D - QX[0].Th) * sin(A[0].FWJ);
}
else
{
Z[0].ZHX = JD[0].x -(A[0].D - QX[0].Th) * cos(A[0].FWJ);
Z[0].ZHY = JD[0].y -(A[0].D - QX[0].Th) * sin(A[0].FWJ);
}
for (int i = 0; i < n - 2; i++) 
{
x0.push_back(QX[i].l - pow(QX[i].l, 3) / (40 * pow(QX[i].r, 2)));
y0.push_back(pow(QX[i].l, 2) / (6 * QX[i].r));
if (JD[i + 1].x - JD[i].x > 0)
{
Z[i].HYX = Z[i].ZHX + (x0[i] * cos(A[i].FWJ) - y0[i] * sin(A[i].FWJ));
Z[i].HYY = Z[i].ZHY + (x0[i] * sin(A[i].FWJ) + y0[i] * cos(A[i].FWJ));
}
else
{
Z[i].HYX = Z[i].ZHX - (x0[i] * cos(A[i].FWJ) - y0[i] * sin(A[i].FWJ));
Z[i].HYY = Z[i].ZHY - (x0[i] * sin(A[i].FWJ) + y0[i] * cos(A[i].FWJ));
}
if (JD[i + 2].x > JD[i + 1].x) 
{
Z[i].HZX = JD[i + 1].x + QX[i].Th * cos(A[i + 1].FWJ);
Z[i].HZY = JD[i + 1].y + QX[i].Th * sin(A[i + 1].FWJ);
Z[i].YHX = Z[i].HZX - (x0[i] * cos(A[i + 1].FWJ) + y0[i] * sin(A[i + 1].FWJ));
Z[i].YHY = Z[i].HZY - (x0[i] * sin(A[i + 1].FWJ) - y0[i] * cos(A[i + 1].FWJ));
Z[i + 1].ZHX = JD[i + 1].x + (A[i + 1].D - QX[i].Th - QX[i + 1].Th)* cos(A[i+1].FWJ);
Z[i + 1].ZHY = JD[i + 1].y + (A[i + 1].D - QX[i].Th - QX[i + 1].Th) * sin(A[i + 1].FWJ);
}
else
{
Z[i + 1].ZHX = JD[i + 1].x -(A[i + 1].D - QX[i].Th - QX[i + 1].Th) * cos(A[i + 1].FWJ);
Z[i + 1].ZHY = JD[i + 1].y - (A[i + 1].D - QX[i].Th - QX[i + 1].Th) * sin(A[i + 1].FWJ);
Z[i].HZX = JD[i + 1].x - QX[i].Th * cos(A[i + 1].FWJ);
Z[i].HZY = JD[i + 1].y - QX[i].Th * sin(A[i + 1].FWJ);
Z[i].YHX = Z[i].HZX + (x0[i] * cos(A[i + 1].FWJ) + y0[i] * sin(A[i + 1].FWJ));
Z[i].YHY = Z[i].ZHY + (x0[i] * sin(A[i + 1].FWJ) - y0[i] * cos(A[i + 1].FWJ));
}
}
cout << "坐标计算成功" << endl;
}
void Data::xianshi() 
{
cout << "起点里程：DK" << int(L[0].QD/1000) << "+" << L[0].QD-int(L[0].QD/1000)*1000 << endl;
for (int i = 0; i < n - 2; i++)
{
cout << "第" << i+1 << "个曲线：" << endl;
cout << "缓和曲线转角为：" << QX[i].a << endl;
cout << "缓和曲线内移距为：" << QX[i].p << endl;
cout << "缓和曲线切线增长为：" << QX[i].q << endl;
cout << "缓和曲线切线长为：" << QX[i].Th << endl;
cout << "缓和曲线曲线长为：" << QX[i].Lh << "其中圆曲线长为：" << QX[i].Ly << endl;
cout << "缓和曲线外距为：" << QX[i].Eh << endl;
cout << "切曲差为：" << QX[i].Dh << endl;
cout << "直缓点里程：DK" << int(L[i].ZH / 1000) << "+" << L[i].ZH - int(L[i].ZH / 1000) * 1000 << endl;
cout << "缓圆点里程: DK" << int(L[i].HY / 1000) << "+" << L[i].HY - int(L[i].HY / 1000) * 1000 << endl;
cout << "曲中点里程: DK" << int(L[i].QZ / 1000) << "+" << L[i].QZ - int(L[i].QZ / 1000) * 1000 << endl;
cout << "圆缓点里程：DK" << int(L[i].YH / 1000) << "+" << L[i].YH - int(L[i].YH / 1000) * 1000 << endl;
cout << "缓直点里程：DK" << int(L[i].HZ / 1000) << "+" << L[i].HZ - int(L[i].HZ / 1000) * 1000 << endl;
cout << "直缓点坐标：" << Z[i].ZHX << "," << Z[i].ZHY<< endl;
cout << "缓圆点坐标：" << Z[i].HYX << "," << Z[i].HYY << endl;
cout << "圆缓点坐标：" << Z[i].YHX << "," << Z[i].YHY << endl;
cout << "缓直点坐标：" << Z[i].HZX << "," << Z[i].HZY << endl;
}
}
