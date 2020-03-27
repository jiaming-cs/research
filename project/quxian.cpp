#include<iostream>
#include<string>
#include<math.h>
#include<vector>
using namespace std;
#define MAX 10000//最多交点个数
#define PI 3.1459265359

void menu()
{
	cout << "****************************" << endl;
	cout << "*****  1.添加交点信息  *****" << endl;
	cout << "*****  2.输入起点里程  *****" << endl;
	cout << "*****  3.输出计算结果  *****" << endl;
	cout << "*****  4.交点信息统计  *****" << endl;
	cout << "*****  5.清空交点信息  *****" << endl;
	cout << "*****  0.退出计算系统  *****" << endl;
	cout << "****************************" << endl;
}



//交点信息结构体
struct JD
{
	double jd_x;//交点坐标x
	double jd_y;//交点坐标y
	double jd_l0;//缓和曲线长
	double jd_r;//圆曲线半径
};

//交点统计结构体
struct Book
{
	struct JD jdArray[MAX];//统计保存的交点信息数组
	int jd_size;//统计的交点个数
};

//1.添加交点信息
void addJD(Book * dot)
{
	//判断交点统计存储是否已满
	if (dot->jd_size == MAX)
	{
		cout << "交点统计存储已满，无法添加。" << endl;
		return;
	}
	else
	{
		//x坐标
		double x = 0;
		cout << "请输入x坐标：" << endl;
		cin >> x;
		dot->jdArray[dot->jd_size].jd_x = x;
		//y坐标
		double y = 0;
		cout << "请输入y坐标：" << endl;
		cin >> y;
		dot->jdArray[dot->jd_size].jd_y = y;
		//缓和曲线长
		double l0 = 0;
		cout << "请输入缓和曲线长度l0(m)：" << endl;
		cin >> l0;
		dot->jdArray[dot->jd_size].jd_l0 = l0;
		//曲线半径
		double r = 0;
		cout << "请输入圆曲线半径r(m)：" << endl;
		cin >> r;
		dot->jdArray[dot->jd_size].jd_r = r;

		//更新统计交点个数
		dot->jd_size++;
		
		cout << "添加成功！" << endl;
		system("pause");
		system("cls");
	}
}

//2.输入起点里程
void licheng(Book* dot)
{
	cout << "请输入起始点里程（只输入数字部分）：" << endl;
	int qdlc=0;
	cin >> qdlc;
	int qdlc1 = (qdlc / 1000);
	int qdlc2 = (qdlc % 1000);
	cout << "起始点里程为：DK" << qdlc1 << "+" << qdlc2 << endl;
	cout << "起始点里程输入成功！" << endl;
	system("pause");
	system("cls");
}

//3.显示计算结果
void jisuan(Book* dot)
{
	if (dot->jd_size < 3)
	{
		cout << "交点数量不足，无法计算！" << endl;
	}
	else
	{
		//方位角计算
		cout << "1.方位角计算结果如下：" << endl;
		for (int j = 0; j < (dot->jd_size - 1); j++)
		{
			if (dot->jdArray[j + 1].jd_y == dot->jdArray[j].jd_y)
			{
				cout << "JD" << (j + 1) << "的方位角为：" << "90°" << endl;
			}
			else
			{
				double fwj = atan2((dot->jdArray[j + 1].jd_x - dot->jdArray[j].jd_x) , (dot->jdArray[j + 1].jd_y - dot->jdArray[j].jd_y));
				if (fwj > 0)
				{
					double fwj1 = fwj / PI * 180;
					cout << "JD" << (j + 1) << "的方位角为：" << fwj1 << "°" << endl;
				}
				else if (fwj == 0)
				{
					cout << "JD" << (j + 1) << "的方位角为：" << "0°" << endl;
				}
				else
				{
					double fwj1 = fwj / PI * 180 + 180;
					cout << "JD" << (j + 1) << "的方位角为：" << fwj1 << "°" << endl;
				}
			}
			
		}
		//转角计算


	}
	system("pause");
	system("cls");
}

//4.交点信息统计
void showJD(Book* dot)
{
	if (dot->jd_size == 0)
	{
		cout << "当前记录为空！" << endl;
	}
	else
	{
		for (int i = 0; i < dot->jd_size; i++)
		{
			cout <<"JD"<< (i + 1) << "  ";
			cout << "x：" << dot->jdArray[i].jd_x << "\t";
			cout << "y：" << dot->jdArray[i].jd_y << "\t";
			cout << "l0(m)：" << dot->jdArray[i].jd_l0 << "\t";
			cout << "r(m)：" << dot->jdArray[i].jd_r << endl;
		}
	}
	system("pause");
	system("cls");
}

//5.清空交点信息
void cleanJD(Book* dot)
{
	dot->jd_size = 0;
	cout << "交点信息已清空！" << endl;
	system("pause");
	system("cls");
}

int main()
{
	//创建交点统计结构体变量
	Book dot;
	//初始化交点统计个数
	dot.jd_size = 0;

	int select = 0;
	while (true)
	{
		menu();

		cin >> select;
		switch (select)
		{
		case 1://1.添加交点信息
			addJD(&dot);
			break;
		case 2://2.输入起点里程
			licheng(&dot);
			break;
		case 3://3.显示计算结果
			jisuan(&dot);
			break;
		case 4://4.交点信息统计
			showJD(&dot);
			break;
		case 5://5.清空交点信息
			cleanJD(&dot);
			break;
		case 0://0.退出计算系统
			cout << "欢迎下次使用" << endl;
			system("pause");
			return 0;
			break;
		default:
			break;
		}
	}
	

	system("pause");
	return 0;
}
