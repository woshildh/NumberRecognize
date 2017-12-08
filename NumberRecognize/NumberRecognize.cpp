// NumberRecognize.cpp: 手写数字识别
//
#define _CRT_SECURE_NO_WARNINGS
#include "stdafx.h"
#include<iostream>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include "preprocess.hpp"
#include"OCR.h"
using namespace std;
using namespace cv;

int red, green, blue;
Mat imagen;
Mat screenBuffer;
int drawing;//用来标记是否正在动并且已经按下了
int r, last_x, last_y;//r表示每个点绘图时的半径,last_x,last_y记录上一个点的坐标

void draw(int x, int y)
{
	circle(imagen, Point(x, y), r, Scalar(blue, green, red), -1, LINE_AA, 0);
	imagen.copyTo(screenBuffer);
	imshow("手写板", screenBuffer);
}

void drawCursor(int x, int y)
{
	imagen.copyTo(screenBuffer);
	circle(screenBuffer, Point(x, y), r, Scalar(red, green, blue), 1, LINE_AA, 0);
}

void on_mouse(int event, int x, int y, int flags, void *param)
{
	last_x = x; last_y = y; //用来记录上一个点
	drawCursor(x, y);
	//选择鼠标事件
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		drawing = 1;
		drawCursor(x, y);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		drawing = 0;
	}
	else if (event == CV_EVENT_MOUSEMOVE && flags && CV_EVENT_FLAG_LBUTTON)
	{
		if (drawing == 1)
			draw(x, y);
	}
}

int main(int argc, char** argv)
{
	printf("                                手写数字识别\n"
		"快捷键: \n"
		"\tq - 退出程序\n"
		"\tr - 重置白版\n"
		"\t+ - 笔迹增粗 ++\n"
		"\t- - 笔迹减细 --\n"
		"\ts - 保存输入为 out.pbm\n"	//输入可以作为样本再次部署进去
		"\tc - 输入分类识别, 结果在console显示\n"
		"\tESC - 退出程序\n");
	drawing = 0;
	r = 4;
	red = green = blue = 0;
	last_x = last_y = 0;

	OCR example;
	example.setTrainNum(70);//设置训练数据的图片数目
	example.read_data();//读入每个类中的100张图片数据
	example.train();
	float accuracy = example.test();

	//创建图像
	imagen.create(Size(128, 128), CV_8UC3);
	imagen.setTo(Scalar(255, 255, 255)); //把整幅图像改为白板图像 setTo函数设置矩阵的值为一样的值

	screenBuffer = imagen.clone();

	namedWindow("手写板", 0);
	resizeWindow("手写板", 512, 512);
	setMouseCallback("手写板", on_mouse); //设置回调函数
	
	int cishu = 0;//用来标识第几次预测
	cout << endl << endl << "次数"<<"			"<<"预测结果"<<endl;
	for (;;)
	{
		int c;//因为waiteKey()只能返回整数
		imshow("手写板", screenBuffer);
		c = waitKey(10);
		if ((char)c == 'r')
		{
			imagen.setTo(Scalar(255, 255, 255));
			drawCursor(last_x, last_y);
		}
		else if ((char)c == '+')
		{
			r++;
			drawCursor(last_x, last_y);
		}
		else if ((char)c == '-' && r>1)
		{
			r--;
			drawCursor(last_x, last_y);
		}
		else if ((char)c == 'q')
		{
			break;
		}
		else if ((char)c == 's')
		{
			imwrite("out.pbm", imagen);
		}
		else if ((char)c == 'c')
		{
			cishu++;
			imwrite("out.pbm",imagen);
			Mat temp = imread("out.pbm", 0);//以灰度方式读入
			float res=example.predict_pre(temp);
			cout <<cishu<<"			"<< res << endl;
		}
	}
	destroyWindow("手写板");//销毁窗口
	return 0;
}