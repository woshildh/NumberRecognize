#pragma once
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include"stdafx.h"
using namespace cv;
using namespace std;

void findBorder(Mat image, Rect &border);
void preprocess(Mat srcImage, Rect border, Mat &dstImage, int user_width, int user_height);
//pre_path指的是原始数据存放的路径,pro_path指的是处理后的数据存放的路径,user_height,user_weight指的是图像处理后的宽和高
int compress(char *pre_path, char *pro_path, int user_weight, int user_height);
