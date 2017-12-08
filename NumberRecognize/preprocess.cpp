#include"stdafx.h"
#include<iostream>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include"preprocess.hpp"
#include<io.h>
#include<cstring>
#define FILENAME_SIZE 150
using namespace cv;
using namespace std;
void findBorder(Mat image,Rect &border)
{
	int min_x = image.cols-1, max_x=0, min_y=image.rows-1, max_y=0;
	int rowNum = image.rows,colNum=image.cols;
	//cout << rowNum << " " << colNum << endl;
	//int flag = 0;
	for (int i = 0; i < rowNum; i++)
	{
		uchar *data = image.ptr<uchar>(i);
		for (int j = 0; j < colNum; j++)
		{
			if (data[j] >120)
			{
				 continue;
			}
			else
			{
				//flag = 1;//测试接下来的一段会不会到达
				if (i < min_y)   min_y = i;
				if (i > max_y)   max_y = i;
				if (j < min_x)   min_x = j;
				if (j > max_x)   max_x = j;
			}
		}
	}
	//cout << flag << endl;
	//cout << min_x << " " << max_x << " " << min_y << " " << max_y << endl;
	border.x = min_x; border.y = min_y;
	border.height = max_y - min_y+1;
	border.width = max_x - min_x+1;
}

void preprocess(Mat srcImage, Rect border,Mat &dstImage,int user_width, int user_height)
{
	Mat subImage = srcImage(border);   //获得图像的最中间部分
	int length = (border.height > border.width ? border.height : border.width);
	Mat  midImage(length,length,CV_8UC1);     //将图像转为一个正方形，并且使subImage处于中间
	Mat_<uchar>::iterator it=midImage.begin<uchar>(), itend = midImage.end<uchar>();
	for (; it != itend; it++)
	{
		(*it) = 255;
	}
	int x =(length-border.width)/2 ;
	int y =(length-border.height)/2;
	Mat ROI = midImage(Rect(x,y,border.width,border.height));
	subImage.copyTo(ROI);//ROI是掩膜区域
	resize(midImage, dstImage, Size(user_width, user_height), CV_INTER_NN);
}
//这个函数对原始数据进行统一的预处理，返回值为1表示处理成功，0表示处理失败.pre_path 和propath均以  /结尾
int compress(char *pre_path1, char *pro_path, int user_weight, int user_height)//pre_path指的是原始数据存放的路径,pro_path指的是处理后的数据存放的路径,user_height,user_weight指的是图像处理后的宽和高
{
	int num = 0;
	//cout << pre_path << endl; return 1;
	char pre_path[100];
	strcpy(pre_path, pre_path1);
	if (_access(pre_path,0) == -1)
	{
		cout << "没有找到相应的文件夹" << endl;
		return -1;
	}
	strcat(pre_path, "*");//必须有通配符*来表示找什么文件
	//cout << pre_path << endl; return 1;
	struct _finddata_t fileinfo;
	intptr_t handle;
	handle = _findfirst(pre_path, &fileinfo);
	//cout << handle;//用于测试
	//return 1; //用于测试
	if (handle == -1)
	{
		cout << "文件夹是空的 " << endl;    return -1;
	}
	do {
		
		if (fileinfo.attrib & _A_SUBDIR  && strcmp(fileinfo.name, "..") != 0 && strcmp(fileinfo.name, ".")!=0)
		{
			char subpath[FILENAME_SIZE]; strcpy(subpath, pre_path1); strcat(subpath, fileinfo.name); strcat(subpath, "/*.pbm");
			//cout << subpath << endl;
			struct _finddata_t subfile; 
			intptr_t subhandle;//此处用long会出错，很坑
			subhandle = _findfirst(subpath, &subfile);
			//cout << subhandle << endl;
			if (subhandle == -1)
				continue;
			subpath[strlen(subpath) - 6] = '\0';
			do {
				//cout << "hello" << endl;
				//求读取图片的路径
				char path[FILENAME_SIZE]; strcpy(path, subpath);
				strcat(path, "/");
				strcat(path, subfile.name);
				
				//对图像进行操作
				Mat pre_img = imread(path, 0);  Mat pro_img;
				Rect border; findBorder(pre_img, border);
				preprocess(pre_img, border, pro_img, 64, 64);

				//求保存图片的路径
				char save_path[FILENAME_SIZE]; strcpy(save_path, pro_path);
				strcat(save_path, fileinfo.name); strcat(save_path, "/"); strcat(save_path, subfile.name);
	
				imwrite(save_path,pro_img);
				num++;
			}while(_findnext(subhandle,&subfile)==0);
			_findclose(subhandle);
		}
	}while(_findnext(handle,&fileinfo)==0);
	_findclose(handle);
	cout << "压缩成功了"<<num<<"张图片"<< endl;
	return 1;
}