/*********************************
头文件
**********************************/
#include <fcntl.h> //vidoe文件打开头文件
#include <unistd.h>  //video文件关闭头文件
#include <sys/ioctl.h> 
#include <signal.h> //信号处理头文件
#include <stdio.h>   //标准库头文件
#include <stdlib.h> //标准库
#include <stdint.h> //数据类型 uint8_t的头文件
#include <string.h> //字符串函数头文件
#include <sys/mman.h> //mmap函数映射对象到内存的头文件
#include <malloc.h> //内存申请函数
#include "opencv2/opencv.hpp" //opencv库
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;


int get_filenames(const std::string& dir, std::vector<std::string>& filenames)
{
    fs::path path(dir);
    if (!fs::exists(path))
    {
        return -1;
    }
 
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter!=end_iter; ++iter)
    {
        if (fs::is_regular_file(iter->status()))
        {
            filenames.push_back(iter->path().string());
        }
 
        if (fs::is_directory(iter->status()))
        {
            get_filenames(iter->path().string(), filenames);
        }
    }
 
    return filenames.size();
}

//异常信号处理
void signal_handle(int signo)
{
    printf("force exit signo %d !!!\n",signo);
    exit(0);
}
void signal_exit_handler(int sig)
{
    exit(0);
}
void signal_crash_handler(int sig)
{
    exit(-1);
}

/*********************************
程序入口
**********************************/
int main (int argc, char *argv[])
{
    //异常信号处理
    signal(SIGINT, signal_handle);
    signal(SIGTERM, signal_exit_handler);
    //signal(SIGINT, signal_exit_handler);
    signal(SIGPIPE, SIG_IGN);
    signal(SIGBUS, signal_crash_handler);
    signal(SIGSEGV, signal_crash_handler);
    signal(SIGFPE, signal_crash_handler);
    signal(SIGABRT, signal_crash_handler);

    //处理图像
    int j = 1;
    int img_num = 20; //标定图片张数
    const int image_count = img_num;
    int count = 0; //保存所有图片角点数量
    int successImageNum = 0;	//成功提取角点的棋盘图数量	
    Size board_size = Size(6,4); //标定板角点尺寸
    vector<Mat>  image_Seq;
    vector<vector<Point2f>>  corners_Seq; //保存每张图片识别的角点
    //用图片标定
    while(img_num)
    {           string str_rgb = "../data/img/"+to_string(j)+".jpg";
                Mat img_rgb = imread(str_rgb);  //rgb图像
                Mat img_rgbgray;  //灰度图
                cvtColor(img_rgb, img_rgbgray, CV_RGB2GRAY); //将BGRA图像转换为灰度图
                //从拍摄的图片中提取角点并进行亚像素精确化
                vector<Point2f> corners; //检测到的角点
                printf("findChessboardCorners....\n");
                bool patternfound = findChessboardCorners(img_rgb, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		        if (!patternfound)
			        printf("can not find chessboard corners!\n");
                else
                {
                    // 亚像素精确化 
			        cornerSubPix(img_rgbgray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
                    //保存识别的角点
                    count = count + corners.size();
                    successImageNum = successImageNum + 1;
                    corners_Seq.push_back(corners); 
                    image_Seq.push_back(img_rgb);
                    printf("findChessboardCorners success!\n");
                    //将像素点绘制到图上
                    Mat img_circle = img_rgb.clone();
                    printf("draw circle on image  %d!\n", j);
                    for(unsigned int j=0; j < corners.size(); j++)
                    {
                        circle(img_circle, corners[j], 10, Scalar(0, 0, 255), 2, 8, 0);
                    }
                    string str_rgb = "../data/corners/"+to_string(j)+".jpg";
                    imwrite(str_rgb, img_circle);
                    j++;
                    img_num--;
                }
        }
    
    //鱼眼相机定标
    printf("fisheye camera conrners 2d to 3d ....\n");
    Size square_size = Size(20, 20);
	vector<vector<Point3f>>  object_Points;        // 保存定标板上角点的三维坐标 
    Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));  //保存提取的所有角点
    vector<int>  point_counts; //每张图片的角点数量
        //初始化定标板上角点的三维坐标
    for (int t = 0; t<successImageNum; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i<board_size.height; i++)
		{
			for (int j = 0; j<board_size.width; j++)
			{
				// 假设定标板放在世界坐标系中z=0的平面上 
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}
    for (int i = 0; i< successImageNum; i++) //获取每张图片的角点数量
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
        //开始定标
    Size image_size = image_Seq[0].size(); //保存图片大小
    cv::Matx33d intrinsic_matrix;    //    摄像机内参数矩阵    
	cv::Vec4d distortion_coeffs;     // 摄像机的4个畸变系数：k1,k2,k3,k4
	std::vector<cv::Vec3d> rotation_vectors;                           // 每幅图像的旋转向量 
	std::vector<cv::Vec3d> translation_vectors;                        // 每幅图像的平移向量 
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
	printf("fisheye camera conrners 2d to 3d success!\n");
        //保存定标文件
    string datFileName = "../data/camParam.dat";
    FILE *camParam = fopen(datFileName.c_str(), "wb");
    if (camParam == NULL) {
        printf("can not create data file: %s !!!\n", datFileName.c_str());
		return -1;
	}
    fwrite(&intrinsic_matrix, sizeof(cv::Matx33d), 1, camParam);
	fwrite(&distortion_coeffs, sizeof(cv::Vec4d), 1, camParam);
	fwrite(&image_size, sizeof(Size), 1, camParam);
	fclose(camParam);
    printf("save data file : %s success!\n", datFileName.c_str());
    //评价定标结果
    ofstream fout("/root/img/caliberation_result.txt");  //    保存定标结果的文件    
    cout << "开始评价定标结果………………" << endl;
	double total_err = 0.0;                   // 所有图像的平均误差的总和 
	double err = 0.0;                        // 每幅图像的平均误差 
	vector<Point2f>  image_points2;             //   保存重新计算得到的投影点    
	cout << "每幅图像的定标误差" << endl;
    fout << "每幅图像的定标误差" << endl << endl;
	for (int i = 0; i<image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		//    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     
		fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
		// 计算新的投影点和旧的投影点之间的误差
		vector<Point2f> tempImagePoint = corners_Seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (size_t i = 0; i != tempImagePoint.size(); i++)
		{
			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		cout << "第" << i + 1 << "幅图像的平均误差" << err << "像素" << endl;
        fout << "第" << i + 1 << "幅图像的平均误差" << err << "像素" << endl;
	}
	cout << "总体平均误差" << total_err / image_count << "像素" << endl;
    fout << "总体平均误差" << total_err / image_count << "像素" << endl << endl;
	cout << "评价完成" << endl;
    //保存定标结果
    cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 保存每幅图像的旋转矩阵 
	fout << "相机内参数矩阵" << endl;
	fout << intrinsic_matrix << endl;
	fout << "畸变系数\n";
	fout << distortion_coeffs << endl;
	for (int i = 0; i<image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量" << endl;
		fout << rotation_vectors[i] << endl;

		// 将旋转向量转换为相对应的旋转矩阵 
		Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量" << endl;
		fout << translation_vectors[i] << endl;
	}
	cout << "完成保存" << endl;
	fout << endl;
    //显示定标结果
    Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "保存矫正图像" << endl;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
		Mat t = image_Seq[i].clone();
		cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR);
		string imageFileName = "../data/img/result"+to_string(i+1)+".jpg";
		imwrite(imageFileName, t);
	}
	cout << "保存结束" << endl;
    return 0;
}
