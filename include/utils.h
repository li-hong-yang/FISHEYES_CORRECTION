/*********************************
头文件
**********************************/

#include <opencv2/opencv.hpp> //opencv库
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

namespace fs = boost::filesystem;


int get_filenames(const std::string& dir, std::vector<std::string>& filenames);

