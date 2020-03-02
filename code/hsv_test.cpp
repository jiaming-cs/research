#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric> //accumulate
#include <cmath>
#include <climits> //INT_MAX
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
int main(int argc, char const *argv[])
{
    

    string folder = "/home/jiaming/research/img/";
    Mat img1 = imread(folder + "test1.png", IMREAD_COLOR);
    Mat img2 = imread(folder + "test2.png", IMREAD_COLOR);
    Mat img1_hsv;
    Mat img2_hsv;

    cvtColor(img1, img1_hsv, COLOR_BGR2HSV_FULL);
    cvtColor(img2, img2_hsv, COLOR_BGR2HSV_FULL); 
 
    imshow("img1", img1_hsv);
    imshow("Img2", img2_hsv);
    waitKey(0);
    return 0;
}