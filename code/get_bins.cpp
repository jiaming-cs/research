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



Mat get_mask(Mat img, Point2f center, int r){
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    circle(mask, center, r, Scalar(255), -1);
    return mask;
}

Mat get_hist(Mat img_hvs, Point2f center, int r = 10, int bins = 60){
    Mat mask = get_mask(img_hvs, center, r);
    int hist_size[] = {bins};
    float hranges[] = {0, 180};
    const float* ranges[] = {hranges};
    int chanels [] = {0};
    MatND hist;
    calcHist(&img_hvs, 1, chanels, mask, hist, 1, hist_size, ranges);
    cout<<format(hist, Formatter::FMT_PYTHON)<<endl;
    hist = hist.reshape(1, 1);
    cout<<format(hist, Formatter::FMT_PYTHON)<<endl;
    Scalar totoal = sum(hist);
    cout<<totoal[0]<<endl;
    hist = hist * 1.0 / totoal[0];
    cout<<format(hist, Formatter::FMT_PYTHON)<<endl;
    return hist;
}





int main(int argc, char const *argv[])
{
    // Usage <Image> <x> <y> <r>
    string folder = "/home/jiaming/research/img/";
    string img1_name = argv[1];
    int x = strtol(argv[2], NULL, 10);
    int y = strtol(argv[3], NULL, 10);
    int r = strtol(argv[4], NULL, 10);
    Mat img1 = imread(folder + img1_name, IMREAD_COLOR);
    Mat img1_hsv;
    cvtColor(img1, img1_hsv, COLOR_BGR2HSV_FULL); 
    Point center = Point(x, y);
    circle(img1_hsv, center, r, Scalar(0, 0, 255), 2);
    circle(img1_hsv, Point(200, 150), r, Scalar(0, 0, 255), 2);
    get_hist(img1_hsv, center, r);
    imshow("HSV_IMAGE", img1_hsv);
    imwrite("HSV_Image.png", img1_hsv);
    waitKey(100000);
    
    return 0;
}

