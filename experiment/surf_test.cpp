#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;



void draw_match_knn(Mat img1, int minHessian){
   

    Ptr<SURF>  detector = SURF::create(minHessian, 4, 3, false);
    vector<KeyPoint> key_points_img1;
    detector->detect(img1, key_points_img1);
   
    
    //Mat descriptor_img1, descriptor_img2;
    //detector->compute(img1, key_points_img1, descriptor_img1);
    cout<<minHessian<<","<<key_points_img1.size()<<endl;
    
}

//Usage: <Img> <Angle> <Scale> <MatchNum>

int main(int argc, char const *argv[])
{
    
    string img_name = "/home/jiaming/Documents/research/img/test.png";
    
    Mat img = imread(img_name, IMREAD_COLOR);
    for (int i=50; i < 1001; i+=50){
    draw_match_knn(img, i); 
    }
    
    return 0;
}

