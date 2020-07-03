
#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

int get_kp_num(Mat img, int minHessian, bool extended = false){
    SurfFeatureDetector  detector(minHessian, 4, 3, extended);
    vector<KeyPoint>key_points_img;
    detector.detect(img, key_points_img);
    
    return key_points_img.size();
}

int main(int argc, char const *argv[])
{

    string img_loc = "/home/jiaming/opencv24/img/img1.png";
    Mat img1 = imread(img_loc, IMREAD_COLOR);
    
    fstream file;
    file.open("out.csv", ios::out);
    for(int i=5; i<1000; i+=2){
        file<<i<<","<<get_kp_num(img1, i)<<endl;
    }
    file.close();
    return 0;
}

