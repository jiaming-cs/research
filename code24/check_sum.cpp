#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

Mat diff(Mat img1, Mat img2){
    Mat out = Mat::zeros(img1.size(), img1.type());
    for (int i=0; i<img1.rows; i++){
        for (int j=0; j<img1.cols; j++){
            unsigned p1 = img1.at<uchar>(j, i);
            unsigned p2 = img2.at<uchar>(j, i);
            unsigned d = abs((int)p1 - (int)p2);
            cout<<d<<endl;
            out.at<uchar>(j, i) =  d;
            }
        }
    return out;
}

long long get_difference(Mat& img1, Mat& img2){
    long long sum = 0;
    int num = 0;
    
    for (int i=0; i<img1.rows; i++){
        for (int j=0; j<img1.cols; j++){
            unsigned p1 = img1.at<uchar>(j, i);
            unsigned p2 = img2.at<uchar>(j, i);
            int d = abs((int)p1 - (int)p2); 
            sum += d;
            if(d!=0){
                cout<<j<<", "<< i<<endl;
                cout<<p1<<" "<<p2<<endl;
                num++;
                //cout<<"img1: "<< (int)img1.at<uchar>(j, i)<<endl;
                //cout<<"img2: "<< (int)img2.at<uchar>(j, i)<<endl;
                
                //circle(img2, Point(j, i), 5, Scalar(0), 2);
            }

        }
    }
    cout<<"NUM "<<num<<endl;
}

int main(int argc, char const *argv[])
{
    Mat img1 = imread("/home/jiaming/opencv24/code24/img2/83.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("/home/jiaming/opencv24/code24/img2/84.png", IMREAD_GRAYSCALE);
    
    //circle(img1, Point(471, 428), 10, Scalar(0), 2);
    //circle(img2, Point(471, 428), 10, Scalar(0), 2);
    Mat out;
    absdiff(img1, img2, out);
    Scalar ss = sum(out);
    cout<<ss[0]<<endl;
    
    for (int i=0; i<out.rows; i++){
        for (int j=0; j<out.cols; j++){
            if(out.at<uchar>(i, j)> 10){
            cout<<(int)out.at<uchar>(i, j)<<endl;
            out.at<uchar>(i, j) = 255;
            }
            }
        }
    
    imshow("Differences", out);
    waitKey(0);
    return 0;
}
