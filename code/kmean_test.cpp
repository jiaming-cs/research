#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    string folder = "/home/jiaming/research/img/";
    string img1_name = "pencil_bag.png";
    string img2_name = "large.png";
    Mat img1 = imread(folder + img1_name, IMREAD_GRAYSCALE);
    Mat img2 = imread(folder + img2_name, IMREAD_COLOR);
    Mat lb;
    img1.convertTo(img1, CV_32F);
    TermCriteria tc = TermCriteria(TermCriteria::EPS, 100, 0.1);
    kmeans(img1, 3, lb, tc, 1, KMEANS_RANDOM_CENTERS);
    for (int i = 0; i < img1.rows; i++)
    {
        for (int j = 0; j < img1.cols; j++)
        {
            if (lb.at<uchar>(i, j) == 0){
                img1.at<float>(i, j) = 0;
            }
            else if (lb.at<uchar>(i, j) == 1){
                img1.at<float>(i, j) = 50;
            }
            else{
                img1.at<float>(i, j) = 100;
            }
        }
        
        
    }
    imshow("Kmeans", img1);
    waitKey(0);
    
    return 0;
}
