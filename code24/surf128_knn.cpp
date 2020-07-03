
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

using namespace cv;
using namespace std;

Mat drawMatch(Mat img1, Mat img2, double ratio = 0.8, bool extended = true){
   
    int minHessian=400;
    SurfFeatureDetector  detector(minHessian, 4, 3, extended);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector.detect(img1, key_points_img1);
    detector.detect(img2, key_points_img2);
 
    
    Mat descriptor_img1, descriptor_img2;
    detector.compute(img1, key_points_img1, descriptor_img1);
    detector.compute(img2, key_points_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;


    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }
    
    sort(good_matches.begin(), good_matches.end());

    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
 
    
    for(unsigned int i = 0; i < min(30, (int)good_matches.size()); ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
    }

    Mat img_matches;

    drawMatches(img1, key_points_img1, img2, key_points_img2, good_matches, img_matches);
  
    Mat H = findHomography(img1_points, img2_points, RANSAC);
    
    vector<Point2f> img1_corners(4);
    img1_corners[0]=Point2f(0,0);
    img1_corners[1]=Point2f(img1.cols,0);
    img1_corners[2]=Point2f(img1.cols, img1.rows);
    img1_corners[3]=Point2f(0,img1.rows);
    
    vector<Point2f> img2_corners(4);
    //cout<<format( H, Formatter::FMT_PYTHON)<<endl;
    if (!H.empty()){
        
        perspectiveTransform(img1_corners, img2_corners, H);
        
        line(img_matches, img2_corners[0]+Point2f(img1.cols,0), img2_corners[1] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[1]+Point2f(img1.cols,0), img2_corners[2] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[2]+Point2f(img1.cols,0), img2_corners[3] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[3]+Point2f(img1.cols,0), img2_corners[0] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);

    }
       
    return img_matches;
  
}

int main(int argc, char const *argv[])
{

    string folder = "/home/jiaming/opencv24/";
    string img1_name = "pencil_bag.png";
    string img2_name = "large.png";

    Mat img1 = imread(folder + img1_name, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name, IMREAD_COLOR);
    time_t start, end;
    start = clock();
    Mat img_match = drawMatch(img1, img2);
    end = clock();
    cout<<"Run time for surf128: "<< (double)(end - start) / CLOCKS_PER_SEC<<endl;
    imshow("Matches", img_match);
    waitKey(0);
    return 0;
}

