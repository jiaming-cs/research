#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


Mat drawMatch(Mat img1, Mat img2, bool extended = true){
   
    int minHessian=400;
    Ptr<SURF>  detector = SURF::create(minHessian, 4, 3, extended);
    vector<KeyPoint>keyPoints_img1, keyPoints_img2;
    detector->detect(img1, keyPoints_img1);
    detector->detect(img2, keyPoints_img2);
 
    
    Mat descriptor_img1, descriptor_img2;
    detector->compute(img1, keyPoints_img1, descriptor_img1);
    detector->compute(img2, keyPoints_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> goodMatches;

    double ratio = 0.8;

    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            goodMatches.push_back(matches[i][0]);
        }
    }
    
    sort(goodMatches.begin(), goodMatches.end());

    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
 
    
    for(unsigned int i = 0; i < min(30, (int)goodMatches.size()); ++i)
    {
        img1_points.push_back(keyPoints_img1[goodMatches[i].queryIdx].pt);
        img2_points.push_back(keyPoints_img2[goodMatches[i].trainIdx].pt);
    }

    Mat img_matches;

    drawMatches(img1, keyPoints_img1, img2, keyPoints_img2, goodMatches, img_matches);
  
    Mat H = findHomography(img1_points, img2_points, RANSAC);
    
    vector<Point2f> img1_corners(4);
    img1_corners[0]=Point2f(0,0);
    img1_corners[1]=Point2f(img1.cols,0);
    img1_corners[2]=Point2f(img1.cols, img1.rows);
    img1_corners[3]=Point2f(0,img1.rows);
    
    vector<Point2f> img2_corners(4);
    cout<<format( H, Formatter::FMT_PYTHON)<<endl;
    if (!H.empty()){
        
        perspectiveTransform(img1_corners, img2_corners, H);
        
        line(img_matches, img2_corners[0]+Point2f(img1.cols,0), img2_corners[1] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[1]+Point2f(img1.cols,0), img2_corners[2] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[2]+Point2f(img1.cols,0), img2_corners[3] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[3]+Point2f(img1.cols,0), img2_corners[0] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);

    }
       
    return img_matches;
  
}

int main()
{
    string folder = "/home/jiaming/research/img/";
    string img1_name = "pencil_bag.png";
    string img2_name = "large.png";
    Mat img1 = imread(folder + img1_name, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name, IMREAD_COLOR);
    Mat out = drawMatch(img1, img2);
    imshow("Matches", out);
    waitKey(0);
    return 0;
}

