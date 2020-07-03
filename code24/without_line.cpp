#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;




void drawMatch(Mat& img1, Mat& img2, Mat& img_matches, double ratio = 0.80, bool extended = false){
   
    int minHessian=400;
    SurfFeatureDetector  detector(minHessian, 4, 3, false);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector.detect(img1, key_points_img1);
    detector.detect(img2, key_points_img2);

    while(key_points_img1.size() < 200 && minHessian > 5){
        minHessian /= 2;
        SurfFeatureDetector detector(minHessian, 4, 3, false);
        detector.detect(img1, key_points_img1);
        detector.detect(img2, key_points_img2);    
    }
    
    if(key_points_img1.size() < 4 || key_points_img2.size() < 4){
        return ;
    }

    Mat descriptor_img1, descriptor_img2;
    detector.compute(img1, key_points_img1, descriptor_img1);
    detector.compute(img2, key_points_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    if(descriptor_img1.empty() || descriptor_img2.empty()){
        return ;
    }
    map<int, DMatch> matches_map;
   
    if (matches.size() < 30){
        ratio = 0.9;
    }

    for(int i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            if(matches_map.count(matches[i][0].trainIdx) == 0){
                matches_map[matches[i][0].trainIdx] = matches[i][0];
            }
            else{
                if (matches_map[matches[i][0].trainIdx].distance > matches[i][0].distance){
                    matches_map[matches[i][0].trainIdx] = matches[i][0];
                }
            }
        }
    }
      

    for (map<int, DMatch>::iterator it = matches_map.begin(); it != matches_map.end(); it++){
        good_matches.push_back(it->second);
    }

    if (good_matches.size() < 4){
        return ;
    }

    
    sort(good_matches.begin(), good_matches.end());


    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
    
    int num_good = min((int)good_matches.size(),100);
    vector<DMatch> first_matches;
    
    for(unsigned int i = 0; i < num_good; ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
        first_matches.push_back(good_matches[i]);
    }

    

   

    //drawMatches(img1, key_points_img1, img2, key_points_img2, first_matches, img_matches);
    
    //cv::putText(img_matches, to_string(key_points_img1.size()) + "   " + to_string(key_points_img2.size()), Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0) );
    Mat H = findHomography(img1_points, img2_points, RANSAC, 10);
    
    vector<Point2f> img1_cross(5);
    Point2f middle = Point2f(img1.cols/2, img1.rows/2);
    img1_cross[0]=middle+Point2f(0,-50);
    img1_cross[1]=middle+Point2f(0, 50);
    img1_cross[2]=middle+Point2f(-50, 0);
    img1_cross[3]=middle+Point2f(50, 0);
    img1_cross[4]=middle;
    vector<Point2f> img2_cross(5);
   
    if (!H.empty()){
        
        perspectiveTransform(img1_cross, img2_cross, H);

        line(img1, img1_cross[0], img1_cross[1], Scalar(0, 255, 0), 2);
        line(img1, img1_cross[2], img1_cross[3], Scalar(0, 255, 0), 2);
        
        line(img2, img2_cross[0], img2_cross[1], Scalar(0, 255, 0), 2);
        line(img2, img2_cross[2], img2_cross[3], Scalar(0, 255, 0), 2);
        circle(img2, img2_cross[4], 5, Scalar(0, 0, 255), 2);
    }
       
    return ;
  
}

 
int main()
{
	
	VideoCapture cp1, cp2;
	Mat frame;
 
	cp1.open("http://admin:admin@10.10.10.101:8081");
	cp2.open("http://admin:admin@10.10.10.115:8081");
    Mat frame1, frame2, out1, out2;
	int index = 0;
    while (1)
	{

		cp1 >> frame1;
        frame1 = frame1(Rect(0, 50, frame1.cols, frame1.rows-50));
        cp2 >> frame2;
        frame2 = frame2(Rect(0, 50, frame2.cols, frame2.rows-50));

        drawMatch(frame1, frame2, out1);
      
        imshow("Reference",frame1);

        
        imshow("Template", frame2);
        
        //imwrite("/home/jiaming/opencv24/code24/img1/"+to_string(index)+".png", frame1);
        //imwrite("/home/jiaming/opencv24/code24/img2/"+to_string(index)+".png", frame2);
        //imwrite("/home/jiaming/opencv24/code24/out_frames/"+to_string(index)+".png", out);
        
        //imshow("Img1", frame1);
        //imshow("Img2", frame2);
		waitKey(30);
	}
 
	return 0;

}
