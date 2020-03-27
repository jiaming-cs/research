#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric> //accumulate
#include <fstream>
#include <cmath>
#include <climits> //INT_MAX
#include <opencv2/core/persistence.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


Mat addPad(Mat img){
    int length = max(img.cols, img.rows);
    Mat shiftImg = Mat::zeros(2 * length, 2 * length, img.type());
    Mat ROI = shiftImg(Rect(length - img.cols / 2, length - img.rows / 2, img.cols, img.rows));
    img.copyTo(ROI);
    return shiftImg;
}

Point2f getPosition(Point2f p, Mat affineMat){
    double x = p.x * affineMat.at<double>(0, 0) + p.y * affineMat.at<double>(0, 1) + affineMat.at<double>(0, 2);
    double y = p.x * affineMat.at<double>(1, 0) + p.y * affineMat.at<double>(1, 1) + affineMat.at<double>(1, 2);
    return Point2f(x, y);
}


Mat rotateImg(Mat img, Mat affineMat, Point2f center){
    
    
    Mat out;
    //cout<<format(affineMat, Formatter::FMT_PYTHON)<<endl;
    warpAffine(img, out, affineMat, img.size() , INTER_LINEAR, BORDER_TRANSPARENT); 
    return out;
}

bool isCorrect(Point2f p1, Point2f p2, double d = 25){
    return d > sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

double get_distance(Point2f p1, Point2f p2){
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

Mat draw_match(Mat img1, Mat img2, Mat affineMat, int angle, double scale, int matchNum , bool extended = true){
   
    int minHessian = matchNum;
    Ptr<SURF>  detector = SURF::create(minHessian, 4, 3, extended);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector->detect(img1, key_points_img1);
    detector->detect(img2, key_points_img2);

    /*
    while (key_points_img1.size() < matchNum && minHessian >25)
    {
        minHessian -= 100;
        key_points_img1.clear();
        key_points_img2.clear();
        detector = SURF::create(minHessian, 4, 3, extended);
        detector->detect(img1, key_points_img1);
        detector->detect(img2, key_points_img2);
    
    }
    */
    
    Mat descriptor_img1, descriptor_img2;
    detector->compute(img1, key_points_img1, descriptor_img1);
    detector->compute(img2, key_points_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    double ratio = 0.8;

    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
        
    }
    
    //sort(goodMatches.begin(), goodMatches.end());

    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
 
    
    for(unsigned int i = 0; i < (int)good_matches.size(); ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
    }

    
   
 
    Mat img_matches = Mat::zeros(img1.rows, 2 * img1.cols, img1.type());
    Mat ROI = img_matches(Rect(0, 0, img1.cols, img1.cols));
    img1.copyTo(ROI);
    ROI = img_matches(Rect(img1.cols, 0, img1.cols, img1.rows));
    img2.copyTo(ROI);

    Point2f p, position_predict, position_real;
    Point2f shift = Point2f(img1.cols, 0);

    int good = 0;


    Mat H = findHomography(img1_points, img2_points, RANSAC, 10);
    
    double d_sum = 0;
    for(unsigned int i = 0; i < (int)good_matches.size(); ++i)
    {
        p = key_points_img1[good_matches[i].queryIdx].pt;
        position_predict = getPosition(p, H);
        position_real = getPosition(p, affineMat);
        d_sum += get_distance(position_predict, position_real);
        if (isCorrect(position_predict, position_real))
        {
            line(img_matches, p, position_predict+shift, Scalar(0, 255, 0));
            good++;
        }
            
        else
        {
            line(img_matches, p, position_predict+shift, Scalar(0, 0, 255));
        }
            

    }
    double mean_distance = d_sum / (int)good_matches.size() ;
    double accuracy = good * 1.0 / (int)good_matches.size() ;
    string ac = "Accuracy: ";
    string ag = "Angle: ";
    string sc = "Scale: ";
    cout<< "Threshold: " << minHessian << endl;
    cout<< "Size: " << key_points_img1.size() << endl;
    cout<< "Mean Distance: " << mean_distance << endl;
    cout<< ac << accuracy <<endl;
    putText(img_matches, ac + to_string(accuracy) + " " + ag + to_string(angle) + " " + sc + to_string(scale), Point(100, 100), HersheyFonts::FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0));


   return img_matches;
}


//Usage: <ImgFoder> <ImgNum1> <ImgNum2> <Weight> <MatchNum>

int main(int argc, char const *argv[])
{
   if (argc != 5){
        cout<<"Usage: <Img> <Angle> <Scale> <MatchNum>"<<endl;
        exit(0);
    }

    string folder = "/home/jiaming/Documents/research/img/";
    string img1_name = argv[1];
    int angle = strtol(argv[2], NULL, 10);
    double scale = strtod(argv[3], NULL);
    int matchNum = strtol(argv[4], NULL, 10);
    Mat img1 = imread(folder + img1_name, IMREAD_COLOR);
  
    Mat pad = addPad(img1);
    Point2f center = Point2f(pad.cols / 2, pad.rows / 2);
    Mat affineMat = getRotationMatrix2D(center, angle, scale);
    Mat rot = rotateImg(pad, affineMat, center);
    
    for(int i=25; i < 800; i+=25)
        Mat matchesImg = draw_match(pad, rot, affineMat, angle, scale, i);
   
    return 0;
}

