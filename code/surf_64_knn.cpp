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

bool isCorrect(Point2f p1, Point2f p2, double d = 10){
    return d > sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

double draw_match_knn(Mat img1, Mat img2, Mat affineMat, int angle, double scale, int matchNum = 50, bool extended = true){
   
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
    
    vector<DMatch> good_matches;

    double ratio = 0.8;

    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
        
    }
    
    sort(good_matches.begin(), good_matches.end());

    
   
 
    Mat img_matches = Mat::zeros(img1.rows, 2 * img1.cols, img1.type());
    Mat ROI = img_matches(Rect(0, 0, img1.cols, img1.cols));
    img1.copyTo(ROI);
    ROI = img_matches(Rect(img1.cols, 0, img1.cols, img1.rows));
    img2.copyTo(ROI);

    Point2f p1, p2, position;
    Point2f shift = Point2f(img1.cols, 0);

    int good = 0;

    for(unsigned int i = 0; i < min(matchNum, (int)good_matches.size()); ++i)
    {
        p1 = keyPoints_img1[good_matches[i].queryIdx].pt;
        p2 = keyPoints_img2[good_matches[i].trainIdx].pt;
        position = getPosition(p1, affineMat);
        if (isCorrect(p2, position))
        {
            line(img_matches, p1, p2+shift, Scalar(0, 255, 0));
            good++;
        }
            
        else
        {
            line(img_matches, p1, p2+shift, Scalar(0, 0, 255));
        }
            

    }

    double accuracy = good * 1.0 / matchNum;
    //cout<<accuracy<<endl;
    string ac = "Accuracy: ";
    string ag = "Angle: ";
    string sc = "Scale: ";
   // cout<< ac << accuracy <<endl;
    putText(img_matches, ac + to_string(accuracy) + " " + ag + to_string(angle) + " " + sc + to_string(scale), Point(100, 100), HersheyFonts::FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0));


   return accuracy;
}

double draw_match(Mat img1, Mat img2, Mat affineMat, int angle, double scale, int matchNum = 50, bool extended = false){
   
    
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
    
    vector<DMatch> good_matches;

    double ratio = 0.8;

    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
        
    }
    
    sort(good_matches.begin(), good_matches.end());

    
   
 
    Mat img_matches = Mat::zeros(img1.rows, 2 * img1.cols, img1.type());
    Mat ROI = img_matches(Rect(0, 0, img1.cols, img1.cols));
    img1.copyTo(ROI);
    ROI = img_matches(Rect(img1.cols, 0, img1.cols, img1.rows));
    img2.copyTo(ROI);

    Point2f p1, p2, position;
    Point2f shift = Point2f(img1.cols, 0);

    int good = 0;

    for(unsigned int i = 0; i < min(matchNum, (int)good_matches.size()); ++i)
    {
        p1 = keyPoints_img1[good_matches[i].queryIdx].pt;
        p2 = keyPoints_img2[good_matches[i].trainIdx].pt;
        position = getPosition(p1, affineMat);
        if (isCorrect(p2, position))
        {
            line(img_matches, p1, p2+shift, Scalar(0, 255, 0));
            good++;
        }
            
        else
        {
            line(img_matches, p1, p2+shift, Scalar(0, 0, 255));
        }
            

    }

    double accuracy = good * 1.0 / matchNum;
    //cout<<accuracy<<endl;
    string ac = "Accuracy: ";
    string ag = "Angle: ";
    string sc = "Scale: ";
   // cout<< ac << accuracy <<endl;
    putText(img_matches, ac + to_string(accuracy) + " " + ag + to_string(angle) + " " + sc + to_string(scale), Point(100, 100), HersheyFonts::FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0));


   return accuracy;
}

//Usage: <Img> <Angle> <Scale> <MatchNum>

int main(int argc, char const *argv[])
{
    if (argc != 5){
        cout<<"Usage: <Img> <Step> <Scale> <MatchNum>"<<endl;
        exit(0);
    }

    string folder = "/home/jiaming/research/img/";
    string img1_name = argv[1];
    int step = strtol(argv[2], NULL, 10);
    double scale = strtod(argv[3], NULL);
    int matchNum = strtol(argv[4], NULL, 10);
    Mat img1 = imread(folder + img1_name, IMREAD_COLOR);
  
    Mat pad = addPad(img1);
    Point2f center = Point2f(pad.cols / 2, pad.rows / 2);
    
    vector<double> vc_acc_knn;
    vector<double> vc_acc_normal;
    vector<double> mean_acc_knn;
    vector<double> mean_acc_normal;

for (int i = 0 ; i < 11; i++){
    scale = 0.5 + i * 0.1;
        for (int angle=0; angle<=360; angle += step){
            Mat affineMat = getRotationMatrix2D(center, angle, scale);
            Mat rot = rotateImg(pad, affineMat, center);
            double acc_knn = draw_match_knn(pad, rot, affineMat, angle, scale, matchNum);
            double acc_normal = draw_match(pad, rot, affineMat, angle, scale, matchNum);
            //cout<<"KNN: "<<acc_knn<<endl;
            //cout<<"Normal: "<<acc_normal<<endl;
            vc_acc_knn.push_back(acc_knn);
            vc_acc_normal.push_back(acc_normal);
    }
    double mean = accumulate(vc_acc_knn.begin(), vc_acc_knn.end(), 0.0) / vc_acc_knn.size();

    mean_acc_knn.push_back(mean);
    mean = accumulate(vc_acc_normal.begin(), vc_acc_normal.end(), 0.0) / (double)vc_acc_normal.size();
    mean_acc_normal.push_back(mean);
}

    for (int i = 0; i<mean_acc_knn.size(); i++){
        cout<<mean_acc_knn[i]<<", ";
        
    }
    cout<<endl;
    for (int i = 0; i<mean_acc_knn.size(); i++){
        cout<<mean_acc_normal[i]<<", ";
        
    }
    cout<<endl;


    
   
    /*
    imshow("match", matches_img);
    imshow("match_knn", matches_img_knn);
    imwrite("matches.png", matches_img);
    imwrite("knn_matches.png", matches_img_knn);
    waitKey(100000);
    */
    return 0;
}

