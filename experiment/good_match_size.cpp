// test frame 83 84


#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <map>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

bool compare_hessian(const KeyPoint &kp1, const KeyPoint &kp2) {
    return kp1.response > kp2.response;
}

bool compare_ratio(const pair<DMatch, double> &p1, const pair<DMatch, double> &p2){
    return p1.second < p2.second;
}

Point2f get_position(const Point2f &p, const Mat& affineMat){
    double x = p.x * affineMat.at<double>(0, 0) + p.y * affineMat.at<double>(0, 1) + affineMat.at<double>(0, 2);
    double y = p.x * affineMat.at<double>(1, 0) + p.y * affineMat.at<double>(1, 1) + affineMat.at<double>(1, 2);
    return Point2f(x, y);
}

double get_distance(const Point2f &p1, const Point2f& p2){
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

Mat get_better_h(const vector<Point2f> &img1_points, const vector<Point2f> &img2_points, Mat h1, Mat h2){
    double error1 = 0;
    double error2 = 0;
    Point2f p1, p2;

    for(int i = 0 ; i < img1_points.size(); i++){
        p1 = get_position(img1_points[i], h1);
        p2 = get_position(img1_points[i], h2);
        error1 += get_distance(p1, img2_points[i]);
        error2 += get_distance(p2, img2_points[i]);
    }
    
    //cout<<"error1:  "<<error1<<endl<<"error2:  "<<error2<<endl;
    if (error1 > error2){
        //cout<<"LEMED"<<endl;
        return h2;
    }
        
    else{
        //cout<<"RANSAC"<<endl;
        return h1;
    }
        
     
}


vector<Point2f> draw_match_double(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SURF> &detector1,
                Ptr<SURF> &detector2,
                int max_size = 300,
                int min_size = 200,
                int max_hessian = 800,
                int min_hessian = 5,
                int good_match_num = 50, 
                bool extended = false){
    
                    
    vector<KeyPoint>key_points_img1, key_points_img2;

 
    detector1->detect(img1, key_points_img1);

    int hessian = detector1->getHessianThreshold();


    while ((key_points_img1.size() < min_size || key_points_img1.size() > max_size) ){
        if ( key_points_img1.size() < min_size ){
            if (hessian <= min_hessian)
                break;
            hessian = (int)(hessian/1.5), min_hessian;
            detector1->setHessianThreshold(hessian);
            detector1->detect(img1, key_points_img1);
        }
        else
        {
            sort(key_points_img1.begin(), key_points_img1.end(), compare_hessian);
            key_points_img1.resize(max_size);
            hessian = (int)(hessian * 1.5 );  
            detector1->setHessianThreshold(hessian);
            break;
        }
    }

    detector2->detect(img2, key_points_img2);

    hessian = detector2->getHessianThreshold();

    while ((key_points_img2.size() < min_size || key_points_img2.size() > max_size) && hessian >= min_hessian){
        if ( key_points_img2.size() < min_size ){
            if (hessian <= min_hessian)
                break;
            hessian = (int)(hessian/1.5);
            detector2->setHessianThreshold(hessian);
            detector2->detect(img2, key_points_img2);
        }

        else
        {
            sort(key_points_img2.begin(), key_points_img2.end(), compare_hessian);
            key_points_img2.resize(max_size);
            hessian = (int)(hessian * 1.5 );
            detector2->setHessianThreshold(hessian);
            break;
        }
        
    }



    Mat descriptor_img1, descriptor_img2;

    detector1->compute(img1, key_points_img1, descriptor_img1);
    detector2->compute(img2, key_points_img2, descriptor_img2);
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
   
    vector<DMatch> good_matches;

   
    
    
    map<int, pair<DMatch, double>> matches_map;

   

    for(int i = 0; i < matches.size(); i++)
    {
        double ratio = matches[i][0].distance / matches[i][1].distance;
        if(matches_map.count(matches[i][0].trainIdx) == 0){

            pair<DMatch, double> p(matches[i][0], ratio);
            matches_map[matches[i][0].trainIdx] = p;
        }

        else{

            if (matches_map[matches[i][0].trainIdx].first.distance > matches[i][0].distance){
                pair<DMatch, double> p(matches[i][0], ratio);
                matches_map[matches[i][0].trainIdx] = p;
            }
        }
    }

    vector<pair<DMatch, double>> matches_and_ratio;

    for (map<int, pair<DMatch, double>>::iterator it =matches_map.begin(); it != matches_map.end(); it++){
        matches_and_ratio.push_back(it->second);
    }     

    sort(matches_and_ratio.begin(), matches_and_ratio.end(), compare_ratio);

    if (good_match_num < matches_and_ratio.size()){
        matches_and_ratio.resize(good_match_num);
    }
 

    for (vector<pair<DMatch, double>>::iterator it = matches_and_ratio.begin(); it != matches_and_ratio.end(); it++){
        good_matches.push_back(it->first);
    }



    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
    
    int num_good = min((int)good_matches.size(), (int)good_matches.size());
    vector<DMatch> first_matches;

    for(unsigned int i = 0; i < num_good; ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
        first_matches.push_back(good_matches[i]);
    }

    drawMatches(img1, key_points_img1, img2, key_points_img2, first_matches, img_matches);

    Mat H = findHomography(img1_points, img2_points, RHO);
 
    // vector<Point2f> img1_cross(5);
    // Point2f middle = Point2f(img1.cols/2, img1.rows/2);
    // img1_cross[0]=middle+Point2f(0,-50);
    // img1_cross[1]=middle+Point2f(0, 50);
    // img1_cross[2]=middle+Point2f(-50, 0);
    // img1_cross[3]=middle+Point2f(50, 0);
    // img1_cross[4]=Point(438, 182);
    // vector<Point2f> img2_cross(5);
   

    vector<Point2f> img1_cross(10);
    img1_cross[0]=Point2f(213, 153);
    img1_cross[1]=Point2f(262, 154);
    img1_cross[2]=Point2f(227, 175);
    img1_cross[3]=Point2f(259, 176);
    img1_cross[4]=Point2f(188, 181);
    img1_cross[5]=Point2f(289, 202);
    img1_cross[6]=Point2f(189, 235);
    img1_cross[7]=Point2f(291, 257);
    img1_cross[8]=Point2f(201, 272);
    img1_cross[9]=Point2f(339, 231);
    
   

    vector<Point2f> img2_cross(10);
    
    for (int i=0; i< 10; i++){

        img2_cross[i] = Point2f(0, 0);
    }


    if (!H.empty()){
        
        perspectiveTransform(img1_cross, img2_cross, H);

    }

    return img2_cross;
  
}

void func(int min_size){

    string folder = "/home/jiaming/Documents/research/img/";
    string img1_name = "v1/";
    string img2_name = "v2/";
    string extention = ".png";
    int i=1;
    Mat out;

    Ptr<SURF> detector_a = SURF::create(800);
    Ptr<SURF> detector_b = SURF::create(800);
    vector<Point2f> central_point;
    time_t start, end;
    start = clock();
    Point2f p_zero = Point2f(0, 0);
    while (i<101)
    {

    Mat img1 = imread(folder + img1_name + to_string(i) + extention, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name + to_string(i) + extention, IMREAD_COLOR);
    vector<Point2f> prediction = draw_match_double(img1, img2, out, detector_a, detector_b);
    
    vector<Point2f> ground_true(10);
    ground_true[0]=Point2f(283, 131);
    ground_true[1]=Point2f(323, 132);
    ground_true[2]=Point2f(298, 149);
    ground_true[3]=Point2f(326, 150);
    ground_true[4]=Point2f(262, 154);
    ground_true[5]=Point2f(349, 172);
    ground_true[6]=Point2f(262, 198);
    ground_true[7]=Point2f(350, 218);
    ground_true[8]=Point2f(274, 230);
    ground_true[9]=Point2f(339, 231);
        //central_point.push_back(p);
    double total = 0;
    for (int i= 0; i<10; i++){
        total += get_distance(prediction[i], ground_true[i]);

    } 
    cout<< i << ","<< total/10 <<endl;
        i++;
    }


}
int main(int argc, char const *argv[])
{


    

    func(50);


    return 0;

}

