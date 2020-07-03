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
        cout<<"RHO"<<endl;
        return h2;
    }
        
    else{
        cout<<"RANSAC"<<endl;
        return h1;
    }
        
     
}

void draw_match_double(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SURF> &detector1,
                Ptr<SURF> &detector2,
                int max_hessian = 800,
                int min_hessian = 5,
                int max_size = 300,
                int min_size = 200,
                int good_match_num = 50, 
                bool extended = false){
    
                    
    vector<KeyPoint>key_points_img1, key_points_img2;
    time_t start, end;
    
    start = clock();
    detector1->detect(img1, key_points_img1);
    end = clock();
    cout << "Detect KeyPoints:" << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl;


    int hessian = detector1->getHessianThreshold();
    //cout << "Threshold_start:" << detector->hessianThreshold <<endl;

    start = clock();
   
    if ( key_points_img1.size() < min_size && hessian > min_hessian){
        
        hessian = max((int)(hessian/1.5), min_hessian);
        //detector1.release();
        //start = clock();
        if (hessian != min_hessian)
        detector1->setHessianThreshold(hessian);
        //end = clock();
        //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
        
    }

    else if (key_points_img1.size() > max_size)
    {
        sort(key_points_img1.begin(), key_points_img1.end(), compare_hessian);
        key_points_img1.resize(max_size);
        //hessian = (int)(hessian * 1.5 + rand()%10); // In case stuck
        hessian = (int)(hessian * 1.5 );  
        //detector1.release();
        //start = clock();
        detector1->setHessianThreshold(hessian);
    }

    end = clock();

    cout<< "Time for Find Right Hessain reshould " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    
    //cout<<"Hessian1: "<<detector1->hessianThreshold<<endl;

    detector2->detect(img2, key_points_img2);

    hessian = detector2->getHessianThreshold();

    
    if ( key_points_img2.size() < min_size && hessian > min_hessian ){
        hessian = max((int)(hessian/1.5), min_hessian);
        //detector2.release();
        //start = clock();
        if (hessian != min_hessian)
        detector2->setHessianThreshold(hessian);
        //end = clock();
        //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
        
    }

    else if (key_points_img2.size() > max_size)
    {
        sort(key_points_img2.begin(), key_points_img2.end(), compare_hessian);
        key_points_img2.resize(max_size);
        //hessian = (int)(hessian * 1.5 + rand()%10); // In case stuck
        hessian = (int)(hessian * 1.5 );
        //detector2.release();
        //start = clock();
        detector2->setHessianThreshold(hessian);
        
    }
        
    

    //cout<<"Hessian2: "<<detector2->hessianThreshold<<endl;
    //cout<<"Size: "<<key_points_img2.size()<<endl;
    
    //cout << "Threshold_end:" << detector->hessianThreshold <<endl;
    if(key_points_img1.size() < 4 || key_points_img2.size() < 4){
        return ;
    }

    Mat descriptor_img1, descriptor_img2;

    start = clock();
    detector1->compute(img1, key_points_img1, descriptor_img1);
    end = clock();

    cout<< "Time for computing descriptor " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    
    detector2->compute(img2, key_points_img2, descriptor_img2);
    

    BFMatcher matcher;
    vector<vector<DMatch>>matches;

    start = clock();
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    end = clock();

    cout<< "Time for Match descriptor " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    
    vector<DMatch> good_matches;

    if(descriptor_img1.empty() || descriptor_img2.empty()){
        return ;
    }
    
    
    map<int, pair<DMatch, double>> matches_map;  // int->index of keypoint from image1.  DMatch->closest pair.  double->closest / second closest
   
    //double ratio = max_ratio  - (max_ratio - min_ratio) * ((key_points_img1.size() - min_size) * 1.0 / (max_size - min_size));
    //cout<<"ratio: "<<ratio<<endl;

    start = clock();

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
    
    end = clock();

    cout<< "Time for finding good matches " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    

    for (vector<pair<DMatch, double>>::iterator it = matches_and_ratio.begin(); it != matches_and_ratio.end(); it++){
        good_matches.push_back(it->first);
    }

    if (good_matches.size() < 4){
        return ;
    }

    //cout<<"Number of vaild point: "<< good_matches.size()<<endl;
    
   


    
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
    
    cv::putText(img_matches, to_string(key_points_img1.size()) + "   " + to_string(key_points_img2.size()), Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0) );
   


    start = clock();
    Mat H1 = findHomography(img1_points, img2_points, RANSAC, 10);
    end = clock();

    cout<< "Time for RANSAC " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    

    start = clock();
    Mat H2 = findHomography(img1_points, img2_points, RHO, 10);
    end = clock();
    
    cout<< "Time for RHO " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    
    //Mat H2 = findHomography(img1_points, img2_points, RANSAC, 5);
    //Mat H3 = findHomography(img1_points, img2_points, RANSAC, 3);

    start = clock();
    Mat H = get_better_h(img1_points, img2_points, H1, H2);
    end = clock();
    cout<< "Time for finding a better H matrix " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    
    //H = get_better_h(img1_points, img2_points, H, H3);
    
    //Mat H = findHomography(img1_points, img2_points, RANSAC, 10);
    
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

        line(img_matches, img1_cross[0], img1_cross[1], Scalar(0, 255, 0), 2);
        line(img_matches, img1_cross[2], img1_cross[3], Scalar(0, 255, 0), 2);
        
        line(img_matches, img2_cross[0]+Point2f(img1.cols, 0), img2_cross[1]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_cross[2]+Point2f(img1.cols, 0), img2_cross[3]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        circle(img_matches, img2_cross[4]+Point2f(img1.cols, 0), 5, Scalar(0, 0, 255), 2);
    }

    return ;
  
}


void draw_match_single(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SURF> &detector1,
                Ptr<SURF> &detector2,
                int max_hessian = 800,
                int min_hessian = 5,
                int max_size = 300,
                int min_size = 200,
                int good_match_num = 50, 
                bool extended = false){
    
                    
    vector<KeyPoint>key_points_img1, key_points_img2;
    
    time_t start, end;
 
    detector1->detect(img1, key_points_img1);

    int hessian = detector1->getHessianThreshold();
    //cout << "Threshold_start:" << detector->hessianThreshold <<endl;

    while ((key_points_img1.size() < min_size || key_points_img1.size() > max_size) ){
        if ( key_points_img1.size() < min_size ){
            if (hessian <= min_hessian)
                break;
            hessian = (int)(hessian/1.5), min_hessian;
            //detector1.release();
            //start = clock();
            detector1->setHessianThreshold(hessian);
            //end = clock();
            //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
            detector1->detect(img1, key_points_img1);
        }
        else
        {
            sort(key_points_img1.begin(), key_points_img1.end(), compare_hessian);
            key_points_img1.resize(max_size);
            //hessian = (int)(hessian * 1.5 + rand()%10); // In case stuck
            hessian = (int)(hessian * 1.5 );
            //detector1.release();
            //start = clock();

            detector1->setHessianThreshold(hessian);
            break;
        }
        
    }

    //cout<<"Hessian1: "<<detector1->hessianThreshold<<endl;

    detector2->detect(img2, key_points_img2);

    hessian = detector2->getHessianThreshold();

    while ((key_points_img2.size() < min_size || key_points_img2.size() > max_size) && hessian >= min_hessian){
        if ( key_points_img2.size() < min_size ){
            if (hessian <= min_hessian)
                break;
            hessian = (int)(hessian/1.5);
           // detector2.release();
            //start = clock();
            detector2->setHessianThreshold(hessian);
            //end = clock();
            //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
            detector2->detect(img2, key_points_img2);
        }

        else
        {
            sort(key_points_img2.begin(), key_points_img2.end(), compare_hessian);
            key_points_img2.resize(max_size);
            //hessian = (int)(hessian * 1.5 + rand()%10); // In case stuck
            hessian = (int)(hessian * 1.5 );
           // detector2.release();
            //start = clock();
            detector2->setHessianThreshold(hessian);
            break;
        }
        
    }

    //cout<<"Hessian2: "<<detector2->hessianThreshold<<endl;
    //cout<<"Size: "<<key_points_img2.size()<<endl;
    
    //cout << "Threshold_end:" << detector->hessianThreshold <<endl;
    if(key_points_img1.size() < 4 || key_points_img2.size() < 4){
        return ;
    }

    Mat descriptor_img1, descriptor_img2;
    detector1->compute(img1, key_points_img1, descriptor_img1);
    detector2->compute(img2, key_points_img2, descriptor_img2);
    

    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    if(descriptor_img1.empty() || descriptor_img2.empty()){
        return ;
    }
    
    
    map<int, pair<DMatch, double>> matches_map;  // int->index of keypoint from image1.  DMatch->closest pair.  double->closest / second closest
   
    //double ratio = max_ratio  - (max_ratio - min_ratio) * ((key_points_img1.size() - min_size) * 1.0 / (max_size - min_size));
    //cout<<"ratio: "<<ratio<<endl;

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

    if (good_matches.size() < 4){
        return ;
    }

    //cout<<"Number of vaild point: "<< good_matches.size()<<endl;
    
   


    
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
    
    cv::putText(img_matches, to_string(key_points_img1.size()) + "   " + to_string(key_points_img2.size()), Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0) );
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

        line(img_matches, img1_cross[0], img1_cross[1], Scalar(0, 255, 0), 2);
        line(img_matches, img1_cross[2], img1_cross[3], Scalar(0, 255, 0), 2);
        
        line(img_matches, img2_cross[0]+Point2f(img1.cols, 0), img2_cross[1]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_cross[2]+Point2f(img1.cols, 0), img2_cross[3]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        circle(img_matches, img2_cross[4]+Point2f(img1.cols, 0), 5, Scalar(0, 0, 255), 2);
    }

    return ;
  
}


int main(int argc, char const *argv[])
{


    
    
    string folder = "/home/jiaming/Documents/research/code24/img";
    string img1_name = "1/";
    string img2_name = "2/";
    string extention = ".png";
    int i=0;
    Mat out1, out2;

    Ptr<SURF> detector_a = SURF::create(200);
    Ptr<SURF> detector_b = SURF::create(200);
    Ptr<SURF> detector1 = SURF::create(200);
    Ptr<SURF> detector2 = SURF::create(200);
    
    time_t start, end;
    while (i<115)
    {

    Mat img1 = imread(folder + img1_name + to_string(i) + extention, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name + to_string(i) + extention, IMREAD_COLOR);
 
    start = clock();
    draw_match_double(img1, img2, out1, detector_a, detector_b);
    end = clock();
    cout<< "Time for Double H " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    
    if (!out1.empty())
        imshow("Double", out1);

    start = clock();
    draw_match_single(img1, img2, out2, detector1, detector2);
    end = clock();
    
    cout<< "Time for Single H " << (end - start) * 1.0 / CLOCKS_PER_SEC <<endl; 
    

    if (!out2.empty())
        imshow("Single", out2);

    waitKey(0);

    i++;
    }
    
   

    return 0;

}

