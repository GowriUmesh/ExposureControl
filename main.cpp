/**
 * Authors:   Gowri Umesh <mailgowriumesh@gmail.com>
              Abigail Pop  
 * Created:   24.Jun.2020
 * Version:   0.1
 *
 * Description:This header finds the shortest path between
 * a start cell and an end cell of a given 3D grid
 * The co


#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include<numeric>
#include<math.h>


using namespace cv;
using namespace std;

char const *window_name[16] ={"1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"};

void readImagesandTimes(vector<Mat> &images,vector<double> &exposure_times)
{

    vector<String> img;

    vector<double> times={30,15,8,4,2,1,1/2,1/4,1/8,1/15,1/30,1/60,1/152,1/250,1/500,1/1000};
    glob("../images/Memorial_SourceImages/memorial*.png", img, false);
    size_t count = img.size();
    for (size_t i=0; i<count; i++)
    {
        images.push_back(imread(img[i]));
        exposure_times.push_back(times[i]);

    }
    for(int i=0;i<images.size();i++)
    {
        cv::resize(images[i], images[i], cv::Size(512,768), 0, 0, INTER_LINEAR);
    }
    cout << "Found " << images.size() << " images" << endl;
}
int M_perc( cv::Mat mag)
{
    Mat magnitude=mag;
    vector<int> sorted;
    int Mperc = 0;

    for(int j=0;j<magnitude.rows;j++)
    {
        for(int k=0;k<magnitude.cols;k++)
        {
            sorted.push_back((double)magnitude.at<float>(j, k));
        }
    }

    sort(sorted.begin(), sorted.end());
    Mperc = (sorted[(sorted.size())/2]);

    return Mperc;

}
int M_softp( cv::Mat mag)
{
    Mat magnitude=mag;
    vector<double> sorted,weights;
    double p=0.5,pi=3.14,k=0.5,sum=0,M_softperc = 0;

    for(int j=0;j<magnitude.rows;j++)
    {
        for(int k=0;k<magnitude.cols;k++)
        {
            sorted.push_back((double)magnitude.at<float>(j,k));
        }
    }
    sort(sorted.begin(),sorted.end());

    for(int m=0;m<sorted.size();m++)
    {
        double S = round(p*sorted.size());
        if(m<=S)
        {weights.push_back(sin(pow(((pi*m)/(2*S)),k)));}
        else
        {weights.push_back(sin(pow((pi/2)-(pi*(m-S)/(2*sorted.size()-S)),k)));}
        sum+=weights[m];
    }

    for(int m=0;m<sorted.size();m++)
    {
        weights[m]=weights[m]/sum;
        M_softperc += weights[m]*sorted[m];
    }

    return M_softperc;
}
int FastFeatures(cv::Mat src)
{
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    std::vector<KeyPoint> keypoints;
    detector->detect( src, keypoints );
    Mat img_keypoints;
    drawKeypoints( src, keypoints, img_keypoints );
    int feat = keypoints.size();
    imshow("Fast Features", img_keypoints );
    return feat;
}
double dfx(double x, double DerofMsperc) {
    return x+DerofMsperc;
}

double abs_val(double x) {
    return x > 0 ? x : -x;
}

double gradient_descent(double dx, double error, double gamma, unsigned int max_iters, double DerofMsperc) {
    double c_error = error + 1;
    unsigned int iters = 0;
    double p_error;
    while (error < c_error && iters < max_iters) {
        p_error = dx;
        dx += dfx(p_error,DerofMsperc) * gamma;
        c_error = abs_val(p_error-dx);
        iters++;
    }
    return dx;
}


int main(int, char**argv)
{

      vector<Mat> images;
      vector<double> exposure_times;
      Mat src, src_gray,grad,mag,grad_best,DerofGrad,mag_best;
      int Mperc,Msperc,I_best,M_best=0;
      double DerofMsperc=0;
      int ddepth = CV_32F;
      Mat grad_x, grad_y,abs_grad_x, abs_grad_y,res_grad_x,res_grad_y,res_grad_inv,Ires_grad_x,Ires_grad_y,res_grad;
      vector<double> sorted,weights;
      double p=0.5,pi=3.14,k=0.5,sum=0;


      readImagesandTimes(images,exposure_times);


      for(int i=0;i<images.size();i++)
      {
        src = images[i];
        if( !src.data )
        { return -1; }

        GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        Sobel(src_gray, grad_x, ddepth, 1, 0);
        convertScaleAbs(grad_x,abs_grad_x);
        Sobel(src_gray, grad_y, ddepth, 0, 1);
        convertScaleAbs(grad_y,abs_grad_y);
        cv::addWeighted( abs_grad_x,1, abs_grad_y,1, 0, grad);
        namedWindow(window_name[i], WINDOW_AUTOSIZE);
        imshow(window_name[i],grad);

        magnitude(grad_x,grad_y,mag);
        Msperc=M_softp(mag);
        cout<<"Msoft of frame "<<i<<" is "<<Msperc<<endl;
        if(Msperc>M_best)
        {
            M_best = Msperc;
            I_best = i;
            cv::vconcat(abs_grad_x,abs_grad_y,grad_best);
            mag_best =mag;
        }

        Mperc=M_perc(mag);
        cout<<"Mperc of frame "<<i<<" is "<<Mperc<<endl;

        int features = FastFeatures(src_gray);
        cout<<"Fast Features extracted from Frame "<<i<<" is "<<features<<endl;
        waitKey(1000);

      }
      Mat response;
      Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson();
      calibrate->process(images, response, exposure_times);

      Sobel(response, res_grad_x, CV_64F, 1, 0);
      convertScaleAbs(res_grad_x,res_grad_x);
      Sobel(response, res_grad_y, CV_64F, 0, 1);
      convertScaleAbs(res_grad_y,res_grad_y);
      cv::add(res_grad_x,res_grad_y,res_grad);

      res_grad*=exposure_times[I_best];


      cvtColor(res_grad, res_grad, CV_64F);
      res_grad.convertTo(res_grad,CV_64F);
      res_grad_inv.convertTo(res_grad_inv,CV_64F);
      cv::invert(res_grad,res_grad_inv,DECOMP_SVD);
      Sobel(res_grad_inv, Ires_grad_x, CV_64F, 1, 0);
      Sobel(res_grad_inv, Ires_grad_y, CV_64F, 0, 1);



      cv::hconcat(Ires_grad_x,Ires_grad_y,res_grad);



      grad_best.convertTo(grad_best,CV_64FC1);
      res_grad.convertTo(res_grad,CV_64FC1);
      DerofGrad.convertTo(DerofGrad,CV_64FC1);
      transpose(res_grad,res_grad);

      DerofGrad = 2*grad_best*res_grad;

      for(int j=0;j<DerofGrad.rows;j++)
      {
          for(int k=0;k<DerofGrad.cols;k++)
          {
              sorted.push_back((double)DerofGrad.at<double>(j,k));
          }

      }
      sort(sorted.begin(),sorted.end());

      for(int m=0;m<sorted.size();m++)
      {
          double S = round(p*sorted.size());
          if(m<=S)
          {
              weights.push_back(sin(pow(((pi*m)/(2*S)),k)));}

          else
          {
             weights.push_back(sin(pow((pi/2)-(pi*(m-S)/(2*sorted.size()-S)),k)));
          }
          sum+=weights[m];
      }

      for(int m=0;m<sorted.size();m++)
      {

          weights[m]=weights[m]/sum;
          DerofMsperc+=weights[m]*sorted[m];
       }
      cout<<"Derivative of Msoftperc  :"<<DerofMsperc<<endl;
      {
          double dx = exposure_times[I_best];
          double error = 1e-6;
          double gamma = 0.00001;
          unsigned int max_iters = 10000;
          cout<<"Exposure time to be set for the next frame is: "<<gradient_descent(dx, error, gamma, max_iters,DerofMsperc)<<"sec"<<endl;
       }
return 0;

}
