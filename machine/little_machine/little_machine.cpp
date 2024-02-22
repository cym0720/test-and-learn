#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include<string>
#include"Kalman_Fliter.h"

using namespace std;
using namespace cv;

Mat tvecs(3,1,DataType<double>::type);
Mat rvecs(3,1,DataType<double>::type);
Mat camera_matrix= (Mat_<double>(3,3) <<
             2351.55699,    0,       716.95746,
                 0,   2349.89714,  547.81957,
                0,          0     ,  1 );
double zConst = 0,s;//参数坐标s


void video_infr(VideoCapture captrue,double& fps,int& count,int& fourcc,Size& size);
float iou(RotatedRect box1,RotatedRect box2);
void recognition(VideoCapture capture,VideoWriter outVideo);
void video_show(const string& filename);
void print_point(Point3d xyz,Mat output);

int main()
{
    VideoCapture capture("../machine.mp4");
    if (!capture.isOpened()) 
    {
		cerr << "Error: 无法打开视频文件." <<endl;
	}
    int count,fourcc;
    double fps;
    Size frame_size;
    video_infr(capture,fps,count,fourcc,frame_size);//调用函数vedio_infr给参数赋值
    VideoWriter outVideo("output_video.mp4",fourcc,fps,frame_size,true);//改了输入后记得改
    recognition(capture,outVideo);//调用函数recognition
}   

void video_infr(VideoCapture capture,double& fps,int& count,int& fourcc,Size& frame_size)
{
    fps=capture.get(CAP_PROP_FPS);
    count=capture.get(CAP_PROP_FRAME_COUNT);
    frame_size.width=capture.get(CAP_PROP_FRAME_WIDTH);
    frame_size.height=capture.get(CAP_PROP_FRAME_HEIGHT);
    fourcc=capture.get(CAP_PROP_FOURCC);
}

// float iou(RotatedRect box1,RotatedRect box2)
// {
//     float iou;
//     box1Points2D
//     return iou;
// }

Point3f getWorldPoints(Point2f inPoints,Mat revc_)
{
    //获取图像坐标
    cv::Mat imagePoint = cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
	imagePoint.at<double>(0, 0) = inPoints.x;
	imagePoint.at<double>(1, 0) = inPoints.y;
 
	//计算比例参数S
	cv::Mat tempMat, tempMat2;
	tempMat = revc_.inv() * camera_matrix.inv() * imagePoint;
	tempMat2 = revc_.inv() * tvecs;
	s = zConst + tempMat2.at<double>(2, 0);
	s /= tempMat.at<double>(2, 0);
 
    //计算世界坐标
	Mat wcPoint = revc_.inv() * (s * camera_matrix.inv() * imagePoint - tvecs);
	Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
	return worldPoint;
}

void print_point(Point3f p,Mat output)
{
    string xyz=("("+to_string(static_cast<int>(p.x))+","+to_string(static_cast<int>(p.y))+","+to_string(static_cast<int>(p.x))+")");
    putText(output,xyz,Point(8, 20), FONT_HERSHEY_SIMPLEX,1, Scalar(0,255, 0)); 
}

void recognition(VideoCapture capture,VideoWriter outVideo)
{
    while (1)
	{
		Mat frame,output1,output2;   
		capture >> frame;//读取当前帧
		if (frame.empty())
		{
			break;
		}
        cvtColor(frame,output1,COLOR_BGR2GRAY);
        threshold(output1,output2,20,225,THRESH_BINARY);
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	    Mat dilated;
	    dilate(output2,dilated, element);
        vector<vector<cv::Point>>contours1;
        vector<vector<cv::Point>>contours2;
        findContours(dilated,contours1,RETR_EXTERNAL,CHAIN_APPROX_NONE);
        //找到中心R，绘制实心圆与空心圆，并且利用rect类的四个点与RM官网参数解pnp
        for (int i=0;i<contours1.size();i++)
	    {
		double areal = contourArea(contours1[i]);
        if(areal<750&&areal>500)
            {
            Rect Rect =boundingRect(contours1[i]);
            circle(dilated,(Rect.br()+Rect.tl())/2,150, Scalar(0,0,0),-1, 8, 0);
            circle(dilated,(Rect.br()+Rect.tl())/2,270, Scalar(0,0,0),3, 8, 0);
            vector<Point2f> Points2D;
            vector<Point3f> Points3D;
            Points2D.push_back(Point2f(Rect.br().x,Rect.tl().y));
            Points2D.push_back(Point2f(Rect.br()));
            Points2D.push_back(Point2f(Rect.tl().x,Rect.br().y));
            Points2D.push_back(Point2f(Rect.tl()));
            Points3D.push_back(Point3f(53,-53,0));
            Points3D.push_back(Point3f(53,53,0));
            Points3D.push_back(Point3f(-53,53,0));
            Points3D.push_back(Point3f(-53,-53,0));
            Mat distCoeffs=(Mat_<double>(5,1)<<-0.104800, 0.140335, -0.000984, -0.000920, 0.000000);
            solvePnP(Points3D,Points2D,camera_matrix,distCoeffs,rvecs,tvecs);   
            }            
	    }
        //绘制完成后直接旋转矩形
        findContours(dilated,contours2,RETR_EXTERNAL,CHAIN_APPROX_NONE);
        RotatedRect LastRect;
        for (int i=0;i<contours2.size();i++)
	    {
		    double areal = contourArea(contours2[i]);
            // vector<RotatedRect> Rects;
            if(areal<10000&&areal>9375)
            {              
                RotatedRect rotateRect =minAreaRect(contours2[i]);//轮廓最小外接矩形
                // Rects.push_back(rotateRect);
                // if(Rects.size()>=2)
                // {
                //     iou();
                // }
                // else(Rect.size=1)
                // {
                //     LastRect=Rects[0];  
                // }
	            Point2f rect_points[4];
	            rotateRect.points(rect_points);
	            for (int j = 0; j < 4; j++)
	            {
	            line(frame,rect_points[j], rect_points[(j + 1) % 4], Scalar(0,255,0));
	            }
                circle(frame,rotateRect.center, 5, Scalar(0,255,0), -1, 8, 0);
                Mat rvecs_;
                Rodrigues(rvecs, rvecs_);
                Point3f Point3d=getWorldPoints(rotateRect.center,rvecs_);
                print_point(Point3d,frame);
            }
        }
      outVideo<<frame;
    }  
}

void video_show(const string& filename)
{
    VideoCapture capture(filename);
	while (1)
	{
		Mat frame;
		capture >> frame;//读取当前帧
		if (frame.empty())
		{
			break;
		}
		imshow("readvideo", frame);
        waitKey(1);
    }
}    