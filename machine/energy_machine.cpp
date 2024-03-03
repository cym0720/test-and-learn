#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include<string>
#include<cmath>
#include<algorithm>

using namespace std;
using namespace cv;

float get_angle(Point3f a,Point3f b);
const Mat element = getStructuringElement(MORPH_RECT, Size(6,6));
Mat tvecs(3,1,DataType<double>::type);
Mat rvecs(3,1,DataType<double>::type);
string color;
const Mat camera_matrix= (Mat_<double>(3,3) <<
             2351.55699,    0,       716.95746,
                 0,   2349.89714,  547.81957,
                0,          0     ,  1 );
const double zConst = 0;
double s;//参数坐标s
float a=0.913,w=1.942,b=2.090-a;
vector<double> params={a,w};
vector<pair<double,double>> datas;
vector<double> params_gradient;
float angle_point;

void video_infr(VideoCapture captrue,double& fps,int& count,int& fourcc,Size& size);
vector<double> Gradient(const vector<double>& params, const vector<pair<double, double>>& data);
vector<double> gradientDescent(vector<double>& params, const vector<pair<double, double>>& data, double learningRate, int numIterations);
Point3f predict(vector<double> params,Point3f point_now,float t,float angle_p);
void recognition(VideoCapture capture,VideoWriter outVideo,string color);
void video_show(const string& filename);
void print_point(Point3f xyz,Mat output);
float getDistance(Point a,Point b);
Point3f getWorldPoints(Point2f inPoints,Mat revc_);

int main()
{
    cout<<"请输入识别颜色(red/blue)"<<endl;
    cin>>color;
    VideoCapture capture("../big-3.mp4");
    if (!capture.isOpened()) 
    {
		cerr << "Error: 无法打开视频文件." <<endl;
	}
    int count,fourcc;
    double fps;
    Size frame_size;
    video_infr(capture,fps,count,fourcc,frame_size);//调用函数vedio_infr给参数赋值
    VideoWriter outVideo("output_video.mp4",fourcc,fps,frame_size,true);//改了输入后记得改
    recognition(capture,outVideo,color);//调用函数recognition
}   

void video_infr(VideoCapture capture,double& fps,int& count,int& fourcc,Size& frame_size)
{
    fps=capture.get(CAP_PROP_FPS);
    count=capture.get(CAP_PROP_FRAME_COUNT);
    frame_size.width=capture.get(CAP_PROP_FRAME_WIDTH);
    frame_size.height=capture.get(CAP_PROP_FRAME_HEIGHT);
    fourcc=capture.get(CAP_PROP_FOURCC);
}

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

Point2f get2Dpoint(Point3f point,Mat rvec_)
{
    //获取图像坐标
    cv::Mat imagePoint = cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
	imagePoint.at<double>(0, 0) = point.x;
	imagePoint.at<double>(1, 0) = point.y;
    imagePoint.at<double>(1, 0) = point.y;

    //逆运算
    imagePoint=1/2*(camera_matrix*(rvec_*imagePoint+tvecs));
    Point2f point2D(imagePoint.at<double>(0, 0),imagePoint.at<double>(1, 0));
    return point2D;
}

void print_point(Point3f p,Mat output,Point point)
{
    string xyz=("("+to_string(static_cast<int>(p.x))+","+to_string(static_cast<int>(p.y))+","+to_string(static_cast<int>(p.z))+")");
    putText(output,xyz,point, FONT_HERSHEY_SIMPLEX,1, Scalar(0,255, 0)); 
}

float getDistance(Point a,Point b)
{
    float distance;
    distance=powf((a.x-b.x),2)+powf((a.y-b.y),2);
    distance=sqrtf(distance);
    return distance;
}

float get_angle(Point3f a,Point3f b,float& angle_p)
{
    float tan_a,tan_b,angle;
    tan_a=a.x/a.y,tan_b=b.x/b.y;
    angle_p=atan(tan_a);
    angle=angle_p-atan(tan_b);
    return angle;
}

double errorFunction(const std::vector<double>& params, const std::vector<std::pair<double, double>>& data) 
{
    double sumError = 0.0;
    for (const auto& point : data) {
        double t = point.first;
        double y = point.second;
        double yFit = params[0] * std::sin(params[2] * t) + params[1];
        sumError += (y - yFit) * (y - yFit);
    }
    return sumError;
}

// 计算目标函数的梯度
vector<double> Gradient(const std::vector<double>& params, const vector<std::pair<double, double>>& data) {
    vector<double> gradient(2); // 梯度向量，对应于参数a, w
    for (const auto& point : data) {
        double t = point.first;
        double v = point.second; 
        double yFit = params[0] * std::sin(params[1] * t) + 2.09-params[1];
        gradient[0] += 2 * (v - yFit) * (-params[1] * t * std::cos(params[1] * t)); // 对应于a的梯度
        gradient[1] += 2 * (v - yFit) * params[0] * t * std::sin(params[1] * t); // 对应于w的梯度
    }
    return gradient;
}

// 梯度下降法更新参数
vector<double> gradientDescent(std::vector<double>& params, const std::vector<std::pair<double, double>>& data, double learningRate, int numIterations)
{
    vector<double> optimizedParams(2); // 最终优化的参数值
    for (int i = 0; i < numIterations; ++i) 
    {
        vector<double> gradient = Gradient(params, data); // 计算梯度
        for (int j = 0; j < 2; ++j) 
        {
            params[j]-=learningRate*gradient[j]; //更新参数值
        }
    }
    optimizedParams = params; // 将最终的参数值存储在optimizedParams中
    return optimizedParams;
}

Point3f predict(vector<double> params,Point3f point_now,float t,float angle_p)
{
    float b=2.09-params[0];
    float angle_increase=-params[0]/params[1]*cos(params[1]*(t+0.2))+params[0]/params[1]*cos(params[1]*t)+b*0.2;
    float angle_predict=angle_increase+angle_p;
    float length=sqrt(pow(point_now.x,2)+pow(point_now.y,2));
    Point3f point_predict(length*cos(angle_predict),length*sin(angle_predict),point_now.z);
    return point_predict;
}

void recognition(VideoCapture capture,VideoWriter outVideo,string color)
{
    vector<RotatedRect> Boxs_Last;
    float time=0,angle_now=0,angle_last=0;
    Point3f Point_now(0,0,0);
    Point3f Point_last(0,0,0);
    double t=0;
    while (1)
	{
		Mat frame,output1,output2;   
		capture >> frame;//读取当前帧
		if (frame.empty())
		{
			break;
		}
        //转为HSV格式后用inRange分类红色或蓝色
        cvtColor(frame,output1,COLOR_BGR2HSV);
        Mat dilated;
        if(color=="red")
        {
            inRange(output1,Scalar(0,43,46),Scalar(10,255,255),output2);
            inRange(output1,Scalar(156,43,46),Scalar(180,255,255),output2); 
            dilate(output2,output2,element);
        }
        if(color=="blue")
        {
            inRange(output1,Scalar(27,43,46),Scalar(124,255,255),output2);
        }
        dilate(output2,dilated,element); 
        vector<vector<cv::Point>>contours1;
        vector<vector<cv::Point>>contours2;
        findContours(dilated,contours1,RETR_EXTERNAL,CHAIN_APPROX_NONE);
        //找到中心R，绘制实心圆与空心圆，并且利用rect类的四个点与RM官网参数解pnp
        Point2f R_center;
        for (int i=0;i<contours1.size();i++)
	    {
		double areal = contourArea(contours1[i]);
        if(areal<1000&&areal>350)
            {    
                Rect Rect =boundingRect(contours1[i]);
                if(abs((Rect.br().x-Rect.tl().x)-(Rect.br().y-Rect.tl().y))<5)
                {
                    float r_in,r_out,r=abs((Rect.br().x-Rect.tl().x)+(Rect.br().y-Rect.tl().y))/2;
                    r_in=((color=="blue")?6*r:5.5*r);
                    r_out=((color=="blue")?7.8*r:7.5*r);                   
                    R_center=(Rect.br()+Rect.tl())/2;
                    circle(dilated,R_center,r_in, Scalar(0,0,0),-1, 8, 0);
                    circle(dilated,R_center,r_out, Scalar(0,0,0),3, 8, 0);
                    vector<Point2f> Points2D;
                    vector<Point3f> Points3D;
                    Points2D.push_back(Point2f(Rect. br().x,Rect.tl().y));
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
	    }
        findContours(dilated,contours2,RETR_EXTERNAL,CHAIN_APPROX_NONE);
        vector<RotatedRect> Boxs;
        vector<float> Areas;
        float length;
        for(int i=0;i<contours2.size();i++)
        {
            double areal=contourArea(contours2[i]);
            if(areal>3000&&areal<10000)
            {
                RotatedRect rotatedRect =minAreaRect(contours2[i]);
                float distance=getDistance(rotatedRect.center,R_center);
                if(distance>150&&distance<300)
                {
                    Boxs.push_back(rotatedRect);
                    Areas.push_back(areal);
                    length=distance;
                }
            }
        }
        if(Boxs.size()>=1)
        {
            int max= max_element(Areas.begin(),Areas.end()) - Areas.begin();
            Point2f rect_points[4];
	        Boxs[max].points(rect_points);
	        for (int j = 0; j < 4; j++)
	        {
	            line(frame,rect_points[j], rect_points[(j + 1) % 4], Scalar(0,255,0));
	        }
            circle(frame,Boxs[max].center, 5, Scalar(0,255,0), -1, 8, 0);
            Mat rvecs_;
            Rodrigues(rvecs, rvecs_);
            Point3f a=getWorldPoints(Boxs[max].center,rvecs_);
            print_point(a,frame,Point(10,25));
            Point_last=Point_now;
            Point_now=a;
            if(datas.size()<41)
            { 
                if(t>1&&t<1000)
                {
                    angle_now=get_angle(Point_now,Point_last,angle_point);
                    if(angle_now!=0)
                    {
                        double time=0.02*t;
                        double v=angle_now/time;
                        datas.push_back(pair(time,v));
                    }
                }
            }
            if(datas.size()==40)
            {
                params_gradient=gradientDescent(params,datas,0.0001,90);
                cout<<"a="<<params_gradient[0]<<endl<<"w="<<params_gradient[1]<<endl<<"b="<<2.09-params_gradient[0];
            }
            if(datas.size()>39)
            {
                Point3f point_p3D=predict(params,Point_now,t,angle_point);
                string predict="predict:("+to_string(static_cast<int>(point_p3D.x))+","+to_string(static_cast<int>(point_p3D.y))+","+to_string(static_cast<int>(point_p3D.z))+")";
                putText(frame,predict,Point(300,25),FONT_HERSHEY_SIMPLEX,1, Scalar(0,255, 0));
            }
            // Point2f point_p2D=get2Dpoint(point_p3D,rvecs_);
            // circle(frame,point_p2D,5, Scalar(0,255,0),-1, 8,0);
            // Point3f wpoint_p=getWorldPoints(point_p2D,rvecs_);
            // print_point(wpoint_p,frame,Point(5000,20));
            t+=1;
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