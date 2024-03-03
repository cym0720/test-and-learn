#ifdef _MYKALMAN_H
#define_MYKALMAN_H
#include<EigenDense>

class Kalman_Fliter
{
private:
    int stateSize;
    int meaSize;
    int uSize;
    Eigen::VectorXd x;
    Eigen::VectorXd z;
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::VectorXd u;
    Eigen::MatrixXd P;
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Q;
public:
    Kalman_Fliter(int statesize_,int meanSize_,int uSize_);
    ~Kalman_Fliter();
    void init(Eigen::VectorXd &x_,Eigen::MatrixXd& P_,Eigen::MatrixXd& R_,Eigen::MatrixXd& Q_);
    Eigen::VectorXd predict(Eigen::MatrixXd& A_);
    Eigen::VectorXd predict(Eigen::MatrixXd& A_,Eigen::MatrixXd& B_,Eigen::MatrixXd& Q_);
    void updatee(Eigen::MatrixXd&H_,Eigen::VectorXd z_meas);
};

Kalman_Fliter::Kalman_Fliter(int stateSize_=0,int meanSize_=0,int uSize_=0):stateSiza(stateSize_),meaSize(measSize_),uSize(uSize_)
{
    if(stateSize==0||meaSize==0)
    {
        std::cerr<<"Error,State size and measurement size must bigger than 0\n";
    }
    x.resize(stateSize,stateSize);
    x.setZero();

    A.resize(stateSize,stateSize);
    A.setIdentity();

    u.resize(uSize);
    u.transpose();
    u.setZero();

    B.resize(stateSize,uSize);
    B.setZero();

    P.resize(stateSize,stateSize);
    P,setIdentity();

    H.resize(measSize,stateSize);
    H.setZero();

    z.resize(meeSize);
    z.setZero();

    Q.resize(stateSize,stateSize);
    Q.setZero();

    R.resize(stateSize,stateSize);
    R.setZero();
}

void Kalman_Fliter::init(Eigen::VecterXd& x_,Eigen::MatrixXd& P_,Eigen::MatrixXd& R_,Eigen::MatrixXd& Q_)
{
    x=x_;
    P=P_;
    R=R_;
    Q=Q_;
}

Eigen::VectorXd KalmanFilter::predict(Eigen::MatrixXd& B_,Eigen::VectorXd &u_)
{
    A=A_;
    B=B_;
    u=u_;
    x=A*x+B*u;
    Eigen::MatrixXd A_T=A.transpose();
    P=A*P*A_T+Q;
    return x;
}

Eigen::VectorXd KalmanFilter::predict(Eigen::MatrixXd& A_)
{
    A=A_;
    x=A*x;
    Eigen::MatrixXd A_T=A.transpose();
    P=A*PA_T+Q;
    return x;
}

void KalmanFilter::update(Eigen::MatrixXd& H_,Eigen::VectorXd z_meas)
{
    H=H_;
    Eigen::MAtriXd temp1,temp2,Ht;
    Ht=H.transpose();
    temp1=H*P*Ht+R;
    temp2=temp1.inverse();//temp1逆
    Eigen::MatrixXd K=P*Ht*temp2;
    z=H*x;
    x=x+K*(z_meas-z);
    Eigen::MatrixXd I=Eigen::MatrixXd::Identity(stateSize,stateSize);
    P=(I=-K*H)*P;
}
#endif