#include<iostream>
#include<cmath>
#include<Eigen/Dense>
using namespace Eigen;
typedef struct 
{
    double x, y, z;
} Point;
double distance(const Point& a, const Point& b) 
{
    return (Vector3d(a.x,a.y,a.z)-Vector3d(b.x,b.y,b.z)).norm();
}
Point initial_guess(Point points[4],double distances[4]) 
{
    Vector3d ref(points[0].x,points[0].y,points[0].z);
    Matrix3d A;
    Vector3d B;
    for(int i=0;i<3;i++) 
    {
        Vector3d p(points[i+1].x,points[i+1].y,points[i+1].z);
        A.row(i)=2*(p-ref);
        double ref_sq=ref.squaredNorm();
        double p_sq=p.squaredNorm();
        B[i]=distances[0]*distances[0]-distances[i+1]*distances[i+1]-ref_sq+p_sq;
    }
    Vector3d X = A.colPivHouseholderQr().solve(B);
    return {X[0],X[1],X[2]};
}
Point optimize(Point points[4],double distances[4],Point initial) 
{
    Vector3d X(initial.x,initial.y,initial.z);
    double lambda=0.001;
    double prev_error=1e20;
    for(int iter=0;iter<100;iter++) 
    {
        Matrix<double,4,3> J;
        Vector4d residuals;
        double error=0;
        for(int i=0;i<4;i++) 
        {
            Vector3d p(points[i].x,points[i].y,points[i].z);
            double d=(X-p).norm();
            double diff=d-distances[i];
            residuals[i]=diff;
            error+=diff*diff;
            if(d<1e-12) d=1e-12;
            J.row(i)=(X-p).transpose()/d;
        }
        if(std::abs(prev_error-error)<1e-6) break;
        prev_error=error;
        Matrix3d H=J.transpose()*J;
        H.diagonal()*=(1.0+lambda);
        Vector3d g=J.transpose()*residuals;
        Vector3d delta=H.colPivHouseholderQr().solve(-g);
        Vector3d X_new=X+delta;
        double new_error=0;
        for(int i=0;i<4;i++) 
        {
            Vector3d p(points[i].x,points[i].y,points[i].z);
            double diff=(X_new-p).norm()-distances[i];
            new_error+=diff*diff;
        }
        if(new_error<error) 
        {
            X=X_new;
            lambda/=10.0;
        } 
        else 
        {
            lambda*=10.0;
        }
    }
    return{X[0],X[1],X[2]};
}
Point get_point(Point points[4],double distances[4]) 
{
    Point initial=initial_guess(points, distances);
    return optimize(points,distances,initial);
}
int main() 
{
    return 0;
}