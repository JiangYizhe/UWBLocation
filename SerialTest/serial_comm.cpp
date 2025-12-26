#include <iostream>
#include <cmath>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <iomanip>
#include <Eigen/Dense>

using namespace Eigen;

// 定义点结构
typedef struct {
    float x, y, z;
} Point;

// 计算两点间距离
float distance(const Point& a, const Point& b) {
    return (Vector3f(a.x, a.y, a.z) - Vector3f(b.x, b.y, b.z)).norm();
}

// 初始猜测
Point initial_guess(Point points[4], float distances[4]) {
    Vector3f ref(points[0].x, points[0].y, points[0].z);
    Matrix3f A;
    Vector3f B;
    for(int i = 0; i < 3; i++) {
        Vector3f p(points[i+1].x, points[i+1].y, points[i+1].z);
        A.row(i) = 2 * (p - ref);
        float ref_sq = ref.squaredNorm();
        float p_sq = p.squaredNorm();
        B[i] = distances[0] * distances[0] - distances[i+1] * distances[i+1] - ref_sq + p_sq;
    }
    Vector3f X = A.colPivHouseholderQr().solve(B);
    return {X[0], X[1], X[2]};
}

// 优化（高斯-牛顿法）
Point optimize(Point points[4], float distances[4], Point initial) {
    Vector3f X(initial.x, initial.y, initial.z);
    float lambda = 0.001;
    float prev_error = 1e20;
    for(int iter = 0; iter < 100; iter++) {
        Matrix<float, 4, 3> J;
        Vector4f residuals;
        float error = 0;
        for(int i = 0; i < 4; i++) {
            Vector3f p(points[i].x, points[i].y, points[i].z);
            float d = (X - p).norm();
            float diff = d - distances[i];
            residuals[i] = diff;
            error += diff * diff;
            if(d < 1e-12f) d = 1e-12f;
            J.row(i) = (X - p).transpose() / d;
        }
        if(std::abs(prev_error - error) < 1e-6f) break;
        prev_error = error;
        Matrix3f H = J.transpose() * J;
        H.diagonal() *= (1.0 + lambda);
        Vector3f g = J.transpose() * residuals;
        Vector3f delta = H.colPivHouseholderQr().solve(-g);
        Vector3f X_new = X + delta;
        float new_error = 0;
        for(int i = 0; i < 4; i++) {
            Vector3f p(points[i].x, points[i].y, points[i].z);
            float diff = (X_new - p).norm() - distances[i];
            new_error += diff * diff;
        }
        if(new_error < error) {
            X = X_new;
            lambda /= 10.0f;
        } else {
            lambda *= 10.0f;
        }
    }
    return {X[0], X[1], X[2]};
}

// 获取标签坐标
Point get_point(Point points[4], float distances[4]) {
    Point initial = initial_guess(points, distances);
    return optimize(points, distances, initial);
}

// 串口配置函数
bool configure_serial(int fd, int baudrate) {
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error from tcgetattr: " << strerror(errno) << std::endl;
        return false;
    }

    cfsetospeed(&tty, baudrate);
    cfsetispeed(&tty, baudrate);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;   // 无校验位
    tty.c_cflag &= ~CSTOPB;   // 1位停止位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;        // 8位数据位

    tty.c_lflag &= ~(ICANON | ECHO | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN] = 16;      // 每次读取16字节（完整帧）
    tty.c_cc[VTIME] = 1;      // 0.1秒超时

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error from tcsetattr: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

int main() {
    // 定义四个基站的坐标（单位：米）
    Point base_stations[4] = {
        {-0.50, 0.0, 1.50},   // 基站1坐标
        {4.0, 0.0, 1.60},   // 基站2坐标
        {-0.50, 4.50, 1.70},   // 基站3坐标
        {4.0, 4.50, 1.80}    // 基站4坐标
    };

    // 打开接收串口（用于接收UWB数据）
    const char* receive_port = "/dev/ttyS0";
    int receive_fd = open(receive_port, O_RDWR);
    if (receive_fd < 0) {
        std::cerr << "Cannot open receive port: " << receive_port << " - " << strerror(errno) << std::endl;
        return 1;
    }
    if (!configure_serial(receive_fd, B115200)) {
        close(receive_fd);
        return 1;
    }

    // 打开发送串口1（用于发送解算后的坐标）
    const char* send_port1 = "/dev/ttyS7";
    int send_fd1 = open(send_port1, O_RDWR);
    if (send_fd1 < 0) {
        std::cerr << "Cannot open send port1: " << send_port1 << " - " << strerror(errno) << std::endl;
        close(receive_fd);
        return 1;
    }
    if (!configure_serial(send_fd1, B115200)) {
        close(receive_fd);
        close(send_fd1);
        return 1;
    }

    // 打开发送串口2（用于发送解算后的坐标）
    const char* send_port2 = "/dev/ttyS4";
    int send_fd2 = open(send_port2, O_RDWR);
    if (send_fd2 < 0) {
        std::cerr << "Cannot open send port1: " << send_port2 << " - " << strerror(errno) << std::endl;
        close(receive_fd);
        return 1;
    }
    if (!configure_serial(send_fd2, B115200)) {
        close(receive_fd);
        close(send_fd2);
        return 1;
    }

    std::cout << "UWB定位系统已启动..." << std::endl;

    while (true) {
        // 读取16字节的数据帧
        uint8_t frame[16];
        ssize_t bytes_read = read(receive_fd, frame, 16);
        if (bytes_read != 16) {
            // 读取不完整，跳过
            continue;
        }

        // 验证帧头：前两个字节是'm'和'r'
        if (frame[0] != 'm' || frame[1] != 'r') {
            // 帧头错误，跳过
            continue;
        }

        // 解析四个距离（从第7字节到第14字节，每两个字节一个距离）
        float distances[4];
        for (int i = 0; i < 4; i++) {
            // 每个距离由两个字节组成，低字节在前，高字节在后
            uint16_t dist_cm = (frame[6 + i*2 + 1] << 8) | frame[6 + i*2];
            distances[i] = dist_cm / 100.0f; // 厘米转米
        }

        // 解算标签坐标
        Point tag = get_point(base_stations, distances);

        uint8_t Frame[14];
         Frame[0] = 0xAA; // 帧头
        // 将浮点数转为小端字节序（兼容STM32）
        memcpy(&Frame[1], &tag.x, sizeof(float));  // 字节1-4: X坐标
        memcpy(&Frame[5], &tag.y, sizeof(float));  // 字节5-8: Y坐标
        memcpy(&Frame[9], &tag.z, sizeof(float));  // 字节9-12: Z坐标
        Frame[13] = 0x55; // 帧尾

         // ==== 修改点2：发送二进制帧（非字符串） ====
        ssize_t bytes_sent = write(send_fd1, Frame, sizeof(Frame));
        write(send_fd2, Frame, sizeof(Frame));
        // 可选：发送失败处理
        if (bytes_sent != sizeof(Frame)) {
            std::cerr << "发送失败！预期14字节，实际发送：" << bytes_sent << std::endl;
        }

         // 保留终端打印（可选）
        std::cout << "发送坐标: X=" << tag.x << ", Y=" << tag.y << ", Z=" << tag.z << std::endl;

        

        // 打印坐标到终端
        std::cout << "标签坐标: (" 
                  << std::fixed << std::setprecision(2) << tag.x << ", " 
                  << tag.y << ", " << tag.z << ")" << std::endl;
                  
        // 将坐标通过发送串口发送（格式化为字符串）
        std::string coord_str1 = "(" + std::to_string(tag.x) + ", " 
                              + std::to_string(tag.y) + ", " 
                              + std::to_string(tag.z) + ")";
        write(send_fd2, coord_str1.c_str(), coord_str1.size());
        

       std::string coord_str2 = "(" + std::to_string(distances[0]) + ", " 
                            + std::to_string(distances[1]) + ", " 
                             + std::to_string(distances[2]) + ", " 
                             + std::to_string(distances[3]) + ")\n";
       write(send_fd2, coord_str2.c_str(), coord_str2.size());
        
    }

    close(receive_fd);
    close(send_fd1);
    close(send_fd2);
    return 0;
}

/*  g++ -std=c++11 -I/path/to/eigen serial_comm.cpp -o serial_test
    ./serial_test
*/


/*#include <iostream>
#include <cmath>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <iomanip>
#include <Eigen/Dense>

using namespace Eigen;

// 定义点结构
typedef struct {
    float x, y, z;
} Point;

// 计算两点间距离
float distance(const Point& a, const Point& b) {
    return (Vector3d(a.x, a.y, a.z) - Vector3d(b.x, b.y, b.z)).norm();
}

// 初始猜测
Point initial_guess(Point points[4], float distances[4]) {
    Vector3d ref(points[0].x, points[0].y, points[0].z);
    Matrix3d A;
    Vector3d B;
    for(int i = 0; i < 3; i++) {
        Vector3d p(points[i+1].x, points[i+1].y, points[i+1].z);
        A.row(i) = 2 * (p - ref);
        float ref_sq = ref.squaredNorm();
        float p_sq = p.squaredNorm();
        B[i] = distances[0]*distances[0] - distances[i+1]*distances[i+1] - ref_sq + p_sq;
    }
    Vector3d X = A.colPivHouseholderQr().solve(B);
    return {X[0], X[1], X[2]};
}

// 优化（高斯-牛顿法）
Point optimize(Point points[4], float distances[4], Point initial) {
    Vector3d X(initial.x, initial.y, initial.z);
    float lambda = 0.001;
    float prev_error = 1e20;
    for(int iter = 0; iter < 100; iter++) {
        Matrix<float, 4, 3> J;
        Vector4d residuals;
        float error = 0;
        for(int i = 0; i < 4; i++) {
            Vector3d p(points[i].x, points[i].y, points[i].z);
            float d = (X - p).norm();
            float diff = d - distances[i];
            residuals[i] = diff;
            error += diff * diff;
            if(d < 1e-12) d = 1e-12;
            J.row(i) = (X - p).transpose() / d;
        }
        if(std::abs(prev_error - error) < 1e-6) break;
        prev_error = error;
        Matrix3d H = J.transpose() * J;
        H.diagonal() *= (1.0 + lambda);
        Vector3d g = J.transpose() * residuals;
        Vector3d delta = H.colPivHouseholderQr().solve(-g);
        Vector3d X_new = X + delta;
        float new_error = 0;
        for(int i = 0; i < 4; i++) {
            Vector3d p(points[i].x, points[i].y, points[i].z);
            float diff = (X_new - p).norm() - distances[i];
            new_error += diff * diff;
        }
        if(new_error < error) {
            X = X_new;
            lambda /= 10.0;
        } else {
            lambda *= 10.0;
        }
    }
    return {X[0], X[1], X[2]};
}

// 获取标签坐标
Point get_point(Point points[4], float distances[4]) {
    Point initial = initial_guess(points, distances);
    return optimize(points, distances, initial);
}

// 配置单串口（同时支持收发）
bool configure_serial(int fd, int baudrate) {
    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        return false;
    }

    // 设置波特率
    cfsetispeed(&tty, baudrate);
    cfsetospeed(&tty, baudrate);

    // 配置基础参数 [3,9](@ref)
    tty.c_cflag |= (CLOCAL | CREAD);  // 本地连接 + 启用接收
    tty.c_cflag &= ~PARENB;           // 无校验位
    tty.c_cflag &= ~CSTOPB;           // 1位停止位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;               // 8位数据位

    // 禁用流控和信号处理 [3](@ref)
    tty.c_iflag &= ~(IXON | IXOFF | IXANY | INLCR | ICRNL);
    tty.c_oflag &= ~(OPOST | ONLCR);
    tty.c_lflag &= ~(ICANON | ECHO | ISIG);

    // 超时设置：阻塞模式，至少读16字节 [3](@ref)
    tty.c_cc[VMIN] = 16;   // 等待完整16字节帧
    tty.c_cc[VTIME] = 1;   // 0.1秒超时

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

int main() {
    // 定义四个基站的坐标（单位：米）
    Point base_stations[4] = {
        {0.0, 0.0, 0.72},   // 基站1坐标
        {2.0, 0.0, 0.72},   // 基站2坐标
        {0.0, 2.0, 0.72},   // 基站3坐标
        {0.0, 0.0, 1.63}    // 基站4坐标
    };

    // 打开串口（单串口收发）
    const char* port = "/dev/ttyS0";
    int fd = open(port, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        std::cerr << "无法打开串口: " << port << " - " << strerror(errno) << std::endl;
        return 1;
    }

    if (!configure_serial(fd, B115200)) {
        close(fd);
        return 1;
    }

    std::cout << "UWB定位系统已启动（单串口模式）..." << std::endl;

    while (true) {
        // 接收16字节UWB数据帧 [4](@ref)
        uint8_t frame[16];
        ssize_t bytes_read = read(fd, frame, 16);
        if (bytes_read != 16) continue;  // 跳过不完整帧

        // 校验帧头（'m''r'）
        if (frame[0] != 'm' || frame[1] != 'r') continue;

        // 解析4个距离值（小端序，厘米转米）
        float distances[4];
        for (int i = 0; i < 4; i++) {
            uint16_t dist_cm = (frame[6 + i*2 + 1] << 8) | frame[6 + i*2];
            distances[i] = dist_cm / 100.0;  // 转换为米
        }

        // 解算标签坐标
        Point tag = get_point(base_stations, distances);

        // 格式化坐标输出
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "坐标:(" << tag.x << "," << tag.y << "," << tag.z << ")\n";
        std::string coord_str = oss.str();

        // 终端打印
        std::cout << coord_str;

        // 通过同一串口发送坐标
        write(fd, coord_str.c_str(), coord_str.size());
    }

    close(fd);
    return 0;
}*/
