#include "kalmanFilter.h"
#include <Eigen/Cholesky>

namespace byte_kalman
{
    // 卡方分布（Chi-Square Distribution）在 95% 置信度下的阈值表。
    // 因为我们的测量空间是 4 维的（cx,cy,w,h），所以对应的自由度是 4。查表可知 chi2inv95[4] = 9.4877。
    // 追踪器拿到这个函数返回的 square_maha（马氏距离平方）后，会挨个判断：
    // 如果某个 YOLO 框的马氏距离平方 > 9.4877，追踪器就会说：“我不管这个框的 IoU 有多高，在物理法则上，我的目标绝对不可能在 1 帧内跑到那个位置，直接排除！” 然后会在代价矩阵里把这个匹配的代价设为无穷大。
    const double KalmanFilter::chi2inv95[10] = {
        0,
        3.8415,
        5.9915,
        7.8147,
        9.4877,
        11.070,
        12.592,
        14.067,
        15.507,
        16.919};
    KalmanFilter::KalmanFilter(
        float std_weight_position,
        float std_weight_velocity)
    {
        int ndim = 4;
        double dt = 1.;

        this->_motion_mat = Eigen::MatrixXf::Identity(8, 8);
        for (int i = 0; i < ndim; i++)
        {
            this->_motion_mat(i, ndim + i) = dt;
        }
        this->_update_mat = Eigen::MatrixXf::Identity(4, 8);

        // std_weight_position
        //  物理意义：对 YOLO 检测框的位置抖动有多大的包容心？
        //  调大（比如 0.1）：追踪器会变得“很软”，它认为位置发生跳变是正常的。结果就是，预测框会死死地贴着 YOLO 的检测框，YOLO 抖它也跟着抖。
        //  调小（比如 0.01）：追踪器会变得“很硬”，它认为目标不可能瞬间瞬移。它会像加了极其沉重的减震器一样，画出一条极度平滑的轨迹，无视 YOLO 偶尔的瞎跳。
        this->_std_weight_position = std_weight_position;
        // std_weight_velocity
        //  物理意义：我对目标“急刹车、急转弯、突然加速”有多大的包容心？
        //  调大（比如 0.02）：追踪器认为目标非常灵活，随时可能改变速度。当目标突然转弯时，追踪器能瞬间反应过来跟上去。缺点是，对于匀速目标，轨迹可能会画出波浪线。
        //  调小（比如 0.001）：追踪器坚信目标是个“铁憨憨”，只会老老实实做匀速直线运动。
        this->_std_weight_velocity = std_weight_velocity;
    }

    KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement)
    {
        // measurement 是 YOLO 刚刚给出的 [cx, cy, w, h]
        DETECTBOX mean_pos = measurement;

        // 因为目标是刚出生的，我们对它的速度一无所知，所以盲猜速度为 0
        DETECTBOX mean_vel;
        for (int i = 0; i < 4; i++)
            mean_vel(i) = 0;

        KAL_MEAN mean;
        for (int i = 0; i < 8; i++)
        {
            if (i < 4)
                mean(i) = mean_pos(i);
            else
                mean(i) = mean_vel(i - 4);
        }

        KAL_MEAN std;
        // 为什么 X 和 Y 的误差都用高度 h（measurement[3]）来乘？
        // 这是因为 ByteTrack 脱胎于行人追踪。行人的特点是：高度很稳定，但宽度一直在变（侧身、摆臂）。所以用高度 h 作为统一的“标尺”来衡量 X、Y 和 h 的误差，是最稳定的。
        // x 中心点误差，用高度 h 衡量
        std(0) = 2 * this->_std_weight_position * measurement[3];
        // y 中心点误差，用高度 h 衡量
        std(1) = 2 * this->_std_weight_position * measurement[3];
        // 单独处理宽度 w 的误差
        // 宽度的误差由宽度自己决定
        std(2) = 2 * this->_std_weight_position * measurement[2];
        // 高度 h 误差，用高度 h 衡量
        std(3) = 2 * this->_std_weight_position * measurement[3];
        // vx x 方向速度不确定性
        std(4) = 10 * this->_std_weight_velocity * measurement[3];
        // vy y 方向速度不确定性
        std(5) = 10 * this->_std_weight_velocity * measurement[3];
        // v_width 宽度变化率不确定性
        std(6) = 10 * this->_std_weight_velocity * measurement[2];
        // v_height 高度变化率不确定性
        std(7) = 10 * this->_std_weight_velocity * measurement[3];

        // 方差 = 标准差的平方
        KAL_MEAN tmp = std.array().square();
        // 把 8 个方差放到一个 8x8 矩阵的对角线上，形成初始协方差矩阵 P0
        KAL_COVA var = tmp.asDiagonal();
        return std::make_pair(mean, var);
    }

    void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
    {
        // 重新获取当前最新状态的尺度 (高度 mean(3) 和宽度 mean(2))
        DETECTBOX std_pos;
        std_pos << this->_std_weight_position * mean(3),
            this->_std_weight_position * mean(3),
            this->_std_weight_position * mean(2),
            this->_std_weight_position * mean(3);
        DETECTBOX std_vel;
        std_vel << this->_std_weight_velocity * mean(3),
            this->_std_weight_velocity * mean(3),
            this->_std_weight_velocity * mean(2),
            this->_std_weight_velocity * mean(3);
        KAL_MEAN tmp;
        tmp.block<1, 4>(0, 0) = std_pos;
        tmp.block<1, 4>(0, 4) = std_vel;
        // 求平方得到方差
        tmp = tmp.array().square();
        // 生成对角矩阵 Q (过程噪声)
        KAL_COVA motion_cov = tmp.asDiagonal();

        // 预测下一个状态的均值
        KAL_MEAN mean1 = this->_motion_mat * mean.transpose();

        // 预测下一个状态的协方差
        KAL_COVA covariance1 = this->_motion_mat * covariance * (this->_motion_mat.transpose());
        // 为什么要加上 Q？
        // 在信息论中，只要你没有看到新的观测数据（闭着眼睛），随着时间的流逝，你的不确定性（熵）就一定会增加。
        covariance1 += motion_cov;

        mean = mean1;
        covariance = covariance1;
    }

    KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance)
    {
        // 核心目的：统一“语言”
        // 追踪器的内部语言是 8 维的： [cx,cy,w,h,vx,vy,vw,vh]（位置 + 速度）。
        // YOLO 的测量语言是 4 维的： [cx,cy,w,h]（只有位置，YOLO 看不出速度）。

        // 构建“测量噪声矩阵” (对应公式中的 R)
        // 物理意义：评估 YOLO 这个“传感器”的固有误差。
        // 注意，这里又计算了一次误差标准差，并且用的依然是 _std_weight_position 和目标的高度/宽度。
        // 在 predict 阶段，我们计算的叫过程噪声 Q（目标自己乱动的误差）。
        // 在这里，我们计算的叫测量噪声 R（Measurement Noise）。现实中，没有任何传感器是完美的。即使目标绝对静止，YOLO 每次框出来的边界也会有几个像素的像素级抖动。这个矩阵就是在量化 YOLO 边框的这种“手抖”程度。
        DETECTBOX std;
        std << this->_std_weight_position * mean(3),
            this->_std_weight_position * mean(3),
            this->_std_weight_position * mean(2),
            this->_std_weight_position * mean(3);

        // 将均值降维
        // 物理意义： 这个矩阵乘法极其简单粗暴，就是把 8 维向量的后 4 个元素（所有速度项）直接扔掉，只保留前 4 个元素（位置项）。追踪器在说：“好了，现在我只看我预测的位置，不谈速度了”。
        KAL_HMEAN mean1 = this->_update_mat * mean.transpose();

        // 将协方差降维
        KAL_HCOVA covariance1 = this->_update_mat * covariance * (this->_update_mat.transpose());

        // 生成最终的“对比容忍度”
        // 物理意义：不确定性的终极叠加。
        // 这是为了后面的匹配做准备。当我们拿预测框和 YOLO 框去计算距离时，我们该给它们多大的“容忍度”？
        // 容忍度 (S) 来自两部分的总和：
        //  1. HPH^T ：追踪器闭眼盲猜时，自身积累的误差。
        //  2. R：YOLO 传感器自身的固有的抖动误差。
        Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
        // 生成 4x4 测量噪声协方差矩阵 R
        diag = diag.array().square().matrix();
        covariance1 += diag;

        return std::make_pair(mean1, covariance1);
    }

    KAL_DATA KalmanFilter::update(
        const KAL_MEAN &mean,
        const KAL_COVA &covariance,
        const DETECTBOX &measurement)
    {
        KAL_HDATA pa = this->project(mean, covariance);
        // 预测框在 4 维空间的均值
        KAL_HMEAN projected_mean = pa.first;
        // 预测框的综合容忍度矩阵 (S 矩阵)
        KAL_HCOVA projected_cov = pa.second;

        // 计算卡尔曼增益 (Kalman Gain)
        // 物理意义（极其重要）：
        //  卡尔曼增益 K 本质上是一个**“信任度天平”**。
        //  如果 YOLO 的框很准（测量误差 R 小），K 就会变大（接近 1），追踪器就会信任 YOLO。
        //  如果追踪器对自己的预测非常有底气（预测误差 P 小），K 就会变小（接近 0），追踪器就会信任自己的预测。
        Eigen::Matrix<float, 4, 8> B = (covariance * (this->_update_mat.transpose())).transpose();
        Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
        // 计算残差
        // 物理意义：预期与现实的落差。
        //  measurement 是实际的 YOLO 框，projected_mean 是闭眼盲猜的框。它俩相减，就是这次预测偏离现实了多少。
        Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; // eg.1x4

        // 纠正 8 维状态均值 (位置 + 速度)
        // 物理意义：知错就改。
        //  把刚才算出来的“残差”乘以“卡尔曼增益（信任权重）”，加回原先的 8 维预测值里。
        auto tmp = innovation * (kalman_gain.transpose());
        KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();

        // 收缩误差光环 (更新协方差)
        // 物理意义：重拾自信。
        //  在 predict 阶段，随着闭眼盲猜，误差光环是不断膨胀的。现在，终于拿到了实打实的 YOLO 观测数据，不确定性被消除了！所以新的协方差矩阵是用原先的矩阵“减去”一部分。你会看到目标的误差光环瞬间缩小。
        KAL_COVA new_covariance = covariance - kalman_gain * projected_cov * (kalman_gain.transpose());
        return std::make_pair(new_mean, new_covariance);
    }

    Eigen::Matrix<float, 1, -1> KalmanFilter::gating_distance(
        const KAL_MEAN &mean,
        const KAL_COVA &covariance,
        const std::vector<DETECTBOX> &measurements,
        bool only_position)
    {
        // 计算马氏距离 (Mahalanobis Distance)。
        // 作用：在匈牙利算法进行分配之前，提前把那些“在物理上绝对不可能匹配”的 YOLO 框过滤掉。

        // 马氏距离 vs 欧氏距离
        // 欧氏距离 (Euclidean Distance)： 用尺子量两个点的绝对直线距离。
        // 马氏距离 (Mahalanobis Distance)： 考虑了“误差光环形状”的距离。
        // 假设你的车在高速上匀速直线行驶，追踪器对它纵向（前进方向）的预测很不确定，但对它横向（车道）的预测非常确定。此时误差光环是一个“长椭圆”。
        // 如果旁边车道（横向）出现了一个 YOLO 框，虽然绝对距离很近，但马氏距离会判定它极其遥远（因为横向不容忍误差）；反之，如果正前方很远处（纵向）出现了一个框，马氏距离会判定它非常接近（因为纵向容忍度高）。

        // 投影与计算残差
        // 调用 project 获取预测位置和 S 矩阵，然后把所有的 YOLO 框减去预测框，得到一个残差矩阵 d。d 的每一行代表一个 YOLO 框的偏差 (cx,cy,w,h)。
        KAL_HDATA pa = this->project(mean, covariance);
        if (only_position)
        {
            printf("not implement!");
            exit(0);
        }
        // 4维预测中心点
        KAL_HMEAN mean1 = pa.first;
        // 4x4 综合容忍度矩阵 S
        KAL_HCOVA covariance1 = pa.second;

        // Cholesky 分解
        DETECTBOXSS d(measurements.size(), 4);
        int pos = 0;
        for (DETECTBOX box : measurements)
        {
            d.row(pos++) = box - mean1;
        }
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();

        // 矩阵求解
        Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();

        // 点乘求和，得出最终距离
        // 元素级平方 (z^2)
        auto zz = ((z.array()) * (z.array())).matrix();
        // 按列求和
        auto square_maha = zz.colwise().sum();
        // 返回所有 YOLO 框的马氏距离平方
        return square_maha;
    }
}