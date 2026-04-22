#ifndef BYTETRACKER_H
#define BYTETRACKER_H
#pragma once

#include "STrack.h"

namespace ByteTrack
{

    struct Object
    {
        int target_id;
        int class_id;
        float prob;
        cv::Rect_<float> rect;
    };

    class BYTETracker
    {
    public:
        // 死亡倒计时，目标丢失（未匹配到检测框）后，在内存中保留等待重新出现的总帧数
        int max_time_lost;
        // 高分界线，得分大于此值的框为“高分框”，参与第一轮常规匹配。
        float track_high_thresh;
        // 低分界线，得分在此值与高分界线之间的为“低分框”，参与第二轮遮挡修补；低于此值的框直接丢弃。
        float track_low_thresh;
        // 出生门槛，只有得分大于此值的检测框，才能被初始化为全新的追踪目标。
        float new_track_thresh;
        // match_thresh/low_match_thresh/unconfirmed_match_thresh 认亲标准，判定检测框与已有轨迹“是否为同一目标”的匹配代价容忍度（通常基于 IoU）。越高，越容易匹配；越低，越难匹配。
        //  什么时候应该调高（比如 0.8 ~ 0.9）？
        //      视频帧率（FPS）较低时。
        //      目标运动速度极快时。
        //      原因： 这种情况下，目标在相邻两帧之间的位移跨度很大，导致上一帧的预测框和这一帧的检测框交集 (IoU) 很小。如果不把容忍度调高，追踪器会认为老目标消失了，新目标出现了，导致疯狂闪烁和切换 ID。
        //  什么时候应该调低（比如 0.4 ~ 0.5）？
        //      画面极其拥挤密集时（例如密集的车流、十字路口的人群）。
        //      原因： 当多个目标靠得非常近时，如果追踪器太宽容，很容易发生 ID 劫持（比如 A 车和 B 车并排，追踪器把 A 的 ID 错误地连到了 B 的检测框上）。降低阈值能逼迫追踪器变得“严谨”，只认准那个跟历史轨迹重合度最高的目标。
        // match_thresh 已追踪到的轨迹+丢失的规矩 vs 高分目标 的阈值
        // low_match_thresh 剩余已追踪到的轨迹 vs 低分目标 的阈值
        // unconfirmed_match_thresh 候选轨迹 vs 剩余高分目标 的阈值
        float match_thresh;
        float low_match_thresh;
        float unconfirmed_match_thresh;
        // min_hits 连续追踪到多少帧才认定是追踪目标
        int min_hits;

        // std_weight_position
        //  物理意义：对 YOLO 检测框的位置抖动有多大的包容心？
        //  调大（比如 0.1）：追踪器会变得“很软”，它认为位置发生跳变是正常的。结果就是，预测框会死死地贴着 YOLO 的检测框，YOLO 抖它也跟着抖。
        //  调小（比如 0.01）：追踪器会变得“很硬”，它认为目标不可能瞬间瞬移。它会像加了极其沉重的减震器一样，画出一条极度平滑的轨迹，无视 YOLO 偶尔的瞎跳。
        // std_weight_velocity
        //  物理意义：我对目标“急刹车、急转弯、突然加速”有多大的包容心？
        //  调大（比如 0.02）：追踪器认为目标非常灵活，随时可能改变速度。当目标突然转弯时，追踪器能瞬间反应过来跟上去。缺点是，对于匀速目标，轨迹可能会画出波浪线。
        //  调小（比如 0.001）：追踪器坚信目标是个“铁憨憨”，只会老老实实做匀速直线运动。
        BYTETracker(
            int max_time_lost = 15,
            float track_high_thresh = 0.5,
            float track_low_thresh = 0.1,
            float new_track_thresh = 0.6,
            float match_thresh = 0.8,
            float low_match_thresh = 0.5,
            float unconfirmed_match_thresh = 0.8,
            int min_hits = 0,
            float std_weight_position = 1. / 20,
            float std_weight_velocity = 1. / 10);
        ~BYTETracker();

        void update(
            const std::vector<Object> &objects,
            // 追踪的轨迹
            std::vector<STrack> &output_stracks,
            // 临时丢失的轨迹
            std::vector<STrack> &lost_stracks,
            // 需要被永久删除的轨迹
            std::vector<STrack> &removed_stracks);

    private:
        int frame_id;

        std::vector<STrack> tracked_stracks;
        std::vector<STrack> lost_stracks;
        std::vector<STrack> removed_stracks; // 无用变量
        byte_kalman::KalmanFilter kalman_filter;

        std::vector<STrack *> joint_stracks(std::vector<STrack *> &tlista, std::vector<STrack> &tlistb);
        std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

        std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
        void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);

        void linear_assignment(std::vector<std::vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
                               std::vector<std::vector<int>> &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
        std::vector<std::vector<float>> iou_distance(std::vector<STrack *> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size, bool fuse_score = true);
        std::vector<std::vector<float>> iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
        std::vector<std::vector<float>> ious(std::vector<std::vector<float>> &atlbrs, std::vector<std::vector<float>> &btlbrs);

        double lapjv(const std::vector<std::vector<float>> &cost, std::vector<int> &rowsol, std::vector<int> &colsol,
                     bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);
    };

}

#endif // BYTETRACKER_H
