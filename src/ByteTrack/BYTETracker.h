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
        BYTETracker(
            int max_time_lost = 15,
            float track_high_thresh = 0.5,
            float track_low_thresh = 0.1,
            float new_track_thresh = 0.6,
            float match_thresh = 0.8,
            int min_hits = 0);
        ~BYTETracker();

        void update(
            const std::vector<Object> &objects,
            // 追踪的轨迹
            std::vector<STrack> &output_stracks,
            // 临时丢失的轨迹
            std::vector<STrack> &lost_stracks,
            // 需要被永久删除的轨迹
            std::vector<STrack> &removed_stracks
        );

    private:
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

        int max_time_lost; // Number of frames allowable to go missing
        float track_high_thresh;
        float track_low_thresh;
        float new_track_thresh;
        float match_thresh;
        int min_hits; // 只有连续检测到 N 帧才输出

    private:
        int frame_id;

        std::vector<STrack> tracked_stracks;
        std::vector<STrack> lost_stracks;
        std::vector<STrack> removed_stracks; // 无用变量
        byte_kalman::KalmanFilter kalman_filter;
    };

}

#endif // BYTETRACKER_H
