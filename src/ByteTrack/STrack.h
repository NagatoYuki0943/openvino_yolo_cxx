#ifndef STRACK_H
#define STRACK_H
#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

namespace ByteTrack
{

    enum TrackState
    {
        New = 0,
        Tracked,
        Lost,
        Removed
    };

    class STrack
    {
    public:
        STrack(std::vector<float> tlwh_, float score, int class_id, int target_id);
        ~STrack();

        std::vector<float> static tlbr_to_tlwh(std::vector<float> &tlbr);
        void static multi_predict(std::vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);
        void static_tlwh();
        void static_tlbr();
        std::vector<float> tlwh_to_xyah(std::vector<float> tlwh_tmp);
        std::vector<float> to_xyah();
        void mark_lost();
        void mark_removed();
        std::uint64_t next_id();
        int end_frame();

        void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
        void re_activate(STrack &new_track, int frame_id, bool new_id = false);
        void update(STrack &new_track, int frame_id, int min_hits);
        cv::Scalar get_color() const;

    public:
        bool is_activated;
        std::uint64_t track_id;
        int state;

        std::vector<float> _tlwh;
        std::vector<float> tlwh;
        std::vector<float> tlbr;
        int frame_id;
        int tracklet_len;
        int start_frame;

        KAL_MEAN mean;
        KAL_COVA covariance;
        float score;
        int class_id;
        int target_id;

    private:
        int _hits = 0; // 记录该目标累积匹配成功的次数
        byte_kalman::KalmanFilter kalman_filter;
    };

}

#endif // STRACK_H
