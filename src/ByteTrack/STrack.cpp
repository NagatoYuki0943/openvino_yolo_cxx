#include "STrack.h"

namespace ByteTrack
{

    STrack::STrack(std::vector<float> tlwh_, float score, int class_id, int target_id)
    {
        this->_tlwh.resize(4);
        this->_tlwh.assign(tlwh_.begin(), tlwh_.end());

        this->is_activated = false;
        this->track_id = 0;
        this->state = TrackState::New;

        this->tlwh.resize(4);
        this->tlbr.resize(4);

        this->static_tlwh();
        this->static_tlbr();
        this->frame_id = 0;
        this->tracklet_len = 0;
        this->score = score;
        this->class_id = class_id;
        this->target_id = target_id;
        this->start_frame = 0;
    }

    STrack::~STrack()
    {
    }

    void STrack::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id)
    {
        this->kalman_filter = kalman_filter;
        // 【解决id增长过快修改点】：延迟分配 id
        // this->track_id = this->next_id();

        std::vector<float> _tlwh_tmp(4);
        _tlwh_tmp[0] = this->_tlwh[0];
        _tlwh_tmp[1] = this->_tlwh[1];
        _tlwh_tmp[2] = this->_tlwh[2];
        _tlwh_tmp[3] = this->_tlwh[3];
        std::vector<float> xywh = this->tlwh_to_xywh(_tlwh_tmp);
        DETECTBOX xywh_box;
        xywh_box[0] = xywh[0];
        xywh_box[1] = xywh[1];
        xywh_box[2] = xywh[2];
        xywh_box[3] = xywh[3];
        auto mc = this->kalman_filter.initiate(xywh_box);
        this->mean = mc.first;
        this->covariance = mc.second;

        this->static_tlwh();
        this->static_tlbr();

        this->tracklet_len = 0;
        this->state = TrackState::Tracked;

        this->_hits = 1;            // 初始命中次数为 1
        this->track_id = 0;         // 此时还不分配 ID
        this->is_activated = false; // 默认不激活

        // 【解决id增长过快修改点】：判断是否为第一帧
        // if (frame_id == 1)
        // {
        //     this->is_activated = true;
        // }
        // if (frame_id == 1)
        // {
        //     this->is_activated = true;
        //     this->track_id = this->next_id(); // 只有第一帧的框直接转正，发身份证
        // }
        // else
        // {
        //     this->is_activated = false; // 其他帧的新框，进入考察期
        //     this->track_id = 0;         // 分配一个无效的占位 ID
        // }

        this->frame_id = frame_id;
        this->start_frame = frame_id;
    }

    void STrack::re_activate(STrack &new_track, int frame_id, bool new_id)
    {
        std::vector<float> xywh = this->tlwh_to_xywh(new_track.tlwh);
        DETECTBOX xywh_box;
        xywh_box[0] = xywh[0];
        xywh_box[1] = xywh[1];
        xywh_box[2] = xywh[2];
        xywh_box[3] = xywh[3];
        auto mc = this->kalman_filter.update(this->mean, this->covariance, xywh_box);
        this->mean = mc.first;
        this->covariance = mc.second;

        this->static_tlwh();
        this->static_tlbr();

        this->tracklet_len = 0;
        this->state = TrackState::Tracked;
        this->is_activated = true;
        this->frame_id = frame_id;
        this->score = new_track.score;
        // 继承最新一帧对应的检测框索引
        this->target_id = new_track.target_id;

        if (new_id)
            this->track_id = this->next_id();
    }

    void STrack::update(STrack &new_track, int frame_id, int min_hits)
    {
        this->frame_id = frame_id;
        this->tracklet_len++;
        this->_hits++; // 每匹配到一次，计数加 1

        std::vector<float> xywh = this->tlwh_to_xywh(new_track.tlwh);
        DETECTBOX xywh_box;
        xywh_box[0] = xywh[0];
        xywh_box[1] = xywh[1];
        xywh_box[2] = xywh[2];
        xywh_box[3] = xywh[3];

        auto mc = this->kalman_filter.update(this->mean, this->covariance, xywh_box);
        this->mean = mc.first;
        this->covariance = mc.second;

        this->static_tlwh();
        this->static_tlbr();

        this->state = TrackState::Tracked;

        // 【解决id增长过快修改点】：如果是从“未激活”状态变为“激活”状态，此时才分配全局唯一 ID
        // this->is_activated = true;
        // 只有达到命中阈值，才分配 ID 并激活
        if (!this->is_activated && this->_hits >= min_hits)
        {
            this->track_id = this->next_id(); // 考察期通过，正式发身份证
            this->is_activated = true;
        }

        this->score = new_track.score;
        // 继承最新一帧对应的检测框索引
        this->target_id = new_track.target_id;
    }

    void STrack::static_tlwh()
    {
        if (this->state == TrackState::New)
        {
            this->tlwh[0] = this->_tlwh[0];
            this->tlwh[1] = this->_tlwh[1];
            this->tlwh[2] = this->_tlwh[2];
            this->tlwh[3] = this->_tlwh[3];
            return;
        }

        this->tlwh[0] = this->mean[0];
        this->tlwh[1] = this->mean[1];
        this->tlwh[2] = this->mean[2];
        this->tlwh[3] = this->mean[3];

        // 上面直接使用 width, 这里不用再还原
        // this->tlwh[2] *= this->tlwh[3];
        this->tlwh[0] -= this->tlwh[2] / 2;
        this->tlwh[1] -= this->tlwh[3] / 2;
    }

    void STrack::static_tlbr()
    {
        this->tlbr.clear();
        this->tlbr.assign(this->tlwh.begin(), this->tlwh.end());
        this->tlbr[2] += this->tlbr[0];
        this->tlbr[3] += this->tlbr[1];
    }

    std::vector<float> STrack::tlwh_to_xywh(std::vector<float> tlwh_tmp)
    {
        std::vector<float> tlwh_output = tlwh_tmp;
        tlwh_output[0] += tlwh_output[2] / 2; // x_center
        tlwh_output[1] += tlwh_output[3] / 2; // y_center
        // tlwh_output[2] /= tlwh_output[3];  // 宽高比
        return tlwh_output;
    }

    std::vector<float> STrack::to_xywh()
    {
        return this->tlwh_to_xywh(this->tlwh);
    }

    std::vector<float> STrack::tlbr_to_tlwh(std::vector<float> &tlbr)
    {
        std::vector<float> tlwh = tlbr;
        tlwh[2] -= tlwh[0];
        tlwh[3] -= tlwh[1];
        return tlwh;
    }

    void STrack::mark_lost()
    {
        this->state = TrackState::Lost;
    }

    void STrack::mark_removed()
    {
        this->state = TrackState::Removed;
    }

    std::uint64_t STrack::next_id()
    {
        static std::uint64_t _count = 1;
        _count++;
        return _count;
    }

    int STrack::end_frame()
    {
        return this->frame_id;
    }

    void STrack::multi_predict(std::vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter)
    {
        for (int i = 0; i < stracks.size(); i++)
        {
            if (stracks[i]->state != TrackState::Tracked)
            {
                stracks[i]->mean[7] = 0;
            }
            kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
            stracks[i]->static_tlwh();
            stracks[i]->static_tlbr();
        }
    }

    cv::Scalar STrack::get_color() const
    {
        int idx = this->track_id + 3;
        return cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
    }

}
