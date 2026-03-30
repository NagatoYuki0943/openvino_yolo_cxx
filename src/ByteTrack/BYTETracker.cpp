#include "BYTETracker.h"
#include <fstream>

namespace ByteTrack
{

    BYTETracker::BYTETracker(int max_time_lost, float track_high_thresh, float track_low_thresh, float new_track_thresh, float match_thresh)
    {
        set_max_time_lost(max_time_lost);
        set_track_high_thresh(track_high_thresh);
        set_track_low_thresh(track_low_thresh);
        set_new_track_thresh(new_track_thresh);
        set_match_thresh(match_thresh);

        frame_id = 0;
    }

    BYTETracker::~BYTETracker()
    {
    }

    void BYTETracker::update(const std::vector<Object> &objects, std::vector<STrack> &lost_stracks, std::vector<STrack> &output_stracks)
    {
        ////////////////// Step 1: Get detections (获取并划分当前帧的检测结果) //////////////////
        this->frame_id++; // 帧计数器递增
        // 定义不同状态的轨迹集合，用于后续更新状态机
        std::vector<STrack> activated_stracks; // 成功匹配并更新的活跃轨迹
        std::vector<STrack> refind_stracks;    // 从 Lost 状态重新找回的轨迹
        std::vector<STrack> removed_stracks;   // 需要被永久删除的轨迹
        std::vector<STrack> detections;        // 高分检测框集合 (得分 >= track_high_thresh)
        std::vector<STrack> detections_low;    // 低分检测框集合 (得分在 track_low_thresh 和 track_high_thresh 之间)

        std::vector<STrack> detections_cp;
        std::vector<STrack> tracked_stracks_swap;
        std::vector<STrack> resa, resb;

        std::vector<STrack *> unconfirmed;       // 未确认的轨迹（通常是上一帧刚生成，还未稳定追踪的轨迹）
        std::vector<STrack *> tracked_stracks;   // 处于稳定追踪状态的轨迹
        std::vector<STrack *> strack_pool;       // 参与第一轮匹配的轨迹池（包含稳定追踪和丢失的轨迹）
        std::vector<STrack *> r_tracked_stracks; // 第一轮没匹配上，留给第二轮匹配的轨迹

        // 1.1 将当前帧的 YOLO 检测结果按照置信度分为“高分框”和“低分框”
        if (objects.size() > 0)
        {
            for (int i = 0; i < objects.size(); i++)
            {
                std::vector<float> tlbr_;
                tlbr_.resize(4);
                tlbr_[0] = objects[i].rect.x;
                tlbr_[1] = objects[i].rect.y;
                tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
                tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

                float score = objects[i].prob;
                int class_id = objects[i].class_id;
                int target_id = objects[i].target_id;

                // 将检测框转换为追踪器使用的 STrack 对象
                STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, class_id, target_id);
                if (score >= track_high_thresh)
                {
                    detections.push_back(strack); // 优等生：进入高分列表
                }
                else if (score >= track_low_thresh)
                {
                    detections_low.push_back(strack); // 替补队员：进入低分列表，专门用来修补遮挡
                }
            }
        }

        // 1.2 将追踪器中已有的轨迹划分为“未确认 (unconfirmed)”和“已确认 (tracked)”
        for (int i = 0; i < this->tracked_stracks.size(); i++)
        {
            if (!this->tracked_stracks[i].is_activated)
                unconfirmed.push_back(&this->tracked_stracks[i]); // 刚出生还没稳固的轨迹
            else
                tracked_stracks.push_back(&this->tracked_stracks[i]); // 稳定追踪中的轨迹
        }

        ////////////////// Step 2: First association, with IoU (第一轮匹配：高分框匹配) //////////////////
        // 2.1 将稳定追踪的轨迹和丢失(Lost)的轨迹合并，作为第一轮匹配的目标池
        strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);

        // 2.2 使用卡尔曼滤波器预测这些轨迹在当前帧的位置
        STrack::multi_predict(strack_pool, this->kalman_filter);

        // 2.3 计算预测位置与高分检测框之间的 IoU 距离矩阵 (代价矩阵)
        std::vector<std::vector<float>> dists;
        int dist_size = 0, dist_size_size = 0;
        dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

        // 2.4 使用匈牙利算法 (Linear Assignment) 进行最优二分图匹配
        std::vector<std::vector<int>> matches;
        std::vector<int> u_track, u_detection; // u_track: 未匹配上的轨迹, u_detection: 未匹配上的高分框
        linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

        // 2.5 处理第一轮成功匹配的结果
        for (int i = 0; i < matches.size(); i++)
        {
            STrack *track = strack_pool[matches[i][0]];
            STrack *det = &detections[matches[i][1]];
            if (track->state == TrackState::Tracked)
            {
                // 稳定轨迹正常更新状态
                track->update(*det, this->frame_id);
                activated_stracks.push_back(*track);
            }
            else
            {
                // 从 Lost 状态找回的轨迹，重新激活
                track->re_activate(*det, this->frame_id, false);
                refind_stracks.push_back(*track);
            }
        }

        ////////////////// Step 3: Second association, using low score dets (第二轮匹配：ByteTrack核心，低分框匹配遮挡目标) //////////////////
        // 3.1 备份第一轮未匹配的高分框 (留给后面去匹配 unconfirmed 轨迹，或生成新轨迹)
        for (int i = 0; i < u_detection.size(); i++)
        {
            detections_cp.push_back(detections[u_detection[i]]);
        }
        // 清空 detections，把低分框装进来作为第二轮的检测候选
        detections.clear();
        detections.assign(detections_low.begin(), detections_low.end());

        // 3.2 筛选出第一轮没匹配上的“处于稳定追踪状态”的轨迹 (Lost 状态的轨迹不参与低分框匹配)
        for (int i = 0; i < u_track.size(); i++)
        {
            if (strack_pool[u_track[i]]->state == TrackState::Tracked)
            {
                r_tracked_stracks.push_back(strack_pool[u_track[i]]);
            }
        }

        // 3.3 计算这些未匹配轨迹与“低分检测框”之间的 IoU 距离
        dists.clear();
        dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size, false);

        // 3.4 再次使用匈牙利算法匹配 (匹配阈值固定设为 0.5)
        matches.clear();
        u_track.clear();
        u_detection.clear();
        linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

        // 3.5 处理第二轮成功匹配的结果 (成功挽救了被遮挡的目标)
        for (int i = 0; i < matches.size(); i++)
        {
            STrack *track = r_tracked_stracks[matches[i][0]];
            STrack *det = &detections[matches[i][1]];
            if (track->state == TrackState::Tracked)
            {
                track->update(*det, this->frame_id);
                activated_stracks.push_back(*track);
            }
            else
            {
                track->re_activate(*det, this->frame_id, false);
                refind_stracks.push_back(*track);
            }
        }

        // 3.6 经过两轮匹配依然没有找到对应框的轨迹，标记为 Lost (丢失)
        for (int i = 0; i < u_track.size(); i++)
        {
            STrack *track = r_tracked_stracks[u_track[i]];
            if (track->state != TrackState::Lost)
            {
                track->mark_lost();
                lost_stracks.push_back(*track);
            }
        }

        // Deal with unconfirmed tracks, usually tracks with only one beginning frame
        // 处理未确认的轨迹 (通常是上一帧刚产生的新轨迹)
        detections.clear();
        detections.assign(detections_cp.begin(), detections_cp.end()); // 拿出刚才备份的、第一轮没用掉的“高分框”

        dists.clear();
        dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

        matches.clear();
        std::vector<int> u_unconfirmed;
        u_detection.clear();
        // 匹配未确认轨迹与剩余高分框 (匹配阈值较严格，0.7)
        linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

        for (int i = 0; i < matches.size(); i++)
        {
            // 匹配成功，更新状态，此时轨迹内部的 is_activated 会变成 true
            unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
            activated_stracks.push_back(*unconfirmed[matches[i][0]]);
        }

        for (int i = 0; i < u_unconfirmed.size(); i++)
        {
            // 没匹配上的未确认轨迹，说明上一帧很可能是个误检，直接永久标记为 Removed 抹杀掉
            STrack *track = unconfirmed[u_unconfirmed[i]];
            track->mark_removed();
            removed_stracks.push_back(*track);
        }

        ////////////////// Step 4: Init new stracks (初始化新轨迹) //////////////////
        // 对于到了最后依然没有被任何历史轨迹认领的高分检测框，将其视为新目标的出现
        for (int i = 0; i < u_detection.size(); i++)
        {
            STrack *t = &detections[u_detection[i]];
            if (t->score < new_track_thresh) // 如果高分框里有得分不够 new_track_thresh 的，不予创建
                continue;
            // 激活新轨迹 (此时分配 Track ID，但 is_activated 状态取决于这是不是第1帧)
            t->activate(this->kalman_filter, this->frame_id);
            activated_stracks.push_back(*t);
        }

        ////////////////// Step 5: Update state (更新追踪器内部的全局状态) //////////////////
        // 5.1 检查所有处于 Lost 状态的轨迹，如果丢失时间超过 max_time_lost，则彻底删除 (Removed)
        for (int i = 0; i < this->lost_stracks.size(); i++)
        {
            if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
            {
                this->lost_stracks[i].mark_removed();
                removed_stracks.push_back(this->lost_stracks[i]);
            }
        }

        // 5.2 整理当前活跃的 tracked_stracks
        for (int i = 0; i < this->tracked_stracks.size(); i++)
        {
            if (this->tracked_stracks[i].state == TrackState::Tracked)
            {
                tracked_stracks_swap.push_back(this->tracked_stracks[i]);
            }
        }
        this->tracked_stracks.clear();
        this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

        // 将本帧新匹配成功、新创建的、以及重新找回的轨迹合并到 tracked_stracks 列表中
        this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
        this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

        // 5.3 从全局 lost_stracks 列表中剔除掉那些刚才被找回(Tracked)或被删除(Removed)的轨迹
        this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
        for (int i = 0; i < lost_stracks.size(); i++)
        {
            this->lost_stracks.push_back(lost_stracks[i]); // 加上本帧新标记为 lost 的轨迹
        }

        this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
        for (int i = 0; i < removed_stracks.size(); i++)
        {
            this->removed_stracks.push_back(removed_stracks[i]);
        }

        // 5.4 剔除可能存在的重复轨迹 (通常基于 IoU 去重，防止同一个目标产生了多条轨迹)
        remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

        this->tracked_stracks.clear();
        this->tracked_stracks.assign(resa.begin(), resa.end());
        this->lost_stracks.clear();
        this->lost_stracks.assign(resb.begin(), resb.end());

        // 5.5 组装最终要输出给外部的结果
        for (int i = 0; i < this->tracked_stracks.size(); i++)
        {
            // 只有“已激活/已确认”的轨迹才能输出给前端，过滤掉“一闪而过”的幽灵误检框
            if (this->tracked_stracks[i].is_activated)
            {
                output_stracks.push_back(this->tracked_stracks[i]);
            }
        }

        // 更新函数外部传入的 lost_stracks 引用，便于外部调用者可能需要的统计逻辑
        lost_stracks.clear();
        for (int i = 0; i < this->lost_stracks.size(); i++)
        {
            lost_stracks.push_back(this->lost_stracks[i]);
        }
    }

}