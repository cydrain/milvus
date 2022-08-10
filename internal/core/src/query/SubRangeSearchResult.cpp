// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <cmath>

#include "exceptions/EasyAssert.h"
#include "query/SubRangeSearchResult.h"

namespace milvus::query {

void
SubRangeSearchResult::merge(const SubRangeSearchResult& right) {
    AssertInfo(metric_type_ == right.metric_type_, "[SubRangeSearchResult]Metric type check failed");
    AssertInfo(num_queries_ == right.num_queries_, "[SubRangeSearchResult]Nq check failed");
    AssertInfo(radius_ == right.radius_, "[SubRangeSearchResult]Radius check failed");

    std::vector<float> buf_distances;
    std::vector<int64_t> buf_ids;
    std::vector<size_t> buf_lims(num_queries_ + 1);

    for (int64_t qn = 0; qn < num_queries_; ++qn) {
        auto left_offset = this->get_lims()[qn];
        auto left_num = this->get_lims()[qn + 1] - this->get_lims()[qn];
        int64_t* __restrict__ left_ids = this->get_seg_offsets() + left_offset;
        float* __restrict__ left_distances = this->get_distances() + left_offset;

        auto right_offset = right.get_lims()[qn];
        auto right_num = right.get_lims()[qn + 1] - right.get_lims()[qn];
        auto right_ids = right.get_seg_offsets() + right_offset;
        auto right_distances = right.get_distances() + right_offset;

        buf_lims[qn + 1] = this->get_lims()[qn + 1] + right.get_lims()[qn + 1];
        auto offset = buf_lims[qn];
        buf_ids.insert(buf_ids.begin() + offset, left_ids, left_ids + left_num);
        buf_ids.insert(buf_ids.begin() + offset + left_num, right_ids, right_ids + right_num);
        buf_distances.insert(buf_distances.begin() + offset, left_distances, left_distances + left_num);
        buf_distances.insert(buf_distances.begin() + offset + left_num, right_distances, right_distances + right_num);
    }
    this->mutable_seg_offsets().swap(buf_ids);
    this->mutable_distances().swap(buf_distances);
    this->mutable_lims().swap(buf_lims);
}

void
SubRangeSearchResult::round_values() {
    if (round_decimal_ == -1) {
        return;
    }
    const float multiplier = pow(10.0, round_decimal_);
    for (auto it = this->distances_.begin(); it != this->distances_.end(); it++) {
        *it = round(*it * multiplier) / multiplier;
    }
}

}  // namespace milvus::query
