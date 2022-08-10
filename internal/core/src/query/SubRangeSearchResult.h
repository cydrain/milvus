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

#pragma once

#include <limits>
#include <utility>
#include <vector>
#include "common/Types.h"

namespace milvus::query {

class SubRangeSearchResult {
 public:
    SubRangeSearchResult(const int64_t num_queries,
                         const float radius,
                         const knowhere::MetricType& metric_type,
                         const int64_t round_decimal)
        : num_queries_(num_queries),
          radius_(radius),
          round_decimal_(round_decimal),
          metric_type_(metric_type),
          lims_(num_queries + 1, 0) {
    }

    SubRangeSearchResult(SubRangeSearchResult&& other)
        : num_queries_(other.num_queries_),
          radius_(other.radius_),
          round_decimal_(other.round_decimal_),
          metric_type_(other.metric_type_),
          seg_offsets_(std::move(other.seg_offsets_)),
          distances_(std::move(other.distances_)),
          lims_(std::move(other.lims_)) {
    }

 public:
    int64_t
    get_num_queries() const {
        return num_queries_;
    }

    float
    get_radius() const {
        return radius_;
    }

    const int64_t*
    get_seg_offsets() const {
        return seg_offsets_.data();
    }

    int64_t*
    get_seg_offsets() {
        return seg_offsets_.data();
    }

    const float*
    get_distances() const {
        return distances_.data();
    }

    float*
    get_distances() {
        return distances_.data();
    }

    const size_t*
    get_lims() const {
        return lims_.data();
    }

    size_t*
    get_lims() {
        return lims_.data();
    }

    auto&
    mutable_seg_offsets() {
        return seg_offsets_;
    }

    auto&
    mutable_distances() {
        return distances_;
    }

    auto&
    mutable_lims() {
        return lims_;
    }

    bool
    empty() const {
        return lims_.back() == 0;
    }

    void
    round_values();

    void
    merge(const SubRangeSearchResult& sub_result);

 private:
    int64_t num_queries_;
    float radius_;
    int64_t round_decimal_;
    knowhere::MetricType metric_type_;
    std::vector<int64_t> seg_offsets_;
    std::vector<float> distances_;
    std::vector<size_t> lims_;
};

}  // namespace milvus::query
