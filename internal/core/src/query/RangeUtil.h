// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <vector>

#include "common/Utils.h"
#include "knowhere/common/Dataset.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/utils/BitsetView.h"
#include "log/Log.h"
#include "utils/Heap.h"

namespace milvus::query {

/*
 *   L2:   0.0        low_bound      high_bound
 *          |------------+---------------|       max_heap   ascending_order
 *         1.0        high_bound     low_bound
 *   IP:    |------------+---------------|       min_heap   descending_order
 */
template <bool is_ip>
int64_t
GetRangeSearchResultForOneNq(const int64_t topk,
                             const float low_bound,
                             const float high_bound,
                             const int64_t size,
                             const float* distances,
                             const int64_t* labels,
                             float* simi,
                             int64_t* idxi,
                             const faiss::BitsetView bitset) {
    if (is_ip) {
        minheap_heapify<float>(topk, simi, idxi);
    } else {
        maxheap_heapify<float>(topk, simi, idxi);
    }

    for (auto j = 0; j < size; j++) {
        auto dis = distances[j];
        auto id = labels[j];
        assert(bitset.empty() || !bitset.test(id));
        if (is_ip) {
            assert(dis > low_bound);
            if (dis <= high_bound && dis >= simi[0]) {
                minheap_replace_top<float>(topk, simi, idxi, dis, id);
            }
        } else {
            assert(dis < high_bound);
            if (dis >= low_bound && dis <= simi[0]) {
                maxheap_replace_top<float>(topk, simi, idxi, dis, id);
            }
        }
    }

    if (is_ip) {
        return minheap_reorder<float>(topk, simi, idxi);
    } else {
        return maxheap_reorder<float>(topk, simi, idxi);
    }
}

/* This API will re-generate TOPK sorted range search results */
inline knowhere::DatasetPtr
ReGenRangeSearchResult(const knowhere::DatasetPtr res,
                       const knowhere::MetricType& metric_type,
                       const int64_t nq,
                       const int64_t topk,
                       const float low_bound,
                       const float high_bound,
                       const faiss::BitsetView bitset) {
    int64_t cnt = 0;
    int64_t* p_id = new int64_t[topk * nq];
    float* p_dist = new float[topk * nq];

    float lb = IsMetricType(metric_type, knowhere::metric::L2) ? (low_bound * low_bound) : low_bound;
    float hb = IsMetricType(metric_type, knowhere::metric::L2) ? (high_bound * high_bound) : high_bound;

    auto lims = knowhere::GetDatasetLims(res);
    auto labels = knowhere::GetDatasetIDs(res);
    auto distances = knowhere::GetDatasetDistance(res);
    for (auto i = 0; i < nq; i++) {
        if (metric_type == knowhere::metric::IP) {
            cnt += GetRangeSearchResultForOneNq<true>(topk, lb, hb, lims[i + 1] - lims[i], distances + lims[i],
                                                      labels + lims[i], p_dist + i * topk, p_id + i * topk, bitset);
        } else {
            cnt += GetRangeSearchResultForOneNq<false>(topk, lb, hb, lims[i + 1] - lims[i], distances + lims[i],
                                                       labels + lims[i], p_dist + i * topk, p_id + i * topk, bitset);
        }
    }
    LOG_SEGCORE_DEBUG_ << "Range search metric type: " << metric_type << ", radius (" << low_bound << ", " << high_bound
                       << "), valid result num: " << cnt;

    return knowhere::GenResultDataset(p_id, p_dist);
}

}  // namespace milvus::query
