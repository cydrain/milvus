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

#include <string>
#include <vector>

#include "SearchBruteForce.h"
#include "SubRangeSearchResult.h"
#include "SubSearchResult.h"
#include "log/Log.h"
#include "knowhere/archive/BruteForce.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

namespace milvus::query {

SubSearchResult
BruteForceSearch(const dataset::SearchDataset& dataset,
                 const void* chunk_data_raw,
                 int64_t chunk_rows,
                 const BitsetView& bitset) {
    LOG_SEGCORE_DEBUG_ << "CYD - BruteForceSearch";
    SubSearchResult sub_result(dataset.num_queries, dataset.topk, dataset.metric_type, dataset.round_decimal);
    try {
        auto nq = dataset.num_queries;
        auto dim = dataset.dim;
        auto topk = dataset.topk;

        auto base_dataset = knowhere::GenDataset(chunk_rows, dim, chunk_data_raw);
        auto query_dataset = knowhere::GenDataset(nq, dim, dataset.query_data);
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, dataset.metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::TOPK, topk},
        };
        auto result = knowhere::BruteForce::Search(base_dataset, query_dataset, config, bitset);

        sub_result.mutable_seg_offsets().resize(nq * topk);
        sub_result.mutable_distances().resize(nq * topk);

        std::copy_n(knowhere::GetDatasetIDs(result), nq * topk, sub_result.get_seg_offsets());
        std::copy_n(knowhere::GetDatasetDistance(result), nq * topk, sub_result.get_distances());
    } catch (std::exception& e) {
        PanicInfo(e.what());
    }
    sub_result.round_values();
    return sub_result;
}

SubRangeSearchResult
BruteForceRangeSearch(const dataset::RangeSearchDataset& dataset,
                      const void* chunk_data_raw,
                      int64_t chunk_rows,
                      const BitsetView& bitset) {
    LOG_SEGCORE_DEBUG_ << "CYD - BruteForceRangeSearch";
    SubRangeSearchResult sub_result(dataset.num_queries, dataset.radius, dataset.metric_type, dataset.round_decimal);
    try {
        auto nq = dataset.num_queries;
        auto dim = dataset.dim;
        auto radius = dataset.radius;

        auto base_dataset = knowhere::GenDataset(chunk_rows, dim, chunk_data_raw);
        auto query_dataset = knowhere::GenDataset(nq, dim, dataset.query_data);
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, dataset.metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::RADIUS, radius},
        };
        auto result = knowhere::BruteForce::RangeSearch(base_dataset, query_dataset, config, bitset);
        auto lims = knowhere::GetDatasetLims(result);
        auto total_num = lims[nq];

        sub_result.mutable_lims().resize(nq + 1);
        sub_result.mutable_seg_offsets().resize(total_num);
        sub_result.mutable_distances().resize(total_num);

        std::copy_n(knowhere::GetDatasetIDs(result), total_num, sub_result.get_seg_offsets());
        std::copy_n(knowhere::GetDatasetDistance(result), total_num, sub_result.get_distances());
        std::copy_n(knowhere::GetDatasetLims(result), nq + 1, sub_result.get_lims());
    } catch (std::exception& e) {
        PanicInfo(e.what());
    }
    sub_result.round_values();
    return sub_result;
}

}  // namespace milvus::query
