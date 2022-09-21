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

#include "common/Consts.h"
#include "exceptions/EasyAssert.h"
#include "knowhere/archive/BruteForce.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "query/RangeUtil.h"
#include "query/SearchBruteForce.h"
#include "query/SubSearchResult.h"

namespace milvus::query {

SubSearchResult
BruteForceSearch(const dataset::SearchDataset& dataset,
                 const void* chunk_data_raw,
                 const int64_t chunk_rows,
                 const knowhere::Config& config,
                 const BitsetView& bitset) {
    auto metric_type = dataset.metric_type;
    auto nq = dataset.num_queries;
    auto dim = dataset.dim;
    auto topk = dataset.topk;
    auto sub_result = SubSearchResult(nq, topk, metric_type, dataset.round_decimal);

    auto base_dataset = knowhere::GenDataset(chunk_rows, dim, chunk_data_raw);
    auto query_dataset = knowhere::GenDataset(nq, dim, dataset.query_data);
    auto conf = knowhere::Config{
        {knowhere::meta::METRIC_TYPE, metric_type},
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, topk},
    };

    try {
        /* if radius_low_bound and radius_high_bound are set, do range search, otherwise do search */
        knowhere::DatasetPtr result;
            float low_bound = -1.0;
            float high_bound = 1000000;

            if (metric_type == knowhere::metric::IP) {
                knowhere::SetMetaRadius(conf, low_bound);
            } else {
                knowhere::SetMetaRadius(conf, high_bound);
            }

            auto res = knowhere::BruteForce::RangeSearch(base_dataset, query_dataset, conf, bitset);
            result = ReGenRangeSearchResult(res, metric_type, nq, topk, low_bound, high_bound, bitset);

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

}  // namespace milvus::query
