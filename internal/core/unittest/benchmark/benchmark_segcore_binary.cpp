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

#include <gtest/gtest.h>
#include <vector>

#include "benchmark_segcore.h"
#include "segcore/segcore_init_c.h"

class Benchmark_segcore_binary : public Benchmark_segcore {
 public:
    void
    test_binary_idmap(const knowhere::Config& cfg) {
        auto conf = cfg;

        std::string dsl_template = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "params": {
                                "nprobe": 1
                            },
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4
                        }
                    }
                }]
            }
        })";

        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            for (auto k : TOPKs_) {
                auto dsl = boost::format(dsl_template) % metric_type_.c_str() % k;
                CALC_TIME_SPAN(auto result = Search(dsl.str(), nq, k, conf));
                auto ids = result->seg_offsets_.data();
                float recall = CalcRecall(ids, nq, k);
                printf("  nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", nq, k, t_diff, recall);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_binary_ivf(const knowhere::Config& cfg) {
        auto conf = cfg;
        std::string nlist = conf[knowhere::indexparam::NLIST];

        std::string dsl_template = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "params": {
                                "nprobe": %d
                            },
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4
                        }
                    }
                }]
            }
        })";

        printf("\n[%0.3f s] %s | %s | nlist=%s\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               nlist.c_str());
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            for (auto nq : NQs_) {
                for (auto k : TOPKs_) {
                    auto dsl = boost::format(dsl_template) % metric_type_.c_str() % nprobe % k;
                    CALC_TIME_SPAN(auto result = Search(dsl.str(), nq, k, conf));
                    auto ids = result->seg_offsets_.data();
                    float recall = CalcRecall(ids, nq, k);
                    printf("  nprobe = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", nprobe, nq, k, t_diff,
                           recall);
                    std::fflush(stdout);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        // set_ann_test_name("sift-128-euclidean");
        set_ann_test_name("sift-4096-hamming");
        parse_ann_test_name();
        load_hdf5_data<true>();

        assert(metric_str_ == METRIC_HAM_STR || metric_str_ == METRIC_JAC_STR || metric_str_ == METRIC_TAN_STR);
        metric_type_ = (metric_str_ == METRIC_HAM_STR)   ? knowhere::metric::HAMMING
                       : (metric_str_ == METRIC_JAC_STR) ? knowhere::metric::JACCARD
                                                         : knowhere::metric::TANIMOTO;
        vector_data_type_ = milvus::DataType::VECTOR_BINARY;
        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
        cfg_[knowhere::meta::DIM] = std::to_string(dim_);

        SegcoreSetSimdType("avx2");
        CreateSchema();
        PrepareRawData();
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    const std::vector<int32_t> NQs_ = {10000};
    const std::vector<int32_t> TOPKs_ = {100};

    // IVF index params
    const std::vector<int32_t> NLISTs_ = {1024};
    const std::vector<int32_t> NPROBEs_ = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
};

TEST_F(Benchmark_segcore_binary, TEST_BINARY_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP;

    knowhere::Config conf = cfg_;
    CreateAndLoadSealedSegment(index_type_, vector_data_type_, conf);
    test_binary_idmap(conf);
}

TEST_F(Benchmark_segcore_binary, TEST_BINARY_IVF_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = std::to_string(nlist);
        CreateAndLoadSealedSegment(index_type_, vector_data_type_, conf);
        test_binary_ivf(conf);
    }
}
