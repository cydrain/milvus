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

const float RADIUS_LOW_BOUND_IP = -1.0;
const float RADIUS_HIGH_BOUND_IP = 1.0;
const float RADIUS_LOW_BOUND_L2 = 0.0;
const float RADIUS_HIGH_BOUND_L2 = 400.0;

class Benchmark_segcore_float : public Benchmark_segcore {
 public:
    void
    test_idmap(const knowhere::Config& cfg) {
        auto conf = cfg;

        std::string dsl_fmt = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4,
                            "params": {
                                "nprobe": 1
                            }
                        }
                    }
                }]
            }
        })";

        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            for (auto k : TOPKs_) {
                auto dsl = boost::format(dsl_fmt) % metric_type_.c_str() % k;
                CALC_TIME_SPAN(auto result = Search(dsl.str(), nq, k, conf));
                auto ids = result->seg_offsets_.data();
                float recall = CalcRecall(ids, nq, k);
                printf("  nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", nq, k, t_diff, recall);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_idmap_range(const knowhere::Config& cfg) {
        auto conf = cfg;

        std::string dsl_fmt = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4,
                            "params": {
                                "nprobe": 1,
                                "radius_low_bound": %f,
                                "radius_high_bound": %f
                            }
                        }
                    }
                }]
            }
        })";

        float low_bound = (metric_type_ == knowhere::metric::IP) ? RADIUS_LOW_BOUND_IP : RADIUS_LOW_BOUND_L2;
        float high_bound = (metric_type_ == knowhere::metric::IP) ? RADIUS_HIGH_BOUND_IP : RADIUS_HIGH_BOUND_L2;
        conf[RADIUS_LOW_BOUND] = low_bound;
        conf[RADIUS_HIGH_BOUND] = high_bound;

        printf("\n[%0.3f s] %s | %s, radius_low_bound = %.3f, radius_high_bound = %.3f \n", get_time_diff(),
               ann_test_name_.c_str(), index_type_.c_str(), low_bound, high_bound);
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            for (auto k : TOPKs_) {
                auto dsl = boost::format(dsl_fmt) % metric_type_.c_str() % k % low_bound % high_bound;
                CALC_TIME_SPAN(auto result = Search(dsl.str(), nq, k, conf));
                auto ids = result->seg_offsets_.data();
                float recall = CalcRecall(ids, nq, k);
                printf("  nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", nq, k, t_diff, recall);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_ivf(const knowhere::Config& cfg) {
        auto conf = cfg;
        std::string nlist = conf[knowhere::indexparam::NLIST];

        std::string dsl_fmt = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4,
                            "params": {
                                "nprobe": %d
                            }
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
                    auto dsl = boost::format(dsl_fmt) % metric_type_.c_str() % k % nprobe;
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

    void
    test_ivf_range(const knowhere::Config& cfg) {
        auto conf = cfg;
        std::string nlist = conf[knowhere::indexparam::NLIST];

        std::string dsl_fmt = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4,
                            "params": {
                                "nprobe": %d,
                                "radius_low_bound": %f,
                                "radius_high_bound": %f
                            }
                        }
                    }
                }]
            }
        })";

        float low_bound = (metric_type_ == knowhere::metric::IP) ? RADIUS_LOW_BOUND_IP : RADIUS_LOW_BOUND_L2;
        float high_bound = (metric_type_ == knowhere::metric::IP) ? RADIUS_HIGH_BOUND_IP : RADIUS_HIGH_BOUND_L2;
        conf[RADIUS_LOW_BOUND] = low_bound;
        conf[RADIUS_HIGH_BOUND] = high_bound;

        printf("\n[%0.3f s] %s | %s | nlist=%s, radius_low_bound = %.3f, radius_high_bound = %.3f\n",
               get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(), nlist.c_str(), low_bound, high_bound);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            for (auto nq : NQs_) {
                for (auto k : TOPKs_) {
                    auto dsl = boost::format(dsl_fmt) % metric_type_.c_str() % k % nprobe % low_bound % high_bound;
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

    void
    test_hnsw(const knowhere::Config& cfg) {
        auto conf = cfg;
        std::string M = conf[knowhere::indexparam::HNSW_M];
        std::string efConstruction = conf[knowhere::indexparam::EFCONSTRUCTION];

        std::string dsl_fmt = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4,
                            "params": {
                                "ef": %d
                            }
                        }
                    }
                }]
            }
        })";

        printf("\n[%0.3f s] %s | %s | M=%s | efConstruction=%s\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), M.c_str(), efConstruction.c_str());
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            for (auto nq : NQs_) {
                for (auto k : TOPKs_) {
                    auto dsl = boost::format(dsl_fmt) % metric_type_.c_str() % k % ef;
                    CALC_TIME_SPAN(auto result = Search(dsl.str(), nq, k, conf));
                    auto ids = result->seg_offsets_.data();
                    float recall = CalcRecall(ids, nq, k);
                    printf("  ef = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", ef, nq, k, t_diff, recall);
                    std::fflush(stdout);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_hnsw_range(const knowhere::Config& cfg) {
        auto conf = cfg;
        std::string M = conf[knowhere::indexparam::HNSW_M];
        std::string efConstruction = conf[knowhere::indexparam::EFCONSTRUCTION];

        std::string dsl_fmt = R"({
            "bool": {
                "must": [{
                    "vector": {
                        "vec": {
                            "metric_type": "%s",
                            "query": "$0",
                            "topk": %d,
                            "round_decimal": 4,
                            "params": {
                                "ef": %d,
                                "range_k": %d,
                                "radius_low_bound": %f,
                                "radius_high_bound": %f
                            }
                        }
                    }
                }]
            }
        })";

        float low_bound = (metric_type_ == knowhere::metric::IP) ? RADIUS_LOW_BOUND_IP : RADIUS_LOW_BOUND_L2;
        float high_bound = (metric_type_ == knowhere::metric::IP) ? RADIUS_HIGH_BOUND_IP : RADIUS_HIGH_BOUND_L2;
        conf[RADIUS_LOW_BOUND] = low_bound;
        conf[RADIUS_HIGH_BOUND] = high_bound;

        printf("\n[%0.3f s] %s | %s | M=%s | efConstruction=%s, radius_low_bound = %.3f, radius_high_bound = %.3f\n",
               get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(), M.c_str(), efConstruction.c_str(),
               low_bound, high_bound);
        printf("================================================================================\n");
        for (auto ef: EFs_) {
            for (auto range_k : HNSW_Ks_) {
                for (auto nq: NQs_) {
                    for (auto k: TOPKs_) {
                        auto dsl = boost::format(dsl_fmt) % metric_type_.c_str() % k % ef % range_k % low_bound % high_bound;
                        CALC_TIME_SPAN(auto result = Search(dsl.str(), nq, k, conf));
                        auto ids = result->seg_offsets_.data();
                        float recall = CalcRecall(ids, nq, k);
                        printf("  ef = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", ef, nq, k, t_diff,
                               recall);
                        std::fflush(stdout);
                    }
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
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();

        assert(metric_str_ == METRIC_IP_STR || metric_str_ == METRIC_L2_STR);
        metric_type_ = (metric_str_ == METRIC_IP_STR) ? knowhere::metric::IP : knowhere::metric::L2;
        vector_data_type_ = milvus::DataType::VECTOR_FLOAT;
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

    // IVFPQ index params
    const std::vector<int32_t> Ms_ = {8, 16, 32};
    const int32_t NBITS_ = 8;

    // HNSW index params
    const std::vector<int32_t> HNSW_Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {200};
    const std::vector<int32_t> EFs_ = {128, 256, 512};
    const std::vector<int32_t> HNSW_Ks_ = {20};
};

TEST_F(Benchmark_segcore_float, TEST_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    knowhere::Config conf = cfg_;
    CreateAndLoadSealedSegment(index_type_, vector_data_type_, conf);
    test_idmap(conf);
    test_idmap_range(conf);
}

TEST_F(Benchmark_segcore_float, TEST_IVF_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = std::to_string(nlist);
        CreateAndLoadSealedSegment(index_type_, vector_data_type_, conf);
        test_ivf(conf);
        test_ivf_range(conf);
    }
}

TEST_F(Benchmark_segcore_float, TEST_IVF_SQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = std::to_string(nlist);
        CreateAndLoadSealedSegment(index_type_, vector_data_type_, conf);
        test_ivf(conf);
        test_ivf_range(conf);
    }
}

TEST_F(Benchmark_segcore_float, TEST_IVF_PQ) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFPQ;

    knowhere::Config conf = cfg_;
    conf[knowhere::indexparam::NBITS] = std::to_string(NBITS_);
    for (auto m : Ms_) {
        conf[knowhere::indexparam::M] = std::to_string(m);
        for (auto nlist : NLISTs_) {
            conf[knowhere::indexparam::NLIST] = std::to_string(nlist);
            CreateAndLoadSealedSegment(index_type_, vector_data_type_, conf);
            test_ivf(conf);
            test_ivf_range(conf);
        }
    }
}

TEST_F(Benchmark_segcore_float, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Config conf = cfg_;
    for (auto M : HNSW_Ms_) {
        conf[knowhere::indexparam::HNSW_M] = std::to_string(M);
        for (auto efc : EFCONs_) {
            conf[knowhere::indexparam::EFCONSTRUCTION] = std::to_string(efc);
            CreateAndLoadSealedSegment(index_type_, vector_data_type_, conf);
            test_hnsw(conf);
            test_hnsw_range(conf);
        }
    }
}
