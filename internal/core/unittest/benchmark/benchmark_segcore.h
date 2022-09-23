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

#include <boost/format.hpp>
#include <vector>

#include "benchmark_sift.h"
#include "common/Schema.h"
#include "index/IndexFactory.h"
#include "knowhere/index/IndexType.h"
#include "segcore/SegmentSealedImpl.h"
#include "segcore/Utils.h"

class Benchmark_segcore : public Benchmark_sift {
 public:
    void
    CreateSchema() {
        printf("[%.3f s] Creating schema\n", get_time_diff());
        schema_ = std::make_shared<milvus::Schema>();
        vec_fid_ = schema_->AddDebugField("vec", vector_data_type_, dim_, metric_type_);
        pk_fid_ = schema_->AddDebugField("pk", milvus::DataType::INT64);
        schema_->set_primary_field_id(pk_fid_);
    }

    void
    PrepareRawData() {
        printf("[%.3f s] Preparing raw data\n", get_time_diff());
        raw_data_ = std::make_unique<milvus::InsertData>();
        auto insert_cols = [&](void* data, int64_t count, auto &field_meta) {
            auto array = milvus::segcore::CreateDataArrayFrom(data, count, field_meta);
            raw_data_->mutable_fields_data()->AddAllocated(array.release());
        };

        // insert vector
        auto vec_meta = schema_->operator[](vec_fid_);
        insert_cols(xb_, nb_, vec_meta);

        // insert primary key
        auto pk_meta = schema_->operator[](pk_fid_);
        std::vector<int64_t> pk(nb_);
        for (int64_t i = 0; i < nb_; i++) {
            pk[i] = i;
        }
        insert_cols(pk.data(), nb_, pk_meta);

        raw_data_->set_num_rows(nb_);

        // create row_ids_ and timestamps_
        row_ids_.resize(nb_);
        timestamps_.resize(nb_);
        for (int64_t i = 0; i < nb_; i++) {
            row_ids_[i] = i;
            timestamps_[i] = 0;
        }
    }

    milvus::index::IndexBasePtr
    CreateIndex(knowhere::IndexType& index_type,
                milvus::DataType data_type,
                knowhere::Config& conf) {
        printf("[%.3f s] Building index on %d vectors\n", get_time_diff(), nb_);
        milvus::index::CreateIndexInfo info;
        info.metric_type = knowhere::GetMetaMetricType(conf);
        info.index_type = index_type;
        info.field_type = data_type;

        auto database = knowhere::GenDataset(nb_, dim_, xb_);
        auto index = milvus::index::IndexFactory::GetInstance().CreateIndex(info, nullptr);
        index->BuildWithDataset(database, conf);
        return index;
    }

    void
    CreateAndLoadSealedSegment(knowhere::IndexType& index_type,
                               milvus::DataType data_type,
                               knowhere::Config& conf) {
        auto load_field_data = [&](int64_t fid, int64_t count, milvus::DataArray* array) {
            LoadFieldDataInfo info;
            info.field_id = fid;
            info.row_count = count;
            info.field_data = array;
            sealed_segment_->LoadFieldData(info);
        };

        printf("[%.3f s] Creating sealed segment\n", get_time_diff());
        // load index for vec field, load raw data for scalar filed
        sealed_segment_ = milvus::segcore::CreateSealedSegment(schema_);

        // load field row_ids
        printf("[%.3f s] Loading row ids for sealed segment\n", get_time_diff());
        assert(row_ids_.size() == nb_);
        milvus::FieldMeta row_ids_meta(milvus::FieldName("RowID"), RowFieldID, milvus::DataType::INT64);
        auto row_ids_array = milvus::segcore::CreateScalarDataArrayFrom(row_ids_.data(), nb_, row_ids_meta);
        load_field_data(RowFieldID.get(), nb_, row_ids_array.release());

        // load field timestamp
        printf("[%.3f s] Loading timestamps for sealed segment\n", get_time_diff());
        assert(timestamps_.size() == nb_);
        milvus::FieldMeta ts_meta(milvus::FieldName("Timestamp"), TimestampFieldID, milvus::DataType::INT64);
        auto ts_array = milvus::segcore::CreateScalarDataArrayFrom(timestamps_.data(), nb_, ts_meta);
        load_field_data(TimestampFieldID.get(), nb_, ts_array.release());

        // load user raw_data
        printf("[%.3f s] Loading raw data for sealed segment\n", get_time_diff());
        for (auto field_data : raw_data_->fields_data()) {
            load_field_data(field_data.field_id(), nb_, &field_data);
        }

        auto index = CreateIndex(index_type, data_type, conf);
        milvus::index::LoadIndexInfo load_info;
        load_info.field_id = vec_fid_.get();
        load_info.index = std::move(index);
        load_info.index_params["metric_type"] = metric_type_;

        // load index for vec field
        printf("[%.3f s] Loading index for sealed segment\n", get_time_diff());
        sealed_segment_->DropFieldData(vec_fid_);
        sealed_segment_->LoadIndex(load_info);
    }

    std::unique_ptr<milvus::SearchResult>
    Search(std::string dsl, int64_t nq, int64_t topk, knowhere::Config& conf) {
        auto plan = milvus::query::CreatePlan(*schema_, dsl);

        milvus::proto::common::PlaceholderGroup raw_group;
        milvus::proto::common::PlaceholderType place_holder_type;
        size_t code_size;

        if (vector_data_type_ == milvus::DataType::VECTOR_FLOAT) {
            place_holder_type = milvus::proto::common::PlaceholderType::FloatVector;
            code_size = dim_ * sizeof(float);
        } else if (vector_data_type_ == milvus::DataType::VECTOR_BINARY) {
            place_holder_type = milvus::proto::common::PlaceholderType::BinaryVector;
            code_size = dim_ / 8;
        } else {
            assert(false);
        }

        auto value = raw_group.add_placeholders();
        value->set_tag("$0");
        value->set_type(place_holder_type);
        for (int i = 0; i < nq; ++i) {
            value->add_values((const char*)xq_ + code_size * i, code_size);
        }
        auto place_holder_group = ParsePlaceholderGroup(plan.get(), raw_group.SerializeAsString());

        milvus::SearchInfo searchInfo;
        searchInfo.metric_type_ = knowhere::GetMetaMetricType(conf);
        searchInfo.topk_ = topk;
        searchInfo.search_params_ = conf;

        return sealed_segment_->Search(plan.get(), place_holder_group.get(), std::numeric_limits<int64_t>::max());
    }

 protected:
    knowhere::MetricType metric_type_;
    knowhere::BinarySet binary_set_;
    knowhere::IndexType index_type_;
    knowhere::VecIndexPtr index_ = nullptr;
    knowhere::Config cfg_;

    milvus::DataType vector_data_type_;
    milvus::FieldId vec_fid_;
    milvus::FieldId pk_fid_;
    milvus::SchemaPtr schema_;
    std::vector<milvus::idx_t> row_ids_;
    std::vector<milvus::Timestamp> timestamps_;
    std::unique_ptr<milvus::InsertData> raw_data_;
    milvus::segcore::SegmentSealedPtr sealed_segment_;
};
