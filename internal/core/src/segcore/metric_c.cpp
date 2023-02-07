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

#include "knowhere/metrics/Prometheus.h"
#include "knowhere/metrics/SystemInfoCollector.h"
#include "log/Log.h"
#include "segcore/metric_c.h"

int64_t
GetNumOfDiskIO() {
//    auto res = knowhere::IndicatorCollector::GetInstance().Get(knowhere::IndicatorType::DISK_IO_NUM).Get();
    int64_t res = 0;
    LOG_SEGCORE_DEBUG_ << "CYD - segcore get disk IO num " << res;
    return res;
}

char*
GetKnowhereAllMetrics() {
    knowhere::SystemInfoCollector::GetInstance().Start();
    sleep(5);
    auto str = knowhere::Prometheus::GetInstance().GetMetrics();
    LOG_SEGCORE_DEBUG_ << "CYD - segcore get knowhere all metrics:\n" << str;
    size_t len = str.length();
    char* res = (char*)malloc(len + 1);
    std::memset(res, 0, len + 1);
    std::memcpy(res, str.data(), len);
    return res;
}