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

#include "knowhere/common/Indicator.h"
#include "log/Log.h"
#include "segcore/metric_c.h"

int64_t
GetNumOfDiskIO() {
    auto res = knowhere::IndicatorCollector::GetInstance().Get(knowhere::IndicatorType::DISK_IO_NUM).Get();
    LOG_SEGCORE_DEBUG_ << "CYD - segcore get disk IO num " << res;
    return res;
}
