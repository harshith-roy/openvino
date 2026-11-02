// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/istreams_executor.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/util/log.hpp"
#include "parallel_custom_arena.hpp"

namespace ov {
namespace threading {

IStreamsExecutor::~IStreamsExecutor() {}

void IStreamsExecutor::Config::set_property(const std::string& key, const ov::Any& value) {
    set_property({{key, value}});
}

void IStreamsExecutor::Config::set_property(const ov::AnyMap& property) {
    for (const auto& it : property) {
        const auto& key = it.first;
        const auto& value = it.second;
        if (key == ov::num_streams) {
            auto streams = value.as<ov::streams::Num>();
            if (streams == ov::streams::NUMA) {
                _streams = get_num_numa_nodes();
            } else if (streams == ov::streams::AUTO) {
                // bare minimum of streams (that evenly divides available number of cores)
                _streams = get_default_num_streams();
            } else if (streams.num >= 0) {
                _streams = streams.num;
            } else {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::num_streams.name(),
                               ". Expected non negative numbers (#streams) or ",
                               "ov::streams::NUMA|ov::streams::AUTO, Got: ",
                               streams);
            }
        } else if (key == ov::inference_num_threads) {
            int val_i;
            try {
                val_i = value.as<int>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::inference_num_threads.name(),
                               ". Expected only non negative integer numbers.");
            }
            OPENVINO_ASSERT(val_i >= 0,
                            "Wrong value for property key ",
                            ov::inference_num_threads.name(),
                            ". Expected only non negative integer numbers. Got: ",
                            val_i);
            _threads = val_i;
        } else if (key == ov::internal::threads_per_stream) {
            _threads_per_stream = static_cast<int>(value.as<size_t>());
        } else {
            OPENVINO_THROW("Not recognized property key ", key);
        }
    }
}

ov::Any IStreamsExecutor::Config::get_property(const std::string& key) const {
    if (key == ov::supported_properties) {
        std::vector<std::string> properties{
            ov::num_streams.name(),
            ov::inference_num_threads.name(),
            ov::internal::threads_per_stream.name(),
        };
        return properties;
    } else if (key == ov::num_streams) {
        return decltype(ov::num_streams)::value_type{_streams};
    } else if (key == ov::inference_num_threads) {
        return decltype(ov::inference_num_threads)::value_type{_threads};
    } else if (key == ov::internal::threads_per_stream) {
        return decltype(ov::internal::threads_per_stream)::value_type{_threads_per_stream};
    } else {
        OPENVINO_THROW("Wrong value for property key ", key);
    }
    return {};
}

int IStreamsExecutor::Config::get_default_num_streams() {
    // bare minimum of streams (that evenly divides available number of core)
    const auto proc_type_table = get_proc_type_table();
    if (proc_type_table.empty()) {
        return 1;
    }
    const auto num_cores = proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC] + proc_type_table[0][LP_EFFICIENT_CORE_PROC];
    if (0 == num_cores % 4)
        return std::max(4, num_cores / 4);
    else if (0 == num_cores % 5)
        return std::max(5, num_cores / 5);
    else if (0 == num_cores % 3)
        return std::max(3, num_cores / 3);
    else  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
        return 1;
}

IStreamsExecutor::Config IStreamsExecutor::Config::make_default_multi_threaded(
    const IStreamsExecutor::Config& initial) {
    const auto proc_type_table = get_proc_type_table();
    auto streamConfig = initial;

    if (proc_type_table.empty()) {
        return streamConfig;
    }

    int num_cores = proc_type_table[0][ALL_PROC];

    if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0 || proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0) && proc_type_table[0][MAIN_CORE_PROC] > 0) {
        if (streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::ANY_CORE) {
            num_cores = proc_type_table[0][ALL_PROC];
        } else if (streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY) {
            num_cores = proc_type_table[0][MAIN_CORE_PROC];
        } else if (streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
            num_cores = proc_type_table[0][EFFICIENT_CORE_PROC] + proc_type_table[0][LP_EFFICIENT_CORE_PROC];
        }
    }

    const auto threads = streamConfig._threads ? streamConfig._threads : num_cores;
    int threads_per_stream = streamConfig._streams ? std::max(1, threads / streamConfig._streams) : threads;
    if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0 || proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0) && proc_type_table[0][MAIN_CORE_PROC] > 0 &&
        streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::ANY_CORE) {
        if (streamConfig._streams > 1) {
            threads_per_stream =
                std::min(std::min(proc_type_table[0][MAIN_CORE_PROC], proc_type_table[0][EFFICIENT_CORE_PROC] + proc_type_table[0][LP_EFFICIENT_CORE_PROC]),
                         threads_per_stream);
            while (1) {
                int streams_num = proc_type_table[0][MAIN_CORE_PROC] / threads_per_stream +
                                  proc_type_table[0][HYPER_THREADING_PROC] / threads_per_stream +
                                  proc_type_table[0][EFFICIENT_CORE_PROC] / threads_per_stream +
                                  proc_type_table[0][LP_EFFICIENT_CORE_PROC] / threads_per_stream;
                if (streams_num >= streamConfig._streams) {
                    break;
                } else if (threads_per_stream > 1) {
                    threads_per_stream--;
                } else {
                    // No further reduction can increase the number of streams.
                    break;
                }
            }
        }
    }
    streamConfig._threads_per_stream = threads_per_stream;
    streamConfig._threads = streamConfig._threads_per_stream * streamConfig._streams;
    streamConfig.update_executor_config();
    return streamConfig;
}

void IStreamsExecutor::Config::update_executor_config() {
    const auto proc_type_table = get_proc_type_table();
    bool streams_info_available = false;

    if (proc_type_table.empty() || proc_type_table[0][ALL_PROC] == 0) {
        if (_cpu_reservation) {
            OPENVINO_THROW("No CPU resources available for reservation. All available CPUs may already be reserved "
                           "by previously compiled models. Release those models or recompile them with a lower "
                           "INFERENCE_NUM_THREADS value to reserve fewer CPUs, then retry.");
        } else {
            return;
        }
    }

    if (!_streams_info_table.empty()) {
        streams_info_available = true;
        std::vector<int> threads_proc_type(HYPER_THREADING_PROC + 1, 0);
        int threads_all = 0;
        for (size_t i = 0; i < _streams_info_table.size(); i++) {
            if (_streams_info_table[i][NUMBER_OF_STREAMS] > 0) {
                int num_threads =
                    _streams_info_table[i][THREADS_PER_STREAM] * _streams_info_table[i][NUMBER_OF_STREAMS];
                threads_proc_type[_streams_info_table[i][PROC_TYPE]] += num_threads;
                threads_all += num_threads;
            }
        }
        if (threads_all == 0) {
            OPENVINO_THROW("streams_info_table is invalid!");
        }
        for (size_t i = ALL_PROC; i < threads_proc_type.size(); i++) {
            if (threads_proc_type[i] > proc_type_table[0][i]) {
                OPENVINO_THROW("Not enough CPU resources!");
            }
        }
    }

    if (!streams_info_available) {
        _streams_info_table.clear();

        const auto total_num_cores = proc_type_table[0][ALL_PROC];
        const auto total_num_big_cores = proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
        const auto total_num_little_cores = proc_type_table[0][EFFICIENT_CORE_PROC] + proc_type_table[0][LP_EFFICIENT_CORE_PROC];

        if ((total_num_little_cores == 0 && _thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) ||
            (total_num_big_cores == 0 && _thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY) ||
            (proc_type_table.size() > 1 && _thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY)) {
            _thread_preferred_core_type = ov::hint::SchedulingCoreType::ANY_CORE;
        }

        int num_cores = total_num_cores;
        if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY) {
            num_cores = total_num_big_cores;
        } else if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
            num_cores = total_num_little_cores;
        }

        _streams = _streams > 0 ? (_cores_limit == true ? std::min(_streams, num_cores) : _streams) : _streams;
        if (_streams == 0) {
            set_config_zero_stream();
            return;
        }

        _threads_per_stream = (_threads_per_stream > 0 && _cores_limit)
                                  ? std::min(num_cores, _streams * _threads_per_stream) / _streams
                                  : 0;
        // _threads_per_stream = 0: not use tbb to create threads
        if (_threads_per_stream == 0) {
            _cpu_reservation = false;
            _cpu_pinning = false;
            return;
        }

        // create stream_info_table based on core type
        std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE, 0);
        stream_info[THREADS_PER_STREAM] = _threads_per_stream;
        stream_info[STREAM_NUMA_NODE_ID] = 0;
        stream_info[STREAM_SOCKET_ID] = 0;
        int cur_threads = _streams * _threads_per_stream;
        if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
            struct EcorePool {
                int proc_type;
                int numa_node_id;
                int socket_id;
                int cores;
                int stream_capacity;
                int streams = 0;
                int remaining_cores = 0;
            };

            std::vector<EcorePool> pools;
            const size_t first_domain = proc_type_table.size() > 1 ? 1 : 0;
            int total_cores = 0;
            int total_stream_capacity = 0;
            for (size_t row = first_domain; row < proc_type_table.size(); ++row) {
                for (const int proc_type : {EFFICIENT_CORE_PROC, LP_EFFICIENT_CORE_PROC}) {
                    const int cores = proc_type_table[row][proc_type];
                    if (cores > 0) {
                        const int capacity = cores / _threads_per_stream;
                        pools.push_back({proc_type,
                                         proc_type_table[row][PROC_NUMA_NODE_ID],
                                         proc_type_table[row][PROC_SOCKET_ID],
                                         cores,
                                         capacity});
                        total_cores += cores;
                        total_stream_capacity += capacity;
                    }
                }
            }

            const int target_streams = std::min(_streams, total_stream_capacity);
            int allocated_streams = 0;
            for (auto& pool : pools) {
                const long long weighted_stream_quota =
                    total_cores > 0 ? static_cast<long long>(target_streams) * pool.cores / total_cores : 0;
                pool.streams = std::min(pool.stream_capacity, static_cast<int>(weighted_stream_quota));
                allocated_streams += pool.streams;
            }
            while (allocated_streams < target_streams) {
                auto best_pool = pools.end();
                long long best_deficit = std::numeric_limits<long long>::min();
                for (auto pool = pools.begin(); pool != pools.end(); ++pool) {
                    if (pool->streams >= pool->stream_capacity) {
                        continue;
                    }
                    const long long deficit =
                        static_cast<long long>(target_streams) * pool->cores -
                        static_cast<long long>(pool->streams) * total_cores;
                    if (deficit > best_deficit) {
                        best_deficit = deficit;
                        best_pool = pool;
                    }
                }
                if (best_pool == pools.end()) {
                    break;
                }
                ++best_pool->streams;
                ++allocated_streams;
            }

            for (auto& pool : pools) {
                if (pool.streams > 0) {
                    auto row_info = stream_info;
                    row_info[NUMBER_OF_STREAMS] = pool.streams;
                    row_info[PROC_TYPE] = pool.proc_type;
                    row_info[STREAM_NUMA_NODE_ID] = pool.numa_node_id;
                    row_info[STREAM_SOCKET_ID] = pool.socket_id;
                    _streams_info_table.push_back(std::move(row_info));
                }
                pool.remaining_cores = pool.cores - pool.streams * _threads_per_stream;
            }

            int remaining_cores = 0;
            for (const auto& pool : pools) {
                remaining_cores += pool.remaining_cores;
            }
            const int composite_streams =
                std::min(_streams - allocated_streams, remaining_cores / _threads_per_stream);
            for (int stream = 0; stream < composite_streams; ++stream) {
                int composite_capacity = 0;
                for (const auto& pool : pools) {
                    composite_capacity += pool.remaining_cores;
                }
                std::vector<int> component_cores(pools.size(), 0);
                std::vector<long long> remainders(pools.size(), -1);
                int assigned_cores = 0;
                for (size_t i = 0; i < pools.size(); ++i) {
                    if (pools[i].remaining_cores > 0) {
                        const long long numerator =
                            static_cast<long long>(_threads_per_stream) * pools[i].remaining_cores;
                        component_cores[i] = static_cast<int>(numerator / composite_capacity);
                        remainders[i] = numerator % composite_capacity;
                        assigned_cores += component_cores[i];
                    }
                }
                while (assigned_cores < _threads_per_stream) {
                    size_t best_pool = pools.size();
                    for (size_t i = 0; i < pools.size(); ++i) {
                        if (component_cores[i] < pools[i].remaining_cores &&
                            (best_pool == pools.size() || remainders[i] > remainders[best_pool])) {
                            best_pool = i;
                        }
                    }
                    OPENVINO_ASSERT(best_pool < pools.size(),
                                    "Unable to distribute ECORE_ONLY composite stream across available pools.");
                    ++component_cores[best_pool];
                    remainders[best_pool] = -1;
                    ++assigned_cores;
                }

                int composite_numa_node = -1;
                int composite_socket = -1;
                bool first_component = true;
                bool spans_domains = false;
                for (size_t i = 0; i < pools.size(); ++i) {
                    if (component_cores[i] == 0) {
                        continue;
                    }
                    if (first_component) {
                        composite_numa_node = pools[i].numa_node_id;
                        composite_socket = pools[i].socket_id;
                        first_component = false;
                    } else if (composite_numa_node != pools[i].numa_node_id ||
                               composite_socket != pools[i].socket_id) {
                        spans_domains = true;
                    }
                }

                std::vector<int> composite_info(CPU_STREAMS_TABLE_SIZE, 0);
                composite_info[NUMBER_OF_STREAMS] = 1;
                composite_info[PROC_TYPE] = ALL_PROC;
                composite_info[THREADS_PER_STREAM] = _threads_per_stream;
                composite_info[STREAM_NUMA_NODE_ID] = spans_domains ? -1 : composite_numa_node;
                composite_info[STREAM_SOCKET_ID] = spans_domains ? -1 : composite_socket;
                _streams_info_table.push_back(std::move(composite_info));

                for (size_t i = 0; i < pools.size(); ++i) {
                    if (component_cores[i] == 0) {
                        continue;
                    }
                    std::vector<int> component_info(CPU_STREAMS_TABLE_SIZE, 0);
                    component_info[PROC_TYPE] = pools[i].proc_type;
                    component_info[THREADS_PER_STREAM] = component_cores[i];
                    component_info[STREAM_NUMA_NODE_ID] = pools[i].numa_node_id;
                    component_info[STREAM_SOCKET_ID] = pools[i].socket_id;
                    _streams_info_table.push_back(std::move(component_info));
                    pools[i].remaining_cores -= component_cores[i];
                }
            }
        } else {
            int start = proc_type_table.size() > 1 ? 1 : 0;
            std::vector<int> core_types;
            const auto active_core_types =
                (proc_type_table[0][MAIN_CORE_PROC] > 0) +
                (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) +
                (proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0) +
                (proc_type_table[0][HYPER_THREADING_PROC] > 0);
            const bool has_multiple_numa_nodes = proc_type_table.size() > 2;
            bool composite_stream = false;
            // Use a composite stream when the request crosses processor types or NUMA allocation domains.
            if (_streams == 1 && _threads_per_stream > proc_type_table[start][ov::MAIN_CORE_PROC] &&
                (active_core_types > 1 || has_multiple_numa_nodes)) {
                composite_stream = true;
                stream_info[NUMBER_OF_STREAMS] = _streams;
                stream_info[PROC_TYPE] = ALL_PROC;
                stream_info[STREAM_NUMA_NODE_ID] = proc_type_table.size() > 1 ? -1 : 0;
                stream_info[STREAM_SOCKET_ID] = proc_type_table.size() > 1 ? -1 : 0;
                _streams_info_table.push_back(stream_info);
                stream_info[NUMBER_OF_STREAMS] = 0;
            }
            if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY &&
                (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 || proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0)) {
                core_types = {MAIN_CORE_PROC, HYPER_THREADING_PROC};
            } else {
                core_types = {MAIN_CORE_PROC, EFFICIENT_CORE_PROC, LP_EFFICIENT_CORE_PROC, HYPER_THREADING_PROC};
            }
            for (int j : core_types) {
                for (size_t i = start; i < proc_type_table.size(); i++) {
                    if (proc_type_table[i][j] > 0 && cur_threads > 0) {
                        auto row_stream_info = stream_info;
                        row_stream_info[THREADS_PER_STREAM] = _threads_per_stream;
                        if (_threads_per_stream > proc_type_table[i][j]) {
                            row_stream_info[THREADS_PER_STREAM] = std::min(proc_type_table[i][j], cur_threads);
                            row_stream_info[NUMBER_OF_STREAMS] = composite_stream ? 0 : 1;
                            cur_threads -= row_stream_info[THREADS_PER_STREAM];
                        } else {
                            row_stream_info[NUMBER_OF_STREAMS] =
                                std::min(proc_type_table[i][j], cur_threads) / _threads_per_stream;
                            if (!composite_stream && row_stream_info[NUMBER_OF_STREAMS] == 0 && cur_threads > 0) {
                                row_stream_info[THREADS_PER_STREAM] = std::min(proc_type_table[i][j], cur_threads);
                                row_stream_info[NUMBER_OF_STREAMS] = 1;
                            }
                            cur_threads -= row_stream_info[NUMBER_OF_STREAMS] * row_stream_info[THREADS_PER_STREAM];
                        }
                        if (composite_stream) {
                            row_stream_info[NUMBER_OF_STREAMS] = 0;
                        }
                        row_stream_info[PROC_TYPE] = j;
                        row_stream_info[STREAM_NUMA_NODE_ID] = proc_type_table[i][PROC_NUMA_NODE_ID];
                        row_stream_info[STREAM_SOCKET_ID] = proc_type_table[i][PROC_SOCKET_ID];
                        if (composite_stream || row_stream_info[NUMBER_OF_STREAMS] > 0) {
                            _streams_info_table.push_back(std::move(row_stream_info));
                        }
                    }
                }
            }
        }
    }

    OPENVINO_ASSERT(!_streams_info_table.empty(), "No CPU stream allocation could be created.");

    if (_cpu_pinning || _cpu_reservation) {
        reserve_available_cpus(_streams_info_table, _stream_processor_ids, _cpu_reservation ? CPU_USED : NOT_USED);
    }

    // Recaculate _streams, _threads and _threads_per_stream by _streams_info_table
    int num_streams = 0;
    _threads = 0;
    _sub_streams = 0;
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        if (_streams_info_table[i][NUMBER_OF_STREAMS] > 0) {
            num_streams += _streams_info_table[i][NUMBER_OF_STREAMS];
            _threads += _streams_info_table[i][NUMBER_OF_STREAMS] * _streams_info_table[i][THREADS_PER_STREAM];
        } else if (_streams_info_table[i][NUMBER_OF_STREAMS] == -1) {
            _sub_streams += 1;
        }
    }
    _threads_per_stream = _streams_info_table[0][THREADS_PER_STREAM];
    _streams = _streams > 0 ? num_streams : _streams;

#ifdef ENABLE_OPENVINO_DEBUG
    OPENVINO_DEBUG("[ threading ] proc_type_table:");
    for (size_t i = 0; i < proc_type_table.size(); i++) {
        OPENVINO_DEBUG(proc_type_table[i][ALL_PROC],
                       proc_type_table[i][MAIN_CORE_PROC],
                       " ",
                       proc_type_table[i][EFFICIENT_CORE_PROC],
                       " ",
                       proc_type_table[i][HYPER_THREADING_PROC],
                       " ",
                       proc_type_table[i][PROC_NUMA_NODE_ID],
                       " ",
                       proc_type_table[i][PROC_SOCKET_ID]);
    }

    OPENVINO_DEBUG("[ threading ] streams_info_table:");
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        OPENVINO_DEBUG(_streams_info_table[i][NUMBER_OF_STREAMS],
                       " ",
                       _streams_info_table[i][PROC_TYPE],
                       " ",
                       _streams_info_table[i][THREADS_PER_STREAM],
                       " ",
                       _streams_info_table[i][STREAM_NUMA_NODE_ID],
                       " ",
                       _streams_info_table[i][STREAM_SOCKET_ID]);
    }
    OPENVINO_DEBUG("[ threading ] )", _name, ": ", _streams, "(", _threads, ")");
#endif
}

void IStreamsExecutor::Config::update_executor_config(bool lock) {
    if (lock) {
        {
            std::lock_guard<std::mutex> lock{_streams_executor_mutex};
            update_executor_config();
        }
    } else {
        update_executor_config();
    }
}

void IStreamsExecutor::Config::set_config_zero_stream() {
    std::vector<std::vector<int>> proc_type_table = get_proc_type_table();
    int core_type = MAIN_CORE_PROC;
    int numa_id = 0;
    int socket_id = 0;

    if (proc_type_table.size() > 0) {
        core_type = proc_type_table[0][MAIN_CORE_PROC] > 0
                        ? MAIN_CORE_PROC
                        : (proc_type_table[0][EFFICIENT_CORE_PROC] > 0
                               ? EFFICIENT_CORE_PROC
                               : (proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0 ? LP_EFFICIENT_CORE_PROC : HYPER_THREADING_PROC));
        numa_id = std::max(0, proc_type_table[0][PROC_NUMA_NODE_ID]);
        socket_id = std::max(0, proc_type_table[0][PROC_SOCKET_ID]);
    }
    _streams_info_table.push_back({1, core_type, 1, numa_id, socket_id});
    _cpu_reservation = false;
    _cpu_pinning = false;
}

}  // namespace threading
}  // namespace ov
