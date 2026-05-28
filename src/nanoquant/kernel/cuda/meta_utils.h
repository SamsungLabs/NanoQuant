// Copyright (c) 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// For TORCH_CHECK_VALUE
#include <torch/extension.h>

#include <sstream>
#include <string>
#include <type_traits>

// Helper to sweep through a compile-time list of values
// and call a function with a std::integral_constant for the matching runtime value.
template <typename T, T... values>
struct parameter_sweep {
    template <typename F, typename U>
    void operator()(const char* name, U runtime_value, F&& func) const {
        bool found = false;

        (
            [&]() {
                if (!found && runtime_value == values) {
                    std::forward<decltype(func)>(func)(std::integral_constant<T, values>{});
                    found = true;
                }
            }(),
            ...);

        auto gen_choices = [&]() -> std::string {
            std::stringstream ss;
            bool is_first = true;
            (((is_first ? ss : (ss << ", ")) << values, is_first = false), ...);
            return ss.str();
        };

        TORCH_CHECK_VALUE(found, name, " should be one of [", gen_choices(), "], but ", runtime_value, " was given");
    }
};
