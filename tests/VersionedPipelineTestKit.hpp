#pragma once

#include <concepts>
#include <type_traits>
#include <string_view>
#include <functional>
#include <stdexcept>
#include <initializer_list>

// ===========================================================================================
// GENERIC VERSIONED PIPELINE TEST KIT
// Domain-agnostic framework for testing multiple algorithm implementations
// ===========================================================================================

namespace VersionedPipelineTestKit {

// ===========================================================================================
// A) SPEC INTERFACE - Domain-specific plugin that defines types and invariants
// ===========================================================================================

template<typename S>
concept PipelineSpec = requires {
    // Associated types for THIS domain
    typename S::Input;     // What goes in (e.g., particles, keys, text)
    typename S::State;     // Per-run mutable state (or std::monostate if stateless)
    typename S::Output;    // What comes out (e.g., partitions, sorted data, parsed tree)
    
    // Required static functions
    { S::gen_input(std::size_t{}, unsigned{}) } -> std::same_as<typename S::Input>;
    { S::check_invariants(std::declval<const typename S::Input&>(),
                          std::declval<const typename S::Output&>()) } -> std::same_as<void>;
};

// ===========================================================================================
// B) VERSION INTERFACE - Implementation that works for a given Spec
// ===========================================================================================

template<typename V, typename Spec>
concept VersionFor = PipelineSpec<Spec> && requires(typename Spec::State& state,
                                                    const typename Spec::Input& input) {
    // Identity
    { V::name() } -> std::convertible_to<std::string_view>;
    
    // Lifecycle
    { V::make_state() } -> std::same_as<typename Spec::State>;
    { V::set_input(state, input) } -> std::same_as<void>;
    { V::run(state) } -> std::same_as<typename Spec::Output>;
};

// Optional: Versions that support stage-by-stage execution
template<typename V, typename Spec>
concept StagedVersionFor = VersionFor<V, Spec> && requires(typename Spec::State& state) {
    { V::has_stages } -> std::convertible_to<bool>;
    { V::stage_count() } -> std::convertible_to<int>;
    { V::run_stage(state, int{}) } -> std::same_as<void>;
    { V::stage_name(int{}) } -> std::convertible_to<std::string_view>;
};

// ===========================================================================================
// C) ADAPTER TEMPLATES - Convert various patterns into VersionFor interface
// ===========================================================================================

// 1) Adapt three "prime" functions into a single Version
template<typename Spec, typename Prepare, typename Process, typename Finalize>
struct VersionFromPrimes {
    static std::string_view name() { 
        return Prepare::name(); // or combine all names
    }
    
    static typename Spec::State make_state() { 
        return Prepare::make_state(); 
    }
    
    static void set_input(typename Spec::State& state, const typename Spec::Input& input) {
        Prepare::set_input(state, input);
    }
    
    static typename Spec::Output run(typename Spec::State& state) {
        Prepare::prepare(state);
        Process::process(state);
        return Finalize::finalize(state);
    }
};

// 2) Adapt a "system" object with methods into a Version
template<typename Spec, typename System>
struct VersionFromSystem {
    static std::string_view name() { 
        return System::name(); 
    }
    
    static typename Spec::State make_state() { 
        return typename Spec::State{}; 
    }
    
    static void set_input(typename Spec::State& state, const typename Spec::Input& input) {
        System::load(state, input);
    }
    
    static typename Spec::Output run(typename Spec::State& state) {
        return System::execute(state);
    }
};

// 3) Adapt a pure functional interface
template<typename Spec, auto RunFunction>
struct VersionFromFunction {
    static std::string_view name() { 
        return RunFunction.name(); 
    }
    
    static typename Spec::State make_state() { 
        return typename Spec::State{}; 
    }
    
    static void set_input(typename Spec::State& state, const typename Spec::Input& input) {
        // Store input in state for pure function call
        if constexpr (requires { state.input = input; }) {
            state.input = input;
        }
    }
    
    static typename Spec::Output run(typename Spec::State& state) {
        if constexpr (requires { RunFunction(state.input); }) {
            return RunFunction(state.input);
        } else {
            return RunFunction(state);
        }
    }
};

// ===========================================================================================
// D) TYPE LIST AND COMPILE-TIME UTILITIES
// ===========================================================================================

template<typename... Ts> 
struct type_list {};

template<template<typename> class Fn, typename... Ts>
inline void for_each_type(type_list<Ts...>) { 
    (Fn<Ts>{}(), ...); 
}

// Convert type_list to GoogleTest Types for TYPED_TEST_SUITE
template<typename... Ts> 
struct as_gtest_types;

template<typename... Ts> 
struct as_gtest_types<type_list<Ts...>> { 
    using type = ::testing::Types<Ts...>; 
};

// ===========================================================================================
// E) TEST HARNESS UTILITIES
// ===========================================================================================

// Run one version: create state, set input, run, return output
template<typename Version, typename Spec>
requires VersionFor<Version, Spec>
inline typename Spec::Output run_once(const typename Spec::Input& input) {
    auto state = Version::make_state();
    Version::set_input(state, input);
    return Version::run(state);
}

// Test property for single version
template<typename Version, typename Spec, typename Property>
requires VersionFor<Version, Spec>
inline void check_property(const typename Spec::Input& input, Property&& property) {
    auto output = run_once<Version, Spec>(input);
    property(input, output);
}

// Compare N versions for equivalence using custom equality predicate
template<typename Spec, typename Equal, typename V0, typename... Vs>
requires (VersionFor<V0, Spec> && (VersionFor<Vs, Spec> && ...)) && PipelineSpec<Spec>
inline void expect_equivalence(const typename Spec::Input& input, Equal&& eq) {
    auto output0 = run_once<V0, Spec>(input);
    
    auto check_version = [&]<typename V>() {
        auto output = run_once<V, Spec>(input);
        if (!eq(output0, output)) {
            throw std::runtime_error(
                std::string(V::name()) + " not equivalent to " + std::string(V0::name())
            );
        }
    };
    
    (check_version.template operator()<Vs>(), ...);
}

// Overload for default equality (operator==)
template<typename Spec, typename V0, typename... Vs>
requires (VersionFor<V0, Spec> && (VersionFor<Vs, Spec> && ...)) && PipelineSpec<Spec>
inline void expect_equivalence(const typename Spec::Input& input) {
    expect_equivalence<Spec>(input, std::equal_to<typename Spec::Output>{});
}

// Run all versions and check invariants for each
template<typename Spec, typename... Versions>
requires (VersionFor<Versions, Spec> && ...) && PipelineSpec<Spec>
inline void check_all_invariants(const typename Spec::Input& input) {
    auto check_version = [&]<typename V>() {
        auto output = run_once<V, Spec>(input);
        Spec::check_invariants(input, output);
    };
    
    (check_version.template operator()<Versions>(), ...);
}

// Staged execution for debugging/profiling
template<typename Version, typename Spec>
requires StagedVersionFor<Version, Spec>
inline typename Spec::Output run_with_stages(const typename Spec::Input& input) {
    auto state = Version::make_state();
    Version::set_input(state, input);
    
    for (int stage = 0; stage < Version::stage_count(); ++stage) {
        // Could add timing/logging here
        Version::run_stage(state, stage);
    }
    
    return Version::get_output(state); // Assuming this method exists
}

// ===========================================================================================
// F) COMMON PROPERTY PREDICATES (reusable across domains)
// ===========================================================================================

// Check determinism: same input always produces same output
template<typename Version, typename Spec>
requires VersionFor<Version, Spec>
struct DeterminismProperty {
    void operator()(const typename Spec::Input& input) {
        auto output1 = run_once<Version, Spec>(input);
        auto output2 = run_once<Version, Spec>(input);
        
        if (!(output1 == output2)) {
            throw std::runtime_error(
                std::string(Version::name()) + " is not deterministic"
            );
        }
    }
};

// Check that invariants hold
template<typename Spec>
struct InvariantProperty {
    void operator()(const typename Spec::Input& input, const typename Spec::Output& output) {
        Spec::check_invariants(input, output);
    }
};

// Performance measurement property
template<typename Version, typename Spec>
struct PerformanceProperty {
    int trials;
    std::function<void(std::chrono::nanoseconds)> report;
    
    void operator()(const typename Spec::Input& input) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < trials; ++i) {
            [[maybe_unused]] auto output = run_once<Version, Spec>(input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        if (report) report(duration / trials);
    }
};

// ===========================================================================================
// G) GOOGLETEST INTEGRATION MACROS (optional convenience)
// ===========================================================================================

#ifdef GTEST_VERSION

#define DEFINE_SPEC_TEST_SUITE(TestName, Spec, VersionList) \
    template<typename Version> \
    class TestName : public ::testing::Test { \
    protected: \
        using SpecType = Spec; \
        using VersionType = Version; \
    }; \
    TYPED_TEST_SUITE(TestName, typename as_gtest_types<VersionList>::type)

#define TYPED_SPEC_TEST(TestName, TestCase) \
    TYPED_TEST(TestName, TestCase)

#define EXPECT_INVARIANTS_HOLD(input) \
    do { \
        auto output = run_once<TypeParam, SpecType>(input); \
        EXPECT_NO_THROW(SpecType::check_invariants(input, output)); \
    } while(0)

#define EXPECT_DETERMINISTIC(input) \
    do { \
        auto output1 = run_once<TypeParam, SpecType>(input); \
        auto output2 = run_once<TypeParam, SpecType>(input); \
        EXPECT_EQ(output1, output2) << TypeParam::name() << " is not deterministic"; \
    } while(0)

#endif // GTEST_VERSION

} // namespace VersionedPipelineTestKit

