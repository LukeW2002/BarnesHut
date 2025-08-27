#ifndef BH_TESTING
#define BH_TESTING
#endif
#include <gtest/gtest.h>
#include "BarnesHutParticleSystem.h"
#include "EventSystem.h"
#include "BHTestHooks.h"
#include <random>
#include <iostream>
#include <sstream>

// Utility to suppress stdout during tests
class StdoutCapture {
    std::ostringstream buffer;
    std::streambuf* old_cout;
public:
    StdoutCapture() : old_cout(std::cout.rdbuf(buffer.rdbuf())) {}
    ~StdoutCapture() { std::cout.rdbuf(old_cout); }
    std::string str() const { return buffer.str(); }
};

// Clean, minimal test output formatter
struct CleanTestOutput : testing::EmptyTestEventListener {
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    std::unique_ptr<StdoutCapture> capture;
    
    void OnTestProgramStart(const testing::UnitTest& unit_test) override {
        std::cout << "\nBarnes-Hut Test Suite\n";
        std::cout << "====================\n";
        std::cout << "Tests to run: " << unit_test.test_to_run_count() << "\n\n";
    }
    
    void OnTestSuiteStart(const testing::TestSuite& test_suite) override {
        if (test_suite.test_to_run_count() > 0) {
            std::cout << test_suite.name() << ":\n";
        }
    }
    
    void OnTestStart(const testing::TestInfo&) override {
        // Capture system output during test execution
        capture = std::make_unique<StdoutCapture>();
    }
    
    void OnTestEnd(const testing::TestInfo& test_info) override {
        // Release capture and get any output
        std::string captured_output;
        if (capture) {
            captured_output = capture->str();
            capture.reset();
        }
        
        total_tests++;
        
        if (test_info.result()->Passed()) {
            passed_tests++;
            std::cout << "  PASS " << test_info.name() << "\n";
        } else {
            failed_tests++;
            std::cout << "  FAIL " << test_info.name() << "\n";
            
            // Show failure details
            auto* result = test_info.result();
            for (int i = 0; i < result->total_part_count(); i++) {
                const auto& part = result->GetTestPartResult(i);
                if (part.failed()) {
                    std::cout << "       Expected: " << part.summary() << "\n";
                    
                    // Extract more readable failure info
                    std::string message = part.message();
                    if (!message.empty()) {
                        // Find the key failure line
                        size_t pos = message.find("Expected:");
                        if (pos != std::string::npos) {
                            size_t end = message.find('\n', pos);
                            if (end == std::string::npos) end = message.length();
                            std::cout << "       " << message.substr(pos, end - pos) << "\n";
                            
                            // Look for actual value
                            pos = message.find("Actual:", end);
                            if (pos != std::string::npos) {
                                end = message.find('\n', pos);
                                if (end == std::string::npos) end = message.length();
                                std::cout << "       " << message.substr(pos, end - pos) << "\n";
                            }
                        } else {
                            // Fallback: show first line of message
                            size_t end = message.find('\n');
                            if (end == std::string::npos) end = std::min(message.length(), size_t(80));
                            std::cout << "       " << message.substr(0, end) << "\n";
                        }
                    }
                    break; // Only show first failure
                }
            }
        }
    }
    
    void OnTestSuiteEnd(const testing::TestSuite& test_suite) override {
        if (test_suite.test_to_run_count() > 0) {
            std::cout << "\n";
        }
    }
    
    void OnTestProgramEnd(const testing::UnitTest& unit_test) override {
        std::cout << "Results\n";
        std::cout << "=======\n";
        std::cout << "Passed: " << passed_tests << "\n";
        std::cout << "Failed: " << failed_tests << "\n";
        std::cout << "Total:  " << total_tests << "\n";
        
        if (failed_tests == 0) {
            std::cout << "\nAll tests passed.\n";
        } else {
            std::cout << "\n" << failed_tests << " test(s) failed.\n";
        }
    }
};

// Basic smoke test to verify the system is working
TEST(AutoDiscovery, SystemBasicFunctionality) {
    EventBus bus;
    BarnesHutParticleSystem::Config cfg;
    BarnesHutParticleSystem sys(100, bus, cfg);
    
    // Add a few test particles
    ASSERT_TRUE(sys.add_particle({0.0f, 0.0f}, {0.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}));
    ASSERT_TRUE(sys.add_particle({1.0f, 0.0f}, {0.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}));
    ASSERT_TRUE(sys.add_particle({0.0f, 1.0f}, {0.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}));
    
    sys.set_boundary(-2.0f, 2.0f, -2.0f, 2.0f);
    
    EXPECT_EQ(sys.get_particle_count(), 3);
    
    // Update should not crash
    EXPECT_NO_THROW(sys.update(1e-6f));
    
    // All forces should be finite
    for (size_t i = 0; i < sys.get_particle_count(); ++i) {
        auto force = sys.get_force(i);
        EXPECT_TRUE(std::isfinite(force.x)) << "Force X not finite for particle " << i;
        EXPECT_TRUE(std::isfinite(force.y)) << "Force Y not finite for particle " << i;
    }
}

// Test that verifies BH_TESTING hooks work
TEST(AutoDiscovery, TestHooksAccessible) {
#ifdef BH_TESTING
    EventBus bus;
    BarnesHutParticleSystem::Config cfg;
    BarnesHutParticleSystem sys(50, bus, cfg);
    
    // Add particles
    for (int i = 0; i < 10; ++i) {
        sys.add_particle({static_cast<float>(i) * 0.1f, 0.0f}, {0.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f});
    }
    
    sys.set_boundary(-1.0f, 2.0f, -1.0f, 1.0f);
    sys.update(1e-6f);
    
    // Test hooks should work
    auto snapshot = BHTestHooks::snapshot(sys);
    
    EXPECT_EQ(snapshot.N, 10);
    EXPECT_FALSE(snapshot.nodes.empty());
    EXPECT_NE(snapshot.root, UINT32_MAX);
#else
    GTEST_SKIP() << "BH_TESTING not defined";
#endif
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Replace default output with clean formatter
    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new CleanTestOutput());
    
    return RUN_ALL_TESTS();
}
