#!/bin/bash

# Parse command line arguments
BUILD_TESTS=true
AUTO_RUN_TESTS=true

for arg in "$@"; do
    case $arg in
        --no-test|--no-tests)
            BUILD_TESTS=false
            AUTO_RUN_TESTS=false
            echo "Tests disabled via command line option"
            ;;
        --help|-h)
            echo "Barnes-Hut Performance Build + Auto-Testing"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --no-test, --no-tests    Skip building and running tests"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Default behavior: Build with comprehensive auto-testing"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
done

if $BUILD_TESTS; then
    echo "BARNES-HUT PERFORMANCE BUILD + AUTO-TESTING"
    echo "==========================================="
else
    echo "BARNES-HUT PERFORMANCE BUILD (NO TESTS)"
    echo "======================================="
fi

# Check if we're in the right directory
if [[ ! -d "header" || ! -d "src" ]]; then
    echo "Error: Please run this script from the BarnesHutMrkII project root"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Function to install dependencies
install_dependencies() {
    echo "Checking dependencies..."
    
    local needs_install=false
    
    # Check for OpenMP
    if ! brew list libomp &> /dev/null; then
        echo "Installing OpenMP for maximum parallelization..."
        brew install libomp
        needs_install=true
    else
        echo "OpenMP found"
    fi
    
    # Only check for GoogleTest if we're building tests
    if $BUILD_TESTS; then
        if ! brew list googletest &> /dev/null; then
            echo "Installing GoogleTest for comprehensive testing..."
            brew install googletest
            needs_install=true
        else
            echo "GoogleTest found"
        fi
    else
        echo "Skipping GoogleTest (tests disabled)"
    fi
    
    # Check for Eigen (usually comes with other packages)
    if [[ ! -d "/opt/homebrew/include/eigen3" ]] && [[ ! -d "/usr/local/include/eigen3" ]]; then
        echo "Installing Eigen3 for linear algebra..."
        brew install eigen
        needs_install=true
    else
        echo "Eigen3 found"
    fi
    
    if $needs_install; then
        echo "Dependencies installed successfully"
    else
        echo "All dependencies already installed"
    fi
}

# Install dependencies
if command -v brew &> /dev/null; then
    install_dependencies
else
    echo "Warning: Homebrew not found, skipping dependency check"
fi

# Setup libs directory
mkdir -p libs

# Download ImGui if not present
if [[ ! -f "libs/imgui/imgui.h" ]]; then
    echo "Downloading ImGui..."
    git clone https://github.com/ocornut/imgui.git libs/imgui
    echo "ImGui downloaded"
else
    echo "ImGui already present"
fi

# Build configuration
echo ""
if $BUILD_TESTS; then
    echo "CONFIGURING BUILD WITH AUTO-TESTING"
    echo "===================================="
    BUILD_DIR="build_test"
else
    echo "CONFIGURING PERFORMANCE BUILD (NO TESTS)"
    echo "========================================"
    BUILD_DIR="build_release"
fi

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# Get CPU count for parallel compilation
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
echo "Using $CPU_CORES CPU cores for compilation"

# Configure CMake based on test preference
echo "Configuring CMake..."
if $BUILD_TESTS; then
    # Build with tests and auto-run
    cmake \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DENABLE_OPENMP=ON \
        -DBUILD_TESTS=ON \
        -DRUN_TESTS_AFTER_BUILD=$AUTO_RUN_TESTS \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -fno-math-errno -ffp-contract=fast -march=native -mtune=native" \
        -DCMAKE_OBJCXX_FLAGS_RELWITHDEBINFO="-O3 -g -fno-math-errno -ffp-contract=fast -march=native -mtune=native" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ..
else
    # Pure performance build without tests
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_OPENMP=ON \
        -DBUILD_TESTS=OFF \
        -DRUN_TESTS_AFTER_BUILD=OFF \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -fno-math-errno -flto -ffast-math -funroll-loops" \
        -DCMAKE_OBJCXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -flto -ffast-math -funroll-loops" \
        -DCMAKE_EXE_LINKER_FLAGS_RELEASE="-flto -Wl,-dead_strip" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ..
fi

if [[ $? -ne 0 ]]; then
    echo "CMake configuration failed"
    exit 1
fi

echo ""
if $BUILD_TESTS; then
    echo "BUILDING PROJECT + TESTS"
    echo "======================="
else
    echo "BUILDING HIGH-PERFORMANCE EXECUTABLE"
    echo "==================================="
fi

# Build everything
make -j$CPU_CORES

build_status=$?

if [[ $build_status -eq 0 ]]; then
    echo ""
    if $BUILD_TESTS; then
        echo "BUILD + TESTS COMPLETE!"
        echo "======================"
    else
        echo "HIGH-PERFORMANCE BUILD COMPLETE!"
        echo "==============================="
    fi
    echo ""
    echo "OPTIMIZATION FEATURES:"
    if $BUILD_TESTS; then
        echo "  O3 optimization with debug symbols (RelWithDebInfo)"
    else
        echo "  Maximum O3 optimization + LTO (Release)"
        echo "  Fast math and aggressive loop unrolling"
        echo "  Dead code stripping"
    fi
    echo "  CPU-specific optimizations (-march=native)"
    echo "  OpenMP parallelization"
    echo "  Fast math optimizations"
    echo ""
    
    if $BUILD_TESTS; then
        echo "TESTING FEATURES:"
        echo "  Auto-discovery of all test files in tests/"
        echo "  Comprehensive Barnes-Hut validation"
        echo "  NEON SIMD correctness tests"
        echo "  Tree structure invariant checks"
        echo "  Force calculation accuracy tests"
        echo ""
        
        # Show discovered test files
        if [[ -f "bh_tests" ]]; then
            echo "TEST EXECUTABLE READY:"
            echo "  ./bh_tests                 - Run all tests"
            echo "  ./bh_tests --gtest_help    - Show test options"
            echo ""
            
            # Show available test categories
            echo "TEST CATEGORIES AVAILABLE:"
            echo "  make test_quick            - Fast tests only"
            echo "  make test_critical         - Critical bug detection"
            echo "  make test_neon            - SIMD optimization tests"
            echo "  make test_tree_structure  - Tree validation"
            echo "  make test_traversal       - Force calculation tests"
            echo "  make test_comprehensive   - All tests with details"
            echo ""
        fi
    fi
    
    # Show main executable
    if [[ -f "BarnesHutMrkII" ]]; then
        FILE_SIZE=$(ls -lh BarnesHutMrkII | awk '{print $5}')
        echo "MAIN EXECUTABLE READY:"
        echo "  ./BarnesHutMrkII          - Run particle system"
        echo "  Binary size: $FILE_SIZE"
        if ! $BUILD_TESTS; then
            echo "  Build type: Release (maximum performance)"
        else
            echo "  Build type: RelWithDebInfo (optimized + debuggable)"
        fi
        echo ""
    fi
    
    if $BUILD_TESTS; then
        echo "DEVELOPMENT WORKFLOW:"
        echo "  1. Add new test files to tests/ directory"
        echo "  2. Run 'make' to rebuild and auto-test"
        echo "  3. Use 'make test_quick' for rapid iteration"
        echo "  4. Use 'make test_comprehensive' for full validation"
        echo ""
    else
        echo "PRODUCTION WORKFLOW:"
        echo "  1. Edit source files"
        echo "  2. Run 'make' in $BUILD_DIR/ to rebuild"
        echo "  3. Use './setup.sh' (with tests) for validation"
        echo "  4. Use './setup.sh --no-test' for production builds"
        echo ""
    fi
    
    # Show performance expectations
    echo "PERFORMANCE EXPECTATIONS:"
    if ! $BUILD_TESTS; then
        echo "  Maximum performance optimizations enabled"
        echo "  Link-time optimization (LTO) active"
    fi
    echo "  Force calculation: <1ms for 1000 particles"
    echo "  Tree building: <5ms for 10k particles"
    echo "  Multi-threaded when available"
    echo "  NEON SIMD acceleration on ARM64"
    echo ""
    
else
    echo ""
    echo "BUILD FAILED"
    echo "============"
    echo "Check the error messages above for details"
    echo "Common issues:"
    echo "  - Missing dependencies (run brew install <package>)"
    echo "  - Compilation errors in source files"
    if $BUILD_TESTS; then
        echo "  - Test failures preventing build completion"
        echo ""
        echo "Try building without tests: $0 --no-test"
    fi
    echo ""
    exit 1
fi

# Return to project root
cd ..

if $BUILD_TESTS; then
    echo "Ready for development with automatic testing!"
    echo "Edit source files and run 'make' in $BUILD_DIR/ to rebuild + retest"
    echo ""
    echo "For production builds without tests, use: $0 --no-test"
else
    echo "Ready for production deployment!"
    echo "Edit source files and run 'make' in $BUILD_DIR/ to rebuild"
    echo ""
    echo "For development with tests, use: $0"
fi
