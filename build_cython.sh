#!/bin/bash
# Build Cython extensions for vector-db
#
# This script compiles the Cython (.pyx) files to C and then to
# shared libraries (.so on Linux/Mac, .pyd on Windows).
#
# Usage:
#   ./build_cython.sh        # Build in development mode
#   ./build_cython.sh clean  # Clean build artifacts

set -e  # Exit on error

if [ "$1" == "clean" ]; then
    echo "Cleaning Cython build artifacts..."
    rm -rf build/
    rm -rf src/**/*.c
    rm -rf src/**/*.so
    rm -rf src/**/*.html
    echo "Clean complete."
    exit 0
fi

echo "Building Cython extensions..."
echo "=============================="

# Build the extensions
python setup.py build_ext --inplace

echo ""
echo "Build complete!"
echo ""
echo "Generated files:"
echo "  - src/utils/distance_cy.c (C source)"
echo "  - src/utils/distance_cy*.so (compiled extension)"
echo "  - src/utils/distance_cy.html (optimization annotations)"
echo ""
echo "You can now import with: from src.utils.distance_cy import euclidean_distance"
