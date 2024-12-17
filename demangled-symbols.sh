#!/bin/bash
dyld-shared-cache-extractor \
   /System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e \
   /tmp/libraries
cd /tmp/libraries/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
nm -gU libSparse.dylib | /opt/homebrew/opt/llvm/bin/llvm-cxxfilt
# remove "/opt/homebrew/opt/llvm/bin/llvm-cxxfilt" for raw symbols.
# install llvm and dyld-shared-cache-extractor with homebrew first.