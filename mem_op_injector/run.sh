#!/bin/bash
make clean; make
# export TOOL_VERBOSE=1; LD_PRELOAD=pf_injector.so ../test-apps/simple_add/simple_add;
time export TOOL_VERBOSE=1; LD_PRELOAD=pf_injector.so ../test-apps/vectoradd/vectoradd;
#export TOOL_VERBOSE=1; LD_PRELOAD=../../mem_printf/mem_printf.so ../test-apps/vectoradd/vectoradd;
