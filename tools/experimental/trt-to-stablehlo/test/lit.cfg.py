import os
import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "TRT-TO-STABLEHLO"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.trt_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

config.trt_tools_dir = os.path.join(config.trt_obj_root, "bin")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.trt_tools_dir, config.llvm_tools_dir]
tools = ["trt-opt"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
