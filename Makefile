include Makefile.config

DEBUG ?= 0
RELEASE ?= 0

ifeq ($(DEBUG), 1)
	CXXFLAGS += -O0 -g #-ggdb
	LDFLAGS  += -O0 -g -ggdb
else
	CXXFLAGS += -O3
	LDFLAGS  += -O3
endif

TARGET = ray
CXX = g++
CXXFLAGS += -std=c++17
WARNING = -Wall -W

SOURCE_DIR = ./src
OBJECT_DIR = ./obj
INCLUDE_DIR = ./include

SOURCES = $(shell find $(SOURCE_DIR) -name '*.cpp')

# --- 修正後 ---
# 1. パスを WSL2 内の Linux 用 LibTorch に変更
LIBTORCH_DIR = /home/ama/libtorch

# 2. ライブラリ指定に -ltorch_cuda を追加し、cpu を cuda に置き換える
LIBTORCH_INC = -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include
# --- 修正後 ---
LIBTORCH_LIB = -L$(LIBTORCH_DIR)/lib -ltorch -ltorch_cuda -ltorch_cpu -lc10 -Wl,-rpath,$(LIBTORCH_DIR)/lib

INCLUDE += $(LIBTORCH_INC)
# ABI設定を 1 に（最新のLibTorch用）
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
LDFLAGS  += $(LIBTORCH_LIB)

OBJECTS = $(subst $(SOURCE_DIR), $(OBJECT_DIR), $(SOURCES))
OBJECTS := $(subst .cpp,.o,$(OBJECTS))
OBJECTS := $(subst .cu,.o,$(OBJECTS))

DEPENDS = $(OBJECTS:.o=.d)

INCLUDE += -I$(INCLUDE_DIR)
# --- 修正後 ---
#CUDA_PATH = /usr/local/cuda
#INCLUDE += -I$(CUDA_PATH)/include   # ← INCLUDE に変更
#LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart # ← リンクエラー防止のためこれも追加推奨

ifeq ($(RELEASE), 1)
	LDFLAGS += -static -lm -Wl,--whole-archive -lpthread -Wl,--no-whole-archive
else
	LDFLAGS += -lm -lpthread
endif

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIB_CUDA) $(LIB_REDIS)

$(SOURCE_DIR)/%.cpp: $(SOURCE_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) --cuda $< -o $@

$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(CXX) $(WARNING) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

clean:
	-rm -f *~ $(TARGET) $(OBJECTS) $(SOURCE_DIR)/*~ $(SOURCE_DIR)/*/*~ $(SOURCE_DIR)/*/*/*~ $(SOURCE_DIR)/*/*/*/*~ $(INCLUDE_DIR)/*~ $(INCLUDE_DIR)/*/*~ $(INCLUDE_DIR)/*/*/*~ $(INCLUDE_DIR)/*/*/*/*~

-include $(DEPENDS)
