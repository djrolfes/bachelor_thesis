TARGET ?= main
SRC_DIRS ?= ./src
LIB_DIRS ?= ./lib
EXT_INC_DIRS ?= ./ext_inc
SYS_LIBS ?= -lqhullcpp -lqhull_r -lutil
CXXFLAGS+= --std=c++17 -O3

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
LIBS := $(shell find $(LIB_DIRS) -name *.a)
DEPS := $(OBJS:.o=.d)


INC_DIRS = $(EXT_INC_DIRS)
INC_DIRS += $(shell find $(SRC_DIRS) -type d)


INC_FLAGS := $(addprefix -I,$(INC_DIRS))
#INC_Flags += -i/usr/local/cuda-11.3/include

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP


$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) $(LIBS) $(SYS_LIBS) -o $@ $(LOADLIBES) $(LDLIBS)


.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS) $(CUDAOBJS)

#-include $(DEPS)
