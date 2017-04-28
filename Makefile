################################################################################
# Result name for release version
NAME   = test.exe
# Result name for debuag version
NAME_D = test_d.exe
################################################################################
# Directory layout
SRC_DIR      = src
INCLUDE_DIR  = include
BUILD_DIR    = build
BUILD_DIR_D  = build_d
RESULT_DIR   = .
RESULT_DIR_D = $(RESULT_DIR)
################################################################################
# Not profiling
OPTIMAZE_COMMON = -O3
################################################################################
# Processor specific optimization
OPTIMAZE_SPECIFIC = -std=c++11 -qopenmp
################################################################################
# Optimization options
OPTIMAZE   = -pipe $(OPTIMAZE_COMMON) $(OPTIMAZE_SPECIFIC)
OPTIMAZE_D = -pipe -g $(OPTIMAZE_SPECIFIC)
################################################################################
# Compiler
CC   = /opt/intel/bin/icpc
################################################################################
# Linker
LINK = /opt/intel/bin/icpc
################################################################################
# Compiler flags
CFLAGS   = -c -I$(INCLUDE_DIR) $(OPTIMAZE)
CFLAGS_D = -c -I$(INCLUDE_DIR) $(OPTIMAZE_D)
################################################################################
# Linker flags
LDFLAGS   = $(OPTIMAZE_SPECIFIC) -static-intel -static-libstdc++
LDFLAGS_D = -g $(OPTIMAZE_SPECIFIC) -static-intel -static-libstdc++
################################################################################
# Linker additional libraries
LIB  = -lm
LIB += -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread
#LIB += -lcublas -lcusparse -lcudart -lcudadevrt
################################################################################
RESULT   = $(RESULT_DIR)/$(NAME)
RESULT_D = $(RESULT_DIR_D)/$(NAME_D)

OBJS     = \
             $(BUILD_DIR)/tools.o  \
             $(BUILD_DIR)/main.o

OBJS_D   = $(subst $(BUILD_DIR)/, $(BUILD_DIR_D)/, $(OBJS))
DEPFILES = $(subst .o,.d,$(OBJS))
DEPFILES_D = $(subst .o,.d,$(OBJS_D))
################################################################################
# Build rules
all: clean debug release

clean: clean_r clean_d


clean_r:
	rm -f $(OBJS) $(RESULT) $(DEPFILES)
clean_d:
	rm -f $(OBJS_D) $(RESULT_D) $(DEPFILES_D)

debug: $(RESULT_D)

release: $(RESULT)


$(RESULT): $(OBJS)
	$(LINK) $(LDFLAGS) $(OBJS) $(LIB) -o $(RESULT)

$(RESULT_D): $(OBJS_D)
	$(LINK) $(LDFLAGS_D) $(OBJS_D) $(LIB) -o $(RESULT_D)
################################################################################
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -MMD $< -o $@

$(BUILD_DIR_D)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_D) -MMD $< -o $@
################################################################################