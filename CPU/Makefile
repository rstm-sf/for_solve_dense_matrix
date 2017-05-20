####################################################################################################
# Result name for release version
NAME   = test_mkl.exe
# Result name for debuag version
NAME_D = test_mkl_d.exe
####################################################################################################
# define library directories
MKLROOT  =
####################################################################################################
# Directory layout
SRC_DIR      = src
INCLUDE_DIR  = include
BUILD_DIR    = build
BUILD_DIR_D  = build_d
RESULT_DIR   = .
RESULT_DIR_D = $(RESULT_DIR)
####################################################################################################
# Compiler
CXX   = icpc
####################################################################################################
# Linker
LD = $(CXX)
####################################################################################################
OPTIMAZE_SPECIFIC = -std=c++11
OPTIMAZE          = $(OPTIMAZE_SPECIFIC) -fast
OPTIMAZE_D        = $(OPTIMAZE_SPECIFIC) -O0 -g
####################################################################################################
# Compiler flags
# -DIS_DOUBLE use double precision
CPPFLAGS = -I$(INCLUDE_DIR)     \
           -I$(MKLROOT)/include

CXXFLAGS   = -Wall -MMD -pipe $(OPTIMAZE)
CXXFLAGS_D = -Wall -MMD -pipe $(OPTIMAZE_D)
####################################################################################################
# Linker flags
LDFLAGS   = -Wall $(OPTIMAZE_SPECIFIC) -qopenmp -static-intel -static-libstdc++
LDFLAGS_D = $(LDFLAGS)
####################################################################################################
# Linker additional libraries
# parallel: -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group
# sequential: -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group
LIBDIRS = -L$(MKLROOT)/lib

LIBS  = -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group \
        -lpthread -lstdc++ -lm
####################################################################################################
RESULT     = $(RESULT_DIR)/$(NAME)
RESULT_D   = $(RESULT_DIR_D)/$(NAME_D)

OBJS       = $(BUILD_DIR)/tools.o          \
             $(BUILD_DIR)/mkl_solver.o     \
             $(BUILD_DIR)/main.o

OBJS_D     = $(subst $(BUILD_DIR)/,$(BUILD_DIR_D)/,$(OBJS))
DEPFILES   = $(subst .o,.d,$(OBJS))
DEPFILES_D = $(subst .o,.d,$(OBJS_D))
####################################################################################################
# Build rules
all: clean debug release

clean: clean_r clean_d


clean_r:
	rm -f $(OBJS) $(RESULT) $(DEPFILES)
clean_d:
	rm -f $(OBJS_D) $(RESULT_D) $(DEPFILES_D)

debug: $(RESULT_D)

release: $(RESULT)
####################################################################################################
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(BUILD_DIR_D)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS_D) $(CPPFLAGS) -c -o $@ $<


$(RESULT): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS) $(LIBDIRS)

$(RESULT_D): $(OBJS_D)
	$(LD) $(LDFLAGS_D) -o $@ $^ $(LIBS) $(LIBDIRS)
####################################################################################################