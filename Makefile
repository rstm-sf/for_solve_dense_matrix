####################################################################################################
# Result name for release version
NAME   = test.exe
# Result name for debuag version
NAME_D = test_d.exe
####################################################################################################
# define library directories
MAGMADIR =
CUDADIR  =
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
#NVCC  = nvcc
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
CPPFLAGS = -I$(INCLUDE_DIR) -DIS_DOUBLE  \
           -I$(MAGMADIR)/include -DADD_  \
           -I$(CUDADIR)/include          \
           -I$(MKLROOT)/include

CXXFLAGS   = -Wall -MMD -pipe $(OPTIMAZE)
CXXFLAGS_D = -Wall -MMD -pipe $(OPTIMAZE_D)

#NV_SM        = -gencode arch=compute_50,code=sm_50
#NV_COMP      = -gencode arch=compute_50,code=compute_50
#NVCCFLAGS    = $(NV_SM) $(NV_COMP) -Wno-deprecated-gpu-targets -ccbin=$(CXX)
#NVCCFLAGS_D := $(NVCCFLAGS) -G -O0 -Xcompiler "$(CXXFLAGS_D)"
#NVCCFLAGS   += -O3 -Xcompiler "$(CXXFLAGS)"
####################################################################################################
# Linker flags
LDFLAGS   = -Wall $(OPTIMAZE_SPECIFIC) -qopenmp -static-intel -static-libstdc++
LDFLAGS_D = $(LDFLAGS)
####################################################################################################
# Linker additional libraries
LIBS  = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -lm
LIBS += -L$(CUDADIR)/lib64 -lcublas -lcusparse -lcusolver -lcudart -lcudadevrt
LIBS += -L$(MAGMADIR)/lib -lmagma
####################################################################################################
RESULT     = $(RESULT_DIR)/$(NAME)
RESULT_D   = $(RESULT_DIR_D)/$(NAME_D)

OBJS       = $(BUILD_DIR)/tools.o          \
			 $(BUILD_DIR)/magma_solver.o   \
             $(BUILD_DIR)/mkl_solver.o     \
             $(BUILD_DIR)/cu_solver.o      \
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
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS) 

$(RESULT_D): $(OBJS_D)
	$(LD) $(LDFLAGS_D) -o $@ $^ $(LIBS)
####################################################################################################