CXX=g++

SHARE_LOCATION=./N3LP
BOOST_LOCATION=pathToBoost #pathtoBoost # TODO: Modify the path to Boost

BUILD_DIR=objs

#CXXFLAGS =-Wall
CXXFLAGS+=-O3

EIGEN_LOCATION=pathToEigen #pathtoEigen # TODO: Modify the path to Eigen
CXXFLAGS+=-std=c++0x
CXXFLAGS+=-mfpmath=sse
CXXFLAGS+=-mmmx
CXXFLAGS+=-lm
CXXFLAGS+=-fomit-frame-pointer
CXXFLAGS+=-fno-schedule-insns2
CXXFLAGS+=-fexceptions
CXXFLAGS+=-funroll-loops
CXXFLAGS+=-march=native
CXXFLAGS+=-m64
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
CXXFLAGS+=-DEIGEN_NO_DEBUG
CXXFLAGS+=-DEIGEN_NO_STATIC_ASSERT
CXXFLAGS+=-I$(EIGEN_LOCATION)
CXXFLAGS+=-I$(BOOST_LOCATION)
CXXFLAGS+=-I$(SHARE_LOCATION)
CXXFLAGS+=-fopenmp

SRCS=$(shell ls *.cpp)
OBJS=$(SRCS:.cpp=.o)

PROGRAM=tree2seq

all : $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,$(PROGRAM))

$(BUILD_DIR)/%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/$(PROGRAM) : $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
	$(CXX) $(CXXFLAGS) $(CXXLIBS) -o $@ $^
	mv $(BUILD_DIR)/$(PROGRAM) ./
	rm -f ?*~

clean:
	rm -f $(BUILD_DIR)/* $(PROGRAM) ?*~
