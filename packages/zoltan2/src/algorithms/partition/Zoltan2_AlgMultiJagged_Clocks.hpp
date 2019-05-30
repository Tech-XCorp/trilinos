// @HEADER
//
// ***********************************************************************
//
//   Zoltan2: A package of combinatorial algorithms for scientific computing
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Karen Devine      (kddevin@sandia.gov)
//                    Erik Boman        (egboman@sandia.gov)
//                    Siva Rajamanickam (srajama@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
/*! \file Zoltan2_AlgMultiJagged_Clocks.hpp
  \brief Temporary clock code - plan to delete this.
 */

// TODO: Remove this option and remove all clock code.
// This is temporary and turns off clocking for most clocks.
#define ZOLTAN2_MJ_DISABLE_CLOCKS false

// TODO: Delete all clock stuff. These were temporary timers for profiling.
class Clock {
  typedef typename std::chrono::time_point<std::chrono::steady_clock> clock_t;
  public:
    Clock(std::string clock_name, bool bStart, bool bAlwaysOnIn = false) :
      name(clock_name), bAlwaysOn(bAlwaysOnIn) {
      if(!ZOLTAN2_MJ_DISABLE_CLOCKS || bAlwaysOnIn) {
        reset();
        if(bStart) {
          start();
        }
      }
    }
    void reset() {
      if(!ZOLTAN2_MJ_DISABLE_CLOCKS || bAlwaysOn) {
        time_sum_ns = 0;
        counter_start = 0;
        counter_stop = 0;
      }
    }
    int time() const {
      if(!ZOLTAN2_MJ_DISABLE_CLOCKS || bAlwaysOn) {
        if(counter_start != counter_stop) {
          printf("Clock %s bad counters for time!\n", name.c_str());
          throw std::logic_error("bad timer counters for time!\n");
        }
        return time_sum_ns;
      } else {
        return 0.0;
      }
    }
    void start() {
      if(!ZOLTAN2_MJ_DISABLE_CLOCKS || bAlwaysOn) {
        if(counter_start != counter_stop) {
          printf("Clock %s bad counters for start!\n", name.c_str());
          throw std::logic_error("bad timer counters for start!\n");
        }
        ++counter_start;
        start_time = std::chrono::steady_clock::now();
      }
    }
    void stop(bool bPrint = false) {
      if(!ZOLTAN2_MJ_DISABLE_CLOCKS || bAlwaysOn) {
        if(counter_start-1 != counter_stop) {
          printf("Clock %s bad counters for stop!\n", name.c_str());
          throw std::logic_error("bad timer counters for stop!\n");
        }
        ++counter_stop;
        clock_t now_time = std::chrono::steady_clock::now();
        time_sum_ns += static_cast<int>(std::chrono::duration_cast<
          std::chrono::nanoseconds>(now_time - start_time).count());

        if(bPrint) {
          print();
        }
      }
    }
    void print() {
      if(!ZOLTAN2_MJ_DISABLE_CLOCKS || bAlwaysOn) {
        printf("%s: %d us    Count: %d\n", name.c_str(),
          time()/1000, counter_stop);
      }
    }
  private:
    std::string name;
    int counter_start;
    int counter_stop;
    clock_t start_time;
    int time_sum_ns;
    bool bAlwaysOn;
};

// TODO: Also delete all of this temp profiling code
static Clock clock_mj_1D_part_init(
  "        clock_mj_1D_part_init", false);
static Clock clock_mj_1D_part_init2(
  "        clock_mj_1D_part_init2", false);
static Clock clock_mj_1D_part_while_loop(
  "        clock_mj_1D_part_while_loop", false);
static Clock clock_swap("          clock_swap", false);
static Clock clock_host_copies("          clock_host_copies", false);
static Clock clock_mj_1D_part_get_weights_init(
  "          clock_mj_1D_part_get_weights_init", false);
static Clock clock_mj_1D_part_get_weights_setup(
  "          clock_mj_1D_part_get_weights_setup", false);
static Clock clock_mj_1D_part_get_weights(
  "          clock_mj_1D_part_get_weights", false);
static Clock clock_weights1("            clock_weights1", false);
static Clock clock_weights_new_to_optimize(
  "              clock_weights_new_to_optimize", false);
static Clock clock_weights2("            clock_weights2", false);
static Clock clock_weights3("            clock_weights3", false);
static Clock clock_functor_weights(
  "              clock_functor_weights", false);
static Clock clock_weights4("            clock_weights4", false);
static Clock clock_mj_combine_rightleft_and_weights(
  "          clock_mj_combine_rightleft_and_weights", false);
static Clock clock_mj_get_new_cut_coordinates_init(
  "          clock_mj_get_new_cut_coordinates_init", false);
static Clock clock_mj_get_new_cut_coordinates(
  "          clock_mj_get_new_cut_coordinates", false);
static Clock clock_mj_get_new_cut_coordinates_end(
  "          clock_mj_get_new_cut_coordinates_end", false);
static Clock clock_write_globals(
  "          clock_write_globals", false);
static Clock clock_mj_1D_part_end(
  "        clock_mj_1D_part_end", false);
static Clock clock_mj_create_new_partitions(
  "         clock_mj_create_new_partitions", false);
static Clock clock_mj_create_new_partitions_1(
  "           clock_mj_create_new_partitions_1", false);
static Clock clock_mj_create_new_partitions_2(
  "           clock_mj_create_new_partitions_2", false);
static Clock clock_mj_create_new_partitions_3(
  "           clock_mj_create_new_partitions_3", false);
static Clock clock_mj_create_new_partitions_4(
  "           clock_mj_create_new_partitions_4", false);
static Clock clock_mj_create_new_partitions_5(
  "           clock_mj_create_new_partitions_5", false);
static Clock clock_mj_create_new_partitions_6(
  "           clock_mj_create_new_partitions_6", false);
static Clock clock_mj_migrate_coords(
  " ------- migrate_coords", false);
