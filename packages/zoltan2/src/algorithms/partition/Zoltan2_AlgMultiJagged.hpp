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
/*! \file Zoltan2_AlgMultiJagged.hpp
  \brief Contains the Multi-jagged algorthm.
 */

#ifndef _ZOLTAN2_ALGMultiJagged_HPP_
#define _ZOLTAN2_ALGMultiJagged_HPP_

#include <Zoltan2_MultiJagged_ReductionOps.hpp>
#include <Zoltan2_CoordinateModel.hpp>
#include <Zoltan2_Parameters.hpp>
#include <Zoltan2_Algorithm.hpp>
#include <Zoltan2_IntegerRangeList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <Tpetra_Distributor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Zoltan2_CoordinatePartitioningGraph.hpp>
#include <new>          // ::operator new[]
#include <algorithm>    // std::sort
#include <Zoltan2_Util.hpp>
#include <vector>

// TODO: This is a temporary setting to be removed and calculated based on
// conditions of the system and the algorithm.
#define SET_NUM_TEAMS_ReduceWeightsFunctor 60
#define SET_NUM_TEAMS_RightLeftClosestFunctor 30 // tuned to my local machine - needs work
#define SET_NUM_TEAMS_mj_create_new_partitions_clock 500

#define SET_MAX_TEAMS 200 // to do - optimize

// TODO: Delete all clock stuff. There were temporary timers for profiling.
class Clock {
  typedef typename std::chrono::time_point<std::chrono::steady_clock> clock_t;
  public:
    Clock(std::string clock_name, bool bStart) :
      name(clock_name) {
      reset();
      if(bStart) {
        start();
      }
    }
    void reset() {
      time_sum = 0;
      counter_start = 0;
      counter_stop = 0;
    }
    double time() const {
      if(counter_start != counter_stop) {
        printf("Clock %s bad counters for time!\n", name.c_str());
        throw std::logic_error("bad timer counters for time!\n");
      }
      return time_sum;
    }
    void start() {
      if(counter_start != counter_stop) {
        printf("Clock %s bad counters for start!\n", name.c_str());
        throw std::logic_error("bad timer counters for start!\n");
      }
      ++counter_start;
      start_time = std::chrono::steady_clock::now();
    }
    void stop(bool bPrint = false) {
      if(counter_start-1 != counter_stop) {
        printf("Clock %s bad counters for stop!\n", name.c_str());
        throw std::logic_error("bad timer counters for stop!\n");
      }
      ++counter_stop;
      clock_t now_time = std::chrono::steady_clock::now();
      time_sum += std::chrono::duration_cast<std::chrono::duration<double> >(now_time - start_time).count();
      if(bPrint) {
        print();
      }
    }
    void print() {
      printf("%s: %.2f ms    Count: %d\n", name.c_str(), (float)(time() * 1000.0), counter_stop);
    }
  private:
    std::string name;
    int counter_start;
    int counter_stop;
    clock_t start_time;
    double time_sum;
};

// TODO: Also delete all of this temp profiling code
static Clock clock_mj_1D_part_init("        clock_mj_1D_part_init", false);
static Clock clock_mj_1D_part_init2("        clock_mj_1D_part_init2", false);
static Clock clock_mj_1D_part_while_loop("        clock_mj_1D_part_while_loop", false);
static Clock clock_swap("          clock_swap", false);
static Clock clock_host_copies("          clock_host_copies", false);
static Clock clock_mj_1D_part_get_weights_init("          clock_mj_1D_part_get_weights_init", false);
static Clock clock_mj_1D_part_get_weights_setup("          clock_mj_1D_part_get_weights_setup", false);
static Clock clock_mj_1D_part_get_weights("          clock_mj_1D_part_get_weights", false);
static Clock clock_weights1("            clock_weights1", false);
static Clock clock_weights2("            clock_weights2", false);
static Clock clock_weights3("            clock_weights3", false);
static Clock clock_functor_weights("              clock_functor_weights", false);
static Clock clock_weights4("            clock_weights4", false);
static Clock clock_weights5("            clock_weights5", false);
static Clock clock_weights6("            clock_weights6", false);
static Clock clock_functor_rightleft_closest("              clock_functor_rightleft_closest", false);
static Clock clock_mj_accumulate_thread_results("          clock_mj_accumulate_thread_results", false);
static Clock clock_mj_get_new_cut_coordinates_init("          clock_mj_get_new_cut_coordinates_init", false);
static Clock clock_mj_get_new_cut_coordinates("          clock_mj_get_new_cut_coordinates", false);
static Clock clock_mj_get_new_cut_coordinates_end("          clock_mj_get_new_cut_coordinates_end", false);
static Clock clock_write_globals("          clock_write_globals", false);
static Clock clock_mj_1D_part_end("        clock_mj_1D_part_end", false);
static Clock clock_mj_create_new_partitions("           clock_mj_create_new_partitions", false);

#if defined(__cplusplus) && __cplusplus >= 201103L
#include <unordered_map>
#else
#include <Teuchos_Hashtable.hpp>
#endif // C++11 is enabled

#ifdef ZOLTAN2_USEZOLTANCOMM
#ifdef HAVE_ZOLTAN2_MPI
#define ENABLE_ZOLTAN_MIGRATION
#include "zoltan_comm_cpp.h"
#include "zoltan_types.h" // for error codes
#endif
#endif

#define LEAST_SIGNIFICANCE 0.0001
#define SIGNIFICANCE_MUL 1000

//if the (last dimension reduce all count) x the mpi world size
//estimated to be bigger than this number then migration will be forced
//in earlier iterations.
#define FUTURE_REDUCEALL_CUTOFF 1500000

//if parts right before last dimension are estimated to have less than
//MIN_WORK_LAST_DIM many coords, migration will be forced in earlier iterations.
#define MIN_WORK_LAST_DIM 1000

#define ZOLTAN2_ABS(x) ((x) >= 0 ? (x) : -(x))

//imbalance calculation. Wreal / Wexpected - 1
#define imbalanceOf(Wachieved, totalW, expectedRatio) \
        (Wachieved) / ((totalW) * (expectedRatio)) - 1
#define imbalanceOf2(Wachieved, wExpected) \
        (Wachieved) / (wExpected) - 1


#define ZOLTAN2_ALGMULTIJAGGED_SWAP(a,b,temp) temp=(a);(a)=(b);(b)=temp;

namespace Teuchos{

/*! \brief Zoltan2_BoxBoundaries is a reduction operation
 * to all reduce the all box boundaries.
*/
template <typename Ordinal, typename T>
class Zoltan2_BoxBoundaries  : public ValueTypeReductionOp<Ordinal,T>
{
private:
  Ordinal size;
  T _EPSILON;

public:
  /*! \brief Default Constructor
   */
  Zoltan2_BoxBoundaries (): size(0),
    _EPSILON (std::numeric_limits<T>::epsilon()){}

  /*! \brief Constructor
   *   \param nsum  the count of how many sums will be computed at the
   *             start of the list.
   *   \param nmin  following the sums, this many minimums will be computed.
   *   \param nmax  following the minimums, this many maximums will be computed.
   */
  Zoltan2_BoxBoundaries (Ordinal s_):
    size(s_), _EPSILON (std::numeric_limits<T>::epsilon()){}

  /*! \brief Implement Teuchos::ValueTypeReductionOp interface
   */
  void reduce( const Ordinal count, const T inBuffer[], T inoutBuffer[]) const
  {
    for (Ordinal i=0; i < count; i++){
      if (Z2_ABS(inBuffer[i]) >  _EPSILON){
        inoutBuffer[i] = inBuffer[i];
      }
    }
  }
};
} // namespace Teuchos

namespace Zoltan2{

/*! \brief Allocates memory for the given size.
 *
 */
template <typename T>
T *allocMemory(size_t size){
  if (size > 0){
    T * a = new T[size];
    if (a == NULL) {
      throw  "cannot allocate memory";
    }
    return a;
  }
  else {
    return NULL;
  }
}

/*! \brief Frees the given array.
 *
 */
template <typename T>
void freeArray(T *&array){
  if(array != NULL){
    delete [] array;
    array = NULL;
  }
}


/*! \brief Class for sorting items with multiple values.
 * First sorting with respect to val[0], then val[1] then ... val[count-1].
 * The last tie breaking is done with index values.
 * Used for task mapping partitioning where the points on a cut line needs to be
 * distributed consistently.
 *
 */
template <typename IT, typename CT, typename WT>
class uMultiSortItem
{
public:
  //TODO: Why volatile?
  //no idea, another intel compiler faiulure.
  volatile IT index;
  volatile CT count;
  //unsigned int val;
  volatile WT *val;
  volatile WT _EPSILON;

  uMultiSortItem(){
    this->index = 0;
    this->count = 0;
    this->val = NULL;
    this->_EPSILON = std::numeric_limits<WT>::epsilon() * 100;
  }

  uMultiSortItem(IT index_ ,CT count_, WT *vals_){
    this->index = index_;
    this->count = count_;
    this->val = vals_;
    this->_EPSILON = std::numeric_limits<WT>::epsilon() * 100;
  }

  uMultiSortItem( const uMultiSortItem<IT,CT,WT>& other ) {
    this->index = other.index;
    this->count = other.count;
    this->val = other.val;
    this->_EPSILON = other._EPSILON;
  }

  ~uMultiSortItem(){
    //freeArray<WT>(this->val);
  }

  void set(IT index_ ,CT count_, WT *vals_) {
    this->index = index_;
    this->count = count_;
    this->val = vals_;
  }

  uMultiSortItem<IT,CT,WT> operator=(const uMultiSortItem<IT,CT,WT>& other) {
    this->index = other.index;
    this->count = other.count;
    this->val = other.val;
    return *(this);
  }

  bool operator<(const uMultiSortItem<IT,CT,WT>& other) const{
    assert (this->count == other.count);
    for(CT i = 0; i < this->count; ++i){
      //if the values are equal go to next one.
      if (ZOLTAN2_ABS(this->val[i] - other.val[i]) < this->_EPSILON){
        continue;
      }
      //if next value is smaller return true;
      if(this->val[i] < other.val[i]){
        return true;
      }
      //if next value is bigger return false;
      else {
        return false;
      }
    }
    //if they are totally equal.
    return this->index < other.index;
  }
  bool operator>(const uMultiSortItem<IT,CT,WT>& other) const{
    assert (this->count == other.count);
    for(CT i = 0; i < this->count; ++i){
      //if the values are equal go to next one.
      if (ZOLTAN2_ABS(this->val[i] - other.val[i]) < this->_EPSILON){
        continue;
      }
      //if next value is bigger return true;
      if(this->val[i] > other.val[i]){
        return true;
      }
      //if next value is smaller return false;
      else //(this->val[i] > other.val[i])
      {
        return false;
      }
    }
    //if they are totally equal.
    return this->index > other.index;
  }
};// uSortItem;

/*! \brief Sort items for quick sort function.
 *
 */
template <class IT, class WT>
struct uSortItem
{
  IT id;
  //unsigned int val;
  WT val;
};// uSortItem;

/*! \brief Quick sort function.
 *      Sorts the arr of uSortItems, with respect to increasing vals.
 */
template <class IT, class WT>
void uqsort(IT n, uSortItem<IT, WT> * arr)
{
  int NSTACK = 50;
  int M = 7;
  IT         i, ir=n, j, k, l=1;
  IT         jstack=0, istack[50];
  WT aval;
  uSortItem<IT,WT>    a, temp;

  --arr;
  for (;;) {
    if (ir-l < M) {
      for (j=l+1;j<=ir;j++) {
        a=arr[j];
        aval = a.val;
        for (i=j-1;i>=1;i--) {
          if (arr[i].val <= aval)
            break;
          arr[i+1] = arr[i];
        }
        arr[i+1]=a;
      }
      if (jstack == 0)
          break;
      ir=istack[jstack--];
      l=istack[jstack--];
    }
    else {
      k=(l+ir) >> 1;
      ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[k],arr[l+1], temp)
      if (arr[l+1].val > arr[ir].val) {
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[l+1],arr[ir],temp)
      }
      if (arr[l].val > arr[ir].val) {
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[l],arr[ir],temp)
      }
      if (arr[l+1].val > arr[l].val) {
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[l+1],arr[l],temp)
      }
      i=l+1;
      j=ir;
      a=arr[l];
      aval = a.val;
      for (;;) {
        do i++; while (arr[i].val < aval);
        do j--; while (arr[j].val > aval);
        if (j < i) break;
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[i],arr[j],temp);
      }
      arr[l]=arr[j];
      arr[j]=a;
      jstack += 2;
      if (jstack > NSTACK) {
        std::cout << "uqsort: NSTACK too small in sort." << std::endl;
        exit(1);
      }
      if (ir-i+1 >= j-l) {
        istack[jstack]=ir;
        istack[jstack-1]=i;
        ir=j-1;
      }
      else {
        istack[jstack]=j-1;
        istack[jstack-1]=l;
        l=i;
      }
    }
  }
}

template <class IT, class WT, class SIGN>
struct uSignedSortItem
{
  IT id;
  //unsigned int val;
  WT val;
  SIGN signbit; // 1 means positive, 0 means negative.
  bool operator<(const uSignedSortItem<IT, WT, SIGN>& rhs) const {
    /*if I am negative, the other is positive*/
    if (this->signbit < rhs.signbit){
      return true;
    }
    /*if both has the same sign*/
    else if (this->signbit == rhs.signbit) {
      if (this->val < rhs.val){//if my value is smaller,
        return this->signbit;//then if we both are positive return true.
                            //if we both are negative, return false.
      }
      else if (this->val > rhs.val){//if my value is larger,
        return !this->signbit; //then if we both are positive return false.
                              //if we both are negative, return true.
      }
      else { //if both are equal.
        return false;
      }
    }
    else {
      /*if I am positive, the other is negative*/
      return false;
    }
  }
  bool operator>(const uSignedSortItem<IT, WT, SIGN>& rhs) const {
    /*if I am positive, the other is negative*/
    if (this->signbit > rhs.signbit){
      return true;
    }
    /*if both has the same sign*/
    else if (this->signbit == rhs.signbit){
      if (this->val < rhs.val){//if my value is smaller,
        return !this->signbit;//then if we both are positive return false.
                            //if we both are negative, return true.
      }
      else if (this->val > rhs.val){//if my value is larger,
        return this->signbit; //then if we both are positive return true.
                              //if we both are negative, return false.
      }
      else { // if they are equal
        return false;
      }
    }
    else {
      /*if I am negative, the other is positive*/
      return false;
    }
  }
  bool operator<=(const uSignedSortItem<IT, WT, SIGN>& rhs) {
    return !(*this > rhs);
  }
  bool operator>=(const uSignedSortItem<IT, WT, SIGN>& rhs) {
    return !(*this  < rhs);
  }
};

/*! \brief Quick sort function.
 *      Sorts the arr of uSignedSortItems, with respect to increasing vals.
 */
template <class IT, class WT, class SIGN>
void uqSignsort(IT n, uSignedSortItem<IT, WT, SIGN> * arr){
  IT NSTACK = 50;
  IT M = 7;
  IT         i, ir=n, j, k, l=1;
  IT         jstack=0, istack[50];
  uSignedSortItem<IT,WT,SIGN>    a, temp;

  --arr;
  for (;;) {
    if (ir < M + l) {
      for (j=l+1;j<=ir;j++) {
        a=arr[j];
        for (i=j-1;i>=1;i--) {
          if (arr[i] <= a) {
              break;
          }
          arr[i+1] = arr[i];
        }
        arr[i+1]=a;
      }
      if (jstack == 0) {
        break;
      }
      ir=istack[jstack--];
      l=istack[jstack--];
    }
    else {
      k=(l+ir) >> 1;
      ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[k],arr[l+1], temp)
      if (arr[l+1] > arr[ir]) {
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[l+1],arr[ir],temp)
      }
      if (arr[l] > arr[ir]) {
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[l],arr[ir],temp)
      }
      if (arr[l+1] > arr[l]) {
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[l+1],arr[l],temp)
      }
      i=l+1;
      j=ir;
      a=arr[l];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        ZOLTAN2_ALGMULTIJAGGED_SWAP(arr[i],arr[j],temp);
      }
      arr[l]=arr[j];
      arr[j]=a;
      jstack += 2;
      if (jstack > NSTACK){
        std::cout << "uqsort: NSTACK too small in sort." << std::endl;
        exit(1);
      }
      if (ir+l+1 >= j+i) {
        istack[jstack]=ir;
        istack[jstack-1]=i;
        ir=j-1;
      }
      else {
        istack[jstack]=j-1;
        istack[jstack-1]=l;
        l=i;
      }
    }
  }
}

/*! \brief Multi Jagged coordinate partitioning algorithm.
 *
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
class AlgMJ
{

// TODO: For use of extended host lambdas added for CUDA this was changed to
// public. I did this for CUDA only.
#ifdef KOKKOS_HAVE_CUDA
public:
#else
private:
#endif

  typedef typename mj_node_t::device_type device_t;
  typedef coordinateModelPartBox<mj_scalar_t, mj_part_t> mj_partBox_t;
  typedef std::vector<mj_partBox_t> mj_partBoxVector_t;
  
  RCP<const Environment> mj_env;          // the environment object
  RCP<const Comm<int> > mj_problemComm;   // initial comm object
  RCP<Comm<int> > comm; // comm object than can be altered during execution
  double imbalance_tolerance;             // input imbalance tolerance.
  int recursion_depth; // number of steps that partitioning will be solved in.
  int coord_dim;                          // coordinate dim
  int num_weights_per_coord;              // # of weights per coord
  size_t initial_num_loc_coords;          // initial num local coords.
  global_size_t initial_num_glob_coords;  // initial num global coords.
  mj_lno_t num_local_coords;              // number of local coords.
  mj_gno_t num_global_coords;             // number of global coords.
  mj_scalar_t sEpsilon;                   // epsilon for mj_scalar_t
  
  // can distribute points on same coordiante to different parts.
  bool distribute_points_on_cut_lines;
  
  // how many parts we can calculate concurrently.
  mj_part_t max_concurrent_part_calculation;

  bool mj_run_as_rcb; // means recursion depth is adjusted to maximum value.
  int mj_user_recursion_depth; // the recursion depth value provided by user.
  bool mj_keep_part_boxes; // if the boxes need to be kept.

  // whether to migrate=1, avoid migrate=2, or leave decision to MJ=0
  int check_migrate_avoid_migration_option;

  // when doing the migration, 0 will aim for perfect load-imbalance, 1 - will 
  // aim for minimized number of messages with possibly bad load-imbalance
  int migration_type;

  // when MJ decides whether to migrate, the minimum imbalance for migration.
  mj_scalar_t minimum_migration_imbalance;

  mj_part_t total_num_cut ;           // how many cuts will be totally
  mj_part_t total_num_part;           // how many parts will be totally

  mj_part_t max_num_part_along_dim ;  // maximum part count along a dimension.
  mj_part_t max_num_cut_along_dim;    // maximum cut count along a dimension.
  
  // maximum part+cut count along a dimension.
  size_t max_num_total_part_along_dim;

  mj_part_t total_dim_num_reduce_all;  // estimate on #reduceAlls can be done.
  
  // max no of parts that might occur during the partition before the last
  // partitioning dimension.
  mj_part_t last_dim_num_part;
  
  // input part array specifying num part to divide along each dim.
  Kokkos::View<mj_part_t *, device_t> kokkos_part_no_array;

  Kokkos::View<mj_scalar_t **, Kokkos::LayoutLeft, device_t>
    kokkos_mj_coordinates; // two dimension coordinate array
    
  // two dimension weight array
  Kokkos::View<mj_scalar_t **, device_t> kokkos_mj_weights;
  
  // if the target parts are uniform
  Kokkos::View<bool *, device_t> kokkos_mj_uniform_parts;

  // target part weight sizes.
  Kokkos::View<mj_scalar_t **, device_t> kokkos_mj_part_sizes; 
    
  // if the coordinates have uniform weights
  Kokkos::View<bool *, device_t> kokkos_mj_uniform_weights; 

  size_t num_global_parts; // the targeted number of parts

  // vector of all boxes for all parts, constructed if mj_keep_part_boxes true
  RCP<mj_partBoxVector_t> kept_boxes;

  RCP<mj_partBox_t> global_box;
  
  int myRank;           // processor rank
  int myActualRank;     // initial rank

  bool divide_to_prime_first;
  
  // initial global ids of the coordinates.
  Kokkos::View<const mj_gno_t*, device_t> kokkos_initial_mj_gnos;
  
  // current global ids of the coordinates, might change during migration.
  Kokkos::View<mj_gno_t*, device_t> kokkos_current_mj_gnos;

  // the actual processor owner of the coordinate, to track after migrations.
  Kokkos::View<int*, device_t> kokkos_owner_of_coordinate;
  
  // permutation of coordinates, for partitioning.
  Kokkos::View<mj_lno_t*, device_t> kokkos_coordinate_permutations;
  
  // permutation work array.
  Kokkos::View<mj_lno_t*, device_t> kokkos_new_coordinate_permutations;
  
  // the part ids assigned to coordinates.
  Kokkos::View<mj_part_t*, device_t> kokkos_assigned_part_ids;
  
  // beginning and end of each part.
  Kokkos::View<mj_lno_t *, device_t> kokkos_part_xadj;
    
  // work array for beginning and end of each part.
  Kokkos::View<mj_lno_t *, device_t> kokkos_new_part_xadj;

  Kokkos::View<mj_scalar_t *, device_t> kokkos_all_cut_coordinates;
  
  // how much weight should a MPI put left side of the each cutline
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_process_cut_line_weight_to_put_left;
    
  // weight percentage each thread in MPI puts left side of the each outline
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_cut_line_weight_to_put_left;

  // work array to manipulate coordinate of cutlines in different iterations.
  // necessary because previous cut line information is used for determining
  // the next cutline information. therefore, cannot update the cut work array
  // until all cutlines are determined.
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_coordinates_work_array;
  
  // Used for swapping above kokkos_cut_coordinates_work_array
  Kokkos::View<mj_scalar_t *, device_t> kokkos_temp_cut_coords;

  // cumulative part weight array.
  Kokkos::View<mj_scalar_t *, device_t> kokkos_target_part_weights;

  // upper bound coordinate of a cut line
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_upper_bound_coordinates;
  
  // lower bound coordinate of a cut line
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_lower_bound_coordinates;

  // lower bound weight of a cut line
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_lower_bound_weights;
  
  // upper bound weight of a cut line  
  Kokkos::View<mj_scalar_t *, device_t> kokkos_cut_upper_bound_weights;

  // combined array to exchange the min and max coordinate, and total
  // weight of part.
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_process_local_min_max_coord_total_weight;
  
  // global combined array with the results for min, max and total weight.
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_global_min_max_coord_total_weight;

  // isDone is used to determine if a cutline is determined already. If a cut
  // line is already determined, the next iterations will skip this cut line.
  Kokkos::View<bool *, device_t> kokkos_is_cut_line_determined;

  // my_incomplete_cut_count count holds the number of cutlines that have not
  // been finalized for each part when concurrentPartCount>1, using this
  // information, if my_incomplete_cut_count[x]==0, then no work is done
  // for this part.
  Kokkos::View<mj_part_t *, device_t> kokkos_my_incomplete_cut_count;

  // local part weights of each thread.
  Kokkos::View<double *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_part_weights;

  // the work manupulation array for partweights.
  Kokkos::View<double *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_part_weight_work;

  // thread_cut_left_closest_point to hold the closest coordinate
  // to a cutline from left (for each thread).
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_cut_left_closest_point;

  // thread_cut_right_closest_point to hold the closest coordinate
  // to a cutline from right (for each thread)
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_cut_right_closest_point;

  // to store how many points in each part a thread has.
  Kokkos::View<mj_lno_t *, Kokkos::LayoutLeft, device_t>
    kokkos_thread_point_counts;

  Kokkos::View<mj_scalar_t *, device_t> kokkos_process_rectilinear_cut_weight;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_global_rectilinear_cut_weight;

  // for faster communication, concatanation of
  // totalPartWeights sized 2P-1, since there are P parts and P-1 cut lines
  // leftClosest distances sized P-1, since P-1 cut lines
  // rightClosest distances size P-1, since P-1 cut lines.
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_total_part_weight_left_right_closests;
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_global_total_part_weight_left_right_closests;

  /* \brief Either the mj array (part_no_array) or num_global_parts should be
   * provided in the input. part_no_array takes precedence if both are
   * provided. Depending on these parameters, total cut/part number, maximum
   * part/cut number along a dimension, estimated number of reduceAlls,
   * and the number of parts before the last dimension is calculated.
   * */
  void set_part_specifications();

  /* \brief Tries to determine the part number for current dimension,
   * by trying to make the partitioning as square as possible.
   * \param num_total_future how many more partitionings are required.
   * \param root how many more recursion depth is left.
   */
  inline mj_part_t get_part_count(
    mj_part_t num_total_future,
    double root);

  /* \brief Allocates all required memory for the mj partitioning algorithm.
   *
   */
  void allocate_set_work_memory();

  /* \brief for part communication we keep track of the box boundaries.
   * This is performed when either asked specifically, or when geometric
   * mapping is performed afterwards. This function initializes a single box
   * with all global min and max coordinates.
   * \param initial_partitioning_boxes the input and output vector for boxes.
   */
  void init_part_boxes(RCP<mj_partBoxVector_t> & outPartBoxes);

  /* \brief compute global bounding box:  min/max coords of global domain */
  void compute_global_box();

  /* \brief Function returns how many parts that will be obtained after this
   * dimension partitioning. It sets how many parts each current part will be
   * partitioned into in this dimension to view_num_partitioning_in_current_dim
   * vector, sets how many total future parts each obtained part will be
   * partitioned into in next_future_num_parts_in_parts vector, If part boxes
   * are kept, then sets initializes the output_part_boxes as its ancestor.
   * \param view_num_partitioning_in_current_dim: output. How many parts each
   * current part will be partitioned into.
   * \param future_num_part_in_parts: input, how many future parts each
   * current part will be partitioned into.
   * \param next_future_num_parts_in_parts: output, how many future parts
   * each obtained part will be partitioned into.
   * \param future_num_parts: output, max number of future parts that will be
   * obtained from a single
   * \param current_num_parts: input, how many parts are there currently.
   * \param current_iteration: input, current dimension iteration number.
   * \param input_part_boxes: input, if boxes are kept, current boxes.
   * \param output_part_boxes: output, if boxes are kept, the initial box
   * boundaries for obtained parts.
   */
  mj_part_t update_part_num_arrays(
    Kokkos::View<mj_part_t*, device_t> & view_num_partitioning_in_current_dim,
    std::vector<mj_part_t> *future_num_part_in_parts,
    std::vector<mj_part_t> *next_future_num_parts_in_parts,
    mj_part_t &future_num_parts,
    mj_part_t current_num_parts,
    int current_iteration,
    RCP<mj_partBoxVector_t> input_part_boxes,
    RCP<mj_partBoxVector_t> output_part_boxes,
    mj_part_t atomic_part_count);

  /*! \brief Function to determine the local minimum and maximum coordinate,
   * and local total weight
   * in the given set of local points.
   * TODO: Fix parameters doc
   */
  void mj_get_local_min_max_coord_totW(
    mj_part_t current_work_part,
    mj_part_t current_concurrent_num_parts,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords);

  /*! \brief Function to determine the local minimum and maximum coordinate,
   * and local total weight
   * in the given set of local points.
   * TODO: Fix parameters
   */
  void mj_taskmapper_get_local_min_max_coord_totW(
    mj_part_t current_work_part,
    mj_part_t current_concurrent_num_parts,
    int kk,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords);

  /*! \brief Function that reduces global minimum and maximum coordinates with
   * global total weight from given local arrays.
   * \param current_concurrent_num_parts is the number of parts whose cut
   * lines will be calculated concurrently.
   * \param local_min_max_total is the array holding local min and max
   * coordinate values with local total weight.
   * First current_concurrent_num_parts entries are minimums of the parts,
   * next current_concurrent_num_parts entries are max and then total weights.
   * \param global_min_max_total is the output array holding global min and
   * global coordinate values with global total weight.
   * The structure is same as local_min_max_total.
   */
  void mj_get_global_min_max_coord_totW(
    mj_part_t current_concurrent_num_parts,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_local_min_max_total,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_global_min_max_total);

  /*! \brief Function that calculates the new coordinates for the cut lines.
   * Function is called inside the parallel region.
   * \param min_coord minimum coordinate in the range.
   * \param max_coord maximum coordinate in the range.
   * \param num_cuts holds number of cuts in current partitioning dimension.
   * \param global_weight holds the global total weight in the current part.
   * \param initial_cut_coords is the output array for the initial cut lines.
   * \param target_part_weights is the output array holding the cumulative
   * ratios of parts in current partitioning.
   * For partitioning to 4 uniformly, target_part_weights will be
   * (0.25 * globalTotalWeight, 0.5 *globalTotalWeight , 0.75 *
   * globalTotalWeight, globalTotalWeight).
   * \param future_num_part_in_parts is the vector that holds how many more
   * parts each part will be divided into more
   * for the parts at the beginning of this coordinate partitioning
   * \param next_future_num_parts_in_parts is the vector that holds how many
   * more parts each part will be divided into more for the parts that will be
   * obtained at the end of this coordinate partitioning.
   * \param concurrent_current_part is the index of the part in the
   * future_num_part_in_parts vector.
   * \param obtained_part_index holds the amount of shift in the
   * next_future_num_parts_in_parts for the output parts.
   */
  void mj_get_initial_cut_coords_target_weights(
    mj_scalar_t min_coord,
    mj_scalar_t max_coord,
    mj_part_t num_cuts/*p-1*/ ,
    mj_scalar_t global_weight,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_initial_cut_coords,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_target_part_weights,
    std::vector <mj_part_t> *future_num_part_in_parts,
    std::vector <mj_part_t> *next_future_num_parts_in_parts,
    mj_part_t concurrent_current_part,
    mj_part_t obtained_part_index);

  /*! \brief Function that calculates the new coordinates for the cut lines.
   * Function is called inside the parallel region.
   * \param max_coordinate maximum coordinate in the range.
   * \param min_coordinate minimum coordinate in the range.
   * \param concurrent_current_part_index is the index of the part in the
   * inTotalCounts vector.
   * \param coordinate_begin_index holds the beginning of the coordinates
   * in current part.
   * \param coordinate_end_index holds end of the coordinates in current part.
   * \param mj_current_coordinate_permutations is the permutation array, holds
   * the real indices of coordinates on mj_current_dim_coords array.
   * \param mj_current_dim_coords is the 1D array holding the coordinates.
   * \param mj_part_ids is the array holding the partIds of each coordinate.
   * \param partition_count is the number of parts that the current part will
   * be partitioned into.
   */
  void set_initial_coordinate_parts(
    mj_scalar_t &max_coordinate,
    mj_scalar_t &min_coordinate,
    mj_part_t &concurrent_current_part_index,
    mj_lno_t coordinate_begin_index,
    mj_lno_t coordinate_end_index,
    Kokkos::View<mj_lno_t *, device_t>
      kokkos_mj_current_coordinate_permutations,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords,
    Kokkos::View<mj_part_t *, device_t> kokkos_mj_part_ids,
    mj_part_t &partition_count);

  /*! \brief Function that is responsible from 1D partitioning of the given
   * range of coordinates.
   * \param mj_current_dim_coords is 1 dimensional array holding coordinate
   * values.
   * \param imbalanceTolerance is the maximum allowed imbalance ratio.
   * \param current_work_part is the beginning index of concurrentPartCount
   * parts.
   * \param current_concurrent_num_parts is the number of parts whose cut
   * lines will be calculated concurrently.
   * \param current_cut_coordinates is the array holding the coordinates of
   * the cut.
   * \param total_incomplete_cut_count is the number of cut lines whose
   * positions should be calculated.
   * \param view_num_partitioning_in_current_dim is the vector that holds how
   * many parts each part will be divided into.
   */
  void mj_1D_part(
    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords,
    mj_scalar_t imbalanceTolerance,
    mj_part_t current_work_part,
    mj_part_t current_concurrent_num_parts,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_coordinates,
    mj_part_t total_incomplete_cut_count,
    Kokkos::View<mj_part_t*, device_t> &
      view_num_partitioning_in_current_dim,
    Kokkos::View<mj_part_t *, device_t> view_rectilinear_cut_count,
    Kokkos::View<size_t*, device_t> view_total_reduction_size);

  /*! \brief Function that calculates the weights of each part according to
   * given part cut coordinates. Function is called inside the parallel
   * region. Thread specific work arrays are provided as function parameter.
   *
   * \param total_part_count is the sum of number of cutlines and number of
   * parts. Simply it is 2*P - 1.
   * \param num_cuts is the number of cut lines. P - 1.
   * \param max_coord is the maximum coordinate in the part.
   * \param min_coord is the min coordinate in the part.
   * \param coordinate_begin_index is the index of the first coordinate in
   * current part.
   * \param coordinate_end_index is the index of the last coordinate in
   * current part.
   * \param mj_current_dim_coords is 1 dimensional array holding coordinate
   * values.
   * \param temp_current_cut_coords is the array holding the coordinates of
   * each cut line. Sized P - 1.
   * \param current_cut_status is the boolean array to determine if the
   * correct position for a cut line is found.
   * \param my_current_part_weights is the array holding the part weights for
   * the calling thread.
   * \param my_current_left_closest is the array holding the coordinate of the
   * closest points to the cut lines from left for the calling thread.
   * \param my_current_right_closest is the array holding the coordinate of
   * the closest points to the cut lines from right for the calling thread.
   * \param partIds is the array that holds the part ids of the coordinates
   */
  void mj_1D_part_get_thread_part_weights(
    Kokkos::View<mj_part_t*, device_t> view_num_partitioning_in_current_dim,
    mj_part_t current_concurrent_num_parts,
    mj_part_t current_work_part,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords);

  /*! \brief Function that reduces the result of multiple threads for
   * left and right closest points and part weights in a single mpi process.
   * \param view_num_partitioning_in_current_dim is the vector that holds the
   * number of cut lines in current dimension for each part.
   * \param current_work_part holds the index of the first part (important
   * when concurrent parts are used.)
   * \param current_concurrent_num_parts is the number of parts whose cut
   * lines will be calculated concurrently.
   */
  void mj_accumulate_thread_results(
    Kokkos::View<mj_part_t*, device_t> view_num_partitioning_in_current_dim,
    mj_part_t current_work_part,
    mj_part_t current_concurrent_num_parts,
    Kokkos::View<bool *, device_t> local_kokkos_is_cut_line_determined,
    Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
      local_kokkos_thread_cut_left_closest_point,
    Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
      local_kokkos_thread_cut_right_closest_point,
    Kokkos::View<mj_scalar_t *, device_t>
      local_kokkos_total_part_weight_left_right_closests,
    Kokkos::View<double *, Kokkos::LayoutLeft, device_t>
      local_kokkos_thread_part_weights);

  /*! \brief Function that calculates the new coordinates for the cut lines.
   * Function is called inside the parallel region. Write the new cut
   * coordinates to new_current_cut_coordinates, and determines if the final
   * position of a cut is found.
   * \param num_total_part is the sum of number of cutlines and number of
   * parts. Simply it is 2*P - 1.
   * \param num_cuts is the number of cut lines. P - 1.
   * \param max_coordinate is the maximum coordinate in the current range of
   * coordinates and in the current dimension.
   * \param min_coordinate is the maximum coordinate in the current range of
   * coordinates and in the current dimension.
   * \param global_total_weight is the global total weight in the current
   * range of coordinates.
   * \param used_imbalance_tolerance is the maximum allowed imbalance ratio.
   * \param current_global_part_weights is the array holding the weight of
   * parts. Assumes there are 2*P - 1 parts (cut lines are seperate parts).
   * \param current_local_part_weights is local totalweight of the processor.
   * \param current_part_target_weights desired cumulative part ratios, size P.
   * \param current_cut_line_determined is the boolean array to determine if
   * the correct position for a cut line is found.
   * \param current_cut_coordinates is the array holding the coordinates of
   * each cut line. Sized P - 1.
   * \param current_cut_upper_bounds is the array holding the upper bound
   * coordinate for each cut line. Sized P - 1.
   * \param current_cut_lower_bounds is the array holding the lower bound
   * coordinate for each cut line. Sized P - 1.
   * \param current_global_left_closest_points is the array holding the
   * closest points to the cut lines from left.
   * \param current_global_right_closest_points is the array holding the
   * closest points to the cut lines from right.
   * \param current_cut_lower_bound_weights is the array holding the weight
   * of the parts at the left of lower bound coordinates.
   * \param current_cut_upper_weights is the array holding the weight of the
   * parts at the left of upper bound coordinates.
   * \param new_current_cut_coordinates is the work array, sized P - 1.
   * \param current_part_cut_line_weight_ratio holds how much weight of the
   * coordinates on the cutline should be put on left side.
   * \param rectilinear_cut_count is the count of cut lines whose balance can
   * be achived via distributing points in same coordinate to different parts.
   * \param my_num_incomplete_cut is the number of cutlines whose position has
   * not been determined yet. For K > 1 it is the count in a single part
   * (whose cut lines are determined).
   */
  void mj_get_new_cut_coordinates(
    mj_part_t current_concurrent_num_parts,
    mj_part_t kk,
    const size_t &num_total_part,
    const mj_part_t &num_cuts,
    const mj_scalar_t &used_imbalance_tolerance,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_current_global_part_weights,
    Kokkos::View<const mj_scalar_t *, device_t>
      kokkos_current_local_part_weights,
    Kokkos::View<const mj_scalar_t *, device_t>
      kokkos_current_part_target_weights,
    Kokkos::View<bool *, device_t> kokkos_current_cut_line_determined,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_coordinates,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_upper_bounds,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_lower_bounds,
    Kokkos::View<mj_scalar_t *, device_t> current_global_left_closest_points,
    Kokkos::View<mj_scalar_t *, device_t> current_global_right_closest_points,
    Kokkos::View<mj_scalar_t *, device_t>
      kokkos_current_cut_lower_bound_weights,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_upper_weights,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_new_current_cut_coordinates,
    Kokkos::View<mj_scalar_t *, device_t> 
    current_part_cut_line_weight_to_put_left,
    Kokkos::View<mj_part_t *, device_t> view_rectilinear_cut_count);

  /*! \brief
   * Function that calculates the next pivot position,
   * according to given coordinates of upper bound and lower bound, the
   * weights at upper and lower bounds, and the expected weight.
   * \param cut_upper_bound is the upper bound coordinate of the cut.
   * \param cut_lower_bound is the lower bound coordinate of the cut.
   * \param cut_upper_weight is the weights at the upper bound of the cut.
   * \param cut_lower_weight is the weights at the lower bound of the cut.
   * \param expected_weight is the expected weight that should be placed on
   * the left of the cut line.
   */
  KOKKOS_INLINE_FUNCTION void mj_calculate_new_cut_position (
    mj_scalar_t cut_upper_bound,
    mj_scalar_t cut_lower_bound,
    mj_scalar_t cut_upper_weight,
    mj_scalar_t cut_lower_weight,
    mj_scalar_t expected_weight,
    mj_scalar_t &new_cut_position);

  /*! \brief Function that determines the permutation indices of coordinates.
   * \param num_parts is the number of parts.
   * \param mj_current_dim_coords is 1 dimensional array holding the
   * coordinate values.
   * \param current_concurrent_cut_coordinate is 1 dimensional array holding
   * the cut coordinates.
   * \param coordinate_begin is the start index of the given partition on
   * partitionedPointPermutations.
   * \param coordinate_end is the end index of the given partition on
   * partitionedPointPermutations.
   * \param used_local_cut_line_weight_to_left holds how much weight of the
   * coordinates on the cutline should be put on left side.
   * \param used_thread_part_weight_work is the two dimensional array holding
   * the weight of parts for each thread. Assumes there are 2*P - 1 parts
   * (cut lines are seperate parts).
   * \param out_part_xadj is the indices of coordinates calculated for the
   * partition on next dimension.
   */
  void mj_create_new_partitions(
    mj_part_t num_parts,
    mj_part_t current_concurrent_work_part,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords,
    Kokkos::View<mj_scalar_t *, device_t>
      kokkos_current_concurrent_cut_coordinate,
    Kokkos::View<mj_scalar_t *, device_t>
      kokkos_used_local_cut_line_weight_to_left,
    Kokkos::View<double *, Kokkos::LayoutLeft, device_t>
      used_thread_part_weight_work,
    Kokkos::View<mj_lno_t *, device_t> kokkos_out_part_xadj,
    Kokkos::View<mj_lno_t *, Kokkos::LayoutLeft, device_t>
      local_kokkos_thread_point_counts,
    bool local_distribute_points_on_cut_lines,
    Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
      local_kokkos_thread_cut_line_weight_to_put_left,
    mj_scalar_t local_sEpsilon,
    Kokkos::View<mj_lno_t*, device_t> local_kokkos_coordinate_permutations,
    Kokkos::View<bool*, device_t> local_kokkos_mj_uniform_weights,
    Kokkos::View<mj_scalar_t**, device_t> local_kokkos_mj_weights,
    Kokkos::View<mj_part_t*, device_t> local_kokkos_assigned_part_ids,
    Kokkos::View<mj_lno_t*, device_t>
      local_kokkos_new_coordinate_permutations);

  /*! \brief Function checks if should do migration or not.
   * It returns true to point that migration should be done when
   * -migration_reduce_all_population are higher than a predetermined value
   * -num_coords_for_last_dim_part that left for the last dimension
   * partitioning is less than a predetermined value - the imbalance of the
   * processors on the parts are higher than given threshold.
   * \param input_num_parts is the number of parts when migration is called.
   * \param output_num_parts is the output number of parts after migration.
   * \param next_future_num_parts_in_parts is the number of total future parts
   * each part is partitioned into. Updated when migration is performed.
   * \param output_part_begin_index is the number that will be used as
   * beginning part number when final solution part numbers are assigned.
   * \param migration_reduce_all_population is the estimated total number of
   * reduceall operations multiplied with number of processors to be used for
   * determining migration.
   * \param num_coords_for_last_dim_part is the estimated number of points in
   * each part, when last dimension partitioning is performed.
   * \param iteration is the string that gives information about the dimension
   * for printing purposes.
   * \param input_part_boxes is the array that holds the part boxes after the
   * migration. (swapped)
   * \param output_part_boxes is the array that holds the part boxes before
   * the migration. (swapped)
   *
   */
  bool mj_perform_migration(
    mj_part_t in_num_parts, //current umb parts
    mj_part_t &out_num_parts, //output umb parts.
    std::vector<mj_part_t> *next_future_num_parts_in_parts,
    mj_part_t &output_part_begin_index,
    size_t migration_reduce_all_population,
    mj_lno_t num_coords_for_last_dim_part,
    std::string iteration,
    RCP<mj_partBoxVector_t> &input_part_boxes,
    RCP<mj_partBoxVector_t> &output_part_boxes);

  /*! \brief Function fills up the num_points_in_all_processor_parts, so that
   * it has the number of coordinates in each processor of each part.
   * to access how many points processor i has on part j,
   * num_points_in_all_processor_parts[i * num_parts + j].
   *
   * \param num_procs is the number of processors for migration operation.
   * \param num_parts is the number of parts in the current partitioning.
   * \param num_points_in_all_processor_parts is the output array that holds
   * the number of coordinates in each part in each processor.
   */
  void get_processor_num_points_in_parts(
    mj_part_t num_procs,
    mj_part_t num_parts,
    mj_gno_t *&num_points_in_all_processor_parts);

  /*! \brief Function checks if should do migration or not.
   * It returns true to point that migration should be done when
   * -migration_reduce_all_population are higher than a predetermined value
   * -num_coords_for_last_dim_part that left for the last dimension
   * partitioning is less than a predetermined value - the imbalance of the
   * processors on the parts are higher than given threshold.
   * \param migration_reduce_all_population is the multiplication of the
   * number of reduceall operations estimated and the number of processors.
   * \param num_coords_for_last_dim_part is the estimated number of
   * coordinates in a part per processor in the last dimension partitioning.
   * \param num_procs is the number of processor attending to migration
   * operation.
   * \param num_parts is the number of parts that exist in the current
   * partitioning.
   * \param num_points_in_all_processor_parts is the input array that holds
   * the number of coordinates in each part in each processor.
   */
  bool mj_check_to_migrate(
    size_t migration_reduce_all_population,
    mj_lno_t num_coords_for_last_dim_part,
    mj_part_t num_procs,
    mj_part_t num_parts,
    mj_gno_t *num_points_in_all_processor_parts);

  /*! \brief Function fills up coordinate_destinations is the output array
   * that holds which part each coordinate should be sent. In addition it
   * calculates the shift amount (output_part_numbering_begin_index) to be
   * done when final numberings of the parts are performed.
   * \param num_points_in_all_processor_parts is the array holding the num
   * points in each part in each proc.
   * \param num_parts is the number of parts that exist in the current
   * partitioning.
   * \param num_procs is the number of processor attending to migration
   * operation.
   * \param send_count_to_each_proc array array storing the number of points
   * to be sent to each part.
   * \param processor_ranks_for_subcomm is the ranks of the processors that
   * will be in the subcommunicator with me.
   * \param next_future_num_parts_in_parts is the vector, how many more parts
   * each part will be divided into in the future.
   * \param out_num_part is the number of parts assigned to the process.
   * \param out_part_indices is the indices of the part to which the processor
   * is assigned.
   * \param output_part_numbering_begin_index is how much the numbers should
   * be shifted when numbering the result parts.
   * \param coordinate_destinations is the output array that holds which part
   * each coordinate should be sent.
   */
  void mj_migration_part_proc_assignment(
    mj_gno_t * num_points_in_all_processor_parts,
    mj_part_t num_parts,
    mj_part_t num_procs,
    mj_lno_t *send_count_to_each_proc,
    std::vector<mj_part_t> &processor_ranks_for_subcomm,
    std::vector<mj_part_t> *next_future_num_parts_in_parts,
    mj_part_t &out_num_part,
    std::vector<mj_part_t> &out_part_indices,
    mj_part_t &output_part_numbering_begin_index,
    int *coordinate_destinations);

  /*! \brief Function that assigned the processors to parts, when there are
   * more processors then parts.
   *  sets the destination of each coordinate in coordinate_destinations, also
   * edits output_part_numbering_begin_index,
   *  and out_part_index, and returns the processor_ranks_for_subcomm which
   * represents the ranks of the processors
   *  that will be used for creating the subcommunicator.
   * \param num_points_in_all_processor_parts is the array holding the num
   * points in each part in each proc.
   * \param num_parts is the number of parts that exist in the current
   * partitioning.
   * \param num_procs is the number of processor attending to migration
   * operation.
   * \param send_count_to_each_proc array array storing the number of points
   * to be sent to each part.
   * \param processor_ranks_for_subcomm is the ranks of the processors that
   * will be in the subcommunicator with me.
   * \param next_future_num_parts_in_parts is the vector, how many more parts
   * each part will be divided into in the future.
   * \param out_part_index is the index of the part to which the processor
   * is assigned.
   * \param output_part_numbering_begin_index is how much the numbers should
   * be shifted when numbering the result parts.
   * \param coordinate_destinations is the output array that holds which part
   * each coordinate should be sent.
   */
  void mj_assign_proc_to_parts(
    mj_gno_t * num_points_in_all_processor_parts,
    mj_part_t num_parts,
    mj_part_t num_procs,
    mj_lno_t *send_count_to_each_proc,
    std::vector<mj_part_t> &processor_ranks_for_subcomm,
    std::vector<mj_part_t> *next_future_num_parts_in_parts,
    mj_part_t &out_part_index,
    mj_part_t &output_part_numbering_begin_index,
    int *coordinate_destinations);

  /*! \brief Function fills up coordinate_destinations is the output array
   * that holds which part each coordinate should be sent.
   * \param num_parts is the number of parts that exist in the
   * current partitioning.
   * \param num_procs is the number of processors attending to
   * migration operation.
   * \param part_assignment_proc_begin_indices ([i]) points to the first
   * processor index that part i will be sent to.
   * \param processor_chains_in_parts the array that holds the linked list
   * structure, started from part_assignment_proc_begin_indices ([i]).
   * \param send_count_to_each_proc array array storing the number of points to
   * be sent to each part.
   * \param coordinate_destinations is the output array that holds which part
   * each coordinate should be sent.
   */
  void assign_send_destinations(
    mj_part_t num_parts,
    mj_part_t *part_assignment_proc_begin_indices,
    mj_part_t *processor_chains_in_parts,
    mj_lno_t *send_count_to_each_proc,
    int *coordinate_destinations);

  /*! \brief Function fills up coordinate_destinations is the output array
   * that holds which part each coordinate should be sent. In addition it
   * calculates the shift amount (output_part_numbering_begin_index) to be done
   * when final numberings of the parts are performed.
   * \param num_parts is the number of parts in the current partitioning.
   * \param sort_item_part_to_proc_assignment is the sorted parts with respect
   * to the assigned processors.
   * \param coordinate_destinations is the output array that holds which part
   * each coordinate should be sent.
   * \param output_part_numbering_begin_index is how much the numbers should be
   * shifted when numbering the result parts.
   * \param next_future_num_parts_in_parts is the vector, how many more parts
   * each part will be divided into in the future.
   */
  void assign_send_destinations2(
    mj_part_t num_parts,
    uSortItem<mj_part_t, mj_part_t> * sort_item_part_to_proc_assignment,
    int *coordinate_destinations,
    mj_part_t &output_part_numbering_begin_index,
    std::vector<mj_part_t> *next_future_num_parts_in_parts);

  /*! \brief Function fills up coordinate_destinations is the output array
   * that holds which part each coordinate should be sent. In addition it
   * calculates the shift amount (output_part_numbering_begin_index) to be done
   * when final numberings of the parts are performed.
   * \param num_points_in_all_processor_parts is the array holding the num
   * points in each part in each proc.
   * \param num_parts is the number of parts that exist in the current
   * partitioning.
   * \param num_procs is the number of processors attending to
   * migration operation.
   * \param send_count_to_each_proc array array storing the number of points to
   * be sent to each part.
   * \param next_future_num_parts_in_parts is the vector, how many more parts
   * each part will be divided into in the future.
   * \param out_num_part is the number of parts assigned to the process.
   * \param out_part_indices is the indices of the part to which the processor
   * is assigned.
   * \param output_part_numbering_begin_index is how much the numbers should be
   * shifted when numbering the result parts.
   * \param coordinate_destinations is the output array that holds which parta
   * each coordinate should be sent.
   */
  void mj_assign_parts_to_procs(
    mj_gno_t * num_points_in_all_processor_parts,
    mj_part_t num_parts,
    mj_part_t num_procs,
    mj_lno_t *send_count_to_each_proc, 
    std::vector<mj_part_t> *next_future_num_parts_in_parts,
    mj_part_t &out_num_part,
    std::vector<mj_part_t> &out_part_indices,
    mj_part_t &output_part_numbering_begin_index,
    int *coordinate_destinations);

  /*! \brief Function fills up coordinate_destinations is the output array
   * that holds which part each coordinate should be sent. In addition it
   * calculates the shift amount (output_part_numbering_begin_index) to be done
   * when final numberings of the parts are performed.
   * \param num_procs is the number of processora attending to
   * migration operation.
   * \param num_new_local_points is the output to represent the new number
   * of local points.
   * \param iteration is the string for the current iteration.
   * \param coordinate_destinations is the output array that holds which part
   * each coordinate should be sent.
   * \param num_parts is the number of parts in the current partitioning.
   */
  void mj_migrate_coords(
    mj_part_t num_procs,
    mj_lno_t &num_new_local_points,
    std::string iteration,
    int *coordinate_destinations,
    mj_part_t num_parts);

  /*! \brief Function creates the new subcomminicator for the processors
   * given in processor_ranks_for_subcomm.
   * \param processor_ranks_for_subcomm is the vector that has the ranks of
   * the processors that will be in the same group.
   */
  void create_sub_communicator(
    std::vector<mj_part_t> &processor_ranks_for_subcomm);

  /*! \brief Function writes the new permutation arrays after the migration.
   * \param output_num_parts is the number of parts assigned to the processor.
   * \param num_parts is the number of parts right before migration.
   */
  void fill_permutation_array(
    mj_part_t output_num_parts,
    mj_part_t num_parts);

  /*! \brief Function checks if should do migration or not.
   * \param current_num_parts is the number of parts in the process.
   * \param output_part_begin_index is the number that will be used as
   * beginning part number
   * \param output_part_boxes is the array that holds the part boxes
   * \param is_data_ever_migrated is the boolean value which is true
   * if the data is ever migrated during the partitioning.
   */
  void set_final_parts(
    mj_part_t current_num_parts,
    mj_part_t output_part_begin_index,
    RCP<mj_partBoxVector_t> &output_part_boxes,
    bool is_data_ever_migrated);

  /*! \brief Function creates consistent chunks for task partitioning. Used only
   * in the case of sequential task partitioning, where consistent handle of the
   * points on the cuts are required.
   * \param num_parts is the number of parts.
   * \param mj_current_dim_coords is 1 dimensional array holding the
   * coordinate values.
   * \param current_concurrent_cut_coordinate is 1 dimensional array holding
   * the cut coordinates.
   * \param coordinate_begin is the start index of the given partition on
   * partitionedPointPermutations.
   * \param coordinate_end is the end index of the given partition on
   * partitionedPointPermutations.
   * \param used_local_cut_line_weight_to_left holds how much weight of the
   * coordinates on the cutline should be put on left side.
   * \param out_part_xadj is the indices of begginning and end of the parts in
  * the output partition.
   * \param coordInd is the index according to which the partitioning is done.
   */
  void create_consistent_chunks(
    mj_part_t num_parts,
    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords,
    Kokkos::View<mj_scalar_t *, device_t> current_concurrent_cut_coordinate,
    mj_lno_t coordinate_begin,
    mj_lno_t coordinate_end,
    Kokkos::View<mj_scalar_t *, device_t> used_local_cut_line_weight_to_left,
    Kokkos::View<mj_lno_t *, device_t> kokkos_out_part_xadj,
    int coordInd,
    bool longest_dim_part,
    uSignedSortItem<int, mj_scalar_t,
    char> *p_coord_dimension_range_sorted);

  /*!
   * \brief Function returns the largest prime factor of a given number.
   * input and output are integer-like.
   */
  mj_part_t find_largest_prime_factor(mj_part_t num_parts){
    mj_part_t largest_factor = 1;
    mj_part_t n = num_parts;
    mj_part_t divisor = 2;
    while (n > 1){
      while (n % divisor == 0){
        n = n / divisor;
        largest_factor = divisor;
      }
      ++divisor;
      if (divisor * divisor > n){
        if (n > 1){
          largest_factor = n;
        }
        break;
      }
    }
    return largest_factor;
  }

public:
  AlgMJ();

  /*! \brief Multi Jagged  coordinate partitioning algorithm.
   *
   * \param env   library configuration and problem parameters
   * \param problemComm the communicator for the problem
   * \param imbalance_tolerance : the input provided imbalance tolerance.
   * \param num_global_parts: number of target global parts.
   * \param part_no_array: part no array, if provided this will be used
   * for partitioning.
   * \param recursion_depth: if part no array is provided, it is the length of
   * part no array, if part no is not provided than it is the number of steps
   * that algorithm will divide into num_global_parts parts.
   * \param coord_dim: coordinate dimension
   * \param num_local_coords: number of local coordinates
   * \param num_global_coords: number of global coordinates
   * \param initial_mj_gnos: the list of initial global id's
   * \param mj_coordinates: the two dimensional coordinate array.
   * \param num_weights_per_coord: number of weights per coordinate
   * \param mj_uniform_weights: if weight index [i] has uniform weight or not.
   * \param mj_weights: the two dimensional array for weights
   * \param mj_uniform_parts: if the target partitioning aims uniform parts
   * \param mj_part_sizes: if the target partitioning does not aim uniform
   * parts, then weight of each part.
   * \param result_assigned_part_ids: Output - 1D pointer, should be provided
   * as null. the result partids corresponding to the coordinates given in
   * result_mj_gnos.
   * \param result_mj_gnos: Output - 1D pointer, should be provided as null. the
   * result coordinate global id's corresponding to the part_ids array.
   */
  void multi_jagged_part(
    const RCP<const Environment> &env,
    RCP<const Comm<int> > &problemComm,
    double imbalance_tolerance,
    size_t num_global_parts,
    Kokkos::View<mj_part_t*, device_t> kokkos_part_no_array,
    int recursion_depth,
    int coord_dim,
    mj_lno_t num_local_coords,
    mj_gno_t num_global_coords,
    Kokkos::View<const mj_gno_t*, device_t> kokkos_initial_mj_gnos,
    Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t>
      kokkos_mj_coordinates,
    int num_weights_per_coord,
    Kokkos::View<bool*, device_t> kokkos_mj_uniform_weights,
    Kokkos::View<mj_scalar_t**, device_t> kokkos_mj_weights,
    Kokkos::View<bool*, device_t> kokkos_mj_uniform_parts,
    Kokkos::View<mj_scalar_t**, device_t> kokkos_mj_part_sizes,
    Kokkos::View<mj_part_t*, device_t> &kokkos_result_assigned_part_ids,
    Kokkos::View<mj_gno_t*, device_t> &kokkos_result_mj_gnos);

  /*! \brief Multi Jagged  coordinate partitioning algorithm.
   *
   * \param distribute_points_on_cut_lines_ : if partitioning can distribute
   * points on same coordinate to different parts.
   * \param max_concurrent_part_calculation_ : how many parts we can calculate
   * concurrently.
   * \param check_migrate_avoid_migration_option_ : whether to migrate=1, avoid
   * migrate=2, or leave decision to MJ=0
   * \param minimum_migration_imbalance_  : when MJ decides whether to migrate,
   * the minimum imbalance for migration.
   * \param migration_type_ : when MJ migration whether to migrate for perfect
   * load-imbalance or less messages
   */
  void set_partitioning_parameters(
    bool distribute_points_on_cut_lines_,
    int max_concurrent_part_calculation_,
    int check_migrate_avoid_migration_option_,
    mj_scalar_t minimum_migration_imbalance_, int migration_type_ = 0);

  /*! \brief Function call, if the part boxes are intended to be kept.
   *
   */
  void set_to_keep_part_boxes();

  /*! \brief Return the global bounding box: min/max coords of global domain
   */
  RCP<mj_partBox_t> get_global_box() const;

  RCP<mj_partBoxVector_t> get_kept_boxes() const;

  RCP<mj_partBoxVector_t> compute_global_box_boundaries(
    RCP<mj_partBoxVector_t> &localPartBoxes) const;

  /*! \brief Special function for partitioning for task mapping.
   * Runs sequential, and performs deterministic partitioning for the
   * partitioning the points along a cutline.
   *
   * \param env library configuration and problem parameters
   * \param num_total_coords number of total coordinates
   * \param num_selected_coords : the number of selected coordinates. This is
   * to set, if there are n processors, but only m<n processors are selected for
   * mapping.
   * \param num_target_part: number of target global parts.
   * \param coord_dim_: coordinate dimension for coordinates
   * \param mj_coordinates_: the coordinates
   * \param inital_adjList_output_adjlist: Array allocated by caller, in the
   * size of num_total_coords, first num_selected_coords elements should list
   * the indices of the selected processors. This is output for output
   * permutation array.
   * \param output_xadj: The output part xadj array, pointing beginning and end
   * of each part on output permutation array (inital_adjList_output_adjlist).
   * Returned in CSR format: part i's info in output_xadj[i] : output_xadj[i+1]
   * \param rd: recursion depth
   * \param part_no_array_: possibly null part_no_array, specifying how many
   * parts each should be divided during partitioning.
   */
  void sequential_task_partitioning(
    const RCP<const Environment> &env,
    mj_lno_t num_total_coords,
    mj_lno_t num_selected_coords,
    size_t num_target_part,
    int coord_dim,
    Kokkos::View<mj_scalar_t **, Kokkos::LayoutLeft, device_t> mj_coordinates_,
    Kokkos::View<mj_lno_t *, device_t>
      kokkos_initial_selected_coords_output_permutation,
    mj_lno_t *output_xadj,
    int recursion_depth,
    Kokkos::View<mj_part_t *, device_t> kokkos_part_no_array,
    bool partition_along_longest_dim,
    int num_ranks_per_node,
    bool divide_to_prime_first_);
};

/*! \brief Multi Jagged  coordinate partitioning algorithm default constructor.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::AlgMJ():
  mj_env(), mj_problemComm(), comm(), imbalance_tolerance(0),
  recursion_depth(0), coord_dim(0),
  num_weights_per_coord(0), initial_num_loc_coords(0),
  initial_num_glob_coords(0),
  num_local_coords(0), num_global_coords(0),
  sEpsilon(std::numeric_limits<mj_scalar_t>::epsilon() * 100),
  distribute_points_on_cut_lines(true),
  max_concurrent_part_calculation(1),
  mj_run_as_rcb(false), mj_user_recursion_depth(0),
  mj_keep_part_boxes(false),
  check_migrate_avoid_migration_option(0), migration_type(0),
  minimum_migration_imbalance(0.30),
  total_num_cut(0), total_num_part(0), max_num_part_along_dim(0),
  max_num_cut_along_dim(0),
  max_num_total_part_along_dim(0),
  total_dim_num_reduce_all(0),
  last_dim_num_part(0),
  num_global_parts(1), kept_boxes(), global_box(),
  myRank(0), myActualRank(0), 
  divide_to_prime_first(false)
{
}

/*! \brief Special function for partitioning for task mapping.
 * Runs sequential, and performs deterministic partitioning for the
 * partitioning the points along a cutline.
 *
 * \param env library configuration and problem parameters
 * \param num_total_coords number of total coordinates
 * \param num_selected_coords : the number of selected coordinates. This is to
 * set, if there are n processors, but only m<n processors are selected for
 * mapping.
 * \param num_target_part: number of target global parts.
 * \param coord_dim_: coordinate dimension for coordinates
 * \param mj_coordinates_: the coordinates
 * \param inital_adjList_output_adjlist: Array allocated by caller, in the size
 * of num_total_coords, first num_selected_coords elements should list the
 * indices of the selected processors. This is output for output permutation
 * array.
 * \param output_xadj: The output part xadj array, pointing beginning and end of
 * each part on output permutation array (inital_adjList_output_adjlist).
 * Returned in CSR format: part i's info in output_xadj[i] : output_xadj[i+1]
 * \param rd: recursion depth
 * \param part_no_array_: possibly null part_no_array, specifying how many parts
 * each should be divided during partitioning.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  sequential_task_partitioning(
  const RCP<const Environment> &env,
  mj_lno_t num_total_coords,
  mj_lno_t num_selected_coords,
  size_t num_target_part,
  int coord_dim_,
  Kokkos::View<mj_scalar_t **, Kokkos::LayoutLeft, device_t>
    kokkos_mj_coordinates_,
  Kokkos::View<mj_lno_t *, device_t> kokkos_inital_adjList_output_adjlist,
  mj_lno_t *output_xadj,
  int rd,
  Kokkos::View<mj_part_t *, device_t> kokkos_part_no_array_,
  bool partition_along_longest_dim,
  int num_ranks_per_node,
  bool divide_to_prime_first_)
{
  this->mj_env = env;
  const RCP<Comm<int> > commN;
  this->mj_problemComm = 
    Teuchos::DefaultComm<int>::getDefaultSerialComm(commN);
  this->comm = 
    Teuchos::rcp_const_cast<Comm<int> >(this->mj_problemComm);
  this->myActualRank = this->myRank = 1;

  this->divide_to_prime_first = divide_to_prime_first_;
  //weights are uniform for task mapping

  //parts are uniform for task mapping
  //as input indices.
  this->imbalance_tolerance = 0;
  this->num_global_parts = num_target_part;
  this->kokkos_part_no_array = kokkos_part_no_array_;
  this->recursion_depth = rd;
  this->coord_dim = coord_dim_;
  this->num_local_coords = num_total_coords;
  this->num_global_coords = num_total_coords;
  
  // will copy the memory to this->kokkos_mj_coordinates.
  this->kokkos_mj_coordinates = kokkos_mj_coordinates_;

  // temporary memory. It is not used here, but the functions
  // require these to be allocated.
  // will copy the memory to this->current_mj_gnos[j].

  this->kokkos_initial_mj_gnos =
    Kokkos::View<mj_gno_t*, device_t>("gids", this->num_local_coords);

  this->num_weights_per_coord = 0;
  Kokkos::View<bool*, device_t>
    kokkos_tmp_mj_uniform_weights("uniform weights", 1);
  this->kokkos_mj_uniform_weights = kokkos_tmp_mj_uniform_weights;
  this->kokkos_mj_uniform_weights(0) = true;
  Kokkos::View<mj_scalar_t**, device_t> kokkos_tmp_mj_weights("weights", 1);
  this->kokkos_mj_weights = kokkos_tmp_mj_weights;
  Kokkos::View<bool*, device_t> kokkos_tmp_mj_uniform_parts("uniform parts", 1);
  this->kokkos_mj_uniform_parts = kokkos_tmp_mj_uniform_parts;
  this->kokkos_mj_uniform_parts(0) = true;
  Kokkos::View<mj_scalar_t**, device_t>
    kokkos_tmp_mj_part_sizes("part sizes", 1);
  this->kokkos_mj_part_sizes = kokkos_tmp_mj_part_sizes;

  this->set_part_specifications();

  this->allocate_set_work_memory();
  // the end of the initial partition is the end of coordinates.
  this->kokkos_part_xadj(0) = static_cast<mj_lno_t>(num_selected_coords);

  for(size_t i = 0; i < static_cast<size_t>(num_total_coords); ++i){
    this->kokkos_coordinate_permutations(i) =
      kokkos_inital_adjList_output_adjlist(i);
  }

  mj_part_t current_num_parts = 1;

  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_coordinates =
    this->kokkos_all_cut_coordinates;

  mj_part_t future_num_parts = this->total_num_part;

  std::vector<mj_part_t> *future_num_part_in_parts =
    new std::vector<mj_part_t> ();
  std::vector<mj_part_t> *next_future_num_parts_in_parts =
    new std::vector<mj_part_t> ();
  next_future_num_parts_in_parts->push_back(this->num_global_parts);
  RCP<mj_partBoxVector_t> t1;
  RCP<mj_partBoxVector_t> t2;

  std::vector <uSignedSortItem<int, mj_scalar_t, char> >
    coord_dimension_range_sorted(this->coord_dim);
  uSignedSortItem<int, mj_scalar_t, char> *p_coord_dimension_range_sorted =
    &(coord_dimension_range_sorted[0]);
  std::vector <mj_scalar_t> coord_dim_mins(this->coord_dim);
  std::vector <mj_scalar_t> coord_dim_maxs(this->coord_dim);

  // Need a device counter - how best to allocate?
  // Putting this allocation in the loops is very costly so moved out here.
  Kokkos::View<mj_part_t*, device_t>
    view_rectilinear_cut_count("view_rectilinear_cut_count", 1);
  Kokkos::View<size_t*, device_t>
    view_total_reduction_size("view_total_reduction_size", 1);

  for (int i = 0; i < this->recursion_depth; ++i){
    // partitioning array. size will be as the number of current partitions
    // and this holds how many parts that each part will be in the current
    // dimension partitioning.
    Kokkos::View<mj_part_t*, device_t> view_num_partitioning_in_current_dim;

    // number of parts that will be obtained at the end of this partitioning.
    // future_num_part_in_parts is as the size of current number of parts.
    // holds how many more parts each should be divided in the further
    // iterations. this will be used to calculate
    // view_num_partitioning_in_current_dim, as the number of parts that the
    // part will be partitioned in the current dimension partitioning.

    // next_future_num_parts_in_parts will be as the size of outnumParts,
    // and this will hold how many more parts that each output part
    // should be divided. this array will also be used to determine the weight
    // ratios of the parts.
    // swap the arrays to use iteratively..
    std::vector<mj_part_t> *tmpPartVect= future_num_part_in_parts;
    future_num_part_in_parts = next_future_num_parts_in_parts;
    next_future_num_parts_in_parts = tmpPartVect;

    // clear next_future_num_parts_in_parts array as
    // getPartitionArrays expects it to be empty.
    // it also expects view_num_partitioning_in_current_dim to be empty as well.
    next_future_num_parts_in_parts->clear();

    // returns the total number of output parts for this dimension partitioning.
    mj_part_t output_part_count_in_dimension =
      this->update_part_num_arrays(
        view_num_partitioning_in_current_dim,
        future_num_part_in_parts,
        next_future_num_parts_in_parts,
        future_num_parts,
        current_num_parts,
        i,
        t1,
        t2, num_ranks_per_node);

    // if the number of obtained parts equal to current number of parts,
    // skip this dimension. For example, this happens when 1 is given in
    // the input part array is given. P=4,5,1,2
    if(output_part_count_in_dimension == current_num_parts) {
      tmpPartVect= future_num_part_in_parts;
      future_num_part_in_parts = next_future_num_parts_in_parts;
      next_future_num_parts_in_parts = tmpPartVect;
      continue;
    }

    //convert i to string to be used for debugging purposes.
    std::string istring = Teuchos::toString<int>(i);

    // alloc Memory to point the indices
    // of the parts in the permutation array.
    this->kokkos_new_part_xadj = Kokkos::View<mj_lno_t*, device_t>(
      "new part xadj", output_part_count_in_dimension);

    // the index where in the outtotalCounts will be written.
    mj_part_t output_part_index = 0;

       // whatever is written to outTotalCounts will be added with previousEnd
    // so that the points will be shifted.
    mj_part_t output_coordinate_end_index = 0;

    mj_part_t current_work_part = 0;
    mj_part_t current_concurrent_num_parts = 1;

    mj_part_t obtained_part_index = 0;

    // get the coordinate axis along which the partitioning will be done.
    int coordInd = i % this->coord_dim;

    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords =
      Kokkos::subview(this->kokkos_mj_coordinates, Kokkos::ALL, coordInd);

    // run for all available parts.

    for (; current_work_part < current_num_parts;
      current_work_part += current_concurrent_num_parts) {

      // current_concurrent_num_parts =
      //  std::min(current_num_parts - current_work_part,
      // this->max_concurrent_part_calculation);

      mj_part_t actual_work_part_count = 0;
      // initialization for 1D partitioning.
      // get the min and max coordinates of each part
      // together with the part weights of each part.
      for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {
        mj_part_t current_work_part_in_concurrent_parts =
          current_work_part + kk;

        // if this part wont be partitioned any further
        //dont do any work for this part.

        mj_part_t partition_count;
        Kokkos::parallel_reduce("Read single", 1,
          KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
          set_single = view_num_partitioning_in_current_dim(
            current_work_part_in_concurrent_parts);
        }, partition_count);
            
        if (partition_count == 1) {
          continue;
        }
        ++actual_work_part_count;

        if(partition_along_longest_dim) {
          auto local_kokkos_process_local_min_max_coord_total_weight =
            this->kokkos_process_local_min_max_coord_total_weight;
          for (int coord_traverse_ind = 0;
            coord_traverse_ind < this->coord_dim; ++coord_traverse_ind) {
            // MD:same for all coordinates, but I will still use this for now.
            this->mj_taskmapper_get_local_min_max_coord_totW(
              current_work_part,
              current_concurrent_num_parts,
              kk,
              Kokkos::subview(this->kokkos_mj_coordinates,
                Kokkos::ALL, coord_traverse_ind));

            coord_dimension_range_sorted[coord_traverse_ind].id =
              coord_traverse_ind;
            coord_dimension_range_sorted[coord_traverse_ind].signbit = 1;

            // TODO: Refactoring and optimizing general MJ leaves
            // us with some awkward issues here
            // for the Task Mapper code. For now just brute force the reads

            // This is the original code we are effecting here
            // coord_dim_mins[coord_traverse_ind] = best_min_coord;
            // coord_dim_maxs[coord_traverse_ind] = best_max_coord;

            Kokkos::parallel_reduce("Read single", 1,
              KOKKOS_LAMBDA(int dummy, mj_scalar_t & set_single) {
              set_single =
                local_kokkos_process_local_min_max_coord_total_weight(kk);
            }, coord_dim_mins[coord_traverse_ind]);

            Kokkos::parallel_reduce("Read single", 1,
              KOKKOS_LAMBDA(int dummy, mj_scalar_t & set_single) {
              set_single =
                local_kokkos_process_local_min_max_coord_total_weight(
                kk + current_concurrent_num_parts);
            }, coord_dim_maxs[coord_traverse_ind]);

            // Temporary - in refactor progress - will need to redo this
            // formatting: TODO
            Kokkos::parallel_reduce("Read single", 1,
              KOKKOS_LAMBDA(int dummy, mj_scalar_t & set_single) {
              set_single =
                local_kokkos_process_local_min_max_coord_total_weight(
                kk + current_concurrent_num_parts) - 
                local_kokkos_process_local_min_max_coord_total_weight(kk);
            }, coord_dimension_range_sorted[coord_traverse_ind].val);
          }

          uqSignsort(this->coord_dim, p_coord_dimension_range_sorted);
          coordInd = p_coord_dimension_range_sorted[this->coord_dim - 1].id;
          kokkos_mj_current_dim_coords =
            Kokkos::subview(this->kokkos_mj_coordinates, Kokkos::ALL, coordInd);
 
          // Note original code tracked unsorted weight but this should be set
          // already. So nothing to do ... but need to investigate if that was
          // intended. TODO:

          // TODO: This also was a relic of the refactor and we might consider
          // a different format
          // Related to above issues - need to clean this up
          auto set_min = coord_dim_mins[coordInd];
          auto set_max = coord_dim_maxs[coordInd];
          Kokkos::parallel_for(
            Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
              (0, 1), KOKKOS_LAMBDA (const int dummy) {
            local_kokkos_process_local_min_max_coord_total_weight(kk) = set_min;
            local_kokkos_process_local_min_max_coord_total_weight(
              kk+ current_concurrent_num_parts) = set_max;
          });
        }
        else {
          throw std::logic_error(
            "Disabled this mj_taskmapper_get_local_min_max_coord_totW B call"
            " for refactor. Need to implement device form.");

          /*
          this->mj_taskmapper_get_local_min_max_coord_totW(
            current_work_part,
            current_concurrent_num_parts,
            kk,
            kokkos_mj_current_dim_coords,
            this->kokkos_process_local_min_max_coord_total_weight(kk),
            this->kokkos_process_local_min_max_coord_total_weight(
              kk + current_concurrent_num_parts), // max coordinate
            this->kokkos_process_local_min_max_coord_total_weight(
              kk + 2*current_concurrent_num_parts) // total weight);
          );
          */
        }
      }

      // 1D partitioning
      if (actual_work_part_count > 0) {
        // obtain global Min max of the part.
        this->mj_get_global_min_max_coord_totW(
          current_concurrent_num_parts,
          this->kokkos_process_local_min_max_coord_total_weight,
          this->kokkos_global_min_max_coord_total_weight);

        // represents the total number of cutlines
        // whose coordinate should be determined.
        mj_part_t total_incomplete_cut_count = 0;

        //Compute weight ratios for parts & cuts:
        //e.g., 0.25  0.25  0.5    0.5  0.75 0.75  1
        //part0  cut0  part1 cut1 part2 cut2 part3
        mj_part_t concurrent_part_cut_shift = 0;
        mj_part_t concurrent_part_part_shift = 0;

        for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {
          mj_scalar_t min_coordinate =
            this->kokkos_global_min_max_coord_total_weight(kk);
          mj_scalar_t max_coordinate =
            this->kokkos_global_min_max_coord_total_weight(kk +
            current_concurrent_num_parts);
          mj_scalar_t global_total_weight =
            this->kokkos_global_min_max_coord_total_weight(kk +
            2 * current_concurrent_num_parts);
          mj_part_t concurrent_current_part_index = current_work_part + kk;

          mj_part_t partition_count;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
            set_single = view_num_partitioning_in_current_dim(
              concurrent_current_part_index);
          }, partition_count);

          Kokkos::View<mj_scalar_t *, device_t> kokkos_usedCutCoordinate =
            Kokkos::subview(kokkos_current_cut_coordinates,
              std::pair<mj_lno_t, mj_lno_t>(
                concurrent_part_cut_shift,
                kokkos_current_cut_coordinates.size()));
          Kokkos::View<mj_scalar_t *, device_t>
            kokkos_current_target_part_weights =
            Kokkos::subview(kokkos_target_part_weights,
              std::pair<mj_lno_t, mj_lno_t>(
                concurrent_part_part_shift,
                kokkos_target_part_weights.size()));

          // shift the usedCutCoordinate array as noCuts.
          concurrent_part_cut_shift += partition_count - 1;
          // shift the partRatio array as noParts.
          concurrent_part_part_shift += partition_count;
          // calculate only if part is not empty,
          // and part will be further partitioend.
          if(partition_count > 1 && min_coordinate <= max_coordinate) {
            // increase allDone by the number of cuts of the current
            // part's cut line number.
            total_incomplete_cut_count += partition_count - 1;
            // set the number of cut lines that should be determined
            // for this part.
            this->kokkos_my_incomplete_cut_count(kk) = partition_count - 1;
            // get the target weights of the parts.
            this->mj_get_initial_cut_coords_target_weights(
              min_coordinate,
              max_coordinate,
              partition_count - 1,
              global_total_weight,
              kokkos_usedCutCoordinate,
              kokkos_current_target_part_weights,
              future_num_part_in_parts,
              next_future_num_parts_in_parts,
              concurrent_current_part_index,
              obtained_part_index);

            mj_lno_t coordinate_end_index =
              this->kokkos_part_xadj(concurrent_current_part_index);
            mj_lno_t coordinate_begin_index =
              concurrent_current_part_index==0 ? 0 :
                this->kokkos_part_xadj[concurrent_current_part_index -1];

            // get the initial estimated part assignments of the coordinates.
            this->set_initial_coordinate_parts(
              max_coordinate,
              min_coordinate,
              concurrent_current_part_index,
              coordinate_begin_index, coordinate_end_index,
              this->kokkos_coordinate_permutations,
              kokkos_mj_current_dim_coords,
              this->kokkos_assigned_part_ids,
              partition_count);
          }
          else {
            // e.g., if have fewer coordinates than parts,
            // don't need to do next dim.
            this->kokkos_my_incomplete_cut_count(kk) = 0;
          }
          obtained_part_index += partition_count;
        }

        // used imbalance, it is always 0, as it is difficult
        // to estimate a range.
        mj_scalar_t used_imbalance = 0;

        // Determine cut lines for k parts here.
        this->mj_env->timerStart(MACRO_TIMERS, "mj_1D_part B()");
        this->mj_1D_part(
          kokkos_mj_current_dim_coords,
          used_imbalance,
          current_work_part,
          current_concurrent_num_parts,
          kokkos_current_cut_coordinates,
          total_incomplete_cut_count,
          view_num_partitioning_in_current_dim,
          view_rectilinear_cut_count,
          view_total_reduction_size);
            
        this->mj_env->timerStop(MACRO_TIMERS, "mj_1D_part B()");
      }
      else {
        obtained_part_index += current_concurrent_num_parts;
      }

      // create part chunks
      {
        mj_part_t output_array_shift = 0;
        mj_part_t cut_shift = 0;
        size_t tlr_shift = 0;
        size_t partweight_array_shift = 0;

        for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {
          mj_part_t current_concurrent_work_part = current_work_part + kk;
          
          mj_part_t num_parts;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
            set_single = view_num_partitioning_in_current_dim(
              current_concurrent_work_part);
          }, num_parts);
          
          // if the part is empty, skip the part.
          if((num_parts != 1  ) &&
            this->kokkos_global_min_max_coord_total_weight(kk) >
              this->kokkos_global_min_max_coord_total_weight(kk +
                current_concurrent_num_parts))
          {
            for(mj_part_t jj = 0; jj < num_parts; ++jj) {
              this->kokkos_new_part_xadj(
                output_part_index + output_array_shift + jj) = 0;
            }
            cut_shift += num_parts - 1;
            tlr_shift += (4 *(num_parts - 1) + 1);
            output_array_shift += num_parts;
            partweight_array_shift += (2 * (num_parts - 1) + 1);
            continue;
          }

          mj_lno_t coordinate_end =
            this->kokkos_part_xadj(current_concurrent_work_part);
          mj_lno_t coordinate_begin =
            current_concurrent_work_part==0 ? 0 :
              this->kokkos_part_xadj(current_concurrent_work_part-1);

          Kokkos::View<mj_scalar_t *, device_t>
            kokkos_current_concurrent_cut_coordinate =
            Kokkos::subview(kokkos_current_cut_coordinates,
              std::pair<mj_lno_t, mj_lno_t>(
                cut_shift,
                kokkos_current_cut_coordinates.size()));
          Kokkos::View<mj_scalar_t *, device_t>
            kokkos_used_local_cut_line_weight_to_left =
            Kokkos::subview(kokkos_process_cut_line_weight_to_put_left,
              std::pair<mj_lno_t, mj_lno_t>(
                cut_shift,
                kokkos_process_cut_line_weight_to_put_left.size()));

          this->kokkos_thread_part_weight_work =
            Kokkos::subview(
              this->kokkos_thread_part_weights,
              std::pair<mj_lno_t, mj_lno_t>(
                partweight_array_shift,
                this->kokkos_thread_part_weights.size()));

          if(num_parts > 1) {
            // Rewrite the indices based on the computed cuts.
            this->create_consistent_chunks(
              num_parts,
              kokkos_mj_current_dim_coords,
              kokkos_current_concurrent_cut_coordinate,
              coordinate_begin,
              coordinate_end,
              kokkos_used_local_cut_line_weight_to_left,
              Kokkos::subview(this->kokkos_new_part_xadj,
                std::pair<mj_lno_t, mj_lno_t>(
                  output_part_index + output_array_shift,
                  this->kokkos_new_part_xadj.size())),
              coordInd,
              partition_along_longest_dim,
              p_coord_dimension_range_sorted);
          }
          else {
            // if this part is partitioned into 1 then just copy
            // the old values.
            mj_lno_t part_size = coordinate_end - coordinate_begin;
            this->kokkos_new_part_xadj(
              output_part_index + output_array_shift) = part_size;

            // TODO optimize
            for(int n = 0; n < part_size; ++n) {
              this->kokkos_new_coordinate_permutations(n+coordinate_begin) =
                this->kokkos_coordinate_permutations(n+coordinate_begin);
            }
          }

          cut_shift += num_parts - 1;
          tlr_shift += (4 *(num_parts - 1) + 1);
          output_array_shift += num_parts;
          partweight_array_shift += (2 * (num_parts - 1) + 1);
        }

        // shift cut coordinates so that all cut coordinates are stored.
        // current_cut_coordinates += cutShift;

        // getChunks from coordinates partitioned the parts and
        // wrote the indices as if there were a single part.
        // now we need to shift the beginning indices.
        for(mj_part_t kk = 0; kk < current_concurrent_num_parts; ++kk) {
          mj_part_t num_parts;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
            set_single = view_num_partitioning_in_current_dim(
              current_work_part + kk);
          }, num_parts);
          
          for (mj_part_t ii = 0;ii < num_parts ; ++ii) {
            //shift it by previousCount
            this->kokkos_new_part_xadj(output_part_index+ii) +=
              output_coordinate_end_index;
            if (ii % 2 == 1) {
              mj_lno_t coordinate_end =
                this->kokkos_new_part_xadj(output_part_index+ii);
              mj_lno_t coordinate_begin =
                this->kokkos_new_part_xadj(output_part_index);

              for (mj_lno_t task_traverse = coordinate_begin;
                task_traverse < coordinate_end; ++task_traverse) {
                mj_lno_t l =
                  this->kokkos_new_coordinate_permutations(task_traverse);

                //MARKER: FLIPPED ZORDER BELOW
                kokkos_mj_current_dim_coords(l) =
                  -kokkos_mj_current_dim_coords(l);
              }
            }
          }
          // increase the previous count by current end.
          output_coordinate_end_index =
            this->kokkos_new_part_xadj(output_part_index + num_parts - 1);
          // increase the current out.
          output_part_index += num_parts;
        }
      }
    }
    // end of this partitioning dimension

    // set the current num parts for next dim partitioning
    current_num_parts = output_part_count_in_dimension;

    //swap the coordinate permutations for the next dimension.
    Kokkos::View<mj_lno_t *, device_t> tmp =
      this->kokkos_coordinate_permutations;
    this->kokkos_coordinate_permutations =
      this->kokkos_new_coordinate_permutations;
    this->kokkos_new_coordinate_permutations = tmp;
    this->kokkos_part_xadj = this->kokkos_new_part_xadj;
    this->kokkos_new_part_xadj = Kokkos::View<mj_lno_t*, device_t>("empty");
  }

  for(mj_lno_t i = 0; i < num_total_coords; ++i){
    kokkos_inital_adjList_output_adjlist(i) =
      this->kokkos_coordinate_permutations(i);
  }

  // Return output_xadj in CSR format
  output_xadj[0] = 0;
  for(size_t i = 0; i < this->num_global_parts ; ++i){
    output_xadj[i+1] = this->kokkos_part_xadj(i);
  }

  delete future_num_part_in_parts;
  delete next_future_num_parts_in_parts;
}

/*! \brief Function returns the part boxes stored
 * returns null if boxes are not stored, and prints warning mesage.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
RCP<typename AlgMJ
  <mj_scalar_t,mj_lno_t,mj_gno_t,mj_part_t,mj_node_t>::mj_partBox_t>
AlgMJ<mj_scalar_t,mj_lno_t,mj_gno_t,mj_part_t, mj_node_t>::
  get_global_box() const
{
  return this->global_box;
}

/*! \brief Function call, if the part boxes are intended to be kept.
 *
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t,
  mj_node_t>::set_to_keep_part_boxes()
{
  this->mj_keep_part_boxes = true;
}

/* \brief Either the mj array (part_no_array) or num_global_parts should be
 * provided in the input. part_no_array takes
 * precedence if both are provided.
 * Depending on these parameters, total cut/part number,
 * maximum part/cut number along a dimension, estimated number of reduceAlls,
 * and the number of parts before the last dimension is calculated.
 * */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  set_part_specifications()
{
  this->total_num_cut = 0; //how many cuts will be totally
  this->total_num_part = 1;    //how many parts will be totally
  this->max_num_part_along_dim = 0; // maximum part count along a dimension.
  this->total_dim_num_reduce_all = 0; // estimate on #reduceAlls can be done.
  this->last_dim_num_part = 1; //max no of parts that might occur
  //during the partition before the
  //last partitioning dimension.
  this->max_num_cut_along_dim = 0;
  this->max_num_total_part_along_dim = 0;

  // TODO - is size() going to work as NULL did?
  if (this->kokkos_part_no_array.size())
  {
    // original code as follows - first line is a simple multiply
    // 2nd line needs kokkos loop 
    // for (int i = 0; i < this->recursion_depth; ++i) {
    //        this->total_dim_num_reduce_all += this->total_num_part;
    //        this->total_num_part *= this->kokkos_part_no_array(i);
    //}

    auto local_kokkos_part_no_array = this->kokkos_part_no_array;
    auto local_recursion_depth = this->recursion_depth; 

    this->total_dim_num_reduce_all =
      this->total_num_part * this->recursion_depth;
    Kokkos::parallel_reduce("Single Reduce", 1,
      KOKKOS_LAMBDA(const int& dummy, mj_part_t & running) {
      running = 1.0;
      for (int i = 0; i < local_recursion_depth; ++i) {
        running *= local_kokkos_part_no_array(i);
      }
    }, this->total_num_part);

    mj_part_t track_max;
    Kokkos::parallel_reduce("MaxReduce", local_recursion_depth,
      KOKKOS_LAMBDA(const int& i, mj_part_t & running_max) {
        if(local_kokkos_part_no_array(i) > running_max) {
          running_max = local_kokkos_part_no_array(i);
        }	
    }, Kokkos::Max<mj_part_t>(track_max));

    auto local_total_num_part = this->total_num_part;
    Kokkos::parallel_reduce("Single Reduce", 1,
      KOKKOS_LAMBDA(const int& dummy, mj_part_t & running) {
      running = local_total_num_part /
        local_kokkos_part_no_array(local_recursion_depth-1);
    }, this->last_dim_num_part);
          
    this->max_num_part_along_dim = track_max;
    this->num_global_parts = this->total_num_part;
  } else {
    mj_part_t future_num_parts = this->num_global_parts;

    // we need to calculate the part numbers now, to determine
    // the maximum along the dimensions.
    for (int i = 0; i < this->recursion_depth; ++i){
      mj_part_t maxNoPartAlongI = this->get_part_count(
        future_num_parts, 1.0f / (this->recursion_depth - i));

      if (maxNoPartAlongI > this->max_num_part_along_dim){
        this->max_num_part_along_dim = maxNoPartAlongI;
      }

      mj_part_t nfutureNumParts = future_num_parts / maxNoPartAlongI;
      if (future_num_parts % maxNoPartAlongI){
        ++nfutureNumParts;
      }
      future_num_parts = nfutureNumParts;
    }
    this->total_num_part = this->num_global_parts;

    if (this->divide_to_prime_first){
      this->total_dim_num_reduce_all = this->num_global_parts * 2;
      this->last_dim_num_part = this->num_global_parts;
    }
    else {
      //this is the lower bound.
      //estimate reduceAll Count here.
      //we find the upperbound instead.
      size_t p = 1;

      for (int i = 0; i < this->recursion_depth; ++i){
        this->total_dim_num_reduce_all += p;
        p *= this->max_num_part_along_dim;
      }

      if (p / this->max_num_part_along_dim > this->num_global_parts){
        this->last_dim_num_part = this->num_global_parts;
      }
      else {
        this->last_dim_num_part  = p / this->max_num_part_along_dim;
      }
    }
  }

  this->total_num_cut = this->total_num_part - 1;
  this->max_num_cut_along_dim = this->max_num_part_along_dim - 1;
  this->max_num_total_part_along_dim = this->max_num_part_along_dim +
    size_t(this->max_num_cut_along_dim);
  // maxPartNo is P, maxCutNo = P-1, matTotalPartcount = 2P-1

  // refine the concurrent part count, if it is given bigger than the maximum
  // possible part count.
  if(this->max_concurrent_part_calculation > this->last_dim_num_part){
    if(this->mj_problemComm->getRank() == 0){
      std::cerr << "Warning: Concurrent part count (" <<
        this->max_concurrent_part_calculation <<
        ") has been set bigger than maximum amount that can be used." <<
        " Setting to:" << this->last_dim_num_part << "." << std::endl;
    }
    this->max_concurrent_part_calculation = this->last_dim_num_part;
  }
}

/* \brief Tries to determine the part number for current dimension,
 * by trying to make the partitioning as square as possible.
 * \param num_total_future how many more partitionings are required.
 * \param root how many more recursion depth is left.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
inline mj_part_t AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  get_part_count(mj_part_t num_total_future, double root)
{
  double fp = pow(num_total_future, root);
  mj_part_t ip = mj_part_t (fp);
  if (fp - ip < std::numeric_limits<float>::epsilon() * 100) {
    return ip;
  }
  else {
    return ip  + 1;
  }
}

/* \brief Function returns how many parts that will be obtained after this
 * dimension partitioning. It sets how many parts each current part will be
 * partitioned into in this dimension to view_num_partitioning_in_current_dim
 * vector, sets how many total future parts each obtained part will be
 * partitioned into in next_future_num_parts_in_parts vector. If part boxes are
 * kept, then sets initializes the output_part_boxes as its ancestor.
 * \param view_num_partitioning_in_current_dim: output. How many parts each
 * current part will be partitioned into.
 * \param future_num_part_in_parts: input, how many future parts each current
 * part will be partitioned into.
 * \param next_future_num_parts_in_parts: output, how many future parts each
 * obtained part will be partitioned into.
 * \param future_num_parts: output, max number of future parts that will be
 * obtained from a single
 * \param current_num_parts: input, how many parts are there currently.
 * \param current_iteration: input, current dimension iteration number.
 * \param input_part_boxes: input, if boxes are kept, current boxes.
 * \param output_part_boxes: output, if boxes are kept, the initial box
 * boundaries for obtained parts.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
mj_part_t AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  update_part_num_arrays(
  Kokkos::View<mj_part_t*, device_t> & view_num_partitioning_in_current_dim,
  std::vector<mj_part_t> *future_num_part_in_parts,
  std::vector<mj_part_t> *next_future_num_parts_in_parts,
  mj_part_t &future_num_parts,
  mj_part_t current_num_parts,
  int current_iteration,
  RCP<mj_partBoxVector_t> input_part_boxes,
  RCP<mj_partBoxVector_t> output_part_boxes,
  mj_part_t atomic_part_count)
{
  // Working on view_num_partitioning_in_current_dim in stages which was
  // originally a std::vector but converting to kokkos view.
  // So here let's pull it back to a std::vector, build the new form,
  // then convert back to a view. Later we can probably handle all this stuff
  // on device but idea is to move the remaining refactor to a more localized
  // area (here) instead of all over.
  
  // This got ugly quickly .... but it's at least partly temporary as I
  // refactor
  typename std::remove_reference<decltype(
    view_num_partitioning_in_current_dim)>::type::HostMirror
    hostArray
    = Kokkos::create_mirror_view(view_num_partitioning_in_current_dim);
  Kokkos::deep_copy(hostArray, view_num_partitioning_in_current_dim);
  std::vector<mj_part_t> vector_num_partitioning_in_current_dim(
    view_num_partitioning_in_current_dim.size());
  for(size_t n = 0; n < view_num_partitioning_in_current_dim.size(); ++n) {
    vector_num_partitioning_in_current_dim[n] = hostArray(n);
  }
  
  // how many parts that will be obtained after this dimension.
  mj_part_t output_num_parts = 0;
  if(this->kokkos_part_no_array.size()) {
    // when the partNo array is provided as input,
    // each current partition will be partition to the same number of parts.
    // we dont need to use the future_num_part_in_parts vector in this case.

    // TODO - what is the right way to read a single value to host
    auto local_kokkos_part_no_array = this->kokkos_part_no_array;
    mj_part_t p;
    Kokkos::parallel_reduce("Read single", 1,
      KOKKOS_LAMBDA(int i, mj_part_t & set_single) {
        set_single = local_kokkos_part_no_array(current_iteration);
    }, p);

    if (p < 1){
      std::cout << "i:" << current_iteration <<
        " p is given as:" << p << std::endl;
      exit(1);
    }
    if (p == 1){
      return current_num_parts;
    }
    for (mj_part_t ii = 0; ii < current_num_parts; ++ii){
      vector_num_partitioning_in_current_dim.push_back(p);
    }
    future_num_parts /= vector_num_partitioning_in_current_dim[0];
    output_num_parts = current_num_parts *
      vector_num_partitioning_in_current_dim[0];
    if (this->mj_keep_part_boxes){
      for (mj_part_t k = 0; k < current_num_parts; ++k){
        //initialized the output boxes as its ancestor.
        for (mj_part_t j = 0; j <
          vector_num_partitioning_in_current_dim[0]; ++j){
          output_part_boxes->push_back((*input_part_boxes)[k]);
        }
      }
    }

    // set the how many more parts each part will be divided.
    // this is obvious when partNo array is provided as input.
    // however, fill this so weights will be calculated according to this array.
    for (mj_part_t ii = 0; ii < output_num_parts; ++ii){
      next_future_num_parts_in_parts->push_back(future_num_parts);
    }
  }
  else {
    // if partNo array is not provided as input, future_num_part_in_parts 
    // holds how many parts each part should be divided. Initially it holds a
    // single number equal to the total number of global parts.

    // calculate the future_num_parts from beginning,
    // since each part might be divided into different number of parts.
    future_num_parts = 1;

    // cout << "i:" << i << std::endl;
    for (mj_part_t ii = 0; ii < current_num_parts; ++ii) {
      // get how many parts a part should be divided.
      mj_part_t future_num_parts_of_part_ii = (*future_num_part_in_parts)[ii];

      // get the ideal number of parts that is close to the
      // (recursion_depth - i) root of the future_num_parts_of_part_ii.
      mj_part_t num_partitions_in_current_dim =
        this->get_part_count(future_num_parts_of_part_ii,
          1.0 / (this->recursion_depth - current_iteration)
                                    );
      if (num_partitions_in_current_dim > this->max_num_part_along_dim){
        std::cerr << "ERROR: maxPartNo calculation is wrong."
          " num_partitions_in_current_dim: "
          << num_partitions_in_current_dim <<  "this->max_num_part_along_dim:"
          << this->max_num_part_along_dim <<
          " this->recursion_depth:" << this->recursion_depth <<
          " current_iteration:" << current_iteration <<
          " future_num_parts_of_part_ii:" << future_num_parts_of_part_ii <<
          " might need to fix max part no calculation for "
          "largest_prime_first partitioning" <<
          std::endl;
        exit(1);
      }
      // add this number to vector_num_partitioning_in_current_dim vector.
      vector_num_partitioning_in_current_dim.push_back(
        num_partitions_in_current_dim);
      mj_part_t largest_prime_factor = num_partitions_in_current_dim;
      if (this->divide_to_prime_first){
        //increase the output number of parts.
        output_num_parts += num_partitions_in_current_dim;
        if (future_num_parts_of_part_ii == atomic_part_count ||
          future_num_parts_of_part_ii % atomic_part_count != 0) {
          atomic_part_count = 1;
        }
        largest_prime_factor = this->find_largest_prime_factor(
          future_num_parts_of_part_ii / atomic_part_count);

        // we divide to  num_partitions_in_current_dim. But we adjust the
        // weights based on largest prime/
        // if num_partitions_in_current_dim = 2, largest prime = 5 --> we
        // divide to 2 parts with weights 3x and 2x.
        // if the largest prime is less than part count, we use the part count
        // so that we divide uniformly.
        if (largest_prime_factor < num_partitions_in_current_dim){
          largest_prime_factor = num_partitions_in_current_dim;
        }
        //ideal number of future partitions for each part.
        mj_part_t ideal_num_future_parts_in_part = (
          future_num_parts_of_part_ii / atomic_part_count) /
          largest_prime_factor;
        // if num_partitions_in_current_dim = 2, largest prime = 5 then ideal
        // weight is 2x
        mj_part_t ideal_prime_scale = largest_prime_factor /
          num_partitions_in_current_dim;

        // std::cout << "current num part:" << ii << " largest_prime_factor:"
        // << largest_prime_factor << " To Partition:" <<
        // future_num_parts_of_part_ii << " ";
        for (mj_part_t iii = 0; iii < num_partitions_in_current_dim; ++iii) {
          // if num_partitions_in_current_dim = 2,
          // largest prime = 5 then ideal weight is 2x
          mj_part_t my_ideal_primescale = ideal_prime_scale;
          // left over weighs. Left side is adjusted to be 3x,
          // right side stays as 2x
          if (iii < (largest_prime_factor) % num_partitions_in_current_dim) {
            ++my_ideal_primescale;
          }
          //scale with 'x';
          mj_part_t num_future_parts_for_part_iii =
            ideal_num_future_parts_in_part * my_ideal_primescale;
          //if there is a remainder in the part increase the part weight.
          if (iii < (future_num_parts_of_part_ii /
            atomic_part_count) % largest_prime_factor) {
            //if not uniform, add 1 for the extra parts.
            ++num_future_parts_for_part_iii;
          }

          next_future_num_parts_in_parts->push_back(
            num_future_parts_for_part_iii * atomic_part_count);

          // if part boxes are stored, initialize the box of the parts as
          // the ancestor.
          if (this->mj_keep_part_boxes) {
            output_part_boxes->push_back((*input_part_boxes)[ii]);
          }
          //set num future_num_parts to maximum in this part.
          if (num_future_parts_for_part_iii > future_num_parts) {
            future_num_parts = num_future_parts_for_part_iii;
          }
        }
      }
      else {
        //increase the output number of parts.
        output_num_parts += num_partitions_in_current_dim;

        if (future_num_parts_of_part_ii == atomic_part_count ||
          future_num_parts_of_part_ii % atomic_part_count != 0) {
          atomic_part_count = 1;
        }
        //ideal number of future partitions for each part.
        mj_part_t ideal_num_future_parts_in_part =
          (future_num_parts_of_part_ii / atomic_part_count) /
          num_partitions_in_current_dim;
        for (mj_part_t iii = 0; iii < num_partitions_in_current_dim; ++iii) {
          mj_part_t num_future_parts_for_part_iii =
            ideal_num_future_parts_in_part;

          //if there is a remainder in the part increase the part weight.
          if (iii < (future_num_parts_of_part_ii / atomic_part_count) %
            num_partitions_in_current_dim) {
            // if not uniform, add 1 for the extra parts.
            ++num_future_parts_for_part_iii;
          }

          next_future_num_parts_in_parts->push_back(
            num_future_parts_for_part_iii * atomic_part_count);

          // if part boxes are stored, initialize the box of the parts as
          // the ancestor.
          if (this->mj_keep_part_boxes){
            output_part_boxes->push_back((*input_part_boxes)[ii]);
          }
          //set num future_num_parts to maximum in this part.
          if (num_future_parts_for_part_iii > future_num_parts)
            future_num_parts = num_future_parts_for_part_iii;
        }
      }
    }
  }

  // Now revert the vector form back to a view
  // TODO: We'd like to avoid above conversion and just do all this on view
  Kokkos::View<mj_part_t*> temp = Kokkos::View<mj_part_t*>(
    "view_num_partitioning_in_current_dim",
    vector_num_partitioning_in_current_dim.size());
  typename decltype(temp)::HostMirror host_view_num_partitioning_in_current_dim
    = Kokkos::create_mirror_view(temp);
  for(int n = 0;
    n < static_cast<int>(vector_num_partitioning_in_current_dim.size()); ++n) {
    host_view_num_partitioning_in_current_dim(n) =
      vector_num_partitioning_in_current_dim[n];
  }
  Kokkos::deep_copy(temp, host_view_num_partitioning_in_current_dim);
  view_num_partitioning_in_current_dim = temp;

  return output_num_parts;
}

/* \brief Allocates and initializes the work memory that will be used by MJ.
 *
 * */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  allocate_set_work_memory()
{
  // points to process that initially owns the coordinate.
  Kokkos::resize(this->kokkos_owner_of_coordinate, 0);

  // Throughout the partitioning execution,
  // instead of the moving the coordinates, hold a permutation array for parts.
  // coordinate_permutations holds the current permutation.
  Kokkos::resize(this->kokkos_coordinate_permutations, this->num_local_coords);

  Kokkos::View<mj_lno_t*, device_t> temp = Kokkos::View<mj_lno_t*, device_t>(
    "kokkos_coordinate_permutations", num_local_coords);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (
    0, this->num_local_coords), KOKKOS_LAMBDA (const int i) {
      temp(i) = i;
  });

  // bring the local data back to the class
  this->kokkos_coordinate_permutations = temp;

  // new_coordinate_permutations holds the current permutation.
  this->kokkos_new_coordinate_permutations = Kokkos::View<mj_lno_t*, device_t>(
    "num_local_coords", this->num_local_coords);
  this->kokkos_assigned_part_ids = Kokkos::View<mj_part_t*, device_t>(
    "assigned parts"); // TODO empty is ok for NULL replacement?
  if(this->num_local_coords > 0){
    this->kokkos_assigned_part_ids = Kokkos::View<mj_part_t*, device_t>(
      "assigned part ids", this->num_local_coords);
  }
  // single partition starts at index-0, and ends at numLocalCoords
  // inTotalCounts array holds the end points in coordinate_permutations array
  // for each partition. Initially sized 1, and single element is set to
  // numLocalCoords.
  this->kokkos_part_xadj = Kokkos::View<mj_lno_t*, device_t>("part xadj", 1);

  // TODO: How do do the above operation on device
  auto local_num_local_coords = this->num_local_coords;
  auto local_kokkos_part_xadj = this->kokkos_part_xadj;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (0, 1),
    KOKKOS_LAMBDA (const int i) {
      // the end of the initial partition is the end of coordinates.
      local_kokkos_part_xadj(0) = static_cast<mj_lno_t>(local_num_local_coords);
  });

  // the ends points of the output, this is allocated later.
  this->kokkos_new_part_xadj = Kokkos::View<mj_lno_t*, device_t>("empty");

  // only store this much if cuts are needed to be stored.
  // this->all_cut_coordinates = allocMemory< mj_scalar_t>(this->total_num_cut);
  this->kokkos_all_cut_coordinates = Kokkos::View<mj_scalar_t*, device_t>(
    "all cut coordinates",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);
    
  // how much weight percentage should a MPI put left side of the each cutline
  this->kokkos_process_cut_line_weight_to_put_left = Kokkos::View<mj_scalar_t*,
    device_t>("empty");
    
  // how much weight percentage should each thread in MPI put left side of
  // each outline
  this->kokkos_thread_cut_line_weight_to_put_left =
    Kokkos::View<mj_scalar_t*, Kokkos::LayoutLeft, device_t>("empty");
    
  // distribute_points_on_cut_lines = false;
  if(this->distribute_points_on_cut_lines){
    this->kokkos_process_cut_line_weight_to_put_left =
      Kokkos::View<mj_scalar_t *, device_t>(
      "kokkos_process_cut_line_weight_to_put_left",
        this->max_num_cut_along_dim * this->max_concurrent_part_calculation);
    this->kokkos_thread_cut_line_weight_to_put_left =
      Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>(
      "kokkos_thread_cut_line_weight_to_put_left", this->max_num_cut_along_dim);
    this->kokkos_process_rectilinear_cut_weight =
      Kokkos::View<mj_scalar_t *, device_t>(
      "kokkos_process_rectilinear_cut_weight", this->max_num_cut_along_dim);
    this->kokkos_global_rectilinear_cut_weight =
      Kokkos::View<mj_scalar_t *, device_t>(
      "kokkos_global_rectilinear_cut_weight", this->max_num_cut_along_dim);
  }

  // work array to manipulate coordinate of cutlines in different iterations.
  // necessary because previous cut line information is used for determining
  // the next cutline information. therefore, cannot update the cut work array
  // until all cutlines are determined.
  this->kokkos_cut_coordinates_work_array =
    Kokkos::View<mj_scalar_t *, device_t>("kokkos_cut_coordinates_work_array",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // cumulative part weight array.
  this->kokkos_target_part_weights = Kokkos::View<mj_scalar_t*, device_t>(
    "kokkos_target_part_weights",
    this->max_num_part_along_dim * this->max_concurrent_part_calculation);
  
  // upper bound coordinate of a cut line
  this->kokkos_cut_upper_bound_coordinates =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_upper_bound_coordinates",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);
    
  // lower bound coordinate of a cut line  
  this->kokkos_cut_lower_bound_coordinates =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_lower_bound_coordinates",
    this->max_num_cut_along_dim* this->max_concurrent_part_calculation);

  // lower bound weight of a cut line
  this->kokkos_cut_lower_bound_weights =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_lower_bound_weights",
    this->max_num_cut_along_dim* this->max_concurrent_part_calculation);

  //upper bound weight of a cut line
  this->kokkos_cut_upper_bound_weights =
    Kokkos::View<mj_scalar_t*, device_t>("kokkos_cut_upper_bound_weights",
    this->max_num_cut_along_dim* this->max_concurrent_part_calculation);

  // combined array to exchange the min and max coordinate,
  // and total weight of part.
  this->kokkos_process_local_min_max_coord_total_weight =
    Kokkos::View<mj_scalar_t*, device_t>(
    "kokkos_process_local_min_max_coord_total_weight",
    3 * this->max_concurrent_part_calculation);

  // global combined array with the results for min, max and total weight.
  this->kokkos_global_min_max_coord_total_weight =
    Kokkos::View<mj_scalar_t*, device_t>(
    "kokkos_global_min_max_coord_total_weight",
    3 * this->max_concurrent_part_calculation);

  // is_cut_line_determined is used to determine if a cutline is
  // determined already. If a cut line is already determined, the next
  // iterations will skip this cut line.
  this->kokkos_is_cut_line_determined = Kokkos::View<bool *, device_t>(
    "kokkos_is_cut_line_determined",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // my_incomplete_cut_count count holds the number of cutlines that have not
  // been finalized for each part when concurrentPartCount>1, using this
  // information, if my_incomplete_cut_count[x]==0, then no work is done for
  // this part.
  this->kokkos_my_incomplete_cut_count =  Kokkos::View<mj_part_t *, device_t>(
    "kokkos_my_incomplete_cut_count", this->max_concurrent_part_calculation);

  // local part weights of each thread.
  this->kokkos_thread_part_weights = Kokkos::View<double *,
    Kokkos::LayoutLeft, device_t>("thread_part_weights",
    this->max_num_total_part_along_dim * this->max_concurrent_part_calculation);

  this->kokkos_thread_cut_left_closest_point = Kokkos::View<mj_scalar_t *,
    Kokkos::LayoutLeft, device_t>("kokkos_thread_cut_left_closest_point",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // thread_cut_right_closest_point to hold the closest coordinate to a
  // cutline from right (for each thread)
  this->kokkos_thread_cut_right_closest_point = Kokkos::View<mj_scalar_t *,
    Kokkos::LayoutLeft, device_t>("kokkos_thread_cut_right_closest_point",
    this->max_num_cut_along_dim * this->max_concurrent_part_calculation);

  // to store how many points in each part a thread has.
  this->kokkos_thread_point_counts = Kokkos::View<mj_lno_t *,
    Kokkos::LayoutLeft, device_t>("kokkos_thread_point_counts",
    this->max_num_part_along_dim);

  // for faster communication, concatanation of
  // totalPartWeights sized 2P-1, since there are P parts and P-1 cut lines
  // leftClosest distances sized P-1, since P-1 cut lines
  // rightClosest distances size P-1, since P-1 cut lines.
  this->kokkos_total_part_weight_left_right_closests =
    Kokkos::View<mj_scalar_t*, device_t>(
      "total_part_weight_left_right_closests",
      (this->max_num_total_part_along_dim + this->max_num_cut_along_dim * 2) *
      this->max_concurrent_part_calculation);

  this->kokkos_global_total_part_weight_left_right_closests =
    Kokkos::View<mj_scalar_t*, device_t>(
      "global_total_part_weight_left_right_closests",
      (this->max_num_total_part_along_dim +
      this->max_num_cut_along_dim * 2) * this->max_concurrent_part_calculation);

  Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t> coord(
    "coord", this->num_local_coords, this->coord_dim);

  auto local_kokkos_mj_coordinates = kokkos_mj_coordinates; 
  auto local_coord_dim = this->coord_dim;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (
      0, local_num_local_coords),
    KOKKOS_LAMBDA (const int j) {
    for (int i=0; i < local_coord_dim; i++){
      coord(j,i) = local_kokkos_mj_coordinates(j,i);
  }});
  this->kokkos_mj_coordinates = coord;

  Kokkos::View<mj_scalar_t**, device_t> weights(
  "weights", this->num_local_coords, this->num_weights_per_coord);

  auto local_kokkos_mj_weights = kokkos_mj_weights;
  auto local_num_weights_per_coord = this->num_weights_per_coord;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (
      0, local_num_local_coords),
    KOKKOS_LAMBDA (const int j) {
    for (int i=0; i < local_num_weights_per_coord; i++){
      weights(j,i) = local_kokkos_mj_weights(j,i);
  }});

  this->kokkos_mj_weights = weights;

  this->kokkos_current_mj_gnos =
  Kokkos::View<mj_gno_t*, device_t>("gids", local_num_local_coords);
  auto local_kokkos_current_mj_gnos = this->kokkos_current_mj_gnos;
  auto local_kokkos_initial_mj_gnos = this->kokkos_initial_mj_gnos;

  this->kokkos_owner_of_coordinate = Kokkos::View<int*, device_t>(
    "kokkos_owner_of_coordinate", this->num_local_coords);

  auto local_kokkos_owner_of_coordinate = this->kokkos_owner_of_coordinate;
  auto local_myActualRank = this->myActualRank;

  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (
      0, local_num_local_coords),
    KOKKOS_LAMBDA (const int j) {
    local_kokkos_current_mj_gnos(j) = local_kokkos_initial_mj_gnos(j);
    local_kokkos_owner_of_coordinate(j) = local_myActualRank;
  });
}

/* \brief compute the global bounding box
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t,mj_lno_t,mj_gno_t,mj_part_t,
          mj_node_t>::compute_global_box()
{
    //local min coords
    mj_scalar_t *mins = allocMemory<mj_scalar_t>(this->coord_dim);
    //global min coords
    mj_scalar_t *gmins = allocMemory<mj_scalar_t>(this->coord_dim);
    //local max coords
    mj_scalar_t *maxs = allocMemory<mj_scalar_t>(this->coord_dim);
    //global max coords
    mj_scalar_t *gmaxs = allocMemory<mj_scalar_t>(this->coord_dim);

    auto local_kokkos_mj_coordinates = this->kokkos_mj_coordinates;
    for (int i = 0; i < this->coord_dim; ++i){
      Kokkos::parallel_reduce("MinReduce", this->num_local_coords,
        KOKKOS_LAMBDA(const mj_lno_t & j, mj_scalar_t & running_min) {
        if(local_kokkos_mj_coordinates(j,i) < running_min) {
          running_min = local_kokkos_mj_coordinates(j,i);
        }
      }, Kokkos::Min<mj_scalar_t>(mins[i]));
      Kokkos::parallel_reduce("MaxReduce", this->num_local_coords,
        KOKKOS_LAMBDA(const mj_lno_t & j, mj_scalar_t & running_max) {
        if(local_kokkos_mj_coordinates(j,i) > running_max) {
          running_max = local_kokkos_mj_coordinates(j,i);
        }
      }, Kokkos::Max<mj_scalar_t>(maxs[i]));
    }

    reduceAll<int, mj_scalar_t>(*this->comm, Teuchos::REDUCE_MIN,
            this->coord_dim, mins, gmins
    );

    reduceAll<int, mj_scalar_t>(*this->comm, Teuchos::REDUCE_MAX,
            this->coord_dim, maxs, gmaxs
    );

    //create single box with all areas.
    global_box = rcp(new mj_partBox_t(0,this->coord_dim,gmins,gmaxs));
    //coordinateModelPartBox <mj_scalar_t, mj_part_t> tmpBox (0, coordDim);
    freeArray<mj_scalar_t>(mins);
    freeArray<mj_scalar_t>(gmins);
    freeArray<mj_scalar_t>(maxs);
    freeArray<mj_scalar_t>(gmaxs);
}

/* \brief for part communication we keep track of the box boundaries.
 * This is performed when either asked specifically, or when geometric mapping
 * is performed afterwards.
 * This function initializes a single box with all global min, max coordinates.
 * \param initial_partitioning_boxes the input and output vector for boxes.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
          typename mj_part_t,
          typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t,
          mj_node_t>::init_part_boxes(
                RCP<mj_partBoxVector_t> & initial_partitioning_boxes
)
{
    mj_partBox_t tmp_box(*global_box);
    initial_partitioning_boxes->push_back(tmp_box);
}

/*! \brief Function to determine the local minimum and maximum coordinate, and
 * local total weight in the given set of local points.
 *  TODO: Repair parameters
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
          typename mj_part_t,
          typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_get_local_min_max_coord_totW(
  mj_part_t current_work_part,
  mj_part_t current_concurrent_num_parts,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords)
{
  auto local_kokkos_part_xadj = this->kokkos_part_xadj;
  auto local_kokkos_coordinate_permutations =
    this->kokkos_coordinate_permutations;
  auto local_kokkos_process_local_min_max_coord_total_weight =
    this->kokkos_process_local_min_max_coord_total_weight;
  auto local_kokkos_mj_weights = this->kokkos_mj_weights;
  auto local_kokkos_mj_uniform_weights = this->kokkos_mj_uniform_weights;

  // pull kokkos_part_xadj to host
  // TODO: Design
  typename decltype(kokkos_part_xadj)::HostMirror
    host_kokkos_part_xadj =
    Kokkos::create_mirror_view(kokkos_part_xadj);
  Kokkos::deep_copy(host_kokkos_part_xadj, kokkos_part_xadj);

  // pull kokkos_mj_uniform_weights to host
  // TODO: Design
  typename decltype(kokkos_mj_uniform_weights)::HostMirror
    host_kokkos_mj_uniform_weights =
    Kokkos::create_mirror_view(kokkos_mj_uniform_weights);
  Kokkos::deep_copy(host_kokkos_mj_uniform_weights,
    kokkos_mj_uniform_weights);
  bool bUniformWeights = host_kokkos_mj_uniform_weights(0) ? true : false;
  
  for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {

    mj_part_t concurrent_current_part = current_work_part + kk;

    mj_lno_t coordinate_begin_index = concurrent_current_part == 0 ? 0 :
      host_kokkos_part_xadj(concurrent_current_part -1);
    mj_lno_t coordinate_end_index =
      host_kokkos_part_xadj(concurrent_current_part);
    // total points to be processed by all teams
    mj_lno_t num_working_points =
      coordinate_end_index - coordinate_begin_index;

    // determine a stride for each team
    // TODO: How to best determine this and should we be concerned if teams
    // get smaller coord counts than their warp sizes?
    const int min_coords_per_team = 32; // abrbitrary ... TODO
    int stride = min_coords_per_team;
    if(stride > num_working_points) {
      stride = num_working_points;
    }

    int num_teams = num_working_points / stride;
    if((num_working_points % stride) > 0) {
      // guarantees no team has no work and the last team has equal or
      // less work than all the others
      ++num_teams; 
    }

    const int max_teams = SET_MAX_TEAMS;
    if(num_teams > max_teams) {
      num_teams = max_teams;
      stride = num_working_points / num_teams;
      if((num_working_points % num_teams) > 0) {
        stride += 1; // make sure we have coverage for the final points
      }
    }

    mj_scalar_t my_thread_min_coord = 0;
    mj_scalar_t my_thread_max_coord = 0;
    mj_scalar_t my_total_weight;

    //if the part is empty.
    //set the min and max coordinates as reverse.
    if(coordinate_begin_index >= coordinate_end_index)
    {
      my_thread_min_coord = std::numeric_limits<mj_scalar_t>::max();
      my_thread_max_coord = -std::numeric_limits<mj_scalar_t>::max();
      my_total_weight = 0;
    }
    else {
      // get min
      Kokkos::parallel_reduce("get min",
        coordinate_end_index - coordinate_begin_index,
        KOKKOS_LAMBDA (const mj_lno_t & j, mj_scalar_t & running_min) {
        int i =
          local_kokkos_coordinate_permutations(j + coordinate_begin_index);
        if(kokkos_mj_current_dim_coords(i) < running_min)
          running_min = kokkos_mj_current_dim_coords(i);
      }, Kokkos::Min<mj_scalar_t>(my_thread_min_coord));
      // get max
      Kokkos::parallel_reduce("get max",
        coordinate_end_index - coordinate_begin_index,
        KOKKOS_LAMBDA (const mj_lno_t & j, mj_scalar_t & running_max) {
        int i =
          local_kokkos_coordinate_permutations(j + coordinate_begin_index);
        if(kokkos_mj_current_dim_coords(i) > running_max)
          running_max = kokkos_mj_current_dim_coords(i);
      }, Kokkos::Max<mj_scalar_t>(my_thread_max_coord));

      if(bUniformWeights) {
        my_total_weight = coordinate_end_index - coordinate_begin_index;
      }
      else {
        my_total_weight = 0;
        Kokkos::parallel_reduce("get weight",
          coordinate_end_index - coordinate_begin_index,
          KOKKOS_LAMBDA (const mj_lno_t & ii, mj_scalar_t & lsum) {
          int i =
            local_kokkos_coordinate_permutations(ii + coordinate_begin_index);
          lsum += local_kokkos_mj_weights(i,0);
        }, my_total_weight);
      }
    }
  
    // single write
    Kokkos::TeamPolicy<typename mj_node_t::execution_space> policy_single(1, 1);
    typedef typename Kokkos::TeamPolicy<typename mj_node_t::execution_space>::
      member_type member_type;
    Kokkos::parallel_for (policy_single, KOKKOS_LAMBDA(member_type team_member)
    {
      local_kokkos_process_local_min_max_coord_total_weight(kk) =
        my_thread_min_coord;
      local_kokkos_process_local_min_max_coord_total_weight(
        kk + current_concurrent_num_parts) = my_thread_max_coord;
      local_kokkos_process_local_min_max_coord_total_weight(
        kk + 2*current_concurrent_num_parts) = my_total_weight;
    });
  }
}

/*! \brief Function to determine the local minimum and maximum coordinate, and
 * local total weight in the given set of local points.
 *  TODO: Repair parameters
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_taskmapper_get_local_min_max_coord_totW(
  mj_part_t current_work_part,
  mj_part_t current_concurrent_num_parts,
  int kk,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords)
{
  mj_part_t current_work_part_in_concurrent_parts = current_work_part + kk;
  auto local_kokkos_part_xadj = this->kokkos_part_xadj;
  auto local_kokkos_coordinate_permutations =
    this->kokkos_coordinate_permutations;
  auto local_kokkos_process_local_min_max_coord_total_weight =
    this->kokkos_process_local_min_max_coord_total_weight;
  auto local_kokkos_mj_weights = this->kokkos_mj_weights;
  auto local_kokkos_mj_uniform_weights = this->kokkos_mj_uniform_weights;

  mj_scalar_t max_scalar = std::numeric_limits<mj_scalar_t>::max();

  Kokkos::TeamPolicy<typename mj_node_t::execution_space> policy1 (1, 1);
  typedef typename Kokkos::TeamPolicy<typename mj_node_t::execution_space>::
    member_type member_type;
  Kokkos::parallel_for (policy1, KOKKOS_LAMBDA(member_type team_member) {

    mj_lno_t coordinate_end_index =
      local_kokkos_part_xadj(current_work_part_in_concurrent_parts);
    mj_lno_t coordinate_begin_index =
      local_kokkos_part_xadj(current_work_part_in_concurrent_parts-1);

    mj_scalar_t my_thread_min_coord = 0;
    mj_scalar_t my_thread_max_coord = 0;
    mj_scalar_t my_total_weight;

    //if the part is empty.
    //set the min and max coordinates as reverse.
    if(coordinate_begin_index >= coordinate_end_index)
    {
      my_thread_min_coord = max_scalar;
      my_thread_max_coord = -max_scalar;
      my_total_weight = 0;
    }
    else {
      // get min
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(
          team_member, coordinate_begin_index, coordinate_end_index),
        [=] (const int& j, mj_scalar_t & running_min) {
        int i = local_kokkos_coordinate_permutations(j);
        if(kokkos_mj_current_dim_coords(i) < running_min)
          running_min = kokkos_mj_current_dim_coords(i);
      }, Kokkos::Min<mj_scalar_t>(my_thread_min_coord));

      // get max
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(
          team_member, coordinate_begin_index, coordinate_end_index),
        [=] (const int& j, mj_scalar_t & running_max) {
        int i = local_kokkos_coordinate_permutations(j);
        if(kokkos_mj_current_dim_coords(i) > running_max)
          running_max = kokkos_mj_current_dim_coords(i);
      }, Kokkos::Max<mj_scalar_t>(my_thread_max_coord));

      // TODO: Note reading the single value should be bool
      // But that doesn't seem to be supported
      int weight0 = local_kokkos_mj_uniform_weights(0) ? 1 : 0;
      if(weight0) {
        my_total_weight = coordinate_end_index - coordinate_begin_index;
      }
      else {
        my_total_weight = 0;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(
            team_member, coordinate_begin_index, coordinate_end_index),
          [=] (int ii, mj_scalar_t & lsum) {
          int i = local_kokkos_coordinate_permutations(ii);
          lsum += local_kokkos_mj_weights(i,0);
        }, my_total_weight);
      }
    }

    local_kokkos_process_local_min_max_coord_total_weight(kk) =
      my_thread_min_coord;
    local_kokkos_process_local_min_max_coord_total_weight(
      kk + current_concurrent_num_parts) = my_thread_max_coord;
    local_kokkos_process_local_min_max_coord_total_weight(
      kk + 2*current_concurrent_num_parts) = my_total_weight;
  });
}

/*! \brief Function that reduces global minimum and maximum coordinates with
 * global total weight from given local arrays.
 * \param current_concurrent_num_parts is the number of parts whose cut lines
 * will be calculated concurrently.
 * \param local_min_max_total is the array holding local min and max coordinate
 * values with local total weight.
 * First concurrentPartCount entries are minimums of the parts, next
 * concurrentPartCount entries are max, and then the total weights.
 * \param global_min_max_total is the output array holding global min and global
 * coordinate values with global total weight.
 * The structure is same as localMinMaxTotal.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t,
  mj_node_t>::mj_get_global_min_max_coord_totW(
  mj_part_t current_concurrent_num_parts,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_local_min_max_total,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_global_min_max_total) {
  // reduce min for first current_concurrent_num_parts elements, reduce
  // max for next concurrentPartCount elements, reduce sum for the last
  // concurrentPartCount elements.
  if(this->comm->getSize()  > 1) {
    Teuchos::MultiJaggedCombinedMinMaxTotalReductionOp<int, mj_scalar_t>
      reductionOp(current_concurrent_num_parts,
        current_concurrent_num_parts, current_concurrent_num_parts);
    try {
      reduceAll<int, mj_scalar_t>(
        *(this->comm),
      reductionOp,
      3 * current_concurrent_num_parts,
      // TODO: Note this is refactored but needs to be improved
      // to avoid the use of direct data() ptr completely.
      kokkos_local_min_max_total.data(),
      kokkos_global_min_max_total.data());
    }
    Z2_THROW_OUTSIDE_ERROR(*(this->mj_env))
  }
  else {
    mj_part_t s = 3 * current_concurrent_num_parts;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename mj_node_t::execution_space,
      mj_part_t> (0, s), KOKKOS_LAMBDA (const mj_part_t i) {
      kokkos_global_min_max_total(i) = kokkos_local_min_max_total(i);
    });
  }
}

/*! \brief Function that calculates the new coordinates for the cut lines.
 * Function is called inside the parallel region.
 * \param min_coord minimum coordinate in the range.
 * \param max_coord maximum coordinate in the range.
 * \param num_cuts holds the number of cuts in current partitioning dimension.
 * \param global_weight holds the global total weight in the current part.
 * \param initial_cut_coords is the output array for the initial cut lines.
 * \param target_part_weights is the output array holding the cumulative ratios
 * of parts in current partitioning.
 * For partitioning to 4 uniformly, target_part_weights will be
 * (0.25 * globalTotalWeight, 0.5 *globalTotalWeight,
 *  0.75 * globalTotalWeight, globalTotalWeight).
 * \param future_num_part_in_parts is the vector that holds how many more parts
 * each part will be divided into more for the parts at the beginning of this
 * coordinate partitioning.
 * \param next_future_num_parts_in_parts is the vector that holds how many more
 * parts each part will be divided into more for the parts that will be obtained
 * at the end of this coordinate partitioning.
 * \param concurrent_current_part is the index of the part in the
 * future_num_part_in_parts vector.
 * \param obtained_part_index holds the amount of shift in the
 * next_future_num_parts_in_parts for the output parts.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_get_initial_cut_coords_target_weights(
  mj_scalar_t min_coord,
  mj_scalar_t max_coord,
  mj_part_t num_cuts/*p-1*/ ,
  mj_scalar_t global_weight,
  /*p - 1 sized, coordinate of each cut line*/
  Kokkos::View<mj_scalar_t *, device_t> kokkos_initial_cut_coords,
  /*cumulative weights, at left side of each cut line. p-1 sized*/
  Kokkos::View<mj_scalar_t *, device_t> current_target_part_weights ,
  std::vector <mj_part_t> *future_num_part_in_parts, //the vecto
  std::vector <mj_part_t> *next_future_num_parts_in_parts,
  mj_part_t concurrent_current_part,
  mj_part_t obtained_part_index)
{
  mj_scalar_t coord_range = max_coord - min_coord;

  // TODO: needs clean up - for now just copy device to host and use
  auto local_kokkos_mj_uniform_parts = this->kokkos_mj_uniform_parts;
  mj_part_t part0;
  Kokkos::parallel_reduce("Read single", 1,
    KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
    set_single = local_kokkos_mj_uniform_parts(0);
  }, part0);

  if(part0) {
    mj_part_t cumulative = 0;
    // how many total future parts the part will be partitioned into.
    mj_scalar_t total_future_part_count_in_part =
      mj_scalar_t((*future_num_part_in_parts)[concurrent_current_part]);
    // how much each part should weigh in ideal case.
    mj_scalar_t unit_part_weight =
      global_weight / total_future_part_count_in_part;
    for(mj_part_t i = 0; i < num_cuts; ++i) {
      cumulative += (*next_future_num_parts_in_parts)[i + obtained_part_index];
      // TODO: We want to refactor these loops
      // For now do temp host to device write
      Kokkos::parallel_for(
        // dummy single loop - to refactor
        Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (0,1),
        KOKKOS_LAMBDA (const int dummy) {
        // set target part weight.
        current_target_part_weights[i] = cumulative * unit_part_weight;
        kokkos_initial_cut_coords(i) = min_coord +
          (coord_range * cumulative) / total_future_part_count_in_part;
      });
    }

    Kokkos::parallel_for(
      // dummy single loop - to refactor
      Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (0,1),
      KOKKOS_LAMBDA (const int dummy) {
      current_target_part_weights[num_cuts] = 1;
    });

    // TODO: needs clean up - for now just copy device to host and use
    auto local_kokkos_mj_uniform_weights = this->kokkos_mj_uniform_weights;
    int uniform_weight0; // should be bool, not int but bool not supported
    Kokkos::parallel_reduce("Read single", 1,
      KOKKOS_LAMBDA(int dummy, int & set_single) {
      set_single = local_kokkos_mj_uniform_weights(0) ? 1 : 0;
    }, uniform_weight0);

    // round the target part weights.
    if(uniform_weight0) {
      Kokkos::parallel_for(
        Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
          (0,num_cuts+1),
        KOKKOS_LAMBDA (const int i) {
        current_target_part_weights[i] =
          long(current_target_part_weights[i] + 0.5);
      });
    }
  }
  else {
    std::cerr << "MJ does not support non uniform part weights" << std::endl;
    exit(1);
  }
}

/*! \brief Function that calculates the new coordinates for the cut lines.
 * Function is called inside the parallel region.
 * \param max_coordinate maximum coordinate in the range.
 * \param min_coordinate minimum coordinate in the range.
 * \param concurrent_current_part_index is the index of the part in the
 * inTotalCounts vector.
 * \param coordinate_begin_index holds the beginning of the coordinates in
 * current part.
 * \param coordinate_end_index holds end of the coordinates in current part.
 * \param mj_current_coordinate_permutations is the permutation array, holds the
 * real indices of coordinates on mj_current_dim_coords array.
 * \param mj_current_dim_coords is the 1D array holding the coordinates.
 * \param mj_part_ids is the array holding the partIds of each coordinate.
 * \param partition_count is the number of parts that the current part will be
 * partitioned into.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  set_initial_coordinate_parts(
  mj_scalar_t &max_coordinate,
  mj_scalar_t &min_coordinate,
  mj_part_t &concurrent_current_part_index,
  mj_lno_t coordinate_begin_index,
  mj_lno_t coordinate_end_index,
  Kokkos::View<mj_lno_t *, device_t> mj_current_coordinate_permutations,
  Kokkos::View<mj_scalar_t *, device_t> mj_current_dim_coords,
  Kokkos::View<mj_part_t *, device_t> kokkos_mj_part_ids,
  mj_part_t &partition_count)
{
  mj_scalar_t coordinate_range = max_coordinate - min_coordinate;

  // if there is single point, or if all points are along a line.
  // set initial part to 0 for all.
  if(ZOLTAN2_ABS(coordinate_range) < this->sEpsilon ) {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename mj_node_t::execution_space,
        int> (coordinate_begin_index, coordinate_end_index),
      KOKKOS_LAMBDA (const int ii) {
      kokkos_mj_part_ids(mj_current_coordinate_permutations[ii]) = 0;
    });
  }
  else {
    //otherwise estimate an initial part for each coordinate.
    //assuming uniform distribution of points.
    mj_scalar_t slice = coordinate_range / partition_count;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename mj_node_t::execution_space,
        int> (coordinate_begin_index, coordinate_end_index),
      KOKKOS_LAMBDA (const int ii) {
      mj_lno_t iii = mj_current_coordinate_permutations[ii];
      mj_part_t pp =
        mj_part_t((mj_current_dim_coords[iii] - min_coordinate) / slice);
      kokkos_mj_part_ids[iii] = 2 * pp;
    });
  }
}

/*! \brief Function that is responsible from 1D partitioning of the given range
 * of coordinates.
 * \param mj_current_dim_coords is 1 dimensional array holding
 * coordinate values.
 * \param imbalanceTolerance is the maximum allowed imbalance ratio.
 * \param current_work_part is the beginning index of concurrentPartCount parts.
 * \param current_concurrent_num_parts is the number of parts whose cut lines
 * will be calculated concurrently.
 * \param current_cut_coordinates is array holding the coordinates of the cut.
 * \param total_incomplete_cut_count is the number of cut lines whose positions
 * should be calculated.
 * \param view_num_partitioning_in_current_dim is the vector that holds how many
 * parts each part will be divided into.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t,mj_node_t>::mj_1D_part(
  Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords,
  mj_scalar_t used_imbalance_tolerance,
  mj_part_t current_work_part,
  mj_part_t current_concurrent_num_parts,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_coordinates,
  mj_part_t total_incomplete_cut_count,
  Kokkos::View<mj_part_t*, device_t> & view_num_partitioning_in_current_dim,
  Kokkos::View<mj_part_t *, device_t> view_rectilinear_cut_count,
  Kokkos::View<size_t*, device_t> view_total_reduction_size)
{
  clock_mj_1D_part_init.start();

  this->kokkos_temp_cut_coords = kokkos_current_cut_coordinates;

  Teuchos::MultiJaggedCombinedReductionOp<mj_part_t, mj_scalar_t>
               *reductionOp = NULL;

  bool bSingleProcess = (this->comm->getSize() == 1);
  
  // Refactor in progress
  // I eliminated the original std::vector
  // and replaced with kokkos view view_num_partitioning_in_current_dim.
  // Now restore a std::vector here just for the reduction op which has not been
  // refactored yet.
  if(!bSingleProcess) {
    // This got ugly quickly .... but it's at least partly temporary as I
    // refactor and perhaps will remove reference type eventually.
    typename std::remove_reference<decltype(
      view_num_partitioning_in_current_dim)>::type::HostMirror
      hostArray
      = Kokkos::create_mirror_view(view_num_partitioning_in_current_dim);
    Kokkos::deep_copy(hostArray, view_num_partitioning_in_current_dim);
    std::vector<mj_part_t> temp(view_num_partitioning_in_current_dim.size());
    for(size_t n = 0; n < view_num_partitioning_in_current_dim.size(); ++n) {
      temp[n] = hostArray(n);
    }

    reductionOp = new Teuchos::MultiJaggedCombinedReductionOp
      <mj_part_t, mj_scalar_t>(
        &temp,
        current_work_part ,
        current_concurrent_num_parts);
  }
  
  // use locals to avoid capturing this for cuda  
  auto local_kokkos_thread_part_weights = kokkos_thread_part_weights;
  auto local_kokkos_thread_cut_left_closest_point =
    kokkos_thread_cut_left_closest_point;
  auto local_kokkos_thread_cut_right_closest_point =
    kokkos_thread_cut_right_closest_point;
  auto local_kokkos_is_cut_line_determined = kokkos_is_cut_line_determined;
  auto local_kokkos_cut_lower_bound_coordinates =
    kokkos_cut_lower_bound_coordinates;
  auto local_kokkos_cut_upper_bound_coordinates =
    kokkos_cut_upper_bound_coordinates;
  auto local_kokkos_cut_upper_bound_weights = kokkos_cut_upper_bound_weights;
  auto local_kokkos_cut_lower_bound_weights = kokkos_cut_lower_bound_weights;
  bool local_distribute_points_on_cut_lines = distribute_points_on_cut_lines;
  auto local_kokkos_process_cut_line_weight_to_put_left =
    kokkos_process_cut_line_weight_to_put_left;
  auto local_kokkos_my_incomplete_cut_count = kokkos_my_incomplete_cut_count;
  auto local_kokkos_temp_cut_coords = kokkos_temp_cut_coords;
  auto local_kokkos_global_total_part_weight_left_right_closests =
    kokkos_global_total_part_weight_left_right_closests;
  auto local_kokkos_total_part_weight_left_right_closests =
    kokkos_total_part_weight_left_right_closests;
  auto local_kokkos_cut_coordinates_work_array =
    kokkos_cut_coordinates_work_array;
  auto local_kokkos_part_xadj = kokkos_part_xadj;
  auto local_kokkos_global_min_max_coord_total_weight =
    kokkos_global_min_max_coord_total_weight;
  auto local_kokkos_target_part_weights =
    kokkos_target_part_weights;
  auto local_kokkos_global_rectilinear_cut_weight =
    kokkos_global_rectilinear_cut_weight;
  auto local_kokkos_process_rectilinear_cut_weight =
    kokkos_process_rectilinear_cut_weight;

  typedef typename Kokkos::TeamPolicy<typename mj_node_t::execution_space>::
    member_type member_type;

  clock_mj_1D_part_init.stop();
  clock_mj_1D_part_init2.start();

  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int dummy) {

    // these need to be initialized
    view_rectilinear_cut_count(0) = 0;
    view_total_reduction_size(0) = 0;

    //initialize the lower and upper bounds of the cuts.
    mj_part_t next = 0;
    for(mj_part_t i = 0; i < current_concurrent_num_parts; ++i){
      mj_part_t num_part_in_dim =
        view_num_partitioning_in_current_dim(current_work_part + i);
      mj_part_t num_cut_in_dim = num_part_in_dim - 1;
      view_total_reduction_size(0) += (4 * num_cut_in_dim + 1);

      for(mj_part_t ii = 0; ii < num_cut_in_dim; ++ii){
        local_kokkos_is_cut_line_determined(next) = false;
        // min coordinate
        local_kokkos_cut_lower_bound_coordinates(next) =
          local_kokkos_global_min_max_coord_total_weight(i);
        // max coordinate
        local_kokkos_cut_upper_bound_coordinates(next) =
          local_kokkos_global_min_max_coord_total_weight(
          i + current_concurrent_num_parts);
        // total weight
        local_kokkos_cut_upper_bound_weights(next) =
          local_kokkos_global_min_max_coord_total_weight(
          i + 2 * current_concurrent_num_parts);
        local_kokkos_cut_lower_bound_weights(next) = 0;
        if(local_distribute_points_on_cut_lines){
          local_kokkos_process_cut_line_weight_to_put_left(next) = 0;
        }
        ++next;
      }
    }
  });

  clock_mj_1D_part_init2.stop();
  clock_mj_1D_part_while_loop.start();

  while (total_incomplete_cut_count != 0) {
    clock_host_copies.start();

    // TODO: Need to eliminate all of this
    // Pull the values for num cuts
    typename std::remove_reference<
      decltype (view_num_partitioning_in_current_dim)>::type::HostMirror
      host_view_num_partitioning_in_current_dim =
      Kokkos::create_mirror_view(view_num_partitioning_in_current_dim);
    Kokkos::deep_copy(host_view_num_partitioning_in_current_dim,
      view_num_partitioning_in_current_dim);

    // Pull the values for incomplete cut cout
    typename decltype (kokkos_my_incomplete_cut_count)::HostMirror
      host_kokkos_my_incomplete_cut_count =
      Kokkos::create_mirror_view(kokkos_my_incomplete_cut_count);
    Kokkos::deep_copy(host_kokkos_my_incomplete_cut_count,
      kokkos_my_incomplete_cut_count);

    clock_host_copies.stop();

    clock_mj_1D_part_get_weights.start();

    this->mj_1D_part_get_thread_part_weights(
      view_num_partitioning_in_current_dim,
      current_concurrent_num_parts,
      current_work_part,
      kokkos_mj_current_dim_coords);

    clock_mj_1D_part_get_weights.stop();

    clock_mj_accumulate_thread_results.start();

    // sum up the results of threads
    // TODO: Now with threads eliminated we should eliminate a lot of this code
    // There is only 1 thread block in the new cuda refactor setup
    this->mj_accumulate_thread_results(
      view_num_partitioning_in_current_dim,
      current_work_part,
      current_concurrent_num_parts,
      local_kokkos_is_cut_line_determined,
      local_kokkos_thread_cut_left_closest_point,
      local_kokkos_thread_cut_right_closest_point,
      local_kokkos_total_part_weight_left_right_closests,
      local_kokkos_thread_part_weights
    );

    clock_mj_accumulate_thread_results.stop();
    clock_write_globals.start();

    // Rewrite as single TODO
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int i) {
      //now sum up the results of mpi processors.
      if(!bSingleProcess){
        // TODO: Ignore this code for cuda right now - not worrying about
        // parallel build yet
        #ifndef KOKKOS_ENABLE_CUDA
        // TODO: Remove use of data() - refactor in progress
        reduceAll<int, mj_scalar_t>( *(this->comm), *reductionOp,
          view_total_reduction_size(0),
          this->kokkos_total_part_weight_left_right_closests.data(),
          this->kokkos_global_total_part_weight_left_right_closests.data());
        #endif
      }
      else {
        // TODO: Optimize and fix this c cast - clean up use of the view
        for(int n = 0; n < (int) view_total_reduction_size(0); ++n) {
          local_kokkos_global_total_part_weight_left_right_closests(n) =
            local_kokkos_total_part_weight_left_right_closests(n);
        }
      }
    });

    clock_write_globals.stop();

    // how much cut will be shifted for the next part in the concurrent
    // part calculation.
    mj_part_t cut_shift = 0;

    // how much the concantaneted array will be shifted for the next part
    // in concurrent part calculation.
    size_t tlr_shift = 0;
    for (mj_part_t kk = 0; kk < current_concurrent_num_parts; ++kk) {

      clock_mj_get_new_cut_coordinates_init.start();

      mj_part_t num_parts =
        host_view_num_partitioning_in_current_dim(current_work_part + kk);

      mj_part_t num_cuts = num_parts - 1;
      size_t num_total_part = num_parts + size_t (num_cuts);

      //if the cuts of this cut has already been completed.
      //nothing to do for this part.
      //just update the shift amount and proceed.
               
      mj_part_t kk_kokkos_my_incomplete_cut_count
        = host_kokkos_my_incomplete_cut_count(kk);

      if (kk_kokkos_my_incomplete_cut_count == 0) {
        cut_shift += num_cuts;
        tlr_shift += (num_total_part + 2 * num_cuts);
        clock_mj_get_new_cut_coordinates_init.stop();
        continue;
      }

      Kokkos::View<mj_scalar_t *, device_t> kokkos_current_local_part_weights =
        Kokkos::subview(local_kokkos_total_part_weight_left_right_closests,
          std::pair<mj_lno_t, mj_lno_t>(
            tlr_shift,
            local_kokkos_total_part_weight_left_right_closests.size()));

      Kokkos::View<mj_scalar_t *, device_t> kokkos_current_global_tlr =
        Kokkos::subview(
          local_kokkos_global_total_part_weight_left_right_closests,
          std::pair<mj_lno_t, mj_lno_t>(
            tlr_shift,
            local_kokkos_global_total_part_weight_left_right_closests.size()));

      Kokkos::View<mj_scalar_t *, device_t>
        kokkos_current_global_left_closest_points =
        Kokkos::subview(kokkos_current_global_tlr,
          std::pair<mj_lno_t, mj_lno_t>(
            num_total_part,
            kokkos_current_global_tlr.size()));
      Kokkos::View<mj_scalar_t *, device_t>
        kokkos_current_global_right_closest_points =
        Kokkos::subview(kokkos_current_global_tlr,
          std::pair<mj_lno_t, mj_lno_t>(
            num_total_part + num_cuts,
            kokkos_current_global_tlr.size()));
      Kokkos::View<mj_scalar_t *, device_t> kokkos_current_global_part_weights =
        kokkos_current_global_tlr;

      Kokkos::View<bool *, device_t> kokkos_current_cut_line_determined =
        Kokkos::subview(local_kokkos_is_cut_line_determined,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift,
            local_kokkos_is_cut_line_determined.size()));
      Kokkos::View<mj_scalar_t *, device_t> kokkos_current_part_target_weights =
        Kokkos::subview(local_kokkos_target_part_weights,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift + kk,
            local_kokkos_target_part_weights.size()));
      Kokkos::View<mj_scalar_t *, device_t>
        kokkos_current_part_cut_line_weight_to_put_left =
        Kokkos::subview(local_kokkos_process_cut_line_weight_to_put_left,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift,
            local_kokkos_process_cut_line_weight_to_put_left.size()));

      mj_part_t initial_incomplete_cut_count =
        kk_kokkos_my_incomplete_cut_count;

      Kokkos::View<mj_scalar_t *, device_t>
        kokkos_current_cut_lower_bound_weights =
        Kokkos::subview(local_kokkos_cut_lower_bound_weights,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift,
            local_kokkos_cut_lower_bound_weights.size()));
      Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_upper_weights =
        Kokkos::subview(local_kokkos_cut_upper_bound_weights,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift,
            local_kokkos_cut_upper_bound_weights.size()));
      Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_upper_bounds =
        Kokkos::subview(local_kokkos_cut_upper_bound_coordinates,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift,
            local_kokkos_cut_upper_bound_coordinates.size()));
      Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_lower_bounds =
        Kokkos::subview(local_kokkos_cut_lower_bound_coordinates,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift,
            local_kokkos_cut_lower_bound_coordinates.size()));

      clock_mj_get_new_cut_coordinates_init.stop();
      clock_mj_get_new_cut_coordinates.start();

      // Now compute the new cut coordinates.
      this->mj_get_new_cut_coordinates(
        current_concurrent_num_parts,
        kk,
        num_total_part,
        num_cuts,
        used_imbalance_tolerance,
        kokkos_current_global_part_weights,
        kokkos_current_local_part_weights,
        kokkos_current_part_target_weights,
        kokkos_current_cut_line_determined,
        Kokkos::subview(local_kokkos_temp_cut_coords,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift, local_kokkos_temp_cut_coords.size())),
        kokkos_current_cut_upper_bounds,
        kokkos_current_cut_lower_bounds,
        kokkos_current_global_left_closest_points,
        kokkos_current_global_right_closest_points,
        kokkos_current_cut_lower_bound_weights,
        kokkos_current_cut_upper_weights,
        Kokkos::subview(local_kokkos_cut_coordinates_work_array,
          std::pair<mj_lno_t, mj_lno_t>(
            cut_shift, local_kokkos_cut_coordinates_work_array.size())),
        kokkos_current_part_cut_line_weight_to_put_left,
        view_rectilinear_cut_count);
  
      clock_mj_get_new_cut_coordinates.stop();
      clock_mj_get_new_cut_coordinates_end.start();

      cut_shift += num_cuts;
      tlr_shift += (num_total_part + 2 * num_cuts);

      Kokkos::parallel_reduce("Read single", 1,
        KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
        set_single = local_kokkos_my_incomplete_cut_count(kk);
      }, kk_kokkos_my_incomplete_cut_count);

      mj_part_t iteration_complete_cut_count =
        initial_incomplete_cut_count - kk_kokkos_my_incomplete_cut_count;
      Kokkos::atomic_add(&total_incomplete_cut_count,
        -iteration_complete_cut_count);

      clock_mj_get_new_cut_coordinates_end.stop();
    }

    { //This unnecessary bracket works around a compiler bug in NVCC when
      // compiling with OpenMP enabled.
      // swap the cut coordinates for next iteration
      // TODO: Need to figure this out - how to swap cleanly with Cuda/Kokkos
      // This is inefficient as a test to get some basic cuda up and running

      clock_swap.start();

      Kokkos::parallel_for((int) local_kokkos_temp_cut_coords.size(),
        KOKKOS_LAMBDA(int n) { 
        // for(int n = 0; n < (int) local_kokkos_temp_cut_coords.size(); ++n) {
        auto t = local_kokkos_temp_cut_coords(n);
        local_kokkos_temp_cut_coords(n) =
          local_kokkos_cut_coordinates_work_array(n);
        local_kokkos_cut_coordinates_work_array(n) = t;
      });

      clock_swap.stop();
    }
  } // end of the while loop

  clock_mj_1D_part_while_loop.stop();
  clock_mj_1D_part_end.start();

  Kokkos::TeamPolicy<typename mj_node_t::execution_space>
    policy3 (1, Kokkos::AUTO());
  Kokkos::parallel_for (policy3, KOKKOS_LAMBDA(member_type team_member) {

    // Needed only if keep_cuts; otherwise can simply swap array pointers
    // cutCoordinates and cutCoordinatesWork.
    // (at first iteration, cutCoordinates == cutCoorindates_tmp).
    // computed cuts must be in cutCoordinates.
    if (kokkos_current_cut_coordinates != local_kokkos_temp_cut_coords) {
      if(team_member.league_rank() == 0) {
        Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
          mj_part_t next = 0;
          for(mj_part_t i = 0; i < current_concurrent_num_parts; ++i) {
            mj_part_t num_parts = -1;
            num_parts =
              view_num_partitioning_in_current_dim(current_work_part + i);
            mj_part_t num_cuts = num_parts - 1;
            for(mj_part_t ii = 0; ii < num_cuts; ++ii){
              kokkos_current_cut_coordinates(next + ii) =
                local_kokkos_temp_cut_coords(next + ii);
            }
            next += num_cuts;
          }
        });
        team_member.team_barrier();  // for end of Kokkos::single
      }
      if(team_member.league_rank() == 0) {
        Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
          // TODO Same as above - need to optimize fix this
          // Work around for the ptr swap setup just to get some cuda running
          // But this is inefficient
          for(int n = 0;
            n < (int) local_kokkos_cut_coordinates_work_array.size(); ++n) {
            local_kokkos_cut_coordinates_work_array(n) =
              local_kokkos_temp_cut_coords(n);
          }
        });
        team_member.team_barrier();  // for end of Kokkos::single
      }
    } // end of if league_rank == 0
  }); // end of outer mj_1D_part loop

  clock_mj_1D_part_end.stop();
  delete reductionOp;
}

template<class scalar_t>
struct ArrayType {
  scalar_t * ptr;
  KOKKOS_INLINE_FUNCTION
  ArrayType(scalar_t * pSetPtr) : ptr(pSetPtr) {};
};

template<class policy_t, class scalar_t, class part_t>
struct ArraySumReducer {

  typedef ArraySumReducer reducer;
  typedef ArrayType<scalar_t> value_type;
  value_type * value;
  size_t value_count;

  KOKKOS_INLINE_FUNCTION ArraySumReducer(
    value_type &val,
    const size_t & count) :
    value(&val), value_count(count)
  {}

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const {
    return *value;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src)  const {
    for(int n = 0; n < value_count; ++n) {
      dst.ptr[n] += src.ptr[n];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type& dst, const volatile value_type& src) const {
    for(int n = 0; n < value_count; ++n) {
      dst.ptr[n] += src.ptr[n];
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type& dst) const {
    for(int n = 0; n < value_count; ++n) {
      dst.ptr[n] = 0;
    }
  }
};

template<class scalar_t, class part_t, class index_t, class node_t>
struct ReduceWeightsFunctorInnerLoop {

  Kokkos::View<index_t*, typename node_t::device_type> permutations;
  Kokkos::View<scalar_t *, typename node_t::device_type> coordinates;
  Kokkos::View<scalar_t**, typename node_t::device_type> weights;
  Kokkos::View<part_t*, typename node_t::device_type> parts;
  Kokkos::View<scalar_t *, typename node_t::device_type> cut_coordinates;
  bool bUniformWeights;
  scalar_t sEpsilon;
  part_t current_concurrent_num_parts;
  
  KOKKOS_INLINE_FUNCTION
  ReduceWeightsFunctorInnerLoop(
    part_t mj_current_concurrent_num_parts,
    Kokkos::View<index_t*, typename node_t::device_type> mj_permutations,
    Kokkos::View<scalar_t *, typename node_t::device_type> mj_coordinates,
    Kokkos::View<scalar_t**, typename node_t::device_type> mj_weights,
    Kokkos::View<part_t*, typename node_t::device_type> mj_parts,
    Kokkos::View<scalar_t *, typename node_t::device_type> mj_cut_coordinates,
    bool mj_bUniformWeights,
    scalar_t mj_sEpsilon
  ) : 
    current_concurrent_num_parts(mj_current_concurrent_num_parts),
    permutations(mj_permutations),
    coordinates(mj_coordinates),
    weights(mj_weights),
    parts(mj_parts),
    cut_coordinates(mj_cut_coordinates),
    bUniformWeights(mj_bUniformWeights),
    sEpsilon(mj_sEpsilon)
  {
  
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const size_t ii, ArrayType<scalar_t>& threadSum) const {
/*
for(int concurrent_part = 0; concurrent_part < current_concurrent_num_parts; ++concurrent_part) {
    auto offset = kk * num_cuts;

    int i = permutations(ii);
    scalar_t coord = coordinates(i);
    scalar_t w = bUniformWeights ? 1 : weights(i,0);
#ifdef NEW_FORM
    auto checking = (int) parts(i);
    scalar_t a;

    checking = 0;

    if(checking < 0) checking = 0;
    if(checking > num_cuts * 2) checking = num_cuts * 2;

    bool bRun = true;
    while(bRun) {
      if(checking % 2 == 0) {
        auto part = checking / 2;
        a = (part > 0) ? cut_coordinates(offset+part-1) : -9999999.9f; // to do fix range
        scalar_t b = (part < num_cuts) ? cut_coordinates(offset+part) : 999999.9f; // to do fix range
        if(coord >= a + sEpsilon && coord <= b - sEpsilon) {
          threadSum.ptr[part*2] += w;
          parts(i) = part*2;
          bRun = false;
        }
      }
      else {
        auto cut = checking / 2;
        a = cut_coordinates(offset+cut);
        if(coord < a + sEpsilon && coord > a - sEpsilon) {
          threadSum.ptr[cut*2+1] += w;
          parts(i) = cut*2+1;
          bRun = false;
        }
      }
      if(coord < a) {
        --checking;
      }           
      else {
        ++checking;
      }
    }
#else
    // check part 0
    scalar_t b = cut_coordinates(offset+0);
    if(coord <= b - sEpsilon) {
      threadSum.ptr[0] += w;
      parts(i) = 0;
    }

    // check cut 0
    if( coord < b + sEpsilon && coord > b - sEpsilon) {
      threadSum.ptr[1] += w;
      parts(i) = 1;
    }

    scalar_t a;

    // now check each part and it's right cut
    for(index_t part = 1; part < num_cuts; ++part) {
      a = b; 
      b = cut_coordinates(offset+part);

      if(coord < b + sEpsilon && coord > b - sEpsilon) {
        threadSum.ptr[part*2+1] += w;
        parts(i) = part*2+1;
      }
      
      if(coord >= a + sEpsilon && coord <= b - sEpsilon) {
        threadSum.ptr[part*2] += w;
        parts(i) = part*2;
      }
    }

    // check last part
    a = b;
    if(coord >= a + sEpsilon) {
      threadSum.ptr[num_cuts*2] += w;
      parts(i) = num_cuts*2;
    }
#endif
}
*/
  }
};

template<class policy_t, class scalar_t, class part_t,
  class index_t, class node_t>
struct ReduceWeightsFunctor {
  typedef typename policy_t::member_type member_type;
  typedef Kokkos::View<scalar_t*> scalar_view_t;
  typedef scalar_t value_type[];

  part_t current_work_part;
  part_t current_concurrent_num_parts;
  int value_count;
  Kokkos::View<index_t*, typename node_t::device_type> permutations;
  Kokkos::View<scalar_t *, typename node_t::device_type> coordinates;
  Kokkos::View<scalar_t**, typename node_t::device_type> weights;
  Kokkos::View<part_t*, typename node_t::device_type> parts;
  Kokkos::View<scalar_t *, typename node_t::device_type> cut_coordinates;
  Kokkos::View<index_t *, typename node_t::device_type> part_xadj;
  Kokkos::View<bool*, typename node_t::device_type> uniform_weights;
  scalar_t sEpsilon;
  
  ReduceWeightsFunctor(
    part_t mj_current_work_part,
    part_t mj_current_concurrent_num_parts,
    const int & mj_weight_array_size,
    Kokkos::View<index_t*, typename node_t::device_type> mj_permutations,
    Kokkos::View<scalar_t *, typename node_t::device_type> mj_coordinates,
    Kokkos::View<scalar_t**, typename node_t::device_type> mj_weights,
    Kokkos::View<part_t*, typename node_t::device_type> mj_parts,
    Kokkos::View<scalar_t *, typename node_t::device_type> mj_cut_coordinates,
    Kokkos::View<index_t *, typename node_t::device_type> mj_part_xadj,
    Kokkos::View<bool*, typename node_t::device_type> mj_uniform_weights,
    scalar_t mj_sEpsilon) :
    current_work_part(mj_current_work_part),
    current_concurrent_num_parts(mj_current_concurrent_num_parts),
    value_count(mj_weight_array_size),
    permutations(mj_permutations),
    coordinates(mj_coordinates),
    weights(mj_weights),
    parts(mj_parts),
    cut_coordinates(mj_cut_coordinates),
    part_xadj(mj_part_xadj),
    uniform_weights(mj_uniform_weights),
    sEpsilon(mj_sEpsilon) {
  }

  size_t team_shmem_size (int team_size) const {
    return sizeof(scalar_t) * value_count * team_size;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type & teamMember, value_type teamSum) const {
    bool bUniformWeights = uniform_weights(0);

for(int concurrent_current_part = 0; concurrent_current_part < current_concurrent_num_parts; ++concurrent_current_part) {
    index_t all_begin = (concurrent_current_part == 0) ? 0 :
      part_xadj(concurrent_current_part -1);
    index_t all_end = part_xadj(concurrent_current_part);
    
    index_t num_working_points = all_end - all_begin;
    int num_teams = teamMember.league_size();
    
    index_t stride = num_working_points / num_teams;
    if((num_working_points % num_teams) > 0) {
      stride += 1; // make sure we have coverage for the final points
    }
        
    index_t begin = all_begin + stride * teamMember.league_rank();
    index_t end = begin + stride;
    if(end > all_end) {
      end = all_end; // the last team may have less work than the other teams
    }

    // create the team shared data - each thread gets one of the arrays
    scalar_t * shared_ptr = (scalar_t *) teamMember.team_shmem().get_shmem(
      sizeof(scalar_t) * value_count * teamMember.team_size());

    // select the array for this thread
    ArrayType<scalar_t>
      array(&shared_ptr[teamMember.team_rank() * value_count]);

    // create reducer which handles the ArrayType class
    ArraySumReducer<policy_t, scalar_t, part_t> arraySumReducer(
      array, value_count);

    // call the reduce
    ReduceWeightsFunctorInnerLoop<scalar_t, part_t,
      index_t, node_t> inner_functor(
      current_concurrent_num_parts,
      permutations,
      coordinates,
      weights,
      parts,
      cut_coordinates,
      bUniformWeights,
      sEpsilon);

    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(teamMember, begin, end),
      inner_functor, arraySumReducer);

    teamMember.team_barrier();

    // collect all the team's results
    Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
      for(int n = 0; n < value_count; ++n) {
        teamSum[n] += array.ptr[n];
      }
    });
}

  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type dst, const value_type src)  const {
    for(int n = 0; n < value_count; ++n) {
      dst[n] += src[n];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type dst, const volatile value_type src) const {
    for(int n = 0; n < value_count; ++n) {
      dst[n] += src[n];
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type dst) const {
    for(int n = 0; n < value_count; ++n) {
      dst[n] = 0;
    }
  }
};

template<class policy_t, class scalar_t, class part_t>
struct ArrayMinMaxReducer {

  typedef ArrayMinMaxReducer reducer;
  typedef ArrayType<scalar_t> value_type;
  value_type * value;
  size_t value_count;
  scalar_t max_scalar;

  KOKKOS_INLINE_FUNCTION ArrayMinMaxReducer(
    value_type &val,
    const size_t & count,
    scalar_t mj_max_scalar) :
    value(&val), value_count(count), max_scalar(mj_max_scalar)
  {}

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const {
    return *value;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src)  const {
    for(int n = 2; n < value_count - 2; n += 2) {
      if(src.ptr[n] > dst.ptr[n]) {
        dst.ptr[n] = src.ptr[n];
      }
      if(src.ptr[n+1] < dst.ptr[n+1]) {
        dst.ptr[n+1] = src.ptr[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type& dst, const volatile value_type& src) const {
   for(int n = 2; n < value_count - 2; n += 2) {
      if(src[n] > dst[n]) {
        dst.ptr[n] = src.ptr[n];
      }
      if(src[n+1] < dst[n+1]) {
        dst.ptr[n+1] = src.ptr[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type& dst) const {
    for(int n = 2; n < value_count - 2; n += 2) {
      dst.ptr[n]   = -max_scalar;
      dst.ptr[n+1] =  max_scalar;
    }
  }
};

template<class policy_t, class scalar_t, class part_t,
  class index_t, class node_t>
struct RightLeftClosestFunctor {
  typedef typename policy_t::member_type member_type;
  typedef Kokkos::View<scalar_t*> scalar_view_t;
  typedef scalar_t value_type[];

  scalar_t max_scalar;
  part_t concurrent_current_part;
  int value_count;
  Kokkos::View<index_t*, typename node_t::device_type> permutations;
  Kokkos::View<scalar_t *, typename node_t::device_type> coordinates;
  Kokkos::View<part_t*, typename node_t::device_type> parts;
  Kokkos::View<scalar_t *, typename node_t::device_type> cut_coordinates;
  Kokkos::View<index_t *, typename node_t::device_type> part_xadj;
  scalar_t sEpsilon;

  RightLeftClosestFunctor(
    scalar_t mj_max_scalar,
    part_t mj_concurrent_current_part,
    const int & num_cuts,
    Kokkos::View<index_t*, typename node_t::device_type> mj_permutations,
    Kokkos::View<scalar_t *, typename node_t::device_type> mj_coordinates,
    Kokkos::View<part_t*, typename node_t::device_type> mj_parts,
    Kokkos::View<scalar_t *, typename node_t::device_type> mj_cut_coordinates,
    Kokkos::View<index_t *, typename node_t::device_type> mj_part_xadj,
    scalar_t mj_sEpsilon) :
    max_scalar(mj_max_scalar),
    concurrent_current_part(mj_concurrent_current_part),
    value_count(num_cuts*2),
    permutations(mj_permutations),
    coordinates(mj_coordinates),
    parts(mj_parts),
    cut_coordinates(mj_cut_coordinates),
    part_xadj(mj_part_xadj),
    sEpsilon(mj_sEpsilon) {
  }

  size_t team_shmem_size (int team_size) const {
    return sizeof(scalar_t) * value_count * team_size;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type & teamMember, value_type teamSum) const {
    index_t all_begin = (concurrent_current_part == 0) ? 0 :
      part_xadj(concurrent_current_part -1);
    index_t all_end = part_xadj(concurrent_current_part);
    index_t num_working_points = all_end - all_begin;
    int num_teams = teamMember.league_size();

    index_t stride = num_working_points / num_teams;
    if((num_working_points % num_teams) > 0) {
      stride += 1; // make sure we have coverage for the final points
    }
        
    index_t begin = all_begin + stride * teamMember.league_rank();
    index_t end = begin + stride;
    if(end > all_end) {
      end = all_end; // the last team may have less work than the other teams
    }

    // create the team shared data - each thread gets one of the arrays
    scalar_t * shared_ptr = (scalar_t *) teamMember.team_shmem().get_shmem(
      sizeof(scalar_t) * value_count * teamMember.team_size());

    // select the array for this thread
    ArrayType<scalar_t>
      array(&shared_ptr[teamMember.team_rank() * value_count]);

    // create reducer which handles the ArrayType class
    ArrayMinMaxReducer<policy_t, scalar_t, part_t> arrayMinMaxReducer(
      array, value_count, max_scalar);

    // call the reduce
    auto local_coordinates = coordinates;
    auto local_permutations = permutations;
    auto local_cut_coordinates = cut_coordinates;

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, begin, end),
      [=] (const size_t ii, ArrayType<scalar_t>& threadSum) {

      int i = local_permutations(ii);
      const scalar_t & coord = local_coordinates(i);
  
      // remove front end buffers - true count here
      part_t num_cuts = value_count / 2 - 2;

      scalar_t * p1 = &threadSum.ptr[2];
      for(int cut = 0; cut < num_cuts; ++cut) {
        scalar_t cut_coord = cut_coordinates(cut);
        if(coord > cut_coord && coord < *(p1+1)) {
          *(p1+1) = coord;
        }
        if(coord < cut_coord && coord > *p1) {
          *p1 = coord;
        }
        p1 += 2;
      }
    }, arrayMinMaxReducer);

    // collect all the team's results
    Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
      for(int n = 2; n < value_count - 2; n += 2) {
        if(array.ptr[n] > teamSum[n]) {
          teamSum[n] = array.ptr[n];
        }
        if(array.ptr[n+1] < teamSum[n+1]) {
          teamSum[n+1] = array.ptr[n+1];
        }
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type dst, const value_type src)  const {
    for(int n = 2; n < value_count - 2; n += 2) {
      if(src[n] > dst[n]) {
        dst[n] = src[n];
      }
    }
    for(int n = 2; n < value_count - 2; n += 2) {
      if(src[n+1] < dst[n+1]) {
        dst[n+1] = src[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type dst, const volatile value_type src) const {
    for(int n = 2; n < value_count - 2; n += 2) {
      if(src[n] > dst[n]) {
        dst[n] = src[n];
      }
    }
    for(int n = 2; n < value_count - 2; n += 2) {
      if(src[n+1] < dst[n+1]) {
        dst[n+1] = src[n+1];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type dst) const {
    for(int n = 2; n < value_count - 2; n += 2) {
      dst[n]   = -max_scalar;
      dst[n+1] =  max_scalar;
    }
  }
};

/*! \brief Function that calculates the weights of each part according to given
 * part cut coordinates. Function is called inside the parallel region. Thread
 * specific work arrays are provided as function parameter.
 * \param total_part_count is the sum of number of cutlines and number of parts.
 * Simply it is 2*P - 1.
 * \param num_cuts is the number of cut lines. P - 1.
 * \param max_coord is the maximum coordinate in the part.
 * \param min_coord is the min coordinate in the part.
 * \param coordinate_begin_index is the index of the first coordinate in
 * current part.
 * \param coordinate_end_index is the index of the last coordinate in
 * current part.
 * \param mj_current_dim_coords is 1 dimensional array holding
 * coordinate values.
 * \param temp_current_cut_coords is the array holding the coordinates of each
 * cut line. Sized P - 1.
 * \param current_cut_status is the boolean array to determine if the correct
 * position for a cut line is found.
 * \param my_current_part_weights is the array holding the part weights for
 * the calling thread.
 * \param my_current_left_closest is the array holding the coordinate of the
 * closest points to the cut lines from left for the calling thread..
 * \param my_current_right_closest is the array holding the coordinate of the
 * closest points to the cut lines from right for the calling thread.
 * \param partIds is the array that holds the part ids of the coordinates
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t,
  mj_part_t, mj_node_t>::
  mj_1D_part_get_thread_part_weights(
  Kokkos::View<mj_part_t*, device_t> view_num_partitioning_in_current_dim,
  mj_part_t current_concurrent_num_parts,
  mj_part_t current_work_part,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords)
{
  clock_mj_1D_part_get_weights_setup.start();
        
  auto local_kokkos_is_cut_line_determined = kokkos_is_cut_line_determined;
  auto local_kokkos_thread_part_weights = kokkos_thread_part_weights;
  auto local_kokkos_thread_cut_left_closest_point =
    kokkos_thread_cut_left_closest_point;
  auto local_kokkos_thread_cut_right_closest_point =
    kokkos_thread_cut_right_closest_point;
  auto local_kokkos_temp_cut_coords = kokkos_temp_cut_coords;
          
  clock_mj_1D_part_get_weights_setup.stop();
        
  // Create some locals so we don't use this inside the kernels
  // which causes problems
  auto local_sEpsilon = this->sEpsilon;
  auto local_kokkos_assigned_part_ids = this->kokkos_assigned_part_ids;
  auto local_kokkos_coordinate_permutations =
    this->kokkos_coordinate_permutations;
  auto local_kokkos_mj_weights = this->kokkos_mj_weights;
  auto local_kokkos_mj_uniform_weights = this->kokkos_mj_uniform_weights;
  auto local_kokkos_part_xadj = this->kokkos_part_xadj;
  auto local_kokkos_global_min_max_coord_total_weight =
    this->kokkos_global_min_max_coord_total_weight;

  typedef Kokkos::TeamPolicy<typename mj_node_t::execution_space> policy_t;
  
  clock_weights1.start();

  auto local_kokkos_my_incomplete_cut_count = kokkos_my_incomplete_cut_count;

  Kokkos::parallel_for (current_concurrent_num_parts, KOKKOS_LAMBDA(size_t kk) {
    mj_part_t num_parts = view_num_partitioning_in_current_dim(current_work_part + kk);
    mj_part_t num_cuts = num_parts - 1;
    size_t total_part_count = num_parts + size_t (num_cuts);  
    if(local_kokkos_my_incomplete_cut_count(kk) > 0) {
      for(int n = 0; n < static_cast<int>(total_part_count); ++n) {
        local_kokkos_thread_part_weights(total_part_count * kk + n) = 0;
      }
    }
  });


  clock_weights1.stop();

  // We need to establish the total working array size
  int array_length = 0;
  Kokkos::parallel_reduce("Get array size", current_concurrent_num_parts,
    KOKKOS_LAMBDA(int kk, int & length) {
    mj_part_t num_parts =
      view_num_partitioning_in_current_dim(current_work_part + kk);
    mj_part_t num_cuts = num_parts - 1;
    length += num_cuts * 2 + 1;
  }, array_length);
    
  ReduceWeightsFunctor<policy_t, mj_scalar_t, mj_part_t, mj_lno_t, mj_node_t>
    teamFunctor(
      current_work_part,
      current_concurrent_num_parts,
      array_length,
      kokkos_coordinate_permutations,
      kokkos_mj_current_dim_coords,
      kokkos_mj_weights,
      kokkos_assigned_part_ids,
      local_kokkos_temp_cut_coords,
      kokkos_part_xadj,
      kokkos_mj_uniform_weights,
      sEpsilon);

  auto policy_ReduceWeightsFunctor =
    policy_t(SET_NUM_TEAMS_ReduceWeightsFunctor, Kokkos::AUTO);

  mj_scalar_t * part_weights = new mj_scalar_t[array_length];

  Kokkos::parallel_reduce(policy_ReduceWeightsFunctor,
    teamFunctor, part_weights);
  
  int offset = 0;
  for (mj_part_t working_kk = 0; working_kk < current_concurrent_num_parts; ++working_kk) {
  
    // TODO: Expensive - optimize it
    mj_part_t num_parts;
    Kokkos::parallel_reduce("Read num parts", 1,
      KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
      set_single = view_num_partitioning_in_current_dim(current_work_part + working_kk);
    }, num_parts);
    mj_part_t num_cuts = num_parts - 1;
    size_t total_part_count = num_parts + size_t (num_cuts);
    
    Kokkos::View<double *, device_t> kokkos_my_current_part_weights =
      Kokkos::subview(local_kokkos_thread_part_weights,
        std::pair<mj_lno_t, mj_lno_t>(working_kk * total_part_count,
         working_kk * total_part_count + total_part_count));
    // Move it from global memory to device memory
    // TODO: Need to figure out how we can better manage this
    typename decltype(kokkos_my_current_part_weights)::HostMirror
      hostArray = Kokkos::create_mirror_view(kokkos_my_current_part_weights);
    for(int i = 0; i < static_cast<int>(total_part_count); ++i) {
      hostArray(i) = part_weights[i+offset];
    }
    Kokkos::deep_copy(kokkos_my_current_part_weights, hostArray);
   
    offset += num_cuts * 2 + 1;
    
  
  }
  
  delete [] part_weights;
  
/*
for (mj_part_t working_kk = 0; working_kk < current_concurrent_num_parts; ++working_kk) {

  // TODO: Expensive - optimize it
  mj_part_t num_parts;
  Kokkos::parallel_reduce("Read num parts", 1,
    KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
    set_single = view_num_partitioning_in_current_dim(current_work_part + working_kk);
  }, num_parts);
  mj_part_t num_cuts = num_parts - 1;
  size_t total_part_count = num_parts + size_t (num_cuts);
  
  // TODO: Expensive - optimize it
  mj_part_t incomplete_cut_count;
  Kokkos::parallel_reduce("Read incomplete_cut_count", 1,
    KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
    set_single = local_kokkos_my_incomplete_cut_count(working_kk);
  }, incomplete_cut_count);
  if(incomplete_cut_count == 0) continue;

  clock_weights2.start();

  int weight_array_size = num_cuts * 2 + 1;
  ReduceWeightsFunctor<policy_t, mj_scalar_t, mj_part_t, mj_lno_t, mj_node_t>
    teamFunctor(
      num_cuts,
      current_work_part,
      working_kk,
      weight_array_size,
      kokkos_coordinate_permutations,
      kokkos_mj_current_dim_coords,
      kokkos_mj_weights,
      kokkos_assigned_part_ids,
      local_kokkos_temp_cut_coords,
      kokkos_part_xadj,
      kokkos_mj_uniform_weights,
      sEpsilon);

  clock_weights2.stop();
  clock_weights3.start();

  auto policy_ReduceWeightsFunctor =
    policy_t(SET_NUM_TEAMS_ReduceWeightsFunctor, Kokkos::AUTO);

  mj_scalar_t * part_weights = new mj_scalar_t[weight_array_size];

  clock_functor_weights.start();

  Kokkos::parallel_reduce(policy_ReduceWeightsFunctor,
    teamFunctor, part_weights);

  clock_functor_weights.stop();

  // TODO: Need to clean this up
  // Be careful with edits here as originally as I was getting some subtle
  // differences between UVM on and UVM off for Cuda only and only some tests.
  // Make sure we only copy the proper elements of the view
  // I think we may want to remap the parent view to be shorter but that
  // was causing an issue I still need to investigate
  Kokkos::View<double *, device_t> kokkos_my_current_part_weights =
    Kokkos::subview(local_kokkos_thread_part_weights,
      std::pair<mj_lno_t, mj_lno_t>(working_kk * total_part_count,
       working_kk * total_part_count + total_part_count));
  // Move it from global memory to device memory
  // TODO: Need to figure out how we can better manage this
  typename decltype(kokkos_my_current_part_weights)::HostMirror
    hostArray = Kokkos::create_mirror_view(kokkos_my_current_part_weights);
  for(int i = 0; i < static_cast<int>(total_part_count); ++i) {
    hostArray(i) = part_weights[i];
  }
  Kokkos::deep_copy(kokkos_my_current_part_weights, hostArray);
 
  delete [] part_weights;

  clock_weights3.stop();
  
} // end phase 1

*/

  clock_weights4.start();

  Kokkos::parallel_for (current_concurrent_num_parts, KOKKOS_LAMBDA(size_t kk) {
    mj_part_t num_parts = view_num_partitioning_in_current_dim(current_work_part + kk);
    mj_part_t num_cuts = num_parts - 1;
    size_t total_part_count = num_parts + size_t (num_cuts);
    auto offset = kk * total_part_count;
    if(local_kokkos_my_incomplete_cut_count(kk) > 0) {
      for (size_t i = 1; i < total_part_count; ++i){
        // check for cuts sharing the same position; all cuts sharing a position
        // have the same weight == total weight for all cuts sharing the position.
        // don't want to accumulate that total weight more than once.
        if(i % 2 == 0 && i > 1 && i < total_part_count - 1 &&
          ZOLTAN2_ABS(local_kokkos_temp_cut_coords(kk * num_cuts + i / 2) -
            local_kokkos_temp_cut_coords(kk * num_cuts + i /2 - 1))
            < local_sEpsilon){
          // i % 2 = 0 when part i represents the cut coordinate.
          // if it is a cut, and if next cut also has the same coordinate, then
          // dont addup.
          local_kokkos_thread_part_weights(offset + i)
            = local_kokkos_thread_part_weights(offset + i-2);
          continue;
        }

        //otherwise do the prefix sum.
        local_kokkos_thread_part_weights(offset + i) +=
          local_kokkos_thread_part_weights(offset + i-1);
      }
    }
  });

  clock_weights4.stop();
  
for (mj_part_t working_kk = 0; working_kk < current_concurrent_num_parts; ++working_kk) {

  // TODO: Expensive - optimize it
  mj_part_t num_parts;
  Kokkos::parallel_reduce("Read num_parts", 1,
    KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
    set_single = view_num_partitioning_in_current_dim(current_work_part + working_kk);
  }, num_parts);
  mj_part_t num_cuts = num_parts - 1;
  
  // TODO: Expensive - optimize it
  mj_part_t incomplete_cut_count;
  Kokkos::parallel_reduce("Read incomplete cut count", 1,
    KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
    set_single = local_kokkos_my_incomplete_cut_count(working_kk);
  }, incomplete_cut_count);
  if(incomplete_cut_count == 0) continue;
  
  Kokkos::View<mj_scalar_t *, device_t> kokkos_my_current_left_closest =
    Kokkos::subview(local_kokkos_thread_cut_left_closest_point,
    std::pair<mj_lno_t, mj_lno_t>(
      working_kk * num_cuts,
      local_kokkos_thread_cut_left_closest_point.size()));
  Kokkos::View<mj_scalar_t *, device_t> kokkos_my_current_right_closest =
    Kokkos::subview(local_kokkos_thread_cut_right_closest_point,
      std::pair<mj_lno_t, mj_lno_t>(
        working_kk * num_cuts,
        local_kokkos_thread_cut_right_closest_point.size()));
  Kokkos::View<mj_scalar_t *, device_t> kokkos_temp_current_cut_coords =
    Kokkos::subview(local_kokkos_temp_cut_coords,
      std::pair<mj_lno_t, mj_lno_t>(
        working_kk * num_cuts,
        local_kokkos_temp_cut_coords.size()));
        
  clock_weights5.start();

  RightLeftClosestFunctor<policy_t, mj_scalar_t, mj_part_t,
    mj_lno_t, mj_node_t> rightLeftClosestFunctor(
    std::numeric_limits<mj_scalar_t>::max(),
    current_work_part + working_kk,
    num_cuts+2, // buffer beginning and end to skip if checks
    kokkos_coordinate_permutations,
    kokkos_mj_current_dim_coords,
    kokkos_assigned_part_ids,
    kokkos_temp_current_cut_coords,
    kokkos_part_xadj,
    sEpsilon);

  clock_weights5.stop();
  clock_weights6.start();

  // will have them as left, right, left, right, etc   2 for each cut
  // add a dummy cut beginning and end so we can skip if checks in the
  // parallel loop
  mj_scalar_t * left_max_right_min_values = new mj_scalar_t[(num_cuts+2)*2];

  clock_functor_rightleft_closest.start();

  auto policy_RightLeftClosestFunctor =
    policy_t(SET_NUM_TEAMS_RightLeftClosestFunctor, Kokkos::AUTO);
 
  Kokkos::parallel_reduce(policy_RightLeftClosestFunctor,
    rightLeftClosestFunctor, left_max_right_min_values);
    
  clock_functor_rightleft_closest.stop();

  // Move it from global memory to device memory
  // TODO: Need to figure out how we can better manage this
  typename decltype(kokkos_my_current_left_closest)::HostMirror
    hostLeftArray = Kokkos::create_mirror_view(kokkos_my_current_left_closest);
  typename decltype(kokkos_my_current_right_closest)::HostMirror
    hostRightArray =
      Kokkos::create_mirror_view(kokkos_my_current_right_closest);
  for(mj_part_t cut = 0; cut < num_cuts; ++cut) {
    // when reading shift right 1 due to the buffer at beginning and end
    hostLeftArray(cut)  = left_max_right_min_values[(cut+1)*2+0];
    hostRightArray(cut) = left_max_right_min_values[(cut+1)*2+1];
  }

  delete [] left_max_right_min_values;

  Kokkos::deep_copy(kokkos_my_current_left_closest, hostLeftArray);
  Kokkos::deep_copy(kokkos_my_current_right_closest, hostRightArray);

  clock_weights6.stop();
} // end phase 3

}

/*! \brief Function that reduces the result of multiple threads
 * for left and right closest points and part weights in a single mpi process.
 * \param view_num_partitioning_in_current_dim is the vector that holds the
 * number of cut lines in current dimension for each part.
 * \param current_work_part holds the index of the first part (important when
 * concurrent parts are used.)
 * \param current_concurrent_num_parts is the number of parts whose cut lines
 * will be calculated concurrently.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_accumulate_thread_results(
  Kokkos::View<mj_part_t*, device_t> view_num_partitioning_in_current_dim,
  mj_part_t current_work_part,
  mj_part_t current_concurrent_num_parts,
  Kokkos::View<bool *, device_t> local_kokkos_is_cut_line_determined,
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    local_kokkos_thread_cut_left_closest_point,
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    local_kokkos_thread_cut_right_closest_point,
  Kokkos::View<mj_scalar_t *, device_t>
    local_kokkos_total_part_weight_left_right_closests,
  Kokkos::View<double *, Kokkos::LayoutLeft, device_t>
    local_kokkos_thread_part_weights)
{

  typedef typename Kokkos::TeamPolicy<typename mj_node_t::execution_space>::
    member_type member_type;

  Kokkos::TeamPolicy<typename mj_node_t::execution_space> policy_single(1, 1);
  Kokkos::parallel_for (policy_single, KOKKOS_LAMBDA(member_type team_member) {

    // needs barrier here, as it requires all threads to finish
    // mj_1D_part_get_thread_part_weights using parallel region here reduces the
    // performance because of the cache invalidates.
    // Note: I confirmed we still need this barrier with the new kokkos refactor
    team_member.team_barrier();
    
    Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
      size_t tlr_array_shift = 0;
      mj_part_t cut_shift = 0;
      // iterate for all concurrent parts to find the left and right closest
      // points in the process.
      for(mj_part_t i = 0; i < current_concurrent_num_parts; ++i) {

        mj_part_t num_parts_in_part =
          view_num_partitioning_in_current_dim(current_work_part + i);
        mj_part_t num_cuts_in_part =
          num_parts_in_part - 1;
        size_t num_total_part_in_part =
          num_parts_in_part + size_t (num_cuts_in_part) ;
        // iterate for cuts in a single part.
        for(mj_part_t ii = 0; ii < num_cuts_in_part ; ++ii){
          mj_part_t next = tlr_array_shift + ii;
          mj_part_t cut_index = cut_shift + ii;

          if(local_kokkos_is_cut_line_determined(cut_index)) continue;
          mj_scalar_t left_closest_in_process =
            local_kokkos_thread_cut_left_closest_point(cut_index);
          mj_scalar_t right_closest_in_process =
            local_kokkos_thread_cut_right_closest_point(cut_index);

          // store the left and right closes points.
          local_kokkos_total_part_weight_left_right_closests(
            num_total_part_in_part + next) = left_closest_in_process;
          local_kokkos_total_part_weight_left_right_closests(
            num_total_part_in_part + num_cuts_in_part + next) =
            right_closest_in_process;
        }
        // set the shift position in the arrays
        tlr_array_shift += (num_total_part_in_part + 2 * num_cuts_in_part);
        cut_shift += num_cuts_in_part;
      }

      tlr_array_shift = 0;
      cut_shift = 0;
      size_t total_part_array_shift = 0;
      //iterate for all concurrent parts to find the total weight in the process.
      for(mj_part_t i = 0; i < current_concurrent_num_parts; ++i) {

        mj_part_t num_parts_in_part =
          view_num_partitioning_in_current_dim(current_work_part + i);
        mj_part_t num_cuts_in_part =
          num_parts_in_part - 1;
        size_t num_total_part_in_part =
          num_parts_in_part + size_t (num_cuts_in_part);

      for(size_t j = 0; j < num_total_part_in_part; ++j) {
        mj_part_t cut_ind = j / 2 + cut_shift;

        // need to check j !=  num_total_part_in_part - 1
        // which is same as j/2 != num_cuts_in_part.
        // we cannot check it using cut_ind, because of the concurrent part
        // concantanetion.
        if(j !=  num_total_part_in_part - 1 &&
          local_kokkos_is_cut_line_determined(cut_ind)) continue;

        double pwj =
          local_kokkos_thread_part_weights(total_part_array_shift + j);

        // size_t jshift = j % total_part_count + i *
        //   (total_part_count + 2 * noCuts);
        local_kokkos_total_part_weight_left_right_closests(tlr_array_shift + j)
          = pwj;
      }
      cut_shift += num_cuts_in_part;
      tlr_array_shift += num_total_part_in_part + 2 * num_cuts_in_part;
      total_part_array_shift += num_total_part_in_part;
      }
    });
    team_member.team_barrier();  // for end of Kokkos::single
  });
}

/*! \brief
 * Function that calculates the next pivot position,
 * according to given coordinates of upper bound and lower bound, the weights at
 * upper and lower bounds, and the expected weight.
 * \param cut_upper_bound is the upper bound coordinate of the cut.
 * \param cut_lower_bound is the lower bound coordinate of the cut.
 * \param cut_upper_weight is the weights at the upper bound of the cut.
 * \param cut_lower_weight is the weights at the lower bound of the cut.
 * \param expected_weight is the expected weight that should be placed on the
 * left of the cut line.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t,
  mj_node_t>::mj_calculate_new_cut_position(mj_scalar_t cut_upper_bound,
  mj_scalar_t cut_lower_bound,
  mj_scalar_t cut_upper_weight,
  mj_scalar_t cut_lower_weight,
  mj_scalar_t expected_weight,
  mj_scalar_t &new_cut_position){

  if(ZOLTAN2_ABS(cut_upper_bound - cut_lower_bound) < this->sEpsilon){
    new_cut_position = cut_upper_bound; //or lower bound does not matter.
  }

  if(ZOLTAN2_ABS(cut_upper_weight - cut_lower_weight) < this->sEpsilon){
    new_cut_position = cut_lower_bound;
  }

  mj_scalar_t coordinate_range = (cut_upper_bound - cut_lower_bound);
  mj_scalar_t weight_range = (cut_upper_weight - cut_lower_weight);
  mj_scalar_t my_weight_diff = (expected_weight - cut_lower_weight);

  mj_scalar_t required_shift = (my_weight_diff / weight_range);
  int scale_constant = 20;
  int shiftint= int (required_shift * scale_constant);
  if (shiftint == 0) shiftint = 1;
  required_shift = mj_scalar_t (shiftint) / scale_constant;
  new_cut_position = coordinate_range * required_shift + cut_lower_bound;
}

/*! \brief Function that determines the permutation indices of the coordinates.
 * \param num_parts is the number of parts.
 * \param mj_current_dim_coords is 1 dimensional array holding the
 * coordinate values.
 * \param current_concurrent_cut_coordinate is 1 dimensional array holding the
 * cut coordinates.
 * \param coordinate_begin is the start index of the given partition on
 * partitionedPointPermutations.
 * \param coordinate_end is the end index of the given partition on
 * partitionedPointPermutations.
 * \param used_local_cut_line_weight_to_left holds how much weight of the
 * coordinates on the cutline should be put on left side.
 * \param used_thread_part_weight_work is the two dimensional array holding the
 * weight of parts for each thread. Assumes there are 2*P - 1 parts
 * (cut lines are seperate parts).
 * \param out_part_xadj is the indices of coordinates calculated for the
 * partition on next dimension.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
mj_create_new_partitions(
  mj_part_t num_parts,
  mj_part_t current_concurrent_work_part,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_mj_current_dim_coords,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_current_concurrent_cut_coordinate,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_used_local_cut_line_weight_to_left,
  Kokkos::View<double *, Kokkos::LayoutLeft, device_t>
    kokkos_used_thread_part_weight_work,
  Kokkos::View<mj_lno_t *, device_t>
    kokkos_out_part_xadj,
  Kokkos::View<mj_lno_t *, Kokkos::LayoutLeft, device_t>
    local_kokkos_thread_point_counts,
  bool local_distribute_points_on_cut_lines,
  Kokkos::View<mj_scalar_t *, Kokkos::LayoutLeft, device_t>
    local_kokkos_thread_cut_line_weight_to_put_left,
  mj_scalar_t local_sEpsilon,
  Kokkos::View<mj_lno_t*, device_t>
    local_kokkos_coordinate_permutations,
  Kokkos::View<bool*, device_t>
    local_kokkos_mj_uniform_weights,
  Kokkos::View<mj_scalar_t**, device_t>
    local_kokkos_mj_weights,
  Kokkos::View<mj_part_t*, device_t>
    local_kokkos_assigned_part_ids,
  Kokkos::View<mj_lno_t*, device_t>
    local_kokkos_new_coordinate_permutations)
{

  clock_mj_create_new_partitions.start();

  auto local_kokkos_part_xadj = this->kokkos_part_xadj;

  mj_part_t num_cuts = num_parts - 1;

  if (local_distribute_points_on_cut_lines) {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename mj_node_t::execution_space, mj_part_t>
        (0, num_cuts), KOKKOS_LAMBDA (const mj_part_t & i) {
      mj_scalar_t left_weight = kokkos_used_local_cut_line_weight_to_left(i);
      if(left_weight > local_sEpsilon) {
        // the weight of thread ii on cut.
        mj_scalar_t thread_ii_weight_on_cut =
          kokkos_used_thread_part_weight_work(i * 2 + 1) -
          kokkos_used_thread_part_weight_work(i * 2);

        if(thread_ii_weight_on_cut < left_weight) {
          // if left weight is bigger than threads weight on cut.
          local_kokkos_thread_cut_line_weight_to_put_left(i) =
            thread_ii_weight_on_cut;
        }
        else {
          // if thread's weight is bigger than space, then put only a portion.
          local_kokkos_thread_cut_line_weight_to_put_left(i) = left_weight;
        }
        left_weight -= thread_ii_weight_on_cut;
      }
      else {
        local_kokkos_thread_cut_line_weight_to_put_left(i) = 0;
      }      
    });
  }

  typedef typename Kokkos::TeamPolicy<typename mj_node_t::execution_space>::
    member_type member_type;

  Kokkos::TeamPolicy<typename mj_node_t::execution_space> policy_single(1, 1);

  if(num_cuts > 0) {

    Kokkos::parallel_for (policy_single, KOKKOS_LAMBDA(member_type team_member) {
      // this is a special case. If cutlines share the same coordinate,
      // their weights are equal. We need to adjust the ratio for that.
      for (mj_part_t i = num_cuts - 1; i > 0 ; --i) {
        if(ZOLTAN2_ABS(kokkos_current_concurrent_cut_coordinate(i) -
          kokkos_current_concurrent_cut_coordinate(i -1)) < local_sEpsilon) {
            local_kokkos_thread_cut_line_weight_to_put_left(i) -=
              local_kokkos_thread_cut_line_weight_to_put_left(i - 1);
        }
        local_kokkos_thread_cut_line_weight_to_put_left(i) =
          int ((local_kokkos_thread_cut_line_weight_to_put_left(i) +
          LEAST_SIGNIFICANCE) * SIGNIFICANCE_MUL) /
          mj_scalar_t(SIGNIFICANCE_MUL);
      }
    });
  }

  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, mj_part_t>
      (0, num_parts), KOKKOS_LAMBDA (const mj_part_t & i) {
    local_kokkos_thread_point_counts(i) = 0;
  });

  Kokkos::View<mj_lno_t *, device_t> record_total_on_cut(
    "track_on_cuts", 1);

  mj_lno_t coordinate_begin_index;
  Kokkos::parallel_reduce("Read coordinate_begin_index", 1,
    KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
    set_single =
      current_concurrent_work_part == 0 ? 0 :
        local_kokkos_part_xadj(current_concurrent_work_part - 1);
  }, coordinate_begin_index);

  mj_lno_t coordinate_end_index;
  Kokkos::parallel_reduce("Read coordinate_end_index", 1,
    KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
    set_single = local_kokkos_part_xadj(current_concurrent_work_part);;
  }, coordinate_end_index);


  mj_lno_t num_working_points = coordinate_end_index - coordinate_begin_index;

  // Found the loops below with atomics won't work properly if they run a team
  // over an index range of 0. So if num_teams is more than points this code
  // fails if we have a range such as (25, 25), or (999,25).
  // So to correct I'll clamp the max teams - probably doesn't make sense to
  // have teams run less than a warp anyways.
  int num_teams = SET_NUM_TEAMS_mj_create_new_partitions_clock;
  // TODO: need to check the system warp size - doesn't really matter
  // since this is just relevant for low coordinate count cases
  if(num_teams > num_working_points/32) {
    num_teams = num_working_points/32;
  }
  if(num_teams == 0) {
    num_teams = 1;
  }

  int stride = num_working_points / num_teams;
  if((num_working_points % num_teams) > 0) {
    stride += 1; // make sure we have coverage for the final points
  }

  Kokkos::TeamPolicy<typename mj_node_t::execution_space>
    policy(num_teams, Kokkos::AUTO());

  Kokkos::parallel_for (policy, KOKKOS_LAMBDA(member_type team_member) {
    mj_lno_t team_begin_index =
      coordinate_begin_index + stride * team_member.league_rank();
    mj_lno_t team_end_index = team_begin_index + stride;
    if(team_end_index > coordinate_end_index) {
      // the last team may have less work than the other teams
      team_end_index = coordinate_end_index;
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
      team_begin_index, team_end_index), [=] (mj_lno_t & ii) {
      mj_lno_t coordinate_index = local_kokkos_coordinate_permutations(ii);
      mj_part_t coordinate_assigned_place =
        local_kokkos_assigned_part_ids(coordinate_index);
      if(coordinate_assigned_place % 2 == 1) {
        Kokkos::atomic_add(&record_total_on_cut(0), 1);
      }
    });
  });

  mj_lno_t total_on_cut;
  Kokkos::parallel_reduce("Read single", 1,
    KOKKOS_LAMBDA(int dummy, int & set_single) {
    set_single = record_total_on_cut(0);
    record_total_on_cut(0) = 0;
  }, total_on_cut);

  Kokkos::View<mj_lno_t *, device_t> track_on_cuts(
    "track_on_cuts", total_on_cut);

  Kokkos::parallel_for (policy, KOKKOS_LAMBDA(member_type team_member) {
    mj_lno_t team_begin_index =
      coordinate_begin_index + stride * team_member.league_rank();
    mj_lno_t team_end_index = team_begin_index + stride;
    if(team_end_index > coordinate_end_index) {
      // the last team may have less work than the other teams
      team_end_index = coordinate_end_index;
    }

    // First collect the number of assignments in our block for each part
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
      team_begin_index, team_end_index), [=] (mj_lno_t & ii) {
      mj_lno_t coordinate_index = local_kokkos_coordinate_permutations(ii);
      mj_part_t coordinate_assigned_place =
        local_kokkos_assigned_part_ids(coordinate_index);
      mj_part_t coordinate_assigned_part = coordinate_assigned_place / 2;
      if(coordinate_assigned_place % 2 == 0) {
        Kokkos::atomic_add(
          &local_kokkos_thread_point_counts(coordinate_assigned_part), 1);
        local_kokkos_assigned_part_ids(coordinate_index) =
          coordinate_assigned_part;
      }
      else {
        // fill a tracking array so we can process these slower points
        // in next cycle
        mj_lno_t set_index =
          Kokkos::atomic_fetch_add(&record_total_on_cut(0), 1);
        track_on_cuts(set_index) = ii;
      }
    });
  });

 
  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (0, 1),
    KOKKOS_LAMBDA (const int & dummy) {
    for(int j = 0; j < total_on_cut; ++j) {
      int ii = track_on_cuts(j);
      mj_lno_t coordinate_index = local_kokkos_coordinate_permutations(ii);
      mj_scalar_t coordinate_weight = local_kokkos_mj_uniform_weights(0) ? 1 :
        local_kokkos_mj_weights(coordinate_index,0);
      mj_part_t coordinate_assigned_place =
        local_kokkos_assigned_part_ids(coordinate_index);
      mj_part_t coordinate_assigned_part = coordinate_assigned_place / 2;
      // if it is on the cut.
      if(local_distribute_points_on_cut_lines &&
        local_kokkos_thread_cut_line_weight_to_put_left(
          coordinate_assigned_part) > local_sEpsilon) {
        // if the rectilinear partitioning is allowed,
        // and the thread has still space to put on the left of the cut
        // then thread puts the vertex to left.
        local_kokkos_thread_cut_line_weight_to_put_left(
          coordinate_assigned_part) -= coordinate_weight;
        // if putting the vertex to left increased the weight more
        // than expected, and if the next cut is on the same coordinate,
        // then we need to adjust how much weight next cut puts to its left as
        // well, in order to take care of the imbalance.
        if(local_kokkos_thread_cut_line_weight_to_put_left(
            coordinate_assigned_part) < 0 && coordinate_assigned_part <
            num_cuts - 1 &&
            ZOLTAN2_ABS(kokkos_current_concurrent_cut_coordinate(
            coordinate_assigned_part+1) -
            kokkos_current_concurrent_cut_coordinate(
            coordinate_assigned_part)) < local_sEpsilon)
        {
          local_kokkos_thread_cut_line_weight_to_put_left(
            coordinate_assigned_part + 1) +=
            local_kokkos_thread_cut_line_weight_to_put_left(
            coordinate_assigned_part);
        }
        ++local_kokkos_thread_point_counts(coordinate_assigned_part);
        local_kokkos_assigned_part_ids(coordinate_index) =
          coordinate_assigned_part;
      }
      else {
        // if there is no more space on the left, put the coordinate to the
        // right of the cut.
        ++coordinate_assigned_part;
        // this while loop is necessary when a line is partitioned into more
        // than 2 parts.
        while(local_distribute_points_on_cut_lines &&
          coordinate_assigned_part < num_cuts)
        {
          // traverse all the cut lines having the same partitiong
          if(ZOLTAN2_ABS(kokkos_current_concurrent_cut_coordinate(
            coordinate_assigned_part) -
            kokkos_current_concurrent_cut_coordinate(
              coordinate_assigned_part - 1)) < local_sEpsilon)
          {
            // if line has enough space on left, put it there.
            if(local_kokkos_thread_cut_line_weight_to_put_left(
              coordinate_assigned_part) > local_sEpsilon &&
              local_kokkos_thread_cut_line_weight_to_put_left(
                coordinate_assigned_part) >=
                ZOLTAN2_ABS(local_kokkos_thread_cut_line_weight_to_put_left(
                  coordinate_assigned_part) - coordinate_weight))
            {
              local_kokkos_thread_cut_line_weight_to_put_left(
                coordinate_assigned_part) -= coordinate_weight;
              // Again if it put too much on left of the cut,
              // update how much the next cut sharing the same coordinate will
              // put to its left.
              if(local_kokkos_thread_cut_line_weight_to_put_left(
                coordinate_assigned_part) < 0 &&
                coordinate_assigned_part < num_cuts - 1 &&
                ZOLTAN2_ABS(kokkos_current_concurrent_cut_coordinate(
                  coordinate_assigned_part+1) -
                kokkos_current_concurrent_cut_coordinate(
                  coordinate_assigned_part)) < local_sEpsilon)
              {
                local_kokkos_thread_cut_line_weight_to_put_left(
                  coordinate_assigned_part + 1) +=
                  local_kokkos_thread_cut_line_weight_to_put_left(
                    coordinate_assigned_part);
              } 
              break;
            }    
          }
          else {
            break;
          }
          ++coordinate_assigned_part;
        }
        local_kokkos_thread_point_counts(coordinate_assigned_part) += 1;
        local_kokkos_assigned_part_ids(coordinate_index) =
          coordinate_assigned_part;
      }
    }
  });

  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, mj_part_t>
      (0, num_parts), KOKKOS_LAMBDA (const mj_part_t & j) {
    kokkos_out_part_xadj(j) = local_kokkos_thread_point_counts(j);
  });

  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, mj_part_t>
      (0, num_parts), KOKKOS_LAMBDA (const mj_part_t & j) {
    local_kokkos_thread_point_counts(j) = 0;
  });

  // TODO: How do we efficiently parallelize this form? 
  // Copy first?
  Kokkos::parallel_for (policy_single, KOKKOS_LAMBDA(member_type team_member) {
    for(mj_part_t j = 1; j < num_parts; ++j) {
      kokkos_out_part_xadj(j) += kokkos_out_part_xadj(j - 1);
      local_kokkos_thread_point_counts(j) += kokkos_out_part_xadj(j - 1);
    }
  });

  Kokkos::parallel_for (policy, KOKKOS_LAMBDA(member_type team_member) {
    mj_lno_t team_begin_index =
      coordinate_begin_index + stride * team_member.league_rank();
    mj_lno_t team_end_index = team_begin_index + stride;
    if(team_end_index > coordinate_end_index) {
      // the last team may have less work than the other teams
      team_end_index = coordinate_end_index;
    }

    // First collect the number of assignments in our block for each part
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, team_begin_index, team_end_index),
      [=] (mj_lno_t & ii) {
        mj_lno_t i = local_kokkos_coordinate_permutations(ii);
        mj_part_t p = local_kokkos_assigned_part_ids(i);

        // We need to atomically read and then increment the write index
        mj_lno_t idx =
          Kokkos::atomic_fetch_add(&local_kokkos_thread_point_counts(p), 1);
        // The actual write is to a single slot so needs no atomic
        local_kokkos_new_coordinate_permutations(
          coordinate_begin_index + idx) = i;
    });
  });

  clock_mj_create_new_partitions.stop();
}

/*! \brief Function that calculates the new coordinates for the cut lines.
 * Function is called inside the parallel region.
 * \param num_total_part is the sum of number of cutlines and number of parts.
 * Simply it is 2*P - 1.
 * \param num_cuts is the number of cut lines. P - 1.
 * \param max_coordinate is the maximum coordinate in the current range of
 * coordinates and in the current dimension.
 * \param min_coordinate is the maximum coordinate in the current range of
 * coordinates and in the current dimension.
 * \param global_total_weight is the global total weight in the current range of
 * coordinates.
 * \param used_imbalance_tolerance is the maximum allowed imbalance ratio.
 * \param current_global_part_weights is the array holding the weight of parts.
 * Assumes there are 2*P - 1 parts (cut lines are seperate parts).
 * \param current_local_part_weights is the local totalweight of the processor.
 * \param current_part_target_weights are the desired cumulative part ratios,
 * sized P.
 * \param current_cut_line_determined is the boolean array to determine if the
 * correct position for a cut line is found.
 * \param current_cut_coordinates is the array holding the coordinates of each
 * cut line. Sized P - 1.
 * \param current_cut_upper_bounds is the array holding the upper bound
 * coordinate for each cut line. Sized P - 1.
 * \param current_cut_lower_bounds is the array holding the lower bound
 * coordinate for each cut line. Sized P - 1.
 * \param current_global_left_closest_points is the array holding the closest
 * points to the cut lines from left.
 * \param current_global_right_closest_points is the array holding the closest
 * points to the cut lines from right.
 * \param current_cut_lower_bound_weights is the array holding the weight of the
 * parts at the left of lower bound coordinates.
 * \param current_cut_upper_weights is the array holding the weight of the parts
 * at the left of upper bound coordinates.
 * \param new_current_cut_coordinates is the work array, sized P - 1.
 * \param current_part_cut_line_weight_ratio holds how much weight of the
 * coordinates on the cutline should be put on left side.
 * \param rectilinear_cut_count is the count of cut lines whose balance can be
 * achived via distributing the points in same coordinate to different parts.
 * \param my_num_incomplete_cut is the number of cutlines whose position has not
 * been determined yet. For K > 1 it is the count in a single part (whose cut
 * lines are determined).
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t,
  mj_node_t>::mj_get_new_cut_coordinates(
  mj_part_t current_concurrent_num_parts,
  mj_part_t kk,
  const size_t &num_total_part,
  const mj_part_t &num_cuts,
  const mj_scalar_t &used_imbalance_tolerance,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_global_part_weights,
  Kokkos::View<const mj_scalar_t *, device_t>
    kokkos_current_local_part_weights,
  Kokkos::View<const mj_scalar_t *, device_t>
    kokkos_current_part_target_weights,
  Kokkos::View<bool *, device_t> kokkos_current_cut_line_determined,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_coordinates,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_upper_bounds,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_lower_bounds,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_current_global_left_closest_points,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_current_global_right_closest_points,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_lower_bound_weights,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_upper_weights,
  Kokkos::View<mj_scalar_t *, device_t> kokkos_new_current_cut_coordinates,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_current_part_cut_line_weight_to_put_left,
  Kokkos::View<mj_part_t *, device_t> view_rectilinear_cut_count)
{
  auto local_sEpsilon = sEpsilon;
  auto local_distribute_points_on_cut_lines = distribute_points_on_cut_lines;
  auto local_kokkos_global_rectilinear_cut_weight =
    kokkos_global_rectilinear_cut_weight;
  auto local_kokkos_process_rectilinear_cut_weight =
    kokkos_process_rectilinear_cut_weight;
  auto local_kokkos_my_incomplete_cut_count =
    kokkos_my_incomplete_cut_count;
  auto local_kokkos_global_min_max_coord_total_weight =
    kokkos_global_min_max_coord_total_weight;

  
  // TODO: Work on this pattern to optimize it
  // Note for a 22 part system I tried removing the outer loop
  // and doing each sub loop as a simple parallel_for over num_cuts.
  // But that was about twice as slow (10ms) as the current form (5ms)
  // so I think the overhead of laucnhing the new global parallel kernels
  // is costly. This form is just running one team so effectively using
  // a single warp to process the cuts. I expect with a lot of parts this
  // might need changing.
  Kokkos::TeamPolicy<typename mj_node_t::execution_space>
    policy2(1, Kokkos::AUTO());
  typedef typename Kokkos::TeamPolicy<typename mj_node_t::execution_space>::
    member_type member_type;
  Kokkos::parallel_for (policy2, KOKKOS_LAMBDA(member_type team_member) {

    mj_scalar_t min_coordinate =
      local_kokkos_global_min_max_coord_total_weight(kk);
    mj_scalar_t max_coordinate =
      local_kokkos_global_min_max_coord_total_weight(
      kk + current_concurrent_num_parts);
    mj_scalar_t global_total_weight =
      local_kokkos_global_min_max_coord_total_weight(
      kk + current_concurrent_num_parts * 2);

    Kokkos::parallel_for(Kokkos::TeamThreadRange (team_member, num_cuts),
      [=] (int & i) {
      // if left and right closest points are not set yet,
      // set it to the cut itself.
      if(min_coordinate -
        kokkos_current_global_left_closest_points(i) > local_sEpsilon) {
        kokkos_current_global_left_closest_points(i) =
          kokkos_current_cut_coordinates(i);
      }
      if(kokkos_current_global_right_closest_points(i) -
        max_coordinate > local_sEpsilon) {
        kokkos_current_global_right_closest_points(i) =
          kokkos_current_cut_coordinates(i);
      }
    });
    team_member.team_barrier(); // for end of Kokkos::TeamThreadRange

    Kokkos::parallel_for(Kokkos::TeamThreadRange (team_member, num_cuts),
      [=] (int & i) {
      // seen weight in the part
      mj_scalar_t seen_weight_in_part = 0;
      // expected weight for part.
      mj_scalar_t expected_weight_in_part = 0;
      // imbalance for the left and right side of the cut.
      mj_scalar_t imbalance_on_left = 0, imbalance_on_right = 0;
      if(local_distribute_points_on_cut_lines) {
        // init the weight on the cut.
        local_kokkos_global_rectilinear_cut_weight(i) = 0;
        local_kokkos_process_rectilinear_cut_weight(i) = 0;
      }
      bool bContinue = false;
      // if already determined at previous iterations,
      // then just write the coordinate to new array, and proceed.
      if(kokkos_current_cut_line_determined(i)) {
        kokkos_new_current_cut_coordinates(i) =
          kokkos_current_cut_coordinates(i);
        bContinue = true;
      }
      if(!bContinue) {
        //current weight of the part at the left of the cut line.
        seen_weight_in_part = kokkos_current_global_part_weights(i * 2);

        //expected ratio
        expected_weight_in_part = kokkos_current_part_target_weights(i);

       //leftImbalance = imbalanceOf(seenW, globalTotalWeight, expected);
        imbalance_on_left = imbalanceOf2(seen_weight_in_part,
          expected_weight_in_part);
        // rightImbalance = imbalanceOf(globalTotalWeight - seenW,
        // globalTotalWeight, 1 - expected);
        imbalance_on_right = imbalanceOf2(global_total_weight -
          seen_weight_in_part, global_total_weight - expected_weight_in_part);
        bool is_left_imbalance_valid = ZOLTAN2_ABS(imbalance_on_left) -
          used_imbalance_tolerance < local_sEpsilon ;
        bool is_right_imbalance_valid = ZOLTAN2_ABS(imbalance_on_right) -
          used_imbalance_tolerance < local_sEpsilon;
        //if the cut line reaches to desired imbalance.
        if(is_left_imbalance_valid && is_right_imbalance_valid) {
          kokkos_current_cut_line_determined(i) = true;
          Kokkos::atomic_add(&local_kokkos_my_incomplete_cut_count(kk), -1);
          kokkos_new_current_cut_coordinates(i) =
            kokkos_current_cut_coordinates(i);
        }
        else if(imbalance_on_left < 0) {
          //if left imbalance < 0 then we need to move the cut to right.
          if(local_distribute_points_on_cut_lines) {
            // if it is okay to distribute the coordinate on
            // the same coordinate to left and right.
            // then check if we can reach to the target weight by including the
            // coordinates in the part.
            if (kokkos_current_global_part_weights(i * 2 + 1) ==
              expected_weight_in_part) {
              // if it is we are done.
              kokkos_current_cut_line_determined(i) = true;
              Kokkos::atomic_add(&local_kokkos_my_incomplete_cut_count(kk), -1);

              //then assign everything on the cut to the left of the cut.
              kokkos_new_current_cut_coordinates(i) =
                kokkos_current_cut_coordinates(i);
              //for this cut all the weight on cut will be put to left.
              kokkos_current_part_cut_line_weight_to_put_left(i) =
                kokkos_current_local_part_weights(i * 2 + 1) -
                kokkos_current_local_part_weights(i * 2);
              bContinue = true;
            }
            else if (kokkos_current_global_part_weights(i * 2 + 1) >
              expected_weight_in_part) {
              // if the weight is larger than the expected weight,
              // then we need to distribute some points to left, some to right.
              kokkos_current_cut_line_determined(i) = true;
              Kokkos::atomic_add(&view_rectilinear_cut_count(0), 1);

              // increase the num cuts to be determined with rectilinear
              // partitioning.
              Kokkos::atomic_add(&local_kokkos_my_incomplete_cut_count(kk), -1);
              kokkos_new_current_cut_coordinates(i) =
                kokkos_current_cut_coordinates(i);
              local_kokkos_process_rectilinear_cut_weight[i] =
                kokkos_current_local_part_weights(i * 2 + 1) -
                kokkos_current_local_part_weights(i * 2);
              bContinue = true;
            }
          }

          if(!bContinue) {

            // we need to move further right,so set lower bound to current line,
            // and shift it to the closes point from right.
            kokkos_current_cut_lower_bounds(i) =
              kokkos_current_global_right_closest_points(i);
            //set the lower bound weight to the weight we have seen.
            kokkos_current_cut_lower_bound_weights(i) = seen_weight_in_part;

            // compare the upper bound with what has been found in the
            // last iteration.
            // we try to make more strict bounds for the cut here.
            for (mj_part_t ii = i + 1; ii < num_cuts ; ++ii) {
              mj_scalar_t p_weight = kokkos_current_global_part_weights(ii * 2);
              mj_scalar_t line_weight =
                kokkos_current_global_part_weights(ii * 2 + 1);
              if(p_weight >= expected_weight_in_part) {
                // if a cut on the right has the expected weight, then we found
                // our cut position. Set up and low coordiantes to this
                // new cut coordinate, but we need one more iteration to
                // finalize the cut position, as wee need to update the part ids.
                if(p_weight == expected_weight_in_part){
                  kokkos_current_cut_upper_bounds(i) =
                    kokkos_current_cut_coordinates(ii);
                  kokkos_current_cut_upper_weights(i) = p_weight;
                  kokkos_current_cut_lower_bounds(i) =
                    kokkos_current_cut_coordinates(ii);
                  kokkos_current_cut_lower_bound_weights(i) = p_weight;
                } else if (p_weight < kokkos_current_cut_upper_weights(i)) {
                  // if a part weight is larger then my expected weight,
                  // but lower than my upper bound weight, update upper bound.
                  kokkos_current_cut_upper_bounds(i) =
                    kokkos_current_global_left_closest_points(ii);
                  kokkos_current_cut_upper_weights(i) = p_weight;
                }
                break;
              }
              // if comes here then pw < ew
              // then compare the weight against line weight.
              if(line_weight >= expected_weight_in_part) {
                // if the line is larger than the expected weight, then we need
                // to reach to the balance by distributing coordinates on
                // this line.
                kokkos_current_cut_upper_bounds(i) =
                  kokkos_current_cut_coordinates(ii);
                kokkos_current_cut_upper_weights(i) = line_weight;
                kokkos_current_cut_lower_bounds(i) =
                  kokkos_current_cut_coordinates(ii);
                kokkos_current_cut_lower_bound_weights(i) = p_weight;
                break;
              }
              // if a stricter lower bound is found,
              // update the lower bound.
              if (p_weight <= expected_weight_in_part && p_weight >=
                kokkos_current_cut_lower_bound_weights(i)){
                kokkos_current_cut_lower_bounds(i) =
                  kokkos_current_global_right_closest_points(ii);
                kokkos_current_cut_lower_bound_weights(i) = p_weight;
              }
            }

            mj_scalar_t new_cut_position = 0;
            this->mj_calculate_new_cut_position(
              kokkos_current_cut_upper_bounds(i),
              kokkos_current_cut_lower_bounds(i),
              kokkos_current_cut_upper_weights(i),
              kokkos_current_cut_lower_bound_weights(i),
              expected_weight_in_part, new_cut_position);

            // if cut line does not move significantly.
            // then finalize the search.
            if (ZOLTAN2_ABS(kokkos_current_cut_coordinates(i) -
              new_cut_position) < local_sEpsilon) {
              kokkos_current_cut_line_determined(i) = true;
              Kokkos::atomic_add(
                &local_kokkos_my_incomplete_cut_count(kk), -1);

              //set the cut coordinate and proceed.
              kokkos_new_current_cut_coordinates(i) =
                kokkos_current_cut_coordinates(i);
            } else {
              kokkos_new_current_cut_coordinates(i) = new_cut_position;
            }
          } // bContinue
        } else {
          // need to move the cut line to left.
          // set upper bound to current line.
          kokkos_current_cut_upper_bounds(i) =
            kokkos_current_global_left_closest_points(i);
          kokkos_current_cut_upper_weights(i) =
            seen_weight_in_part;
          // compare the current cut line weights with
          // previous upper and lower bounds.
          for (int ii = i - 1; ii >= 0; --ii) {
            mj_scalar_t p_weight =
              kokkos_current_global_part_weights(ii * 2);
            mj_scalar_t line_weight =
              kokkos_current_global_part_weights(ii * 2 + 1);
            if(p_weight <= expected_weight_in_part) {
              if(p_weight == expected_weight_in_part) {
                // if the weight of the part is my expected weight
                // then we find the solution.
                kokkos_current_cut_upper_bounds(i) =
                  kokkos_current_cut_coordinates(ii);
                kokkos_current_cut_upper_weights(i) = p_weight;
                kokkos_current_cut_lower_bounds(i) =
                  kokkos_current_cut_coordinates(ii);
                kokkos_current_cut_lower_bound_weights(i) = p_weight;
              }
              else if (p_weight > kokkos_current_cut_lower_bound_weights(i)) {
                // if found weight is bigger than the lower bound
                // then update the lower bound.
                kokkos_current_cut_lower_bounds(i) =
                  kokkos_current_global_right_closest_points(ii);
                kokkos_current_cut_lower_bound_weights(i) = p_weight;

                // at the same time, if weight of line is bigger than the
                // expected weight, then update the upper bound as well.
                // in this case the balance will be obtained by distributing
                // weights on this cut position.
                if(line_weight > expected_weight_in_part){
                  kokkos_current_cut_upper_bounds(i) =
                    kokkos_current_global_right_closest_points(ii);
                  kokkos_current_cut_upper_weights(i) = line_weight;
                }
              }
              break;
            }
            // if the weight of the cut on the left is still bigger than
            // my weight, and also if the weight is smaller than the current
            // upper weight, or if the weight is equal to current upper
            // weight, but on the left of the upper weight, then update
            // upper bound.
            if (p_weight >= expected_weight_in_part &&
              (p_weight < kokkos_current_cut_upper_weights(i) ||
              (p_weight == kokkos_current_cut_upper_weights(i) &&
                kokkos_current_cut_upper_bounds(i) >
                  kokkos_current_global_left_closest_points(ii)))) {
              kokkos_current_cut_upper_bounds(i) =
                kokkos_current_global_left_closest_points(ii);
              kokkos_current_cut_upper_weights(i) = p_weight;
            }
          }
          mj_scalar_t new_cut_position = 0;
          this->mj_calculate_new_cut_position(
            kokkos_current_cut_upper_bounds(i),
            kokkos_current_cut_lower_bounds(i),
            kokkos_current_cut_upper_weights(i),
            kokkos_current_cut_lower_bound_weights(i),
            expected_weight_in_part,
            new_cut_position);

            // if cut line does not move significantly.
            if (ZOLTAN2_ABS(kokkos_current_cut_coordinates(i) -
              new_cut_position) < local_sEpsilon) {
              kokkos_current_cut_line_determined(i) = true;
              Kokkos::atomic_add(
                &local_kokkos_my_incomplete_cut_count(kk), -1);
              //set the cut coordinate and proceed.
              kokkos_new_current_cut_coordinates(i) =
                kokkos_current_cut_coordinates(i);
            } else {
              kokkos_new_current_cut_coordinates(i) =
                new_cut_position;
            }
          }
        }; // bContinue
      });

      team_member.team_barrier(); // for end of Kokkos::TeamThreadRange

      // TODO: This may not be necessary anymore?
        
      // This unnecessary bracket works around a compiler bug in NVCC
      // when enabling OpenMP as well
      {

        Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
          if(view_rectilinear_cut_count(0) > 0) {
          // try
          {
            // For cuda initial testing reduce this to a form ok for device
            // TODO need to opimize this but now we're just assuming on thread
            for(int n = 0; n <
              (int) local_kokkos_process_rectilinear_cut_weight.size(); ++n) {
              local_kokkos_global_rectilinear_cut_weight(n) =
              local_kokkos_process_rectilinear_cut_weight(n);
            }
            /*
              Teuchos::scan<int,mj_scalar_t>(
                *comm, Teuchos::REDUCE_SUM,
                num_cuts,
                // TODO: Note this is refactored but needs to be improved
                // to avoid the use of direct data() ptr completely.
                local_kokkos_process_rectilinear_cut_weight.data(),
                local_kokkos_global_rectilinear_cut_weight.data());
           */
          }
          //  Z2_THROW_OUTSIDE_ERROR(*(this->mj_env))

          for (mj_part_t i = 0; i < num_cuts; ++i) {
            // if cut line weight to be distributed.
            if(local_kokkos_global_rectilinear_cut_weight(i) > 0) {
              // expected weight to go to left of the cut.
              mj_scalar_t expected_part_weight =
                kokkos_current_part_target_weights(i);
              // the weight that should be put to left of the cut.
              mj_scalar_t necessary_weight_on_line_for_left =
                expected_part_weight -
                kokkos_current_global_part_weights(i * 2);

              // the weight of the cut in the process
              mj_scalar_t my_weight_on_line =
                local_kokkos_process_rectilinear_cut_weight(i);

              // the sum of the cut weights upto this process,
              // including the weight of this process.
              mj_scalar_t weight_on_line_upto_process_inclusive =
                local_kokkos_global_rectilinear_cut_weight(i);
              // the space on the left side of the cut after all processes
              // before this process (including this process)
              // puts their weights on cut to left.
              mj_scalar_t space_to_put_left =
                necessary_weight_on_line_for_left -
                weight_on_line_upto_process_inclusive;
              // add my weight to this space to find out how much space
              // is left to me.
              mj_scalar_t space_left_to_me =
                space_to_put_left + my_weight_on_line;

              /*
              cout << "expected_part_weight:" << expected_part_weight
                << " necessary_weight_on_line_for_left:"
                << necessary_weight_on_line_for_left
                << " my_weight_on_line" << my_weight_on_line
                << " weight_on_line_upto_process_inclusive:"
                << weight_on_line_upto_process_inclusive
                << " space_to_put_left:" << space_to_put_left
                << " space_left_to_me" << space_left_to_me << endl;
               */

              if(space_left_to_me < 0) {
                // space_left_to_me is negative and i dont need to put
                // anything to left.
                kokkos_current_part_cut_line_weight_to_put_left(i) = 0;
              }
              else if(space_left_to_me >= my_weight_on_line) {
                // space left to me is bigger than the weight of the
                // processor on cut.
                // so put everything to left.
                kokkos_current_part_cut_line_weight_to_put_left(i) =
                  my_weight_on_line;
                // cout << "setting current_part_cut_line_weight_to_put_left
                // to my_weight_on_line:" << my_weight_on_line << endl;
              }
              else {
                // put only the weight as much as the space.
                kokkos_current_part_cut_line_weight_to_put_left(i) =
                  space_left_to_me;
                // cout << "setting current_part_cut_line_weight_to_put_left
                // to space_left_to_me:" << space_left_to_me << endl;
              }
            }
          }
          view_rectilinear_cut_count(0) = 0;
        }
      });
      team_member.team_barrier(); // for end of Kokkos::single
    } // TODO: This may not be necessary anymore? See comment above
  });
}

/*! \brief Function fills up the num_points_in_all_processor_parts, so that
 * it has the number of coordinates in each processor of each part.
 * to access how many points processor i has on part j,
 * num_points_in_all_processor_parts[i * num_parts + j].
 * \param num_procs is the number of processor attending to migration operation.
 * \param num_parts is the number of parts that exist in current partitioning.
 * \param num_points_in_all_processor_parts is the output array that holds
 * the number of coordinates in each part in each processor.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  get_processor_num_points_in_parts(
  mj_part_t num_procs,
  mj_part_t num_parts,
  mj_gno_t *&num_points_in_all_processor_parts)
{
  // initially allocation_size is num_parts
  size_t allocation_size = num_parts * (num_procs + 1);

  // this will be output
  // holds how many each processor has in each part.
  // last portion is the sum of all processor points in each part.

  // allocate memory for the local num coordinates in each part.
  mj_gno_t *num_local_points_in_each_part_to_reduce_sum =
    allocMemory<mj_gno_t>(allocation_size);

  // this is the portion of the memory which will be used
  // at the summation to obtain total number of processors' points in each part.
  mj_gno_t *my_local_points_to_reduce_sum =
    num_local_points_in_each_part_to_reduce_sum + num_procs * num_parts;

  // this is the portion of the memory where each stores its local number.
  // this information is needed by other processors.
  mj_gno_t *my_local_point_counts_in_each_art =
    num_local_points_in_each_part_to_reduce_sum + this->myRank * num_parts;

  // initialize the array with 0's.
  memset(num_local_points_in_each_part_to_reduce_sum, 0,
    sizeof(mj_gno_t)*allocation_size);

  //write the number of coordinates in each part.
  for (mj_part_t i = 0; i < num_parts; ++i) {
    mj_lno_t part_begin_index = 0;
    if (i > 0) {
      part_begin_index = this->kokkos_new_part_xadj(i - 1);
    }
    mj_lno_t part_end_index = this->kokkos_new_part_xadj(i);
    my_local_points_to_reduce_sum[i] = part_end_index - part_begin_index;
  }

  // copy the local num parts to the last portion of array, so that this portion
  // will represent the global num points in each part after the reduction.
  memcpy (my_local_point_counts_in_each_art, my_local_points_to_reduce_sum,
    sizeof(mj_gno_t) * (num_parts) );

  // reduceAll operation.
  // the portion that belongs to a processor with index p
  // will start from myRank * num_parts.
  // the global number of points will be held at the index
  try{
    reduceAll<int, mj_gno_t>(
      *(this->comm),
      Teuchos::REDUCE_SUM,
      allocation_size,
      num_local_points_in_each_part_to_reduce_sum,
      num_points_in_all_processor_parts);
  }
  Z2_THROW_OUTSIDE_ERROR(*(this->mj_env))
  freeArray<mj_gno_t>(num_local_points_in_each_part_to_reduce_sum);
}



/*! \brief Function checks if should do migration or not.
 * It returns true to point that migration should be done when
 * -migration_reduce_all_population are higher than a predetermined value
 * -num_coords_for_last_dim_part that left for the last dimension partitioning
 * is less than a predetermined value - the imbalance of the processors on the
 * parts are higher than given threshold.
 * \param migration_reduce_all_population is the multiplication of the number of
 * reduceall operations estimated and the number of processors.
 * \param num_coords_for_last_dim_part is the estimated number of coordinates in
 * a part per processor in the last dimension partitioning.
 * \param num_procs is number of processors attending to migration operation.
 * \param num_parts is number of parts that exist in the current partitioning.
 * \param num_points_in_all_processor_parts is the input array that holds
 * the number of coordinates in each part in each processor.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
bool AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_check_to_migrate(
  size_t migration_reduce_all_population,
  mj_lno_t num_coords_for_last_dim_part,
  mj_part_t num_procs,
  mj_part_t num_parts,
  mj_gno_t *num_points_in_all_processor_parts)
{
  // if reduce all count and population in the last dim is too high
  if (migration_reduce_all_population > FUTURE_REDUCEALL_CUTOFF) {
    return true;
  }

  // if the work in a part per processor in the last dim is too low.
  if (num_coords_for_last_dim_part < MIN_WORK_LAST_DIM) {
    return true;
  }
  
  // if migration is to be checked and the imbalance is too high
  if (this->check_migrate_avoid_migration_option == 0) {
    double global_imbalance = 0;
    // global shift to reach the sum of coordiante count in each part.
    size_t global_shift = num_procs * num_parts;

    for (mj_part_t ii = 0; ii < num_procs; ++ii) {
      for (mj_part_t i = 0; i < num_parts; ++i) {
       double ideal_num = num_points_in_all_processor_parts[global_shift + i]
         / double(num_procs);

       global_imbalance += ZOLTAN2_ABS(ideal_num -
         num_points_in_all_processor_parts[ii * num_parts + i]) /  (ideal_num);
      }
    }
    global_imbalance /= num_parts;
    global_imbalance /= num_procs;

    if(global_imbalance <= this->minimum_migration_imbalance) {
      return false;
    }
    else {
      return true;
    }
  }
  else {
    // if migration is forced
    return true;
  }
}

/*! \brief Function fills up coordinate_destinations is the output array
 * that holds which part each coordinate should be sent.
 * \param num_parts is the number of parts that exist in the current
 * partitioning.
 * \param part_assignment_proc_begin_indices ([i]) points to the first processor
 * index that part i will be sent to.
 * \param processor_chains_in_parts the array that holds the linked list
 * structure, started from part_assignment_proc_begin_indices ([i]).
 * \param send_count_to_each_proc array array storing the number of points to
 * be sent to each part.
 * \param coordinate_destinations is the output array that holds which part
 * each coordinate should be sent.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  assign_send_destinations(
  mj_part_t num_parts,
  mj_part_t *part_assignment_proc_begin_indices,
  mj_part_t *processor_chains_in_parts,
  mj_lno_t *send_count_to_each_proc,
  int *coordinate_destinations) {

  for (mj_part_t p = 0; p < num_parts; ++p) {
    mj_lno_t part_begin = 0;
    if (p > 0) part_begin = this->kokkos_new_part_xadj(p - 1);
    mj_lno_t part_end = this->kokkos_new_part_xadj(p);
    // get the first part that current processor will send its part-p.
    mj_part_t proc_to_sent = part_assignment_proc_begin_indices[p];
    // initialize how many point I sent to this processor.
    mj_lno_t num_total_send = 0;
    for (mj_lno_t j=part_begin; j < part_end; j++) {
      mj_lno_t local_ind = this->kokkos_new_coordinate_permutations(j);
      while (num_total_send >= send_count_to_each_proc[proc_to_sent]) {
        // then get the next processor to send the points in part p.
        num_total_send = 0;
        // assign new processor to part_assign_begin[p]
        part_assignment_proc_begin_indices[p] =
          processor_chains_in_parts[proc_to_sent];
        // remove the previous processor
        processor_chains_in_parts[proc_to_sent] = -1;
        // choose the next processor as the next one to send.
        proc_to_sent = part_assignment_proc_begin_indices[p];
      }
      // write the gno index to corresponding position in sendBuf.
      coordinate_destinations[local_ind] = proc_to_sent;
      ++num_total_send;
    }
  }
}

/*! \brief Function fills up coordinate_destinations is the output array
 * that holds which part each coordinate should be sent.
 * \param num_points_in_all_processor_parts is the array holding the num points
 * in each part in each proc.
 * \param num_parts is the number of parts that exist in the
 * current partitioning.
 * \param num_procs is the number of processor attending to migration operation.
 * \param send_count_to_each_proc array array storing the number of points to
 * be sent to each part.
 * \param processor_ranks_for_subcomm is the ranks of the processors that will
 * be in the subcommunicator with me.
 * \param next_future_num_parts_in_parts is the vector, how many more parts
 * each part will be divided into in the future.
 * \param out_part_index is the index of the part to which the processor
 * is assigned.
 * \param output_part_numbering_begin_index is how much the numbers should
 * be shifted when numbering the result parts.
 * \param coordinate_destinations is the output array that holds which part
 * each coordinate should be sent.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_assign_proc_to_parts(
  mj_gno_t * num_points_in_all_processor_parts,
  mj_part_t num_parts,
  mj_part_t num_procs,
  mj_lno_t *send_count_to_each_proc,
  std::vector<mj_part_t> &processor_ranks_for_subcomm,
  std::vector<mj_part_t> *next_future_num_parts_in_parts,
  mj_part_t &out_part_index,
  mj_part_t &output_part_numbering_begin_index,
  int * coordinate_destinations) {

  mj_gno_t *global_num_points_in_parts =
    num_points_in_all_processor_parts + num_procs * num_parts;
  mj_part_t *num_procs_assigned_to_each_part =
    allocMemory<mj_part_t>(num_parts);

  // boolean variable if the process finds its part to be assigned.
  bool did_i_find_my_group = false;

  mj_part_t num_free_procs = num_procs;
  mj_part_t minimum_num_procs_required_for_rest_of_parts = num_parts - 1;

  double max_imbalance_difference = 0;
  mj_part_t max_differing_part = 0;

  // find how many processor each part requires.
  for (mj_part_t i=0; i < num_parts; i++) {

    // scalar portion of the required processors
    double scalar_required_proc = num_procs *
      (double (global_num_points_in_parts[i]) /
      double (this->num_global_coords));

    // round it to closest integer.
    mj_part_t required_proc =
      static_cast<mj_part_t> (0.5 + scalar_required_proc);

    // if assigning the required num procs, creates problems for the rest
    // of the parts, then only assign {num_free_procs -
    // (minimum_num_procs_required_for_rest_of_parts)} procs to this part.
    if (num_free_procs -
      required_proc < minimum_num_procs_required_for_rest_of_parts)  {
        required_proc = num_free_procs -
          (minimum_num_procs_required_for_rest_of_parts);
      }

      // reduce the free processor count
      num_free_procs -= required_proc;

      // reduce the free minimum processor count required for the rest of the
      // part by 1.
      --minimum_num_procs_required_for_rest_of_parts;

      // part (i) is assigned to (required_proc) processors.
      num_procs_assigned_to_each_part[i] = required_proc;

      // because of the roundings some processors might be left as unassigned.
      // we want to assign those processors to the part with most imbalance.
      // find the part with the maximum imbalance here.
      double imbalance_wrt_ideal =
        (scalar_required_proc - required_proc) /  required_proc;
      if (imbalance_wrt_ideal > max_imbalance_difference){
        max_imbalance_difference = imbalance_wrt_ideal;
        max_differing_part = i;
      }
    }

    // assign extra processors to the part with maximum imbalance
    // than the ideal.
    if (num_free_procs > 0){
      num_procs_assigned_to_each_part[max_differing_part] +=  num_free_procs;
    }

    // now find what are the best processors with least migration for each part.

    // part_assignment_proc_begin_indices ([i]) is the array that holds the
    // beginning index of a processor that processor sends its data for part - i
    mj_part_t *part_assignment_proc_begin_indices =
      allocMemory<mj_part_t>(num_parts);

    // the next processor send is found in processor_chains_in_parts,
    // in linked list manner.
    mj_part_t *processor_chains_in_parts = allocMemory<mj_part_t>(num_procs);
    mj_part_t *processor_part_assignments = allocMemory<mj_part_t>(num_procs);

    // initialize the assignment of each processor.
    // this has a linked list implementation.
    // the beginning of processors assigned
    // to each part is hold at  part_assignment_proc_begin_indices[part].
    // then the next processor assigned to that part is located at
    // proc_part_assignments[part_assign_begins[part]], this is a chain
    // until the value of -1 is reached.
    for (int i = 0; i < num_procs; ++i ) {
      processor_part_assignments[i] = -1;
      processor_chains_in_parts[i] = -1;
    }
    for (int i = 0; i < num_parts; ++i ) {
      part_assignment_proc_begin_indices[i] = -1;
    }

    // std::cout << "Before migration: mig type:" <<
    //   this->migration_type << std::endl;
    // Allocate memory for sorting data structure.
    uSignedSortItem<mj_part_t, mj_gno_t, char> *
      sort_item_num_part_points_in_procs =
      allocMemory <uSignedSortItem<mj_part_t, mj_gno_t, char> > (num_procs);

    for(mj_part_t i = 0; i < num_parts; ++i) {
      // the algorithm tries to minimize the cost of migration, by assigning the
      // processors with highest number of coordinates on that part.
      // here we might want to implement a maximum weighted bipartite matching
      // algorithm.
      for(mj_part_t ii = 0; ii < num_procs; ++ii) {
        sort_item_num_part_points_in_procs[ii].id = ii;
        // if processor is not assigned yet.
        // add its num points to the sort data structure.
        if (processor_part_assignments[ii] == -1) {
          sort_item_num_part_points_in_procs[ii].val =
            num_points_in_all_processor_parts[ii * num_parts + i];
          // indicate that the processor has positive weight.
          sort_item_num_part_points_in_procs[ii].signbit = 1;
        }
        else {
          // if processor is already assigned, insert -nLocal - 1 so that it
          // won't be selected again.
          // would be same if we simply set it to -1, but more information with
          // no extra cost (which is used later) is provided.
          // sort_item_num_part_points_in_procs[ii].val =
          // -num_points_in_all_processor_parts[ii * num_parts + i] - 1;

          // UPDATE: Since above gets warning when unsigned is used to
          // represent, we added extra bit to as sign bit to the sort item.
          // It is 1 for positives, 0 for negatives.
          sort_item_num_part_points_in_procs[ii].val =
            num_points_in_all_processor_parts[ii * num_parts + i];
          sort_item_num_part_points_in_procs[ii].signbit = 0;
        }
      }

      // sort the processors in the part.
      uqSignsort<mj_part_t, mj_gno_t,char>
        (num_procs, sort_item_num_part_points_in_procs);

      /*
      for(mj_part_t ii = 0; ii < num_procs; ++ii){
        std::cout << "ii:" << ii << " " <<
          sort_item_num_part_points_in_procs[ii].id <<
          " " << sort_item_num_part_points_in_procs[ii].val <<
          " " << int(sort_item_num_part_points_in_procs[ii].signbit) <<
          std::endl;
      }
      */

      mj_part_t required_proc_count = num_procs_assigned_to_each_part[i];
      mj_gno_t total_num_points_in_part = global_num_points_in_parts[i];
      mj_gno_t ideal_num_points_in_a_proc = Teuchos::as<mj_gno_t>(
        ceil(total_num_points_in_part / double (required_proc_count)));

      // starts sending to least heaviest part.
      mj_part_t next_proc_to_send_index = num_procs - required_proc_count;
      mj_part_t next_proc_to_send_id =
        sort_item_num_part_points_in_procs[next_proc_to_send_index].id;
      mj_lno_t space_left_in_sent_proc = ideal_num_points_in_a_proc -
        sort_item_num_part_points_in_procs[next_proc_to_send_index].val;

      // find the processors that will be assigned to this part, which are the
      // heaviest non assigned processors.
      for(mj_part_t ii = num_procs - 1;
        ii >= num_procs - required_proc_count; --ii) {
        mj_part_t proc_id = sort_item_num_part_points_in_procs[ii].id;
        // assign processor to part - i.
        processor_part_assignments[proc_id] = i;
      }

      bool did_change_sign = false;
      // if processor has a minus count, reverse it.
      for(mj_part_t ii = 0; ii < num_procs; ++ii) {
        // TODO:  THE LINE BELOW PRODUCES A WARNING IF gno_t IS UNSIGNED
        // TODO:  SEE BUG 6194
        if (sort_item_num_part_points_in_procs[ii].signbit == 0){
          did_change_sign = true;
          sort_item_num_part_points_in_procs[ii].signbit = 1;
        }
        else {
          break;
        }
      }
    
      if(did_change_sign) {
        // resort the processors in the part for the rest of the processors that
        // is not assigned.
        uqSignsort<mj_part_t, mj_gno_t>(num_procs - required_proc_count,
          sort_item_num_part_points_in_procs);
      }

      /*
      for(mj_part_t ii = 0; ii < num_procs; ++ii){
        std::cout << "after resort ii:" << ii << " " <<
          sort_item_num_part_points_in_procs[ii].id <<
          " " << sort_item_num_part_points_in_procs[ii].val <<
          " " << int(sort_item_num_part_points_in_procs[ii].signbit ) <<
          std::endl;
      }
      */

      // check if this processors is one of the procs assigned to this part.
      // if it is, then get the group.
      if (!did_i_find_my_group) {
        for(mj_part_t ii = num_procs - 1; ii >=
          num_procs - required_proc_count; --ii) {

        mj_part_t proc_id_to_assign = sort_item_num_part_points_in_procs[ii].id;

        // add the proc to the group.
        processor_ranks_for_subcomm.push_back(proc_id_to_assign);

        if(proc_id_to_assign == this->myRank) {
          // if the assigned process is me, then I find my group.
          did_i_find_my_group = true;
            
          // set the beginning of part i to my rank.
          part_assignment_proc_begin_indices[i] = this->myRank;
          processor_chains_in_parts[this->myRank] = -1;

          // set send count to myself to the number of points that I have
          // in part i.
          send_count_to_each_proc[this->myRank] =
            sort_item_num_part_points_in_procs[ii].val;

          // calculate the shift required for the
          // output_part_numbering_begin_index
          for (mj_part_t in = 0; in < i; ++in){
            output_part_numbering_begin_index +=
              (*next_future_num_parts_in_parts)[in];
          }
          out_part_index = i;
        }
      }

      // if these was not my group,
      // clear the subcomminicator processor array.
      if (!did_i_find_my_group){
        processor_ranks_for_subcomm.clear();
      }
    }

    // send points of the nonassigned coordinates to the assigned coordinates.
    // starts from the heaviest nonassigned processor.
    // TODO we might want to play with this part, that allows more
    // computational imbalance but having better communication balance.
    for(mj_part_t ii = num_procs - required_proc_count - 1; ii >= 0; --ii) {
        mj_part_t nonassigned_proc_id =
          sort_item_num_part_points_in_procs[ii].id;
        mj_lno_t num_points_to_sent =
          sort_item_num_part_points_in_procs[ii].val;

      // we set number of points to -to_sent - 1 for the assigned processors.
      // we reverse it here. This should not happen, as we have already
      // reversed them above.
#ifdef MJ_DEBUG
      if (num_points_to_sent < 0) {
        cout << "Migration - processor assignments - for part:" << i
          << "from proc:" << nonassigned_proc_id << " num_points_to_sent:"
          << num_points_to_sent << std::endl;
        exit(1);
      }
#endif

	    switch (migration_type) {
	      case 0:
	      {
          // now sends the points to the assigned processors.
          while (num_points_to_sent > 0) {
            // if the processor has enough space.
            if (num_points_to_sent <= space_left_in_sent_proc) {
                // reduce the space left in the processor.
                space_left_in_sent_proc -= num_points_to_sent;
                // if my rank is the one that is sending the coordinates.
                if (this->myRank == nonassigned_proc_id){
                  // set my sent count to the sent processor.
                  send_count_to_each_proc[next_proc_to_send_id] =
                    num_points_to_sent;
                  // save the processor in the list (processor_chains_in_parts
                  // and part_assignment_proc_begin_indices)
                  // that the processor will send its point in part-i.
                  mj_part_t prev_begin = part_assignment_proc_begin_indices[i];
                  part_assignment_proc_begin_indices[i] = next_proc_to_send_id;
                  processor_chains_in_parts[next_proc_to_send_id] = prev_begin;
                }
                num_points_to_sent = 0;
            }
            else {
              // there might be no space left in the processor.
              if(space_left_in_sent_proc > 0) {
                num_points_to_sent -= space_left_in_sent_proc;

                //send as the space left in the processor.
                if (this->myRank == nonassigned_proc_id){
                  // send as much as the space in this case.
                  send_count_to_each_proc[next_proc_to_send_id] =
                    space_left_in_sent_proc;
                  mj_part_t prev_begin = part_assignment_proc_begin_indices[i];
                  part_assignment_proc_begin_indices[i] = next_proc_to_send_id;
                  processor_chains_in_parts[next_proc_to_send_id] = prev_begin;
                }
              }
              // change the sent part
              ++next_proc_to_send_index;

#ifdef MJ_DEBUG
              if(next_part_to_send_index <  nprocs - required_proc_count ) {
                  cout << "Migration - processor assignments - for part:"
                    << i
                    <<  " next_part_to_send :" << next_part_to_send_index
                    << " nprocs:" << nprocs
                    << " required_proc_count:" << required_proc_count
                    << " Error: next_part_to_send_index <" <<
                    << " nprocs - required_proc_count" << std::endl;
                  exit(1);
              }
#endif
              // send the new id.
              next_proc_to_send_id =
                sort_item_num_part_points_in_procs[next_proc_to_send_index].id;
              // set the new space in the processor.
              space_left_in_sent_proc = ideal_num_points_in_a_proc -
                sort_item_num_part_points_in_procs[next_proc_to_send_index].val;
            }
          } 
	      }
	      break;
	      default:
	      {
          // to minimize messages, we want each processor to send its
          // coordinates to only a single point.
          // we do not respect imbalances here, we send all points to the
          // next processor.
		      if (this->myRank == nonassigned_proc_id) {
            // set my sent count to the sent processor.
            send_count_to_each_proc[next_proc_to_send_id] = num_points_to_sent;
            // save the processor in the list (processor_chains_in_parts and
            // part_assignment_proc_begin_indices)
            // that the processor will send its point in part-i.
            mj_part_t prev_begin = part_assignment_proc_begin_indices[i];
            part_assignment_proc_begin_indices[i] = next_proc_to_send_id;
            processor_chains_in_parts[next_proc_to_send_id] = prev_begin;
          }
          num_points_to_sent = 0;
          ++next_proc_to_send_index;
		
		      // if we made it to the heaviest processor we round robin and
          // go to beginning
		      if (next_proc_to_send_index == num_procs) {
       		  next_proc_to_send_index = num_procs - required_proc_count;
		      }
          // send the new id.
          next_proc_to_send_id = 
            sort_item_num_part_points_in_procs[next_proc_to_send_index].id;
          // set the new space in the processor.
          space_left_in_sent_proc = ideal_num_points_in_a_proc -
            sort_item_num_part_points_in_procs[next_proc_to_send_index].val;
	      }	
      }
    }
  }
  
  /*
  for (int i = 0; i < num_procs;++i){
    std::cout << "me:" << this->myRank << " to part:" << i << " sends:" << 
      send_count_to_each_proc[i] << std::endl;
  } 
  */  

  this->assign_send_destinations(
    num_parts,
    part_assignment_proc_begin_indices,
    processor_chains_in_parts,
    send_count_to_each_proc,
    coordinate_destinations);

  freeArray<mj_part_t>(part_assignment_proc_begin_indices);
  freeArray<mj_part_t>(processor_chains_in_parts);
  freeArray<mj_part_t>(processor_part_assignments);
  freeArray<uSignedSortItem<mj_part_t, mj_gno_t, char> >
    (sort_item_num_part_points_in_procs);   
  freeArray<mj_part_t > (num_procs_assigned_to_each_part);
}

/*! \brief Function fills up coordinate_destinations is the output array
 * that holds which part each coordinate should be sent. In addition
 * it calculates the shift amount (output_part_numbering_begin_index) to be
 * done when final numberings of the parts are performed.
 * \param num_parts is the number of parts that exist in the
 * current partitioning.
 * \param sort_item_part_to_proc_assignment is the sorted parts with respect
 * to the assigned processors.
 * \param coordinate_destinations is the output array that holds which part
 * each coordinate should be sent.
 * \param output_part_numbering_begin_index is how much the numbers should be
 * shifted when numbering the result parts.
 * \param next_future_num_parts_in_parts is the vector, how many more parts
 * each part will be divided into in the future.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  assign_send_destinations2(
  mj_part_t num_parts,
  uSortItem<mj_part_t, mj_part_t> * sort_item_part_to_proc_assignment,
  int *coordinate_destinations,
  mj_part_t &output_part_numbering_begin_index,
  std::vector<mj_part_t> *next_future_num_parts_in_parts)
{
  mj_part_t part_shift_amount = output_part_numbering_begin_index;
  mj_part_t previous_processor = -1;
  for(mj_part_t i = 0; i < num_parts; ++i){
    mj_part_t p = sort_item_part_to_proc_assignment[i].id;

    // assigned processors are sorted.
    mj_lno_t part_begin_index = 0;

    if (p > 0) {
      part_begin_index = this->kokkos_new_part_xadj(p - 1);
    }

    mj_lno_t part_end_index = this->kokkos_new_part_xadj(p);

    mj_part_t assigned_proc = sort_item_part_to_proc_assignment[i].val;
    if (this->myRank == assigned_proc && previous_processor != assigned_proc) {
        output_part_numbering_begin_index =  part_shift_amount;
    }
    previous_processor = assigned_proc;
    part_shift_amount += (*next_future_num_parts_in_parts)[p];

    for (mj_lno_t j= part_begin_index; j < part_end_index; j++){
      mj_lno_t localInd = this->kokkos_new_coordinate_permutations(j);
      coordinate_destinations[localInd] = assigned_proc;
    }
  }
}


/*! \brief Function fills up coordinate_destinations is the output array
 * that holds which part each coordinate should be sent. In addition it
 * calculates the shift amount (output_part_numbering_begin_index) to be done
 * when final numberings of the parts are performed.
 * \param num_points_in_all_processor_parts is the array holding the num points
 * in each part in each proc.
 * \param num_parts is the number of parts that exist in the
 * current partitioning.
 * \param num_procs is the number of processor attending to migration operation.
 * \param send_count_to_each_proc array array storing the number of points to
 * be sent to each part.
 * \param next_future_num_parts_in_parts is the vector, how many more parts
 * each part will be divided into in the future.
 * \param out_num_part is the number of parts assigned to the process.
 * \param out_part_indices is the indices of the part to which the processor
 * is assigned.
 * \param output_part_numbering_begin_index is how much the numbers should be
 * shifted when numbering the result parts.
 * \param coordinate_destinations is the output array that holds which part
 * each coordinate should be sent.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_assign_parts_to_procs(
  mj_gno_t * num_points_in_all_processor_parts,
  mj_part_t num_parts,
  mj_part_t num_procs,
  mj_lno_t *send_count_to_each_proc,
  std::vector<mj_part_t> *next_future_num_parts_in_parts,
  mj_part_t &out_num_part,
  std::vector<mj_part_t> &out_part_indices,
  mj_part_t &output_part_numbering_begin_index,
  int *coordinate_destinations) {
  out_num_part = 0;

  mj_gno_t *global_num_points_in_parts =
    num_points_in_all_processor_parts + num_procs * num_parts;
  out_part_indices.clear();

  // to sort the parts that is assigned to the processors.
  // id is the part number, sort value is the assigned processor id.
  uSortItem<mj_part_t, mj_part_t> * sort_item_part_to_proc_assignment  =
    allocMemory <uSortItem<mj_part_t, mj_part_t> >(num_parts);
  uSortItem<mj_part_t, mj_gno_t> * sort_item_num_points_of_proc_in_part_i =
    allocMemory <uSortItem<mj_part_t, mj_gno_t> >(num_procs);

  // calculate the optimal number of coordinates that should be assigned
  // to each processor.
  mj_lno_t work_each =
    mj_lno_t (this->num_global_coords / (double (num_procs)) + 0.5f);

  // to hold the left space as the number of coordinates to the optimal
  // number in each proc.
  mj_lno_t *space_in_each_processor = allocMemory <mj_lno_t>(num_procs);

  // initialize left space in each.
  for (mj_part_t i = 0; i < num_procs; ++i) {
    space_in_each_processor[i] = work_each;
  }

  // we keep track of how many parts each processor is assigned to.
  // because in some weird inputs, it might be possible that some
  // processors is not assigned to any part. Using these variables,
  // we force each processor to have at least one part.
  mj_part_t *num_parts_proc_assigned = allocMemory <mj_part_t>(num_procs);
  memset(num_parts_proc_assigned, 0, sizeof(mj_part_t) * num_procs);
  int empty_proc_count = num_procs;

  // to sort the parts with decreasing order of their coordiantes.
  // id are the part numbers, sort value is the number of points in each.
  uSortItem<mj_part_t, mj_gno_t> * sort_item_point_counts_in_parts =
    allocMemory <uSortItem<mj_part_t, mj_gno_t> >(num_parts);

  // initially we will sort the parts according to the number of coordinates
  // they have, so that we will start assigning with the part that has the most
  // number of coordinates.
  for (mj_part_t i = 0; i < num_parts; ++i) {
    sort_item_point_counts_in_parts[i].id = i;
    sort_item_point_counts_in_parts[i].val = global_num_points_in_parts[i];
  }

  // sort parts with increasing order of loads.
  uqsort<mj_part_t, mj_gno_t>(num_parts, sort_item_point_counts_in_parts);

  // assigning parts to the processors
  // traverse the part win decreasing order of load.
  // first assign the heaviest part.
  for (mj_part_t j = 0; j < num_parts; ++j) {
    // sorted with increasing order, traverse inverse.
    mj_part_t i = sort_item_point_counts_in_parts[num_parts - 1 - j].id;
      
    // load of the part
    mj_gno_t load = global_num_points_in_parts[i];

    // assigned processors
    mj_part_t assigned_proc = -1;
      
    // if not fit best processor.
    mj_part_t best_proc_to_assign = 0;

    // sort processors with increasing number of points in this part.
    for (mj_part_t ii = 0; ii < num_procs; ++ii) {
      sort_item_num_points_of_proc_in_part_i[ii].id = ii;

      // if there are still enough parts to fill empty processors, than proceed
      // normally, but if empty processor count is equal to the number of part,
      // then we force to part assignments only to empty processors.
      if (empty_proc_count < num_parts - j ||
        num_parts_proc_assigned[ii] == 0) {
        // how many points processor ii has in part i?
        sort_item_num_points_of_proc_in_part_i[ii].val =
          num_points_in_all_processor_parts[ii * num_parts + i];
      }
      else {
        sort_item_num_points_of_proc_in_part_i[ii].val = -1;
      }
    }

    uqsort<mj_part_t, mj_gno_t>(num_procs,
      sort_item_num_points_of_proc_in_part_i);

    // traverse all processors with decreasing load.
    for (mj_part_t iii = num_procs - 1; iii >= 0; --iii) {
      mj_part_t ii = sort_item_num_points_of_proc_in_part_i[iii].id;
      mj_lno_t left_space = space_in_each_processor[ii] - load;
      //if enought space, assign to this part.
      if(left_space >= 0 ) {
        assigned_proc = ii;
        break;
      }
      //if space is not enough, store the best candidate part.
      if (space_in_each_processor[best_proc_to_assign] <
        space_in_each_processor[ii]) {
        best_proc_to_assign = ii;
      }
    }

    // if none had enough space, then assign it to best part.
    if (assigned_proc == -1) {
      assigned_proc = best_proc_to_assign;
    }

    if (num_parts_proc_assigned[assigned_proc]++ == 0) {
      --empty_proc_count;
    }

    space_in_each_processor[assigned_proc] -= load;
    //to sort later, part-i is assigned to the proccessor - assignment.
    sort_item_part_to_proc_assignment[j].id = i; //part i
    
    // assigned to processor - assignment.
    sort_item_part_to_proc_assignment[j].val = assigned_proc;

    // if assigned processor is me, increase the number.
    if (assigned_proc == this->myRank) {
        out_num_part++;//assigned_part_count;
        out_part_indices.push_back(i);
    }

    // increase the send to that processor by the number of points in that
    // part, as everyone send their coordiantes in this part to the
    // processor assigned to this part.
    send_count_to_each_proc[assigned_proc] +=
      num_points_in_all_processor_parts[this->myRank * num_parts + i];
  }

  freeArray<mj_part_t>(num_parts_proc_assigned);
  freeArray< uSortItem<mj_part_t, mj_gno_t> >
    (sort_item_num_points_of_proc_in_part_i);
  freeArray<uSortItem<mj_part_t, mj_gno_t> >(sort_item_point_counts_in_parts);
  freeArray<mj_lno_t >(space_in_each_processor);

  // sort assignments with respect to the assigned processors.
  uqsort<mj_part_t, mj_part_t>(num_parts, sort_item_part_to_proc_assignment);

  // fill sendBuf.
  this->assign_send_destinations2(
    num_parts,
    sort_item_part_to_proc_assignment,
    coordinate_destinations,
    output_part_numbering_begin_index,
    next_future_num_parts_in_parts);

  freeArray<uSortItem<mj_part_t, mj_part_t> >
    (sort_item_part_to_proc_assignment);
}


/*! \brief Function fills up coordinate_destinations is the output array
 * that holds which part each coordinate should be sent. In addition it
 * calculates the shift amount (output_part_numbering_begin_index) to be done
 * when final numberings of the parts are performed.
 * \param num_points_in_all_processor_parts is the array holding the num points
 * in each part in each proc.
 * \param num_parts is the number of parts that exist in the current
 * partitioning.
 * \param num_procs is the number of processor attending to migration operation.
 * \param send_count_to_each_proc array array storing the number of points to
 * be sent to each part.
 * \param processor_ranks_for_subcomm is the ranks of the processors that will
 * be in the subcommunicator with me.
 * \param next_future_num_parts_in_parts is the vector, how many more parts
 * each part will be divided into in the future.
 * \param out_num_part is the number of parts assigned to the process.
 * \param out_part_indices is the indices of the part to which the processor
 * is assigned.
 * \param output_part_numbering_begin_index is how much the numbers should be
 * shifted when numbering the result parts.
 * \param coordinate_destinations is the output array that holds which part
 * each coordinate should be sent.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_migration_part_proc_assignment(
  mj_gno_t * num_points_in_all_processor_parts,
  mj_part_t num_parts,
  mj_part_t num_procs,
  mj_lno_t *send_count_to_each_proc,
  std::vector<mj_part_t> &processor_ranks_for_subcomm,
  std::vector<mj_part_t> *next_future_num_parts_in_parts,
  mj_part_t &out_num_part,
  std::vector<mj_part_t> &out_part_indices,
  mj_part_t &output_part_numbering_begin_index,
  int *coordinate_destinations)
{
  processor_ranks_for_subcomm.clear();
  // if (this->num_local_coords > 0)
  if (num_procs > num_parts) {
    // if there are more processors than the number of current part
    // then processors share the existing parts.
    // at the end each processor will have a single part,
    // but a part will be shared by a group of processors.
    mj_part_t out_part_index = 0;
    this->mj_assign_proc_to_parts(
      num_points_in_all_processor_parts,
      num_parts,
      num_procs,
      send_count_to_each_proc,
      processor_ranks_for_subcomm,
      next_future_num_parts_in_parts,
      out_part_index,
      output_part_numbering_begin_index,
      coordinate_destinations
    );

    out_num_part = 1;
    out_part_indices.clear();
    out_part_indices.push_back(out_part_index);
  }
  else {
    // there are more parts than the processors.
    // therefore a processor will be assigned multiple parts,
    // the subcommunicators will only have a single processor.
    processor_ranks_for_subcomm.push_back(this->myRank);

    // since there are more parts then procs,
    // assign multiple parts to processors.
    this->mj_assign_parts_to_procs(
      num_points_in_all_processor_parts,
      num_parts,
      num_procs,
      send_count_to_each_proc,
      next_future_num_parts_in_parts,
      out_num_part,
      out_part_indices,
      output_part_numbering_begin_index,
      coordinate_destinations);
  }
}

/*! \brief Function fills up coordinate_destinations is the output array
 * that holds which part each coordinate should be sent. In addition it
 * calculates the shift amount (output_part_numbering_begin_index) to be done
 * when final numberings of the parts are performed.
 * \param num_procs is the number of processor attending to migration operation.
 * \param num_new_local_points is the output to represent the new number of
 * local points.
 * \param iteration is the string for the current iteration.
 * \param coordinate_destinations is the output array that holds which part
 * each coordinate should be sent.
 * \param num_parts is the number of parts that exist in the current
 * partitioning.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_migrate_coords(
  mj_part_t num_procs,
  mj_lno_t &num_new_local_points,
  std::string iteration,
  int *coordinate_destinations,
  mj_part_t num_parts)
{
#ifdef ENABLE_ZOLTAN_MIGRATION
  if (sizeof(mj_lno_t) <= sizeof(int)) {
    // Cannot use Zoltan_Comm with local ordinals larger than ints.
    // In Zoltan_Comm_Create, the cast int(this->num_local_coords)
    // may overflow.
    ZOLTAN_COMM_OBJ *plan = NULL;
    MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*(this->comm));
    int num_incoming_gnos = 0;
    int message_tag = 7859;

    this->mj_env->timerStart(MACRO_TIMERS,
      "MultiJagged - Migration Z1PlanCreating-" + iteration);
    int ierr = Zoltan_Comm_Create(
      &plan,
      int(this->num_local_coords),
      coordinate_destinations,
      mpi_comm,
      message_tag,
      &num_incoming_gnos);
        
    Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
    this->mj_env->timerStop(MACRO_TIMERS,
      "MultiJagged - Migration Z1PlanCreating-" + iteration);

    this->mj_env->timerStart(MACRO_TIMERS,
      "MultiJagged - Migration Z1Migration-" + iteration);
      mj_gno_t *incoming_gnos = allocMemory< mj_gno_t>(num_incoming_gnos);

    // migrate gnos.
    message_tag++;
    ierr = Zoltan_Comm_Do(
      plan,
      message_tag,
      (char *) this->current_mj_gnos,
      sizeof(mj_gno_t),
      (char *) incoming_gnos);

    Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);

    freeArray<mj_gno_t>(this->current_mj_gnos);
    this->current_mj_gnos = incoming_gnos;

    //migrate coordinates
    throw std::logic_error("Did not refactor zoltan code yet for kokkos.");

    // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
    /*
    for (int i = 0; i < this->coord_dim; ++i){
      message_tag++;
      mj_scalar_t *coord = this->mj_coordinates[i];
      this->mj_coordinates[i] = allocMemory<mj_scalar_t>(num_incoming_gnos);
      ierr = Zoltan_Comm_Do(
        plan,
        message_tag,
        (char *) coord,
        sizeof(mj_scalar_t),
        (char *) this->mj_coordinates[i]);
      Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
      freeArray<mj_scalar_t>(coord);
    }
    */

    // migrate weights.
    for (int i = 0; i < this->num_weights_per_coord; ++i) {
      message_tag++;
      mj_scalar_t *weight = this->mj_weights[i];

      this->mj_weights[i] = allocMemory<mj_scalar_t>(num_incoming_gnos);
      ierr = Zoltan_Comm_Do(
        plan,
        message_tag,
        (char *) weight,
        sizeof(mj_scalar_t),
        (char *) this->mj_weights[i]);
      Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
      freeArray<mj_scalar_t>(weight);
    }

    // migrate owners.
    throw std::logic_error("migrate owners not implemented for kokkos yet.");

    // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
    /*
    int *coord_own = allocMemory<int>(num_incoming_gnos);
    message_tag++;
    ierr = Zoltan_Comm_Do(
      plan,
      message_tag,
      (char *) this->owner_of_coordinate,
      sizeof(int), (char *) coord_own);
    Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
    freeArray<int>(this->owner_of_coordinate);
    this->owner_of_coordinate = coord_own;
    */

    // if num procs is less than num parts,
    // we need the part assigment arrays as well, since
    // there will be multiple parts in processor.
    throw std::logic_error("migrate part ids not implemented for kokkos yet.");
    
    // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
    /*
    mj_part_t *new_parts = allocMemory<mj_part_t>(num_incoming_gnos);
    if(num_procs < num_parts) {
      message_tag++;
      ierr = Zoltan_Comm_Do(
        plan,
        message_tag,
        (char *) this->assigned_part_ids,
        sizeof(mj_part_t),
        (char *) new_parts);
      Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
    }
    freeArray<mj_part_t>(this->assigned_part_ids);
    this->assigned_part_ids = new_parts;
    */

    ierr = Zoltan_Comm_Destroy(&plan);
    Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
    num_new_local_points = num_incoming_gnos;
    this->mj_env->timerStop(MACRO_TIMERS,
      "MultiJagged - Migration Z1Migration-" + iteration);
  }
  else
#endif  // ENABLE_ZOLTAN_MIGRATION
  {
    this->mj_env->timerStart(MACRO_TIMERS,
      "MultiJagged - Migration DistributorPlanCreating-" + iteration);

    Tpetra::Distributor distributor(this->comm);
    ArrayView<const mj_part_t> destinations( coordinate_destinations,
      this->num_local_coords);
    mj_lno_t num_incoming_gnos = distributor.createFromSends(destinations);
    this->mj_env->timerStop(MACRO_TIMERS,
      "MultiJagged - Migration DistributorPlanCreating-" + iteration);

    this->mj_env->timerStart(MACRO_TIMERS,
      "MultiJagged - Migration DistributorMigration-" + iteration);

    {
      // migrate gnos.
      ArrayRCP<mj_gno_t> received_gnos(num_incoming_gnos);
      ArrayView<mj_gno_t> sent_gnos(
        this->kokkos_current_mj_gnos.data(), this->num_local_coords);
      distributor.doPostsAndWaits<mj_gno_t>(sent_gnos, 1, received_gnos());
      this->kokkos_current_mj_gnos =
        Kokkos::View<mj_gno_t*, device_t>("gids", num_incoming_gnos);
      memcpy(this->kokkos_current_mj_gnos.data(),
        received_gnos.getRawPtr(), num_incoming_gnos * sizeof(mj_gno_t));
    }

    // migrate coordinates
    Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t>
      temp_coordinates("kokkos_mj_coordinates", num_incoming_gnos,
      this->coord_dim);

    for (int i = 0; i < this->coord_dim; ++i) {
      Kokkos::View<mj_scalar_t *, device_t> sent_subview_kokkos_mj_coordinates
        = Kokkos::subview(this->kokkos_mj_coordinates, Kokkos::ALL, i);
      ArrayView<mj_scalar_t> sent_coord(
        sent_subview_kokkos_mj_coordinates.data(), this->num_local_coords);
      ArrayRCP<mj_scalar_t> received_coord(num_incoming_gnos);
      distributor.doPostsAndWaits<mj_scalar_t>(
        sent_coord, 1, received_coord());
      Kokkos::View<mj_scalar_t *, device_t> subview_kokkos_mj_coordinates =
        Kokkos::subview(temp_coordinates, Kokkos::ALL, i);
      memcpy(subview_kokkos_mj_coordinates.data(),
        received_coord.getRawPtr(), num_incoming_gnos * sizeof(mj_scalar_t));
    }

    this->kokkos_mj_coordinates = temp_coordinates;
        
    // migrate weights.
    Kokkos::View<mj_scalar_t**, device_t> temp_weights(
     "kokkos_mj_weights", num_incoming_gnos, this->num_weights_per_coord);

    for (int i = 0; i < this->num_weights_per_coord; ++i) {
      // TODO: How to optimize this better to use layouts properly
      // I think we can flip the weight layout and then use subviews
      // but need to determine if this will cause problems elsewhere
      ArrayRCP<mj_scalar_t> sent_weight(this->num_local_coords);
      for(int n = 0; n < this->num_local_coords; ++n) {
        sent_weight[n] = this->kokkos_mj_weights(n,i);
      }
      ArrayRCP<mj_scalar_t> received_weight(num_incoming_gnos);
      distributor.doPostsAndWaits<mj_scalar_t>(
        sent_weight(), 1, received_weight());
      for(int n = 0; n < num_incoming_gnos; ++n) {
        temp_weights(n,i) = received_weight[n];
      }
    }
  
    this->kokkos_mj_weights = temp_weights;

    {
      ArrayView<int> sent_owners(
        this->kokkos_owner_of_coordinate.data(), this->num_local_coords);
      ArrayRCP<int> received_owners(num_incoming_gnos);
      distributor.doPostsAndWaits<int>(sent_owners, 1, received_owners());
      this->kokkos_owner_of_coordinate = Kokkos::View<int *, device_t>
        ("owner_of_coordinate", num_incoming_gnos);
      memcpy(this->kokkos_owner_of_coordinate.data(),
        received_owners.getRawPtr(), num_incoming_gnos * sizeof(int));
    }

    // if num procs is less than num parts,
    // we need the part assigment arrays as well, since
    // there will be multiple parts in processor.
    if(num_procs < num_parts) {
      ArrayView<mj_part_t> sent_partids(
        this->kokkos_assigned_part_ids.data(), this->num_local_coords);
      ArrayRCP<mj_part_t> received_partids(num_incoming_gnos);
      distributor.doPostsAndWaits<mj_part_t>(
        sent_partids, 1, received_partids());
      this->kokkos_assigned_part_ids =
        Kokkos::View<mj_part_t *, device_t>
         ("kokkos_assigned_part_ids", num_incoming_gnos);
      memcpy(
        this->kokkos_assigned_part_ids.data(),
        received_partids.getRawPtr(),
        num_incoming_gnos * sizeof(mj_part_t));
    }
    else {
      this->kokkos_assigned_part_ids = Kokkos::View<mj_part_t *, device_t>
        ("kokkos_assigned_part_ids", num_incoming_gnos);
    }

    this->mj_env->timerStop(MACRO_TIMERS,
      "MultiJagged - Migration DistributorMigration-" + iteration);

    num_new_local_points = num_incoming_gnos;
  }
}

/*! \brief Function creates the new subcomminicator for the processors
 * given in processor_ranks_for_subcomm.
 * \param processor_ranks_for_subcomm is the vector that has the ranks of
 * the processors that will be in the same group.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  create_sub_communicator(std::vector<mj_part_t> &processor_ranks_for_subcomm)
{
  mj_part_t group_size = processor_ranks_for_subcomm.size();
  mj_part_t *ids = allocMemory<mj_part_t>(group_size);
  for(mj_part_t i = 0; i < group_size; ++i) {
    ids[i] = processor_ranks_for_subcomm[i];
  }
  ArrayView<const mj_part_t> idView(ids, group_size);
  this->comm = this->comm->createSubcommunicator(idView);
  freeArray<mj_part_t>(ids);
}

/*! \brief Function writes the new permutation arrays after the migration.
 * \param output_num_parts is the number of parts that is assigned to
 * the processor.
 * \param num_parts is the number of parts right before migration.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  fill_permutation_array(
  mj_part_t output_num_parts,
  mj_part_t num_parts)
{
  // if there is single output part, then simply fill the permutation array.
  if (output_num_parts == 1) {
    for(mj_lno_t i = 0; i < this->num_local_coords; ++i) {
      this->kokkos_new_coordinate_permutations(i) = i;
    }
    this->kokkos_new_part_xadj(0) = this->num_local_coords;
  }
  else {
    // otherwise we need to count how many points are there in each part.
    // we allocate here as num_parts, because the sent partids are up to
    // num_parts, although there are outout_num_parts different part.
    mj_lno_t *num_points_in_parts = allocMemory<mj_lno_t>(num_parts);

    // part shift holds the which part number an old part number corresponds to.
    mj_part_t *part_shifts = allocMemory<mj_part_t>(num_parts);

    memset(num_points_in_parts, 0, sizeof(mj_lno_t) * num_parts);

    for(mj_lno_t i = 0; i < this->num_local_coords; ++i) {
      mj_part_t ii = this->kokkos_assigned_part_ids(i);
      ++num_points_in_parts[ii];
    }

    // write the end points of the parts.
    mj_part_t p = 0;
    mj_lno_t prev_index = 0;
    for(mj_part_t i = 0; i < num_parts; ++i){
      if(num_points_in_parts[i] > 0) {
        this->kokkos_new_part_xadj(p) =  prev_index + num_points_in_parts[i];
        prev_index += num_points_in_parts[i];
        part_shifts[i] = p++;
      }
    }

    // for the rest of the parts write the end index as end point.
    mj_part_t assigned_num_parts = p - 1;
    for (;p < num_parts; ++p) {
      this->kokkos_new_part_xadj(p) =
        this->kokkos_new_part_xadj(assigned_num_parts);
    }
    for(mj_part_t i = 0; i < output_num_parts; ++i) {
      num_points_in_parts[i] = this->kokkos_new_part_xadj(i);
    }

    // write the permutation array here.
    // get the part of the coordinate i, shift it to obtain the new part number.
    // assign it to the end of the new part numbers pointer.
    for(mj_lno_t i = this->num_local_coords - 1; i >= 0; --i) {
      mj_part_t part =
        part_shifts[mj_part_t(this->kokkos_assigned_part_ids(i))];
      this->kokkos_new_coordinate_permutations(--num_points_in_parts[part]) = i;
    }

    freeArray<mj_lno_t>(num_points_in_parts);
    freeArray<mj_part_t>(part_shifts);
  }
}

/*! \brief Function checks if should do migration or not.
 * It returns true to point that migration should be done when
 * -migration_reduce_all_population are higher than a predetermined value
 * -num_coords_for_last_dim_part that left for the last dimension partitioning
 * is less than a predetermined value - the imbalance of the processors on the
 * parts are higher than given threshold.
 * \param input_num_parts is the number of parts when migration is called.
 * \param output_num_parts is the output number of parts after migration.
 * \param next_future_num_parts_in_parts is the number of total future parts
 * each part is partitioned into. This will be updated for migration.
 * \param output_part_begin_index is the number that will be used as beginning
 * part number when final solution part numbers are assigned.
 * \param migration_reduce_all_population is the estimated total number of
 * reduceall operations multiplied with number of processors to be used for
 * determining migration.
 * \param num_coords_for_last_dim_part is the estimated number of points in each
 * part, when last dimension partitioning is performed.
 * \param iteration is the string that gives information about the dimension
 * for printing purposes.
 * \param input_part_boxes is the array that holds the part boxes after
 * the migration. (swapped)
 * \param output_part_boxes is the array that holds the part boxes before
 * the migration. (swapped)
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
bool AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  mj_perform_migration(
  mj_part_t input_num_parts,
  mj_part_t &output_num_parts,
  std::vector<mj_part_t> *next_future_num_parts_in_parts,
  mj_part_t &output_part_begin_index,
  size_t migration_reduce_all_population,
  mj_lno_t num_coords_for_last_dim_part,
  std::string iteration,
  RCP<mj_partBoxVector_t> &input_part_boxes,
  RCP<mj_partBoxVector_t> &output_part_boxes)
{
  mj_part_t num_procs = this->comm->getSize();
  this->myRank = this->comm->getRank();

  // this array holds how many points each processor has in each part.
  // to access how many points processor i has on part j,
  // num_points_in_all_processor_parts[i * num_parts + j]
  mj_gno_t *num_points_in_all_processor_parts =
    allocMemory<mj_gno_t>(input_num_parts * (num_procs + 1));

  // get the number of coordinates in each part in each processor.
  this->get_processor_num_points_in_parts(
    num_procs,
    input_num_parts,
    num_points_in_all_processor_parts);

  // check if migration will be performed or not.
  if (!this->mj_check_to_migrate(
    migration_reduce_all_population,
    num_coords_for_last_dim_part,
    num_procs,
    input_num_parts,
    num_points_in_all_processor_parts)) {
    freeArray<mj_gno_t>(num_points_in_all_processor_parts);
    return false;
  }

  mj_lno_t *send_count_to_each_proc = NULL;
  int *coordinate_destinations = allocMemory<int>(this->num_local_coords);
  send_count_to_each_proc = allocMemory<mj_lno_t>(num_procs);
  
  for (int i = 0; i < num_procs; ++i) {
    send_count_to_each_proc[i] = 0;
  }

  std::vector<mj_part_t> processor_ranks_for_subcomm;
  std::vector<mj_part_t> out_part_indices;

  // determine which processors are assigned to which parts
  this->mj_migration_part_proc_assignment(
    num_points_in_all_processor_parts,
    input_num_parts,
    num_procs,
    send_count_to_each_proc,
    processor_ranks_for_subcomm,
    next_future_num_parts_in_parts,
    output_num_parts,
    out_part_indices,
    output_part_begin_index,
    coordinate_destinations);

  freeArray<mj_lno_t>(send_count_to_each_proc);
  std::vector <mj_part_t> tmpv;

  std::sort (out_part_indices.begin(), out_part_indices.end());
  mj_part_t outP = out_part_indices.size();

  mj_gno_t new_global_num_points = 0;
  mj_gno_t *global_num_points_in_parts =
    num_points_in_all_processor_parts + num_procs * input_num_parts;

  if (this->mj_keep_part_boxes) {
    input_part_boxes->clear();
  }

  // now we calculate the new values for next_future_num_parts_in_parts.
  // same for the part boxes.
  for (mj_part_t i = 0; i < outP; ++i) {
    mj_part_t ind = out_part_indices[i];
    new_global_num_points += global_num_points_in_parts[ind];
    tmpv.push_back((*next_future_num_parts_in_parts)[ind]);
    if (this->mj_keep_part_boxes){
      input_part_boxes->push_back((*output_part_boxes)[ind]);
    }
  }

  // swap the input and output part boxes.
  if (this->mj_keep_part_boxes) {
    RCP<mj_partBoxVector_t> tmpPartBoxes = input_part_boxes;
    input_part_boxes = output_part_boxes;
    output_part_boxes = tmpPartBoxes;
  }
  next_future_num_parts_in_parts->clear();
  for (mj_part_t i = 0; i < outP; ++i) {
    mj_part_t p = tmpv[i];
    next_future_num_parts_in_parts->push_back(p);
  }

  freeArray<mj_gno_t>(num_points_in_all_processor_parts);

  mj_lno_t num_new_local_points = 0;

  //perform the actual migration operation here.
  this->mj_migrate_coords(
    num_procs,
    num_new_local_points,
    iteration,
    coordinate_destinations,
    input_num_parts);

  freeArray<int>(coordinate_destinations);

  if(this->num_local_coords != num_new_local_points){
    this->kokkos_new_coordinate_permutations = Kokkos::View<mj_lno_t*, device_t>
      ("kokkos_new_coordinate_permutations", num_new_local_points);
    this->kokkos_coordinate_permutations = Kokkos::View<mj_lno_t*, device_t>
      ("kokkos_coordinate_permutations", num_new_local_points);
  }

  this->num_local_coords = num_new_local_points;
  this->num_global_coords = new_global_num_points;

  // create subcommunicator.
  this->create_sub_communicator(processor_ranks_for_subcomm);
  processor_ranks_for_subcomm.clear();

  // fill the new permutation arrays.
  this->fill_permutation_array(output_num_parts, input_num_parts);
  return true;
}


/*! \brief Function creates consistent chunks for task partitioning. Used only
 * in the case of sequential task partitioning, where consistent handle of the
 * points on the cuts are required.
 * \param num_parts is the number of parts.
 * \param mj_current_dim_coords is 1 dimensional array holding the
 * coordinate values.
 * \param current_concurrent_cut_coordinate is 1 dimensional array holding
 * the cut coordinates.
 * \param coordinate_begin is the start index of the given partition on
 * partitionedPointPermutations.
 * \param coordinate_end is the end index of the given partition on
 * partitionedPointPermutations.
 * \param used_local_cut_line_weight_to_left holds how much weight of the
 * coordinates on the cutline should be put on left side.
 * \param out_part_xadj is the indices of begginning and end of the parts in
 * the output partition.
 * \param coordInd is the index according to which the partitioning is done.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  create_consistent_chunks(
  mj_part_t num_parts,
  Kokkos::View<mj_scalar_t *, device_t> mj_current_dim_coords,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_current_concurrent_cut_coordinate,
  mj_lno_t coordinate_begin,
  mj_lno_t coordinate_end,
  Kokkos::View<mj_scalar_t *, device_t>
    kokkos_used_local_cut_line_weight_to_left,
  Kokkos::View<mj_lno_t *, device_t> kokkos_out_part_xadj,
  int coordInd,
  bool longest_dim_part,
  uSignedSortItem<int, mj_scalar_t, char> * p_coord_dimension_range_sorted)
{
  // mj_lno_t numCoordsInPart =  coordinateEnd - coordinateBegin;
  mj_part_t no_cuts = num_parts - 1;

  // now if the rectilinear partitioning is allowed we decide how
  // much weight each thread should put to left and right.
  if (this->distribute_points_on_cut_lines) {
    for (mj_part_t i = 0; i < no_cuts; ++i) {
      // the left to be put on the left of the cut.
      mj_scalar_t left_weight = kokkos_used_local_cut_line_weight_to_left(i);
      // cout << "i:" << i << " left_weight:" << left_weight << endl;
      if(left_weight > this->sEpsilon) {
        // the weight of thread ii on cut.
        mj_scalar_t thread_ii_weight_on_cut =
          this->kokkos_thread_part_weight_work(i * 2 + 1) -
          this->kokkos_thread_part_weight_work(i * 2);
        if(thread_ii_weight_on_cut < left_weight) {
          this->kokkos_thread_cut_line_weight_to_put_left(i) =
            thread_ii_weight_on_cut;
        }
        else {
          this->kokkos_thread_cut_line_weight_to_put_left(i) = left_weight;
        }
        left_weight -= thread_ii_weight_on_cut;
      }
      else {
        this->kokkos_thread_cut_line_weight_to_put_left(i) = 0;
      }
    }

    if(no_cuts > 0) {
      // this is a special case. If cutlines share the same coordinate,
      // their weights are equal.
      // we need to adjust the ratio for that.
      for (mj_part_t i = no_cuts - 1; i > 0 ; --i) {
        if(ZOLTAN2_ABS(kokkos_current_concurrent_cut_coordinate(i) -
          kokkos_current_concurrent_cut_coordinate(i-1)) < this->sEpsilon) {
          this->kokkos_thread_cut_line_weight_to_put_left(i) -=
            this->kokkos_thread_cut_line_weight_to_put_left(i - 1);
        }
        this->kokkos_thread_cut_line_weight_to_put_left(i) =
          int ((this->kokkos_thread_cut_line_weight_to_put_left(i) +
          LEAST_SIGNIFICANCE) * SIGNIFICANCE_MUL) /
          mj_scalar_t(SIGNIFICANCE_MUL);
      }
    }
  }

  for(mj_part_t ii = 0; ii < num_parts; ++ii) {
    this->kokkos_thread_point_counts(ii) = 0;
  }

  // for this specific case we dont want to distribute the points along the
  // cut position randomly, as we need a specific ordering of them. Instead,
  // we put the coordinates into a sort item, where we sort those
  // using the coordinates of points on other dimensions and the index.

  // some of the cuts might share the same position.
  // in this case, if cut i and cut j share the same position
  // cut_map[i] = cut_map[j] = sort item index.
  mj_part_t *cut_map = allocMemory<mj_part_t> (no_cuts);

  typedef uMultiSortItem<mj_lno_t, int, mj_scalar_t> multiSItem;
  typedef std::vector< multiSItem > multiSVector;
  typedef std::vector<multiSVector> multiS2Vector;

  // to keep track of the memory allocated.
  std::vector<mj_scalar_t *>allocated_memory;

  // vector for which the coordinates will be sorted.
  multiS2Vector sort_vector_points_on_cut;

  // the number of cuts that have different coordinates.
  mj_part_t different_cut_count = 1;
  cut_map[0] = 0;

  // now we insert 1 sort vector for all cuts on the different
  // positins.if multiple cuts are on the same position,
  // they share sort vectors.
  multiSVector tmpMultiSVector;
  sort_vector_points_on_cut.push_back(tmpMultiSVector);

  for (mj_part_t i = 1; i < no_cuts ; ++i){
    // if cuts share the same cut coordinates
    // set the cutmap accordingly.
    if(ZOLTAN2_ABS(kokkos_current_concurrent_cut_coordinate(i) -
      kokkos_current_concurrent_cut_coordinate(i-1)) < this->sEpsilon) {
      cut_map[i] = cut_map[i-1];
    }
    else {
      cut_map[i] = different_cut_count++;
      multiSVector tmp2MultiSVector;
      sort_vector_points_on_cut.push_back(tmp2MultiSVector);
    }
  }

  // now the actual part assigment.
  for (mj_lno_t ii = coordinate_begin; ii < coordinate_end; ++ii) {
    mj_lno_t i = this->kokkos_coordinate_permutations(ii);
    mj_part_t pp = this->kokkos_assigned_part_ids(i);
    mj_part_t p = pp / 2;
    // if the coordinate is on a cut.
    if(pp % 2 == 1 ) {
      mj_scalar_t *vals = allocMemory<mj_scalar_t>(this->coord_dim -1);
      allocated_memory.push_back(vals);

      // we insert the coordinates to the sort item here.
      int val_ind = 0;

      if (longest_dim_part) {
        // std::cout << std::endl << std::endl;
        for(int dim = this->coord_dim - 2; dim >= 0; --dim){
          // uSignedSortItem<int, mj_scalar_t, char>
          //   *p_coord_dimension_range_sorted
          int next_largest_coord_dim = p_coord_dimension_range_sorted[dim].id;
          // std::cout << "next_largest_coord_dim: " <<
          //   next_largest_coord_dim << " ";
          // Note refactor in progress
          vals[val_ind++] =
            this->kokkos_mj_coordinates(i,next_largest_coord_dim);
        }
      }
      else {
        for(int dim = coordInd + 1; dim < this->coord_dim; ++dim){
          vals[val_ind++] = this->kokkos_mj_coordinates(i,dim);
        }
        for(int dim = 0; dim < coordInd; ++dim){
          vals[val_ind++] = this->kokkos_mj_coordinates(i,dim);
        }
      }

      multiSItem tempSortItem(i, this->coord_dim -1, vals);
      //inser the point to the sort vector pointed by the cut_map[p].
      mj_part_t cmap = cut_map[p];
      sort_vector_points_on_cut[cmap].push_back(tempSortItem);
    }
    else {
      //if it is not on the cut, simple sorting.
      ++this->kokkos_thread_point_counts(p);
      this->kokkos_assigned_part_ids(i) = p;
    }
  }

  // sort all the sort vectors.
  for (mj_part_t i = 0; i < different_cut_count; ++i){
    std::sort (sort_vector_points_on_cut[i].begin(),
      sort_vector_points_on_cut[i].end());
  }

  mj_part_t previous_cut_map = cut_map[0];

  // this is how much previous part owns the weight of the current part.
  // when target part weight is 1.6, and the part on the left is given 2,
  // the left has an extra 0.4, while the right has missing 0.4 from the
  // previous cut.
  // This parameter is used to balance this issues.
  // in the above example weight_stolen_from_previous_part will be 0.4.
  // if the left part target is 2.2 but it is given 2,
  // then weight_stolen_from_previous_part will be -0.2.
  mj_scalar_t weight_stolen_from_previous_part = 0;
  for (mj_part_t p = 0; p < no_cuts; ++p) {
    mj_part_t mapped_cut = cut_map[p];

    // if previous cut map is done, and it does not have the same index,
    // then assign all points left on that cut to its right.
    if (previous_cut_map != mapped_cut) {
      mj_lno_t sort_vector_end = (mj_lno_t)
        sort_vector_points_on_cut[previous_cut_map].size() - 1;
      for (; sort_vector_end >= 0; --sort_vector_end) {
        multiSItem t =
          sort_vector_points_on_cut[previous_cut_map][sort_vector_end];
        mj_lno_t i = t.index;
        ++this->kokkos_thread_point_counts(p);
        this->kokkos_assigned_part_ids(i) = p;
      }
      sort_vector_points_on_cut[previous_cut_map].clear();
    }

    // TODO: MD: I dont remember why I have it reverse order here.
    mj_lno_t sort_vector_end = (mj_lno_t)
      sort_vector_points_on_cut[mapped_cut].size() - 1;
    // mj_lno_t sort_vector_begin= 0;
    // mj_lno_t sort_vector_size =
    //   (mj_lno_t)sort_vector_points_on_cut[mapped_cut].size();

    // TODO commented for reverse order
    for (; sort_vector_end >= 0; --sort_vector_end){
      // for (; sort_vector_begin < sort_vector_size; ++sort_vector_begin){
      // TODO COMMENTED FOR REVERSE ORDER
      multiSItem t = sort_vector_points_on_cut[mapped_cut][sort_vector_end];
      //multiSItem t = sort_vector_points_on_cut[mapped_cut][sort_vector_begin];
      mj_lno_t i = t.index;
      mj_scalar_t w = this->kokkos_mj_uniform_weights(0) ? 1 :
        this->kokkos_mj_weights(i,0);
      // part p has enough space for point i, then put it to point i.
      if(this->kokkos_thread_cut_line_weight_to_put_left(p) +
        weight_stolen_from_previous_part> this->sEpsilon &&
        this->kokkos_thread_cut_line_weight_to_put_left(p) +
        weight_stolen_from_previous_part -
        ZOLTAN2_ABS(this->kokkos_thread_cut_line_weight_to_put_left(p) +
        weight_stolen_from_previous_part - w)> this->sEpsilon)
      {
        this->kokkos_thread_cut_line_weight_to_put_left(p) -= w;

        sort_vector_points_on_cut[mapped_cut].pop_back();

        ++this->kokkos_thread_point_counts(p);
        this->kokkos_assigned_part_ids(i) = p;
        // if putting this weight to left overweights the left cut, then
        // increase the space for the next cut using
        // weight_stolen_from_previous_part.
        if(p < no_cuts - 1 &&
          this->kokkos_thread_cut_line_weight_to_put_left(p) < this->sEpsilon) {
            if(mapped_cut == cut_map[p + 1] ) {
              // if the cut before the cut indexed at p was also at the same
              // position special case, as we handle the weight differently here.
              if (previous_cut_map != mapped_cut){
                weight_stolen_from_previous_part =
                  this->kokkos_thread_cut_line_weight_to_put_left(p);
              }
              else {
                // if the cut before the cut indexed at p was also at the same
                // position we assign extra weights cumulatively in this case.
                weight_stolen_from_previous_part +=
                  this->kokkos_thread_cut_line_weight_to_put_left(p);
              }
            }
            else{
              weight_stolen_from_previous_part =
                -this->kokkos_thread_cut_line_weight_to_put_left(p);
            }
            // end assignment for part p
            break;
        }
      } else {
        // if part p does not have enough space for this point
        // and if there is another cut sharing the same positon,
        // again increase the space for the next
        if(p < no_cuts - 1 && mapped_cut == cut_map[p + 1]){
          if (previous_cut_map != mapped_cut){
            weight_stolen_from_previous_part =
              this->kokkos_thread_cut_line_weight_to_put_left(p);
          }
          else {
            weight_stolen_from_previous_part +=
              this->kokkos_thread_cut_line_weight_to_put_left(p);
          }
        }
        else{
          weight_stolen_from_previous_part =
            -this->kokkos_thread_cut_line_weight_to_put_left(p);
        }
        // end assignment for part p
        break;
      }
    }
    previous_cut_map = mapped_cut;
  }

  // TODO commented for reverse order
  // put everything left on the last cut to the last part.
  mj_lno_t sort_vector_end = (mj_lno_t)sort_vector_points_on_cut[
    previous_cut_map].size() - 1;

  // mj_lno_t sort_vector_begin= 0;
  // mj_lno_t sort_vector_size = (mj_lno_t)
  //   sort_vector_points_on_cut[previous_cut_map].size();
  // TODO commented for reverse order
  for (; sort_vector_end >= 0; --sort_vector_end) {
    // TODO commented for reverse order
    multiSItem t = sort_vector_points_on_cut[previous_cut_map][sort_vector_end];
    // multiSItem t =
    //   sort_vector_points_on_cut[previous_cut_map][sort_vector_begin];
    mj_lno_t i = t.index;
    ++this->kokkos_thread_point_counts(no_cuts);
    this->kokkos_assigned_part_ids(i) = no_cuts;
  }

  sort_vector_points_on_cut[previous_cut_map].clear();
  freeArray<mj_part_t> (cut_map);

  //free the memory allocated for vertex sort items .
  mj_lno_t vSize = (mj_lno_t) allocated_memory.size();
  for(mj_lno_t i = 0; i < vSize; ++i){
    freeArray<mj_scalar_t> (allocated_memory[i]);
  }

  // creation of part_xadj as in usual case.
  for(mj_part_t j = 0; j < num_parts; ++j) {
    // TODO: Fnish cleaning this up - refactored out threads but now need to
    // simplify this logic
    mj_lno_t num_points_in_part_j_upto_thread_i = 0;
    mj_lno_t thread_num_points_in_part_j = this->kokkos_thread_point_counts(j);
    this->kokkos_thread_point_counts(j) = num_points_in_part_j_upto_thread_i;
    num_points_in_part_j_upto_thread_i += thread_num_points_in_part_j;
    kokkos_out_part_xadj(j) = num_points_in_part_j_upto_thread_i;
  }

  // perform prefix sum for num_points in parts.
  for(mj_part_t j = 1; j < num_parts; ++j) {
    kokkos_out_part_xadj(j) += kokkos_out_part_xadj(j - 1);
  }

  // shift the num points in threads thread to obtain the
  // beginning index of each thread's private space.
  for(mj_part_t j = 1; j < num_parts; ++j) {
    this->kokkos_thread_point_counts(j) += kokkos_out_part_xadj(j - 1);
  }

  // now thread gets the coordinate and writes the index of coordinate to
  // the permutation array using the part index we calculated.
  for (mj_lno_t ii = coordinate_begin; ii < coordinate_end; ++ii) {
    mj_lno_t i = this->kokkos_coordinate_permutations(ii);
    mj_part_t p =  this->kokkos_assigned_part_ids(i);
    this->kokkos_new_coordinate_permutations(coordinate_begin +
      this->kokkos_thread_point_counts(p)++) = i;
  }
}

/*! \brief Function sends the found partids to the owner of the coordinates, if
 * the data is ever migrated. otherwise, it seets the part numbers and returns.
 * \param current_num_parts is the number of parts in the process.
 * \param output_part_begin_index is the number that will be used as beginning
 * part number
 * \param output_part_boxes is the array that holds the part boxes
 * \param is_data_ever_migrated is the boolean value which is true
 * if the data is ever migrated during the partitioning.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  set_final_parts(
  mj_part_t current_num_parts,
  mj_part_t output_part_begin_index,
  RCP<mj_partBoxVector_t> &output_part_boxes,
  bool is_data_ever_migrated)
{
    this->mj_env->timerStart(MACRO_TIMERS, "MultiJagged - Part_Assignment");

    auto local_kokkos_part_xadj = kokkos_part_xadj;
    auto local_mj_keep_part_boxes = mj_keep_part_boxes;
    auto local_kokkos_coordinate_permutations = kokkos_coordinate_permutations;
    auto local_kokkos_assigned_part_ids = kokkos_assigned_part_ids;

    if(local_mj_keep_part_boxes) {
      for(int i = 0; i < current_num_parts; ++i) {
        (*output_part_boxes)[i].setpId(i + output_part_begin_index);
      }
    }

    Kokkos::TeamPolicy<typename mj_node_t::execution_space> policy(
      current_num_parts, Kokkos::AUTO());
    typedef typename Kokkos::TeamPolicy<typename mj_node_t::execution_space>::
      member_type member_type;
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(member_type team_member) {
      int i = team_member.league_rank();
      Kokkos::parallel_for(Kokkos::TeamThreadRange (team_member, (i != 0) ?
        local_kokkos_part_xadj(i-1) : 0, local_kokkos_part_xadj(i)),
        [=] (int & ii) {
        mj_lno_t k = local_kokkos_coordinate_permutations(ii);
        local_kokkos_assigned_part_ids(k) = i + output_part_begin_index;
      });
    });

    //ArrayRCP<const mj_gno_t> gnoList;
    if(!is_data_ever_migrated){
      //freeArray<mj_gno_t>(this->current_mj_gnos);
      //if(this->num_local_coords > 0){
      //    gnoList = arcpFromArrayView(this->mj_gnos);
      //}
    }
    else {
#ifdef ENABLE_ZOLTAN_MIGRATION
    if (sizeof(mj_lno_t) <=  sizeof(int)) {

      // Cannot use Zoltan_Comm with local ordinals larger than ints.
      // In Zoltan_Comm_Create, the cast int(this->num_local_coords)
      // may overflow.

      // if data is migrated, then send part numbers to the original owners.
      ZOLTAN_COMM_OBJ *plan = NULL;
      MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*(this->mj_problemComm));

      int incoming = 0;
      int message_tag = 7856;

      this->mj_env->timerStart(MACRO_TIMERS,
        "MultiJagged - Final Z1PlanCreating");
      int ierr = Zoltan_Comm_Create( &plan, int(this->num_local_coords),
                      this->owner_of_coordinate, mpi_comm, message_tag,
                      &incoming);
      Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
      this->mj_env->timerStop(MACRO_TIMERS,
        "MultiJagged - Final Z1PlanCreating" );

      mj_gno_t *incoming_gnos = allocMemory< mj_gno_t>(incoming);

      message_tag++;
      this->mj_env->timerStart(MACRO_TIMERS, "MultiJagged - Final Z1PlanComm");
      ierr = Zoltan_Comm_Do( plan, message_tag, (char *) this->current_mj_gnos,
                      sizeof(mj_gno_t), (char *) incoming_gnos);
      Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);

      freeArray<mj_gno_t>(this->current_mj_gnos);
      this->current_mj_gnos = incoming_gnos;

      Kokkos::View<mj_part_t*, device_t>
        kokkos_incoming_partIds(incoming);
      message_tag++;
      ierr = Zoltan_Comm_Do( plan, message_tag,
        (char *) this->kokkos_assigned_part_ids.data(),
        sizeof(mj_part_t), (char *) kokkos_incoming_partIds.data());
      Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);
      this->kokkos_assigned_part_ids = kokkos_incoming_partIds;

      this->mj_env->timerStop(MACRO_TIMERS, "MultiJagged - Final Z1PlanComm");
      ierr = Zoltan_Comm_Destroy(&plan);
      Z2_ASSERT_VALUE(ierr, ZOLTAN_OK);

      this->num_local_coords = incoming;
      //gnoList = arcp(this->current_mj_gnos, 0, this->num_local_coords, true);
    }
    else
#endif  // !ENABLE_ZOLTAN_MIGRATION
    {
      throw std::logic_error("This code not refactored yet - not expected "
        "to work - needs to be on device.");

      //if data is migrated, then send part numbers to the original owners.
      this->mj_env->timerStart(MACRO_TIMERS,
        "MultiJagged - Final DistributorPlanCreating");
      Tpetra::Distributor distributor(this->mj_problemComm);
      ArrayView<const mj_part_t> owners_of_coords(
        this->kokkos_owner_of_coordinate.data(), this->num_local_coords);

      mj_lno_t incoming = distributor.createFromSends(owners_of_coords);
      this->mj_env->timerStop(MACRO_TIMERS,
        "MultiJagged - Final DistributorPlanCreating" );
      this->mj_env->timerStart(MACRO_TIMERS,
        "MultiJagged - Final DistributorPlanComm");
      //migrate gnos to actual owners.
      ArrayRCP<mj_gno_t> received_gnos(incoming);

      ArrayView<mj_gno_t> sent_gnos(this->kokkos_current_mj_gnos.data(),
        this->num_local_coords);
      distributor.doPostsAndWaits<mj_gno_t>(sent_gnos, 1, received_gnos());
      this->kokkos_current_mj_gnos = Kokkos::View<mj_gno_t*, device_t>
        ("kokkos_current_mj_gnos", incoming);
      memcpy( this->kokkos_current_mj_gnos.data(),
        received_gnos.getRawPtr(), incoming * sizeof(mj_gno_t));

      // migrate part ids to actual owners.
      ArrayView<mj_part_t> sent_partids(this->kokkos_assigned_part_ids.data(),
        this->num_local_coords);
      ArrayRCP<mj_part_t> received_partids(incoming);
      distributor.doPostsAndWaits<mj_part_t>(
        sent_partids, 1, received_partids());
      this->kokkos_assigned_part_ids =
        Kokkos::View<mj_part_t*, device_t>
        ("kokkos_assigned_part_ids", incoming);
      memcpy( this->kokkos_assigned_part_ids.data(),
        received_partids.getRawPtr(), incoming * sizeof(mj_part_t));

      this->num_local_coords = incoming;
      this->mj_env->timerStop(MACRO_TIMERS,
        "MultiJagged - Final DistributorPlanComm");
    }
  }

  this->mj_env->timerStop(MACRO_TIMERS, "MultiJagged - Part_Assignment");

  this->mj_env->timerStart(MACRO_TIMERS,
    "MultiJagged - Solution_Part_Assignment");

  // ArrayRCP<mj_part_t> partId;
  // partId = arcp(this->assigned_part_ids, 0, this->num_local_coords, true);

  if (this->mj_keep_part_boxes){
    this->kept_boxes = compute_global_box_boundaries(output_part_boxes);
  }

  this->mj_env->timerStop(MACRO_TIMERS,
    "MultiJagged - Solution_Part_Assignment");
}

/*!\brief Multi Jagged  coordinate partitioning algorithm.
 * \param distribute_points_on_cut_lines_ :  if partitioning can distribute
 * points on same coordinate to different parts.
 * \param max_concurrent_part_calculation_ : how many parts we can calculate
 * concurrently.
 * \param check_migrate_avoid_migration_option_ : whether to migrate=1,
 * avoid migrate=2, or leave decision to MJ=0
 * \param minimum_migration_imbalance_  : when MJ decides whether to migrate,
 * the minimum imbalance for migration.
 * \param migration_type : whether to migrate for perfect load imbalance (0) or
 * less messages.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  set_partitioning_parameters(
  bool distribute_points_on_cut_lines_,
  int max_concurrent_part_calculation_,
  int check_migrate_avoid_migration_option_,
  mj_scalar_t minimum_migration_imbalance_,
  int migration_type_)
{
  this->distribute_points_on_cut_lines = distribute_points_on_cut_lines_;
  this->max_concurrent_part_calculation = max_concurrent_part_calculation_;
  this->check_migrate_avoid_migration_option =
    check_migrate_avoid_migration_option_;
  this->minimum_migration_imbalance = minimum_migration_imbalance_;
  this->migration_type = migration_type_;
}

/*! \brief Multi Jagged  coordinate partitioning algorithm.
 * \param env   library configuration and problem parameters
 * \param problemComm the communicator for the problem
 * \param imbalance_tolerance : the input provided imbalance tolerance.
 * \param num_global_parts: number of target global parts.
 * \param part_no_array: part no array, if provided this will be used for
 * partitioning.
 * \param recursion_depth: if part no array is provided, it is the length of
 * part no array, if part no is not provided than it is the number of steps that
 * algorithm will divide into num_global_parts parts.
 * \param coord_dim: coordinate dimension
 * \param num_local_coords: number of local coordinates
 * \param num_global_coords: number of global coordinates
 * \param initial_mj_gnos: the list of initial global id's
 * \param mj_coordinates: the two dimensional coordinate array.
 * \param num_weights_per_coord: number of weights per coordinate
 * \param mj_uniform_weights: if weight index [i] has uniform weight or not.
 * \param mj_weights: the two dimensional array for weights
 * \param mj_uniform_parts: if the target partitioning aims uniform parts
 * \param mj_part_sizes: if the target partitioning does not aim uniform parts,
 * then weight of each part.
 * \param result_assigned_part_ids: Output - 1D pointer, should be provided as
 * null. Memory is given in the function. the result partids corresponding to
 * the coordinates given in result_mj_gnos.
 * \param result_mj_gnos: Output - 1D pointer, should be provided as null.
 * Memory is given in the function. the result coordinate global id's
 * corresponding to the part_ids array.
 */
template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
void AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t>::
  multi_jagged_part(
  const RCP<const Environment> &env,
  RCP<const Comm<int> > &problemComm,
  double imbalance_tolerance_,
  size_t num_global_parts_,
  Kokkos::View<mj_part_t*, device_t> kokkos_part_no_array_,
  int recursion_depth_,
  int coord_dim_,
  mj_lno_t num_local_coords_,
  mj_gno_t num_global_coords_,
  Kokkos::View<const mj_gno_t*, device_t> kokkos_initial_mj_gnos_,
  Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t>
    kokkos_mj_coordinates_,
  int num_weights_per_coord_,
  Kokkos::View<bool*, device_t> kokkos_mj_uniform_weights_,
  Kokkos::View<mj_scalar_t**, device_t> kokkos_mj_weights_,
  Kokkos::View<bool*, device_t> kokkos_mj_uniform_parts_,
  Kokkos::View<mj_scalar_t**, device_t> kokkos_mj_part_sizes_,
  Kokkos::View<mj_part_t *, device_t> &kokkos_result_assigned_part_ids_,
  Kokkos::View<mj_gno_t*, device_t> &kokkos_result_mj_gnos_)
{
  // purpose of this code is to validate node and UVM status for the tests
  // TODO: Later can remove or make this debug code
  std::cout << "Memory Space: " << mj_node_t::memory_space::name()
    << "  Execution Space: " << mj_node_t::execution_space::name() << std::endl;
      
  clock_mj_create_new_partitions.reset();
  clock_mj_1D_part_while_loop.reset();
  clock_host_copies.reset();
  clock_swap.reset();
  clock_mj_1D_part_init.reset();
  clock_mj_1D_part_init2.reset();
  clock_mj_1D_part_get_weights_init.reset();
  clock_mj_1D_part_get_weights_setup.reset();
  clock_mj_1D_part_get_weights.reset();
  clock_weights1.reset();
  clock_weights2.reset();
  clock_weights3.reset();
  clock_functor_weights.reset();
  clock_weights4.reset();
  clock_weights5.reset();
  clock_weights6.reset();
  clock_functor_rightleft_closest.reset();
  clock_mj_accumulate_thread_results.reset();
  clock_mj_get_new_cut_coordinates_init.reset();
  clock_mj_get_new_cut_coordinates.reset();
  clock_mj_get_new_cut_coordinates_end.reset();
  clock_write_globals.reset();
  clock_mj_1D_part_end.reset();
  
  Clock clock_multi_jagged_part("clock_multi_jagged_part", true);
  Clock clock_multi_jagged_part_init("  clock_multi_jagged_part_init", true);
  Clock clock_multi_jagged_part_init_begin(
    "    clock_multi_jagged_part_init_begin", true);

#ifdef print_debug
  if(comm->getRank() == 0) {
    std::cout << "size of gno:" << sizeof(mj_gno_t) << std::endl;
    std::cout << "size of lno:" << sizeof(mj_lno_t) << std::endl;
    std::cout << "size of mj_scalar_t:" << sizeof(mj_scalar_t) << std::endl;
  }
#endif

  this->mj_env = env;
  this->mj_problemComm = problemComm;
  this->myActualRank = this->myRank = this->mj_problemComm->getRank();
  this->mj_env->timerStart(MACRO_TIMERS, "MultiJagged - Total");
  this->mj_env->debug(3, "In MultiJagged Jagged");
  this->imbalance_tolerance = imbalance_tolerance_;
  this->num_global_parts = num_global_parts_;
  this->kokkos_part_no_array = kokkos_part_no_array_;
  this->recursion_depth = recursion_depth_;
  this->coord_dim = coord_dim_;
  this->num_local_coords = num_local_coords_;
  this->num_global_coords = num_global_coords_;
  this->kokkos_mj_coordinates = kokkos_mj_coordinates_;
  this->kokkos_initial_mj_gnos = kokkos_initial_mj_gnos_;
  this->num_weights_per_coord = num_weights_per_coord_;
  this->kokkos_mj_uniform_weights = kokkos_mj_uniform_weights_;
  this->kokkos_mj_weights = kokkos_mj_weights_;
  this->kokkos_mj_uniform_parts = kokkos_mj_uniform_parts_;
  this->kokkos_mj_part_sizes = kokkos_mj_part_sizes_;

  clock_multi_jagged_part_init_begin.stop();

  // this->set_input_data();

  Clock clock_set_part_specifications(
    "    clock_set_part_specifications", true);
  this->set_part_specifications();
  clock_set_part_specifications.stop();

  Clock clock_allocate_set_work_memory(
    "    clock_allocate_set_work_memory", true);
  this->allocate_set_work_memory();
  clock_allocate_set_work_memory.stop();

  // We duplicate the comm as we create subcommunicators during migration.
  // We keep the problemComm as it is, while comm changes after each migration.
  this->comm = this->mj_problemComm->duplicate();

  // initially there is a single partition
  mj_part_t current_num_parts = 1;
  Kokkos::View<mj_scalar_t *, device_t> kokkos_current_cut_coordinates =
    this->kokkos_all_cut_coordinates;
  this->mj_env->timerStart(MACRO_TIMERS, "MultiJagged - Problem_Partitioning");
  mj_part_t output_part_begin_index = 0;
  mj_part_t future_num_parts = this->total_num_part;
  bool is_data_ever_migrated = false;
  std::vector<mj_part_t> *future_num_part_in_parts =
    new std::vector<mj_part_t> ();
  std::vector<mj_part_t> *next_future_num_parts_in_parts =
    new std::vector<mj_part_t> ();
  next_future_num_parts_in_parts->push_back(this->num_global_parts);
  RCP<mj_partBoxVector_t> input_part_boxes(new mj_partBoxVector_t(), true) ;
  RCP<mj_partBoxVector_t> output_part_boxes(new mj_partBoxVector_t(), true);
  compute_global_box();
  if(this->mj_keep_part_boxes){
    this->init_part_boxes(output_part_boxes);
  }
    
  auto local_kokkos_part_xadj = this->kokkos_part_xadj;

  // Need a device counter - how best to allocate?
  // Putting this allocation in the loops is very costly so moved out here.
  Kokkos::View<mj_part_t*, device_t>
    view_rectilinear_cut_count("view_rectilinear_cut_count", 1);
  Kokkos::View<size_t*, device_t>
    view_total_reduction_size("view_total_reduction_size", 1);

  clock_multi_jagged_part_init.stop();
  Clock clock_multi_jagged_part_loop("  clock_multi_jagged_part_loop", true);

  Clock clock_loopA("    clock_loopA", false);
  Clock clock_main_loop("    clock_main_loop", false);
  Clock clock_main_loop_setup("      clock_main_loop_setup", false);
  Clock clock_mj_get_local_min_max_coord_totW(
    "      clock_mj_get_local_min_max_coord_totW", false);
  Clock clock_mj_get_global_min_max_coord_totW(
    "      clock_mj_get_global_min_max_coord_totW", false);
  Clock clock_main_loop_inner("      clock_main_loop_inner", false);
  Clock clock_main_loop_inner2("      clock_main_loop_inner2", false);
  Clock clock_mj_get_initial_cut_coords_target_weights(
    "      clock_mj_get_initial_cut_coords_target_weights", false);
  Clock clock_set_initial_coordinate_parts(
    "      clock_set_initial_coordinate_parts", false);

  Clock clock_mj_1D_part("      clock_mj_1D_part", false);

  Clock clock_new_part_chunks("      clock_new_part_chunks", false);

  for (int i = 0; i < this->recursion_depth; ++i) {

    clock_loopA.start();

    // convert i to string to be used for debugging purposes.
    std::string istring = Teuchos::toString<int>(i);
    
    Kokkos::View<mj_part_t*, device_t> view_num_partitioning_in_current_dim; 

    // number of parts that will be obtained at the end of this partitioning.
    // future_num_part_in_parts is as the size of current number of parts.
    // holds how many more parts each should be divided in the further
    // iterations. this will be used to calculate
    // view_num_partitioning_in_current_dim, as the number of parts that the
    // part will be partitioned in the current dimension partitioning.

    // next_future_num_parts_in_parts will be as the size of outnumParts,
    // and this will hold how many more parts that each output part
    // should be divided. this array will also be used to determine the weight
    // ratios of the parts. swap the arrays to use iteratively.
    std::vector<mj_part_t> *tmpPartVect= future_num_part_in_parts;
    future_num_part_in_parts = next_future_num_parts_in_parts;
    next_future_num_parts_in_parts = tmpPartVect;

    // clear next_future_num_parts_in_parts array as
    // getPartitionArrays expects it to be empty.
    // it also expects view_num_partitioning_in_current_dim to be empty as well.
    next_future_num_parts_in_parts->clear();
    if(this->mj_keep_part_boxes) {
      RCP<mj_partBoxVector_t> tmpPartBoxes = input_part_boxes;
      input_part_boxes = output_part_boxes;
      output_part_boxes = tmpPartBoxes;
      output_part_boxes->clear();
    }

    // returns the total no. of output parts for this dimension partitioning.
    mj_part_t output_part_count_in_dimension =
      this->update_part_num_arrays(
        view_num_partitioning_in_current_dim,
        future_num_part_in_parts,
        next_future_num_parts_in_parts,
        future_num_parts,
        current_num_parts,
        i,
        input_part_boxes,
        output_part_boxes, 1);
  
    // if the number of obtained parts equal to current number of parts,
    // skip this dimension. For example, this happens when 1 is given in the
    // input part array is given. P=4,5,1,2
    if(output_part_count_in_dimension == current_num_parts) {
      //still need to swap the input output arrays.
      tmpPartVect= future_num_part_in_parts;
      future_num_part_in_parts = next_future_num_parts_in_parts;
      next_future_num_parts_in_parts = tmpPartVect;

      if(this->mj_keep_part_boxes) {
        RCP<mj_partBoxVector_t> tmpPartBoxes = input_part_boxes;
        input_part_boxes = output_part_boxes;
        output_part_boxes = tmpPartBoxes;
      }
      clock_loopA.stop();
      continue;
    }

    // get the coordinate axis along which the partitioning will be done.
    int coordInd = i % this->coord_dim;

    Kokkos::View<mj_scalar_t *, device_t> kokkos_mj_current_dim_coords =
      Kokkos::subview(this->kokkos_mj_coordinates, Kokkos::ALL, coordInd);

    this->mj_env->timerStart(MACRO_TIMERS,
      "MultiJagged - Problem_Partitioning_" + istring);

    // alloc Memory to point the indices
    // of the parts in the permutation array.
    this->kokkos_new_part_xadj = Kokkos::View<mj_lno_t*, device_t>(
      "new part xadj", output_part_count_in_dimension);
 
    // the index where in the new_part_xadj will be written.
    mj_part_t output_part_index = 0;

    // whatever is written to output_part_index will be added with
    // output_coordinate_end_index so that the points will be shifted.
    mj_part_t output_coordinate_end_index = 0;

    mj_part_t current_work_part = 0;
    mj_part_t current_concurrent_num_parts =
      std::min(current_num_parts - current_work_part,
      this->max_concurrent_part_calculation);

    mj_part_t obtained_part_index = 0;

    clock_loopA.stop();
    clock_main_loop.start();

    // run for all available parts.
    for(; current_work_part < current_num_parts;
      current_work_part += current_concurrent_num_parts) {

      clock_main_loop_setup.start();
       
      current_concurrent_num_parts =
        std::min(current_num_parts - current_work_part,
        this->max_concurrent_part_calculation);

      int bDoingWork_int;
      Kokkos::parallel_reduce("Read bDoingWork", 1,
        KOKKOS_LAMBDA(int dummy, int & set_single) {
        set_single = 0;
        for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {
          if(view_num_partitioning_in_current_dim(current_work_part + kk) != 1) {
            set_single = 1;
            break;
          }
        }
      }, bDoingWork_int);
      bool bDoingWork = (bDoingWork_int != 0) ? true : false;

      clock_main_loop_setup.stop();

      clock_mj_get_local_min_max_coord_totW.start();

      this->mj_get_local_min_max_coord_totW(
        current_work_part,
        current_concurrent_num_parts,
        kokkos_mj_current_dim_coords);

      clock_mj_get_local_min_max_coord_totW.stop();

      // 1D partitioning
      if (bDoingWork) {

        clock_mj_get_global_min_max_coord_totW.start();

        // obtain global Min max of the part.
        this->mj_get_global_min_max_coord_totW(
          current_concurrent_num_parts,
          this->kokkos_process_local_min_max_coord_total_weight,
          this->kokkos_global_min_max_coord_total_weight);

        clock_mj_get_global_min_max_coord_totW.stop();

        // represents the total number of cutlines
        // whose coordinate should be determined.
        mj_part_t total_incomplete_cut_count = 0;

        // Compute weight ratios for parts & cuts:
        // e.g., 0.25  0.25  0.5    0.5  0.75 0.75  1
        // part0  cut0  part1 cut1 part2 cut2 part3
        mj_part_t concurrent_part_cut_shift = 0;
        mj_part_t concurrent_part_part_shift = 0;

        for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {


          clock_main_loop_inner.start();

          // same as above - temporary measure to pull these values to host
          // I want to avoid making a parallel loop here for now so I get
          // internal loops running. Then revisit this. TODO: clean it up
          mj_scalar_t min_coordinate;
          auto local_kokkos_global_min_max_coord_total_weight =
            this->kokkos_global_min_max_coord_total_weight;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_scalar_t & set_single) {
            set_single = local_kokkos_global_min_max_coord_total_weight(kk);
          }, min_coordinate);

          mj_scalar_t max_coordinate;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_scalar_t & set_single) {
            set_single = local_kokkos_global_min_max_coord_total_weight(
              kk + current_concurrent_num_parts);
          }, max_coordinate);

          mj_scalar_t global_total_weight;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_scalar_t & set_single) {
            set_single = local_kokkos_global_min_max_coord_total_weight(
              kk + 2*current_concurrent_num_parts);
          }, global_total_weight);

          mj_part_t concurrent_current_part_index = current_work_part + kk;
          
          mj_part_t partition_count;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
            set_single = view_num_partitioning_in_current_dim(
              concurrent_current_part_index);
          }, partition_count);

          Kokkos::View<mj_scalar_t *, device_t> kokkos_usedCutCoordinate =
            Kokkos::subview(kokkos_current_cut_coordinates,
              std::pair<mj_lno_t, mj_lno_t>(
                concurrent_part_cut_shift,
                kokkos_current_cut_coordinates.size()));
          Kokkos::View<mj_scalar_t *, device_t>
            kokkos_current_target_part_weights =
            Kokkos::subview(kokkos_target_part_weights,
              std::pair<mj_lno_t, mj_lno_t>(
                concurrent_part_part_shift,
                kokkos_target_part_weights.size()));
      
          // shift the usedCutCoordinate array as noCuts.
          concurrent_part_cut_shift += partition_count - 1;
          // shift the partRatio array as noParts.
          concurrent_part_part_shift += partition_count;

          clock_main_loop_inner.stop();

          // calculate only if part is not empty,
          // and part will be further partitioned.
          if(partition_count > 1 && min_coordinate <= max_coordinate) {

            clock_main_loop_inner2.start();
          
            // increase num_cuts_do_be_determined by the number of cuts of the
            // current part's cut line number.
            total_incomplete_cut_count += partition_count - 1;
            //set the number of cut lines that should be determined
            //for this part.
            // TODO: eventually this is already in a parallel loop or we
            // clean this up, write to device
            auto local_kokkos_my_incomplete_cut_count =
              this->kokkos_my_incomplete_cut_count;
            Kokkos::parallel_for(
              Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
                (0, 1), KOKKOS_LAMBDA (const int dummy) {
                local_kokkos_my_incomplete_cut_count(kk) = partition_count - 1;
            });
                   
            clock_main_loop_inner2.stop();

            // get the target weights of the parts.
            clock_mj_get_initial_cut_coords_target_weights.start();
            this->mj_get_initial_cut_coords_target_weights(
              min_coordinate,
              max_coordinate,
              partition_count - 1,
              global_total_weight,
              kokkos_usedCutCoordinate,
              kokkos_current_target_part_weights,
              future_num_part_in_parts,
              next_future_num_parts_in_parts,
              concurrent_current_part_index,
              obtained_part_index);

            clock_mj_get_initial_cut_coords_target_weights.stop();

            // TODO: refactor clean up
            mj_lno_t coordinate_end_index;
            Kokkos::parallel_reduce("Read single", 1,
              KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
              set_single =
                local_kokkos_part_xadj(concurrent_current_part_index);
            }, coordinate_end_index);

            // TODO: refactor clean up
            mj_lno_t coordinate_begin_index;
            Kokkos::parallel_reduce("Read single", 1,
              KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
              set_single = concurrent_current_part_index==0 ? 0 :
                local_kokkos_part_xadj(concurrent_current_part_index -1);
            }, coordinate_begin_index);

            // get the initial estimated part assignments of the
            // coordinates.
            this->mj_env->timerStart(MACRO_TIMERS,
              "MultiJagged - Problem_Partitioning_" + istring +
              " set_initial_coordinate_parts()");

            clock_set_initial_coordinate_parts.start();
            
            this->set_initial_coordinate_parts(
              max_coordinate,
              min_coordinate,
              concurrent_current_part_index,
              coordinate_begin_index, coordinate_end_index,
              this->kokkos_coordinate_permutations,
              kokkos_mj_current_dim_coords,
              this->kokkos_assigned_part_ids,
              partition_count);
              this->mj_env->timerStop(MACRO_TIMERS,
                "MultiJagged - Problem_Partitioning_" + istring +
                " set_initial_coordinate_parts()");

            clock_set_initial_coordinate_parts.stop();
          }
          else {
            // e.g., if have fewer coordinates than parts, don't need to do
            // next dim.
            auto local_kokkos_my_incomplete_cut_count =
              this->kokkos_my_incomplete_cut_count;
            Kokkos::parallel_for(
              Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
                (0, 1), KOKKOS_LAMBDA (const int dummy) {
                local_kokkos_my_incomplete_cut_count(kk) = 0;
            });
          }
            
          obtained_part_index += partition_count;
        }
    
        // used imbalance, it is always 0, as it is difficult to
        // estimate a range.
        mj_scalar_t used_imbalance = 0;
        // Determine cut lines for all concurrent parts parts here.
        this->mj_env->timerStart(MACRO_TIMERS,
          "MultiJagged - Problem_Partitioning mj_1D_part()");

        clock_mj_1D_part.start();

        this->mj_1D_part(
          kokkos_mj_current_dim_coords,
          used_imbalance,
          current_work_part,
          current_concurrent_num_parts,
          kokkos_current_cut_coordinates,
          total_incomplete_cut_count,
          view_num_partitioning_in_current_dim,
          view_rectilinear_cut_count,
          view_total_reduction_size);
        
        this->mj_env->timerStop(MACRO_TIMERS,
          "MultiJagged - Problem_Partitioning mj_1D_part()");

        clock_mj_1D_part.stop();
      }

      clock_new_part_chunks.start();
            
      // create new part chunks
      {
        mj_part_t output_array_shift = 0;
        mj_part_t cut_shift = 0;
        size_t tlr_shift = 0;
        size_t partweight_array_shift = 0;
        for(int kk = 0; kk < current_concurrent_num_parts; ++kk) {

          mj_part_t current_concurrent_work_part = current_work_part + kk;

          // TODO: refactor clean up
          mj_part_t num_parts;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
            set_single = view_num_partitioning_in_current_dim(
              current_concurrent_work_part);
          }, num_parts);

          // if the part is empty, skip the part.

          // TODO: Clean up later - for now pull some values to host and keep
          // the algorithm serial host at this point
          int coordinateA_bigger_than_coordinateB = false;
          auto local_kokkos_global_min_max_coord_total_weight =
            this->kokkos_global_min_max_coord_total_weight;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, int & set_single) {
            set_single = local_kokkos_global_min_max_coord_total_weight(kk) >
              (local_kokkos_global_min_max_coord_total_weight(
                kk + current_concurrent_num_parts));
          }, coordinateA_bigger_than_coordinateB);

          if((num_parts != 1) && coordinateA_bigger_than_coordinateB) {
            // we still need to write the begin and end point of the empty part.
            // simply set it zero, the array indices will be shifted later
            auto local_kokkos_new_part_xadj = this->kokkos_new_part_xadj;
            Kokkos::parallel_for(
              Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
                (0, num_parts), KOKKOS_LAMBDA (const int jj) {
                local_kokkos_new_part_xadj(
                  output_part_index + output_array_shift + jj) = 0;
            });

            cut_shift += num_parts - 1;
            tlr_shift += (4 *(num_parts - 1) + 1);
            output_array_shift += num_parts;
            partweight_array_shift += (2 * (num_parts - 1) + 1);
            continue;
          }

          Kokkos::View<mj_scalar_t *, device_t>
            kokkos_current_concurrent_cut_coordinate =
            Kokkos::subview(kokkos_current_cut_coordinates,
              std::pair<mj_lno_t, mj_lno_t>(
                cut_shift,
                kokkos_current_cut_coordinates.size()));
          Kokkos::View<mj_scalar_t *, device_t>
            kokkos_used_local_cut_line_weight_to_left =
            Kokkos::subview(kokkos_process_cut_line_weight_to_put_left,
              std::pair<mj_lno_t, mj_lno_t>(
                cut_shift,
                kokkos_process_cut_line_weight_to_put_left.size()));

          this->kokkos_thread_part_weight_work =
            Kokkos::subview(
              this->kokkos_thread_part_weights,
              std::pair<mj_lno_t, mj_lno_t>(
                partweight_array_shift,
                this->kokkos_thread_part_weights.extent(0)));

          if(num_parts > 1) {
            if(this->mj_keep_part_boxes) {
              // if part boxes are to be stored update the boundaries.
              for (mj_part_t j = 0; j < num_parts - 1; ++j) {
                // TODO: need to refactor output_part_boxes to a View form
                // Then refactor this loop to a parallel_for so all on device
                mj_scalar_t temp_get_val;
                Kokkos::parallel_reduce("Read single", 1,
                  KOKKOS_LAMBDA(int dummy, mj_scalar_t & set_single) {
                  set_single = kokkos_current_concurrent_cut_coordinate(j);
                }, temp_get_val);
                (*output_part_boxes)
                  [output_array_shift + output_part_index + j].
                  updateMinMax(temp_get_val, 1 /*update max*/, coordInd);
                (*output_part_boxes)
                  [output_array_shift + output_part_index + j + 1].
                  updateMinMax(temp_get_val, 0 /*update max*/, coordInd);
              }
            }
          
            // Rewrite the indices based on the computed cuts.
            this->mj_create_new_partitions(
              num_parts,
              current_concurrent_work_part,
              kokkos_mj_current_dim_coords,
              kokkos_current_concurrent_cut_coordinate,
              kokkos_used_local_cut_line_weight_to_left,
              this->kokkos_thread_part_weight_work,
              Kokkos::subview(this->kokkos_new_part_xadj,
                std::pair<mj_lno_t, mj_lno_t>(
                  output_part_index + output_array_shift,
                  this->kokkos_new_part_xadj.size())),  
              this->kokkos_thread_point_counts,
              this->distribute_points_on_cut_lines,
              this->kokkos_thread_cut_line_weight_to_put_left,
              this->sEpsilon,
              this->kokkos_coordinate_permutations,
              this->kokkos_mj_uniform_weights,
              this->kokkos_mj_weights,
              this->kokkos_assigned_part_ids,
              this->kokkos_new_coordinate_permutations);
          }
          else {
            // This should all get simplified into device code
            mj_lno_t coordinate_end;
            Kokkos::parallel_reduce("Read single", 1,
              KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
              set_single =
                local_kokkos_part_xadj[current_concurrent_work_part];;
            }, coordinate_end);

            mj_lno_t coordinate_begin;
            Kokkos::parallel_reduce("Read single", 1,
              KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
              set_single = current_concurrent_work_part==0 ? 0 :
                local_kokkos_part_xadj(current_concurrent_work_part -1);
            }, coordinate_begin);

            // if this part is partitioned into 1 then just copy
            // the old values.
            mj_lno_t part_size = coordinate_end - coordinate_begin;

            // TODO: how to best set 1 value...
            auto local_kokkos_new_part_xadj = this->kokkos_new_part_xadj;                        
            Kokkos::parallel_for(
              Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
              (0, 1), KOKKOS_LAMBDA (const int dummy) {
              local_kokkos_new_part_xadj(
                output_part_index + output_array_shift) = part_size;
            });

            auto local_kokkos_new_coordinate_permutations =
              this->kokkos_new_coordinate_permutations;
            auto local_kokkos_coordinate_permutations =
              this->kokkos_coordinate_permutations;
            Kokkos::parallel_for(
              Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
                (0, part_size), KOKKOS_LAMBDA (const int n) {
              local_kokkos_new_coordinate_permutations(n+coordinate_begin) =
                local_kokkos_coordinate_permutations(n+coordinate_begin);
            });
          }
          cut_shift += num_parts - 1;
          output_array_shift += num_parts;
          partweight_array_shift += (2 * (num_parts - 1) + 1);
        }

        // shift cut coordinates so that all cut coordinates are stored.
        // no shift now because we dont keep the cuts.
        // current_cut_coordinates += cut_shift;
        // mj_create_new_partitions from coordinates partitioned the parts
        // and write the indices as if there were a single part.
        // now we need to shift the beginning indices.
        for(mj_part_t kk = 0; kk < current_concurrent_num_parts; ++kk) {
          // TODO: refactor clean up
          mj_part_t num_parts;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_lno_t & set_single) {
            set_single = view_num_partitioning_in_current_dim(
              current_work_part + kk);
          }, num_parts);

          auto local_kokkos_new_part_xadj = this->kokkos_new_part_xadj;
          Kokkos::parallel_for(
            Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
            (0, num_parts), KOKKOS_LAMBDA (const int ii) {
            local_kokkos_new_part_xadj(output_part_index+ii) +=
              output_coordinate_end_index;
          });
    
          // increase the previous count by current end.
          mj_part_t temp_get;
          Kokkos::parallel_reduce("Read single", 1,
            KOKKOS_LAMBDA(int dummy, mj_part_t & set_single) {
            set_single =
              local_kokkos_new_part_xadj(output_part_index + num_parts - 1);
          }, temp_get);
          output_coordinate_end_index = temp_get;
          //increase the current out.
          output_part_index += num_parts;
        }
      }
      clock_new_part_chunks.stop();
    }

    clock_main_loop.stop();

    // end of this partitioning dimension
    int current_world_size = this->comm->getSize();
    long migration_reduce_all_population =
      this->total_dim_num_reduce_all * current_world_size;
    bool is_migrated_in_current_dimension = false;

    // we migrate if there are more partitionings to be done after this step
    // and if the migration is not forced to be avoided.
    // and the operation is not sequential.
    if (future_num_parts > 1 &&
      this->check_migrate_avoid_migration_option >= 0 &&
      current_world_size > 1) {
      this->mj_env->timerStart(MACRO_TIMERS,
        "MultiJagged - Problem_Migration-" + istring);
      mj_part_t num_parts = output_part_count_in_dimension;

      if (this->mj_perform_migration(
        num_parts,
        current_num_parts, //output
        next_future_num_parts_in_parts, //output
        output_part_begin_index,
        migration_reduce_all_population,
        this->num_local_coords / (future_num_parts * current_num_parts),
        istring,
        input_part_boxes, output_part_boxes) )
      {
        is_migrated_in_current_dimension = true;
        is_data_ever_migrated = true;
        this->mj_env->timerStop(MACRO_TIMERS, 
          "MultiJagged - Problem_Migration-" + istring);
        // since data is migrated, we reduce the number of reduceAll
        // operations for the last part.
        this->total_dim_num_reduce_all /= num_parts;
      }
      else {
        is_migrated_in_current_dimension = false;
        this->mj_env->timerStop(MACRO_TIMERS,
          "MultiJagged - Problem_Migration-" + istring);
      }
    }

    // swap the coordinate permutations for the next dimension.
    Kokkos::View<mj_lno_t*, device_t> tmp =
      this->kokkos_coordinate_permutations;
    this->kokkos_coordinate_permutations =
      this->kokkos_new_coordinate_permutations;
    this->kokkos_new_coordinate_permutations = tmp;
    if(!is_migrated_in_current_dimension) {
      this->total_dim_num_reduce_all -= current_num_parts;
      current_num_parts = output_part_count_in_dimension;
    }

    {
      this->kokkos_part_xadj = this->kokkos_new_part_xadj;
      local_kokkos_part_xadj = this->kokkos_new_part_xadj;

      this->kokkos_new_part_xadj = Kokkos::View<mj_lno_t*, device_t>("empty");
      this->mj_env->timerStop(MACRO_TIMERS,
        "MultiJagged - Problem_Partitioning_" + istring);
    }
  }
  clock_multi_jagged_part_loop.stop();
  Clock clock_multi_jagged_part_finish("  clock_multi_jagged_part_finish", true);

  // Partitioning is done
  delete future_num_part_in_parts;
  delete next_future_num_parts_in_parts;
  this->mj_env->timerStop(MACRO_TIMERS, "MultiJagged - Problem_Partitioning");
  /////////////////////////////End of the partitioning////////////////////////

  //get the final parts of each initial coordinate
  //the results will be written to
  //this->assigned_part_ids for gnos given in this->current_mj_gnos
  this->set_final_parts(
    current_num_parts,
    output_part_begin_index,
    output_part_boxes,
    is_data_ever_migrated);

  kokkos_result_assigned_part_ids_ = this->kokkos_assigned_part_ids;
  kokkos_result_mj_gnos_ = this->kokkos_current_mj_gnos;
  this->mj_env->timerStop(MACRO_TIMERS, "MultiJagged - Total");
  this->mj_env->debug(3, "Out of MultiJagged");

  clock_multi_jagged_part_finish.stop();

  clock_multi_jagged_part.stop();

  printf("-------------------------------------------------------\n");
  clock_multi_jagged_part.print();
  clock_multi_jagged_part_init.print();
  clock_multi_jagged_part_init_begin.print();
  clock_set_part_specifications.print();
  clock_allocate_set_work_memory.print();

  clock_multi_jagged_part_loop.print();
  clock_loopA.print();
  clock_main_loop.print();
  clock_main_loop_setup.print();
  clock_mj_get_local_min_max_coord_totW.print();
  clock_mj_get_global_min_max_coord_totW.print();
  clock_main_loop_inner.print();
  clock_main_loop_inner2.print();
  clock_mj_get_initial_cut_coords_target_weights.print();
  clock_set_initial_coordinate_parts.print();

  clock_mj_1D_part.print();

  clock_mj_1D_part_init.print();
  clock_mj_1D_part_init2.print();
  clock_mj_1D_part_while_loop.print();
  clock_host_copies.print();
  clock_mj_1D_part_get_weights_init.print();
  clock_mj_1D_part_get_weights_setup.print();
  clock_mj_1D_part_get_weights.print();

  clock_weights1.print();
  clock_weights2.print();
  clock_weights3.print();
  clock_functor_weights.print();
  clock_weights4.print();
  clock_weights5.print();
  clock_weights6.print();
  clock_functor_rightleft_closest.print();

  clock_mj_accumulate_thread_results.print();

  clock_write_globals.print();

  clock_mj_get_new_cut_coordinates_init.print();
  clock_mj_get_new_cut_coordinates.print();
  clock_mj_get_new_cut_coordinates_end.print();
  clock_swap.print();

  clock_mj_1D_part_end.print();

  clock_new_part_chunks.print();
  clock_mj_create_new_partitions.print();

  clock_multi_jagged_part_finish.print();
  
  printf("-------------------------------------------------------\n");
}

template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
RCP<typename AlgMJ<mj_scalar_t,mj_lno_t,mj_gno_t,mj_part_t, mj_node_t>::
  mj_partBoxVector_t>
AlgMJ<mj_scalar_t,mj_lno_t,mj_gno_t,mj_part_t, mj_node_t>::
  get_kept_boxes() const
{
  if (this->mj_keep_part_boxes) {
    return this->kept_boxes;
  }
  else {
    throw std::logic_error("Error: part boxes are not stored.");
  }
}

template <typename mj_scalar_t, typename mj_lno_t, typename mj_gno_t,
  typename mj_part_t, typename mj_node_t>
RCP<typename AlgMJ<mj_scalar_t,mj_lno_t,mj_gno_t,mj_part_t, mj_node_t>::
  mj_partBoxVector_t>
AlgMJ<mj_scalar_t,mj_lno_t,mj_gno_t,mj_part_t, mj_node_t>::
  compute_global_box_boundaries(RCP<mj_partBoxVector_t> &localPartBoxes) const
{
  mj_part_t ntasks = this->num_global_parts;
  int dim = (*localPartBoxes)[0].getDim();
  mj_scalar_t *localPartBoundaries = new mj_scalar_t[ntasks * 2 *dim];

  memset(localPartBoundaries, 0, sizeof(mj_scalar_t) * ntasks * 2 *dim);

  mj_scalar_t *globalPartBoundaries = new mj_scalar_t[ntasks * 2 *dim];
  memset(globalPartBoundaries, 0, sizeof(mj_scalar_t) * ntasks * 2 *dim);

  mj_scalar_t *localPartMins = localPartBoundaries;
  mj_scalar_t *localPartMaxs = localPartBoundaries + ntasks * dim;

  mj_scalar_t *globalPartMins = globalPartBoundaries;
  mj_scalar_t *globalPartMaxs = globalPartBoundaries + ntasks * dim;

  mj_part_t boxCount = localPartBoxes->size();
  for (mj_part_t i = 0; i < boxCount; ++i){
    mj_part_t pId = (*localPartBoxes)[i].getpId();

    // cout << "me:" << comm->getRank() << " has:" << pId << endl;

    mj_scalar_t *lmins = (*localPartBoxes)[i].getlmins();
    mj_scalar_t *lmaxs = (*localPartBoxes)[i].getlmaxs();

    for (int j = 0; j < dim; ++j){
      localPartMins[dim * pId + j] = lmins[j];
      localPartMaxs[dim * pId + j] = lmaxs[j];
      
      /*
      std::cout << "me:" << comm->getRank()  <<
              " dim * pId + j:"<< dim * pId + j <<
              " localMin:" << localPartMins[dim * pId + j] <<
              " localMax:" << localPartMaxs[dim * pId + j] << std::endl;
      */
    }
  }

  Teuchos::Zoltan2_BoxBoundaries<int, mj_scalar_t> reductionOp(ntasks * 2 *dim);

  reduceAll<int, mj_scalar_t>(*mj_problemComm, reductionOp,
    ntasks * 2 *dim, localPartBoundaries, globalPartBoundaries);

  RCP<mj_partBoxVector_t> pB(new mj_partBoxVector_t(),true);
  for (mj_part_t i = 0; i < ntasks; ++i) {
    Zoltan2::coordinateModelPartBox <mj_scalar_t, mj_part_t> tpb(
      i, dim, globalPartMins + dim * i, globalPartMaxs + dim * i);

    /*
    for (int j = 0; j < dim; ++j){
        std::cout << "me:" << comm->getRank()  <<
                " dim * pId + j:"<< dim * i + j <<
                " globalMin:" << globalPartMins[dim * i + j] <<
                " globalMax:" << globalPartMaxs[dim * i + j] << std::endl;
    }
    */
    
    pB->push_back(tpb);
  }
  delete []localPartBoundaries;
  delete []globalPartBoundaries;
  //RCP <mj_partBoxVector_t> tmpRCPBox(pB, true);
  return pB;
}

/*! \brief Multi Jagged coordinate partitioning algorithm.
 */
template <typename Adapter>
class Zoltan2_AlgMJ : public Algorithm<Adapter>
{

// TODO: Changed all to public for cuda refactoring - need to to work on design
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef CoordinateModel<typename Adapter::base_adapter_t> coordinateModel_t;
  typedef typename Adapter::scalar_t mj_scalar_t;
  typedef typename Adapter::gno_t mj_gno_t;
  typedef typename Adapter::lno_t mj_lno_t;
  typedef typename Adapter::part_t mj_part_t;
  typedef typename Adapter::node_t mj_node_t;
  typedef coordinateModelPartBox<mj_scalar_t, mj_part_t> mj_partBox_t;
  typedef std::vector<mj_partBox_t> mj_partBoxVector_t;
  typedef typename mj_node_t::device_type device_t;
#endif

   AlgMJ<mj_scalar_t, mj_lno_t, mj_gno_t, mj_part_t, mj_node_t> mj_partitioner;

  RCP<const Environment> mj_env; // the environment object
  RCP<const Comm<int> > mj_problemComm; // initial comm object
  RCP<const coordinateModel_t> mj_coords; // coordinate adapter

  // PARAMETERS
  double imbalance_tolerance; // input imbalance tolerance.
  size_t num_global_parts; // the targeted number of parts

  // input part array specifying num part to divide along each dim.
  Kokkos::View<mj_part_t*, device_t> kokkos_part_no_array;

  // the number of steps that partitioning will be solved in.
  int recursion_depth;

  int coord_dim; // coordinate dimension.
  mj_lno_t num_local_coords; //number of local coords.
  mj_gno_t num_global_coords; //number of global coords.

  // initial global ids of the coordinates.
  Kokkos::View<const mj_gno_t*, device_t> kokkos_initial_mj_gnos;
  
  // two dimension coordinate array.
  Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t>
    kokkos_mj_coordinates;

  int num_weights_per_coord; // number of weights per coordinate

  // if the target parts are uniform.
  Kokkos::View<bool*, device_t> kokkos_mj_uniform_weights;
  
  // two dimensional weight array.
  Kokkos::View<mj_scalar_t**, device_t> kokkos_mj_weights;
  
  // if the target parts are uniform
  Kokkos::View<bool*, device_t> kokkos_mj_uniform_parts;
  
  // target part weight sizes.
  Kokkos::View<mj_scalar_t**, device_t> kokkos_mj_part_sizes;

  // if partitioning can distribute points on same coordiante to
  // different parts.
  bool distribute_points_on_cut_lines;
  
  // how many parts we can calculate concurrently.
  mj_part_t max_concurrent_part_calculation;
  
  // whether to migrate=1, avoid migrate=2, or leave decision to MJ=0
  int check_migrate_avoid_migration_option;
  
  // when doing the migration, 0 will aim for perfect load-imbalance, 
  int migration_type;

  // 1 for minimized messages
    
  // when MJ decides whether to migrate, the minimum imbalance for migration.
  mj_scalar_t minimum_migration_imbalance;
  bool mj_keep_part_boxes; //if the boxes need to be kept.

  // if this is set, then recursion depth is adjusted to its maximum value.
  bool mj_run_as_rcb;
  int mj_premigration_option;
  int min_coord_per_rank_for_premigration;

  // communication graph xadj
  ArrayRCP<mj_part_t> comXAdj_;
  
  // communication graph adj.
  ArrayRCP<mj_part_t> comAdj_;

  void set_up_partitioning_data(
    const RCP<PartitioningSolution<Adapter> >&solution);

  void set_input_parameters(const Teuchos::ParameterList &p);

  RCP<mj_partBoxVector_t> getGlobalBoxBoundaries() const;
    
  bool mj_premigrate_to_subset(
    int used_num_ranks,
    int migration_selection_option,
    RCP<const Environment> mj_env_,
    RCP<const Comm<int> > mj_problemComm_,
    int coord_dim_,
    mj_lno_t num_local_coords_,
    mj_gno_t num_global_coords_,  size_t num_global_parts_,
    Kokkos::View<const mj_gno_t*, device_t> &kokkos_initial_mj_gnos_,
    Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t> &
      kokkos_mj_coordinates_,
    int num_weights_per_coord_,
    Kokkos::View<mj_scalar_t**, device_t> &kokkos_mj_weights_,
    //results
    RCP<const Comm<int> > &result_problemComm_,
    mj_lno_t & result_num_local_coords_,
    Kokkos::View<mj_gno_t*, device_t> &kokkos_result_initial_mj_gnos_,
    Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t> &
      kokkos_result_mj_coordinates_,
    Kokkos::View<mj_scalar_t**, device_t> &kokkos_result_mj_weights_,
    int * &result_actual_owner_rank_);

public:

  Zoltan2_AlgMJ(const RCP<const Environment> &env,
    RCP<const Comm<int> > &problemComm,
    const RCP<const coordinateModel_t> &coords) :
      mj_partitioner(),
      mj_env(env),
      mj_problemComm(problemComm),
      mj_coords(coords),
      imbalance_tolerance(0),
      num_global_parts(1),
      recursion_depth(0),
      coord_dim(0),
      num_local_coords(0),
      num_global_coords(0),
      num_weights_per_coord(0),
      distribute_points_on_cut_lines(true),
      max_concurrent_part_calculation(1),
      check_migrate_avoid_migration_option(0),
      migration_type(0),
      minimum_migration_imbalance(0.30),
      mj_keep_part_boxes(false),
      mj_run_as_rcb(false),
      mj_premigration_option(0),
      min_coord_per_rank_for_premigration(32000),
      comXAdj_(),
      comAdj_()
  {
  }

  ~Zoltan2_AlgMJ()
  {
  }

  /*! \brief Set up validators specific to this algorithm
   */
  static void getValidParameters(ParameterList & pl)
  {
    const bool bUnsorted = true; // this clarifies the flag is for unsrorted
    RCP<Zoltan2::IntegerRangeListValidator<int>> mj_parts_Validator =
    Teuchos::rcp( new Zoltan2::IntegerRangeListValidator<int>(bUnsorted) );
    pl.set("mj_parts", "0", "list of parts for multiJagged partitioning "
      "algorithm. As many as the dimension count.", mj_parts_Validator);

    pl.set("mj_concurrent_part_count", 1, "The number of parts whose cut "
      "coordinates will be calculated concurently.",
      Environment::getAnyIntValidator());

    pl.set("mj_minimum_migration_imbalance", 1.1,
      "mj_minimum_migration_imbalance, the minimum imbalance of the "
      "processors to avoid migration",
      Environment::getAnyDoubleValidator());

    RCP<Teuchos::EnhancedNumberValidator<int>> mj_migration_option_validator =
      Teuchos::rcp( new Teuchos::EnhancedNumberValidator<int>(0, 2) );
    pl.set("mj_migration_option", 1, "Migration option, 0 for decision "
      "depending on the imbalance, 1 for forcing migration, 2 for "
      "avoiding migration", mj_migration_option_validator);

    RCP<Teuchos::EnhancedNumberValidator<int>> mj_migration_type_validator =
      Teuchos::rcp( new Teuchos::EnhancedNumberValidator<int>(0, 1) );
      pl.set("mj_migration_type", 0,
      "Migration type, 0 for migration to minimize the imbalance "
      "1 for migration to minimize messages exchanged the migration.",
      mj_migration_option_validator);

    // bool parameter
    pl.set("mj_keep_part_boxes", false, "Keep the part boundaries of the "
      "geometric partitioning.", Environment::getBoolValidator());

    // bool parameter
    pl.set("mj_enable_rcb", false, "Use MJ as RCB.",
      Environment::getBoolValidator());

    pl.set("mj_recursion_depth", -1, "Recursion depth for MJ: Must be "
      "greater than 0.", Environment::getAnyIntValidator());

    RCP<Teuchos::EnhancedNumberValidator<int>>
      mj_premigration_option_validator =
      Teuchos::rcp( new Teuchos::EnhancedNumberValidator<int>(0, 1024) );

    pl.set("mj_premigration_option", 0,
      "Whether to do premigration or not. 0 for no migration "
      "x > 0 for migration to consecutive processors, "
      "the subset will be 0,x,2x,3x,...subset ranks."
      , mj_premigration_option_validator);

    pl.set("mj_premigration_coordinate_count", 32000, "How many coordinate to "
      "assign each rank in multijagged after premigration"
      , Environment::getAnyIntValidator());
  }

  /*! \brief Multi Jagged  coordinate partitioning algorithm.
   *
   *  \param solution  a PartitioningSolution, on input it
   *      contains part information, on return it also contains
   *      the solution and quality metrics.
   */
  void partition(const RCP<PartitioningSolution<Adapter> > &solution);

  mj_partBoxVector_t &getPartBoxesView() const
  {
    RCP<mj_partBoxVector_t> pBoxes = this->getGlobalBoxBoundaries();
    return *pBoxes;
  }

  mj_part_t pointAssign(int dim, mj_scalar_t *point) const;

  void boxAssign(int dim, mj_scalar_t *lower, mj_scalar_t *upper,
    size_t &nPartsFound, mj_part_t **partsFound) const;

  /*! \brief returns communication graph resulting from MJ partitioning.
   */
  void getCommunicationGraph(
    const PartitioningSolution<Adapter> *solution,
    ArrayRCP<mj_part_t> &comXAdj,
    ArrayRCP<mj_part_t> &comAdj);
};

template <typename Adapter>
bool Zoltan2_AlgMJ<Adapter>::mj_premigrate_to_subset(
  int used_num_ranks, 
  int migration_selection_option,
  RCP<const Environment> mj_env_,
  RCP<const Comm<int> > mj_problemComm_,
  int coord_dim_,
  mj_lno_t num_local_coords_,
  mj_gno_t num_global_coords_, size_t num_global_parts_,
  Kokkos::View<const mj_gno_t*, device_t> &kokkos_initial_mj_gnos_,
  Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t> &
    kokkos_mj_coordinates_,
  int num_weights_per_coord_,
  Kokkos::View<mj_scalar_t**, device_t> &kokkos_mj_weights_,
  //results
  RCP<const Comm<int> > &result_problemComm_,
  mj_lno_t &result_num_local_coords_,
  Kokkos::View<mj_gno_t*, device_t> &kokkos_result_initial_mj_gnos_,
  Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t> &
    kokkos_result_mj_coordinates_,
  Kokkos::View<mj_scalar_t**, device_t> &kokkos_result_mj_weights_,
  int * &result_actual_owner_rank_)
{
  mj_env_->timerStart(MACRO_TIMERS,
    "MultiJagged - PreMigration DistributorPlanCreating");
  
  int myRank = mj_problemComm_->getRank();
  int worldSize = mj_problemComm_->getSize();
  
  mj_part_t groupsize = worldSize / used_num_ranks;

  std::vector<mj_part_t> group_begins(used_num_ranks + 1, 0);

  mj_part_t i_am_sending_to = 0;
  bool am_i_a_reciever = false;

  for(int i = 0; i < used_num_ranks; ++i){
    group_begins[i+ 1]  = group_begins[i] + groupsize;
    if (worldSize % used_num_ranks > i) group_begins[i+ 1] += 1;
    if (i == used_num_ranks) group_begins[i+ 1] = worldSize;
    if (myRank >= group_begins[i] && myRank < group_begins[i + 1]) {
      i_am_sending_to = group_begins[i];
    }
    if (myRank == group_begins[i]) {
      am_i_a_reciever = true;
    }
  }
  
  ArrayView<const mj_part_t> idView(&(group_begins[0]), used_num_ranks );
  result_problemComm_ = mj_problemComm_->createSubcommunicator(idView);

  Tpetra::Distributor distributor(mj_problemComm_);

  std::vector<mj_part_t>
    coordinate_destinations(num_local_coords_, i_am_sending_to);

  ArrayView<const mj_part_t>
    destinations(&(coordinate_destinations[0]), num_local_coords_);
  mj_lno_t num_incoming_gnos = distributor.createFromSends(destinations);
  result_num_local_coords_ = num_incoming_gnos;
  mj_env_->timerStop(MACRO_TIMERS,
    "MultiJagged - PreMigration DistributorPlanCreating");

  mj_env_->timerStart(MACRO_TIMERS,
    "MultiJagged - PreMigration DistributorMigration");
   
  /*
  // migrate gnos.
  {
    ArrayRCP<mj_gno_t> received_gnos(num_incoming_gnos);

    ArrayView<const mj_gno_t> sent_gnos(initial_mj_gnos_, num_local_coords_);
    distributor.doPostsAndWaits<mj_gno_t>(sent_gnos, 1, received_gnos());

    // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
    // started - not tested yet
    // kokkos_result_initial_mj_gnos_ = Kokkos::View<mj_gno_t *, device_t>(
    //  "gids", num_incoming_gnos);
    //for(int n = 0; n < num_incoming_gnos; ++n) {
    //  kokkos_result_initial_mj_gnos_(n) = received_gnos[n];
    //}
    
    result_initial_mj_gnos_ = allocMemory<mj_gno_t>(num_incoming_gnos);
    memcpy(
	  result_initial_mj_gnos_,
	  received_gnos.getRawPtr(),
	  num_incoming_gnos * sizeof(mj_gno_t));
  }
  */

  //migrate coordinates
  kokkos_result_mj_coordinates_ = Kokkos::View<mj_scalar_t **,
    Kokkos::LayoutLeft, device_t>("coords", num_local_coords_);

  // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
  /*
  result_mj_coordinates_ = allocMemory<mj_scalar_t *>(coord_dim_);
  for (int i = 0; i < coord_dim_; ++i){
    ArrayView<const mj_scalar_t>
      sent_coord(mj_coordinates_[i], num_local_coords_);
    ArrayRCP<mj_scalar_t> received_coord(num_incoming_gnos);
    distributor.doPostsAndWaits<mj_scalar_t>(sent_coord, 1, received_coord());
    result_mj_coordinates_[i] = allocMemory<mj_scalar_t>(num_incoming_gnos);
    memcpy(
	    result_mj_coordinates_[i],
	    received_coord.getRawPtr(),
	    num_incoming_gnos * sizeof(mj_scalar_t));
  }
  */

  // migrate weights.
  // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
  /*
  result_mj_weights_ = allocMemory<mj_scalar_t *>(num_weights_per_coord_);
  for (int i = 0; i < num_weights_per_coord_; ++i){
    ArrayView<const mj_scalar_t> sent_weight(mj_weights_[i], num_local_coords_);
    ArrayRCP<mj_scalar_t> received_weight(num_incoming_gnos);
    distributor.doPostsAndWaits<mj_scalar_t>(sent_weight, 1, received_weight());
    result_mj_weights_[i] = allocMemory<mj_scalar_t>(num_incoming_gnos);
    memcpy(
	  result_mj_weights_[i],
	  received_weight.getRawPtr(),
	  num_incoming_gnos * sizeof(mj_scalar_t));
  }
  */

  // migrate the owners of the coordinates
  // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
  /*
  { 
    std::vector<int> owner_of_coordinate(num_local_coords_, myRank);
    ArrayView<int> sent_owners(&(owner_of_coordinate[0]), num_local_coords_);
    ArrayRCP<int> received_owners(num_incoming_gnos);
    distributor.doPostsAndWaits<int>(sent_owners, 1, received_owners());
    result_actual_owner_rank_ = allocMemory<int>(num_incoming_gnos);
    memcpy(
	    result_actual_owner_rank_,
	    received_owners.getRawPtr(),
	    num_incoming_gnos * sizeof(int));
  }
  */

  mj_env_->timerStop(MACRO_TIMERS,
    "MultiJagged - PreMigration DistributorMigration");
  return am_i_a_reciever;
}

/*! \brief Multi Jagged  coordinate partitioning algorithm.
 * \param env   library configuration and problem parameters
 * \param problemComm the communicator for the problem
 * \param coords    a CoordinateModel with user data
 * \param solution  a PartitioningSolution, on input it contains part
 * information, on return it also contains the solution and quality metrics.
 */
template <typename Adapter>
void Zoltan2_AlgMJ<Adapter>::partition(
  const RCP<PartitioningSolution<Adapter> > &solution)
{
  this->mj_env->timerStart(MACRO_TIMERS, "partition() - all");

  {
    this->mj_env->timerStart(MACRO_TIMERS, "partition() - setup");

    this->set_up_partitioning_data(solution);

    this->set_input_parameters(this->mj_env->getParameters());
    if (this->mj_keep_part_boxes) {
        this->mj_partitioner.set_to_keep_part_boxes();
    }

    this->mj_partitioner.set_partitioning_parameters(
      this->distribute_points_on_cut_lines,
      this->max_concurrent_part_calculation,
      this->check_migrate_avoid_migration_option,
      this->minimum_migration_imbalance, this->migration_type);

    RCP<const Comm<int> > result_problemComm = this->mj_problemComm;
    mj_lno_t result_num_local_coords = this->num_local_coords;
    Kokkos::View<mj_gno_t*, device_t> kokkos_result_initial_mj_gnos;
    Kokkos::View<mj_scalar_t**, Kokkos::LayoutLeft, device_t>
      kokkos_result_mj_coordinates = this->kokkos_mj_coordinates;
    Kokkos::View<mj_scalar_t**, device_t> kokkos_result_mj_weights =
      this->kokkos_mj_weights;
    int *result_actual_owner_rank = NULL;

    Kokkos::View<const mj_gno_t*, device_t> kokkos_result_initial_mj_gnos_ =
      this->kokkos_initial_mj_gnos;

    // TODO: MD 08/2017: Further discussion is required.
    // MueLu calls MJ when it has very few coordinates per processors,
    // such as 10. For example, it begins with 1K processor with 1K coordinate
    // in each. Then with coarsening this reduces to 10 coordinate per procesor.
    // It calls MJ to repartition these to 10 coordinates.
    // MJ runs with 1K processor, 10 coordinate in each, and partitions to
    // 10 parts.  As expected strong scaling is problem here, because
    // computation is almost 0, and communication cost of MJ linearly increases. 
    // Premigration option gathers the coordinates to 10 parts before MJ starts
    // therefore MJ will run with a smalller subset of the problem. 
    // Below, I am migrating the coordinates if mj_premigration_option is set,
    // and the result parts are less than the current part count, and the
    // average number of local coordinates is less than some threshold.
    // For example, premigration may not help if 1000 processors are
    // partitioning data to 10, but each of them already have 1M coordinate.
    // In that case, we premigration would not help.
    int current_world_size = this->mj_problemComm->getSize();
    mj_lno_t threshold_num_local_coords =
      this->min_coord_per_rank_for_premigration;
    bool is_pre_migrated = false;
    bool am_i_in_subset = true;
    if (mj_premigration_option > 0 &&
        size_t (current_world_size) > this->num_global_parts &&
        this->num_global_coords < mj_gno_t (
        current_world_size * threshold_num_local_coords))
    {
      if (this->mj_keep_part_boxes) {
        throw std::logic_error("Multijagged: mj_keep_part_boxes and "
          "mj_premigration_option are not supported together yet.");
      }

      is_pre_migrated =true;
      int migration_selection_option = mj_premigration_option;
      if(migration_selection_option * this->num_global_parts >
        (size_t) (current_world_size)) {
        migration_selection_option =
          current_world_size / this->num_global_parts;
      }

      int used_num_ranks = int (this->num_global_coords /
        float (threshold_num_local_coords) + 0.5);

      if (used_num_ranks == 0) {
        used_num_ranks = 1;
      }
  
      am_i_in_subset = this->mj_premigrate_to_subset(
      used_num_ranks,
        migration_selection_option,
        this->mj_env,
        this->mj_problemComm,
        this->coord_dim,
        this->num_local_coords,
        this->num_global_coords,
        this->num_global_parts,
        this->kokkos_initial_mj_gnos,
        this->kokkos_mj_coordinates,
        this->num_weights_per_coord,
        this->kokkos_mj_weights,
        //results
        result_problemComm,
        result_num_local_coords,
        kokkos_result_initial_mj_gnos,
        kokkos_result_mj_coordinates,
        kokkos_result_mj_weights,
        result_actual_owner_rank);

       kokkos_result_initial_mj_gnos_ = kokkos_result_initial_mj_gnos;
     }

    Kokkos::View<mj_part_t *, device_t> kokkos_result_assigned_part_ids;
    Kokkos::View<mj_gno_t*, device_t> kokkos_result_mj_gnos;

    this->mj_env->timerStop(MACRO_TIMERS, "partition() - setup");
    this->mj_env->timerStart(MACRO_TIMERS,
      "partition() - call multi_jagged_part()");

    if (am_i_in_subset){
      this->mj_partitioner.multi_jagged_part(
        this->mj_env,
        result_problemComm, //this->mj_problemComm,
        this->imbalance_tolerance,
        this->num_global_parts,
        this->kokkos_part_no_array,
        this->recursion_depth,
        this->coord_dim,
        result_num_local_coords, //this->num_local_coords,
        this->num_global_coords,
        kokkos_result_initial_mj_gnos_,
        kokkos_result_mj_coordinates,
        this->num_weights_per_coord,
        this->kokkos_mj_uniform_weights,
        kokkos_result_mj_weights,
        this->kokkos_mj_uniform_parts,
        this->kokkos_mj_part_sizes,
        kokkos_result_assigned_part_ids,
        kokkos_result_mj_gnos
      );
    }

    this->mj_env->timerStop(MACRO_TIMERS,
      "partition() - call multi_jagged_part()");
    this->mj_env->timerStart(MACRO_TIMERS, "partition() - cleanup");

    // Reorder results so that they match the order of the input
#if defined(__cplusplus) && __cplusplus >= 201103L
    std::unordered_map<mj_gno_t, mj_lno_t> localGidToLid;
    localGidToLid.reserve(result_num_local_coords);

    // copy to host
    
Clock clock_copy_gnos("clock_copy_gnos", true);
    typename decltype (kokkos_result_initial_mj_gnos_)::HostMirror
      host_kokkos_result_initial_mj_gnos_ =
      Kokkos::create_mirror_view(kokkos_result_initial_mj_gnos_);
    Kokkos::deep_copy(host_kokkos_result_initial_mj_gnos_,
      kokkos_result_initial_mj_gnos_);
clock_copy_gnos.stop(true);

Clock clock_copy_part_ids("clock_copy_part_ids", true);
    typename decltype (kokkos_result_assigned_part_ids)::HostMirror
      host_kokkos_result_assigned_part_ids =
      Kokkos::create_mirror_view(kokkos_result_assigned_part_ids);
    Kokkos::deep_copy(host_kokkos_result_assigned_part_ids,
      kokkos_result_assigned_part_ids);
clock_copy_part_ids.stop(true);

    for (mj_lno_t i = 0; i < result_num_local_coords; i++) {
      localGidToLid[host_kokkos_result_initial_mj_gnos_(i)] = i;
    }
    ArrayRCP<mj_part_t> partId = arcp(new mj_part_t[result_num_local_coords],
        0, result_num_local_coords, true);

    for (mj_lno_t i = 0; i < result_num_local_coords; i++) {
      mj_lno_t origLID = localGidToLid[host_kokkos_result_initial_mj_gnos_(i)];
      partId[origLID] = host_kokkos_result_assigned_part_ids(i);
    }
#else
    Teuchos::Hashtable<mj_gno_t, mj_lno_t>
    localGidToLid(result_num_local_coords);
    for (mj_lno_t i = 0; i < result_num_local_coords; i++)
      localGidToLid.put(kokkos_result_initial_mj_gnos_(i), i);

    ArrayRCP<mj_part_t> partId = arcp(new mj_part_t[result_num_local_coords],
        0, result_num_local_coords, true);

    for (mj_lno_t i = 0; i < result_num_local_coords; i++) {
      mj_lno_t origLID = localGidToLid.get(result_mj_gnos(i));
      partId[origLID] = kokkos_result_assigned_part_ids(i);
    }
#endif // C++11 is enabled

    //now the results are reordered. but if premigration occured,
    //then we need to send these ids to actual owners again. 
    if (is_pre_migrated){
      this->mj_env->timerStart(MACRO_TIMERS,
        "MultiJagged - PostMigration DistributorPlanCreating");
      Tpetra::Distributor distributor(this->mj_problemComm);

      ArrayView<const mj_part_t> actual_owner_destinations(
        result_actual_owner_rank , result_num_local_coords);
      mj_lno_t num_incoming_gnos = distributor.createFromSends(
        actual_owner_destinations);
      if (num_incoming_gnos != this->num_local_coords){
        throw std::logic_error("Zoltan2 - Multijagged Post Migration - "
          "num incoming is not equal to num local coords");
      }

      mj_env->timerStop(MACRO_TIMERS,
        "MultiJagged - PostMigration DistributorPlanCreating");
      mj_env->timerStart(MACRO_TIMERS,
        "MultiJagged - PostMigration DistributorMigration");
      ArrayRCP<mj_gno_t> received_gnos(num_incoming_gnos);
      ArrayRCP<mj_part_t> received_partids(num_incoming_gnos);
      {
        //  ArrayView<const mj_gno_t> sent_gnos(result_num_local_coords);
        throw std::logic_error("Restore distributor!");
        // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
        //ArrayView<const mj_gno_t> sent_gnos(result_initial_mj_gnos_,
        // result_num_local_coords);
        //distributor.doPostsAndWaits<mj_gno_t>(sent_gnos, 1, received_gnos());
      }
      {
        throw std::logic_error("Restore distributor!");
        // TODO RESTORE CODE - COMPLETE REFACTOR FOR KOKKOS
        //ArrayView<mj_part_t> sent_partnos(partId());
        //distributor.doPostsAndWaits<mj_part_t>(sent_partnos, 1,
        // received_partids());
      }
      partId = arcp(new mj_part_t[this->num_local_coords],
                      0, this->num_local_coords, true);

      {
#if defined(__cplusplus) && __cplusplus >= 201103L
      std::unordered_map<mj_gno_t, mj_lno_t> localGidToLid2;
      localGidToLid2.reserve(this->num_local_coords);

      auto local_kokkos_initial_mj_gnos = this->kokkos_initial_mj_gnos;
      for (mj_lno_t i = 0; i < this->num_local_coords; i++)
      {
        // TODO: Change loop so we don't read device to host
        mj_gno_t p;
        Kokkos::parallel_reduce("Read single", 1,
          KOKKOS_LAMBDA(int dummy, mj_gno_t & set_single) {
            set_single = local_kokkos_initial_mj_gnos(i);
        }, p);

        localGidToLid2[p] = i; 
      }

      for (mj_lno_t i = 0; i < this->num_local_coords; i++) {
        mj_lno_t origLID = localGidToLid2[received_gnos[i]];
        partId[origLID] = received_partids[i];
      }
#else
      Teuchos::Hashtable<mj_gno_t, mj_lno_t>
	      localGidToLid2(this->num_local_coords);

      for (mj_lno_t i = 0; i < this->num_local_coords; i++)
        localGidToLid2.put(this->kokkos_initial_mj_gnos(i), i);

      for (mj_lno_t i = 0; i < this->num_local_coords; i++) {
        mj_lno_t origLID = localGidToLid2.get(received_gnos[i]);
        partId[origLID] = received_partids[i];
      }

#endif // C++11 is enabled
      }
      {
        freeArray<int> (result_actual_owner_rank);
      }
      mj_env->timerStop(MACRO_TIMERS,
        "MultiJagged - PostMigration DistributorMigration");
    }
    solution->setParts(partId);
    this->mj_env->timerStop(MACRO_TIMERS, "partition() - cleanup");
  }
  this->mj_env->timerStop(MACRO_TIMERS, "partition() - all");
}

/* \brief Sets the partitioning data for multijagged algorithm.
 * */
template <typename Adapter>
void Zoltan2_AlgMJ<Adapter>::set_up_partitioning_data(
  const RCP<PartitioningSolution<Adapter> > &solution
)
{
  this->coord_dim = this->mj_coords->getCoordinateDim();
  this->num_weights_per_coord = this->mj_coords->getNumWeightsPerCoordinate();
  this->num_local_coords = this->mj_coords->getLocalNumCoordinates();
  this->num_global_coords = this->mj_coords->getGlobalNumCoordinates();
  int criteria_dim = (this->num_weights_per_coord ?
    this->num_weights_per_coord : 1);
  // From the Solution we get part information.
  // If the part sizes for a given criteria are not uniform,
  // then they are values that sum to 1.0.
  this->num_global_parts = solution->getTargetGlobalNumberOfParts();
  // allocate only two dimensional pointer.
  // raw pointer addresess will be obtained from multivector.
  this->kokkos_mj_uniform_parts = Kokkos::View<bool *, device_t>(
    "uniform parts", criteria_dim);
  this->kokkos_mj_part_sizes = Kokkos::View<mj_scalar_t **, device_t>(
    "part sizes", criteria_dim);
  this->kokkos_mj_uniform_weights = Kokkos::View<bool *, device_t>(
    "uniform weights", criteria_dim);
  Kokkos::View<const mj_gno_t *, device_t> kokkos_gnos;
  Kokkos::View<mj_scalar_t **, Kokkos::LayoutLeft, device_t> kokkos_xyz;
  Kokkos::View<mj_scalar_t **, device_t> kokkos_wgts;
  this->mj_coords->getCoordinatesKokkos(kokkos_gnos, kokkos_xyz, kokkos_wgts);
  // obtain global ids.
  this->kokkos_initial_mj_gnos = kokkos_gnos;
  // extract coordinates from multivector.
  this->kokkos_mj_coordinates = kokkos_xyz;
  // if no weights are provided set uniform weight.
  auto local_kokkos_mj_uniform_weights = this->kokkos_mj_uniform_weights;
  if (this->num_weights_per_coord == 0) {
    // originally we did the following:
    // this->kokkos_mj_uniform_weights(0) = true;
    // But I want this to work for UVM off - normally we'd be in a parallel_for
    // but for a single iteration is there a better way? I just do the
    // parallel_for for now
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename mj_node_t::execution_space, int> (0, 1),
      KOKKOS_LAMBDA (const int i) {
        local_kokkos_mj_uniform_weights(i) = true;
      }
    );
    Kokkos::resize(this->kokkos_mj_weights, 0);
  }
  else{
    this->kokkos_mj_weights = kokkos_wgts;

    // Originally in serial - need to allocate properly for UVM
    // for(int wdim = 0; wdim < this->num_weights_per_coord; ++wdim) {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
      (0, this->num_weights_per_coord), KOKKOS_LAMBDA (const int wdim) {
        local_kokkos_mj_uniform_weights(wdim) = false;
    });
  }

  // Here we need solution->criteriaHasUniformPartSizes on device
  // Create a host view and fill it, then copy to device
  // TODO: This got created during the refactor but needs to be cleaned up so
  // it doesn't happen in the first place. 
  typedef Kokkos::View<bool *> view_vector_t;
  view_vector_t device_hasUniformPartSizes(
    "device criteriaHasUniformPartSizes", criteria_dim);
  typename decltype(device_hasUniformPartSizes)::HostMirror
    host_hasUniformPartSizes =
    Kokkos::create_mirror_view(device_hasUniformPartSizes);
  for(int wdim = 0; wdim < criteria_dim; ++wdim) {
    host_hasUniformPartSizes(wdim) =
      solution->criteriaHasUniformPartSizes(wdim);
  }
  Kokkos::deep_copy(device_hasUniformPartSizes, host_hasUniformPartSizes);

  // now we are ready to initialize kokkos_mj_uniform_parts safely on device for
  // UVM off
  // TODO: we could probably refactor this a bit and just copy the view ptr but
  // I want to keep the error checking.
  // Also when we refactor above we may end up with a form similar to this.
  // For now keep the full loop to preserve the original code pattern
  auto local_kokkos_mj_uniform_parts = kokkos_mj_uniform_parts;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<typename mj_node_t::execution_space, int>
      (0, criteria_dim), KOKKOS_LAMBDA (const int wdim) {
      if(device_hasUniformPartSizes(wdim)) {
        local_kokkos_mj_uniform_parts(wdim) = true;
      }
      else {
        printf("Error: MJ does not support non uniform target part weights\n");
        // TODO: Resolve error handling for device
        // exit(1);
      }
  });
}

/* \brief Sets the partitioning parameters for multijagged algorithm.
 * \param pl: is the parameter list provided to zoltan2 call
 * */
template <typename Adapter>
void Zoltan2_AlgMJ<Adapter>::set_input_parameters(
  const Teuchos::ParameterList &pl)
{
  const Teuchos::ParameterEntry *pe = pl.getEntryPtr("imbalance_tolerance");
  if (pe) {
    double tol;
    tol = pe->getValue(&tol);
    this->imbalance_tolerance = tol - 1.0;
  }

  // TODO: May be a more relaxed tolerance is needed. RCB uses 10%
  if (this->imbalance_tolerance <= 0) {
    this->imbalance_tolerance= 10e-4;
  }

  // if an input partitioning array is provided.
  Kokkos::resize(this->kokkos_part_no_array, 0);

  // the length of the input partitioning array.
  this->recursion_depth = 0;

  if (pl.getPtr<Array <mj_part_t> >("mj_parts")) {
    auto mj_parts = pl.get<Array <mj_part_t> >("mj_parts");
    int mj_parts_size = static_cast<int>(mj_parts.size());

    // build the view we'll have data on and copy values from host
    this->kokkos_part_no_array = Kokkos::View<mj_part_t*, device_t>(
      "kokkos_part_no_array", mj_parts_size);
    typename decltype (this->kokkos_part_no_array)::HostMirror 
      host_kokkos_part_no_array = Kokkos::create_mirror_view(
        this->kokkos_part_no_array);
    for(int i = 0; i < mj_parts_size; ++i) {
      host_kokkos_part_no_array(i) = mj_parts.getRawPtr()[i];
    }
    Kokkos::deep_copy(this->kokkos_part_no_array, host_kokkos_part_no_array);

    this->recursion_depth = mj_parts_size - 1;
    this->mj_env->debug(2, "mj_parts provided by user");
  }

  // get mj specific parameters.
  this->distribute_points_on_cut_lines = true;
  this->max_concurrent_part_calculation = 1;

  this->mj_run_as_rcb = false;
  this->mj_premigration_option = 0;
	this->min_coord_per_rank_for_premigration = 32000;

  int mj_user_recursion_depth = -1;
  this->mj_keep_part_boxes = false;
  this->check_migrate_avoid_migration_option = 0;
  this->migration_type = 0;
	this->minimum_migration_imbalance = 0.35;

  pe = pl.getEntryPtr("mj_minimum_migration_imbalance");
  if (pe) {
    double imb;
    imb = pe->getValue(&imb);
    this->minimum_migration_imbalance = imb - 1.0;
  }

  pe = pl.getEntryPtr("mj_migration_option");
  if (pe) {
    this->check_migrate_avoid_migration_option =
      pe->getValue(&this->check_migrate_avoid_migration_option);
  } else {
    this->check_migrate_avoid_migration_option = 0;
  }
  if (this->check_migrate_avoid_migration_option > 1) {
    this->check_migrate_avoid_migration_option = -1;
  }

	///
  pe = pl.getEntryPtr("mj_migration_type");
  if (pe) {
    this->migration_type = pe->getValue(&this->migration_type);
  } else {
    this->migration_type = 0;
  }

	//std::cout << " this->migration_type:" <<  this->migration_type << std::endl;
	///

  pe = pl.getEntryPtr("mj_concurrent_part_count");
  if (pe) {
    this->max_concurrent_part_calculation =
      pe->getValue(&this->max_concurrent_part_calculation);
  } else {
    this->max_concurrent_part_calculation = 1; // Set to 1 if not provided.
  }

  pe = pl.getEntryPtr("mj_keep_part_boxes");
  if (pe) {
    this->mj_keep_part_boxes = pe->getValue(&this->mj_keep_part_boxes);
  } else {
    this->mj_keep_part_boxes = false; // Set to invalid value
  }

  // For now, need keep_part_boxes to do pointAssign and boxAssign.
  // pe = pl.getEntryPtr("keep_cuts");
  // if (pe){
  //      int tmp = pe->getValue(&tmp);
  //      if (tmp) this->mj_keep_part_boxes = true;
  // }

  //need to keep part boxes if mapping type is geometric.
  if (this->mj_keep_part_boxes == false) {
    pe = pl.getEntryPtr("mapping_type");
    if (pe) {
      int mapping_type = -1;
      mapping_type = pe->getValue(&mapping_type);
      if (mapping_type == 0) {
        mj_keep_part_boxes  = true;
      }
    }
  }

  // need to keep part boxes if mapping type is geometric.
  pe = pl.getEntryPtr("mj_enable_rcb");
  if (pe) {
    this->mj_run_as_rcb = pe->getValue(&this->mj_run_as_rcb);
  } else {
    this->mj_run_as_rcb = false; // Set to invalid value
  }

  pe = pl.getEntryPtr("mj_premigration_option");
  if (pe){
    mj_premigration_option = pe->getValue(&mj_premigration_option);
  } else {
     mj_premigration_option = 0;
  }

  pe = pl.getEntryPtr("mj_premigration_coordinate_count");
  if (pe) {
    min_coord_per_rank_for_premigration = pe->getValue(&mj_premigration_option);
  } else {
    min_coord_per_rank_for_premigration = 32000;
  }

  pe = pl.getEntryPtr("mj_recursion_depth");
  if (pe) {
    mj_user_recursion_depth = pe->getValue(&mj_user_recursion_depth);
  } else {
    mj_user_recursion_depth = -1; // Set to invalid value
  }

  bool val = false;
  pe = pl.getEntryPtr("rectilinear");
  if (pe) {
    val = pe->getValue(&val);
  }
  if (val) {
    this->distribute_points_on_cut_lines = false;
  } else {
    this->distribute_points_on_cut_lines = true;
  }

  if (this->mj_run_as_rcb){
    mj_user_recursion_depth =
      (int)(ceil(log ((this->num_global_parts)) / log (2.0)));
  }
  if (this->recursion_depth < 1){
    if (mj_user_recursion_depth > 0){
      this->recursion_depth = mj_user_recursion_depth;
    }
    else {
      this->recursion_depth = this->coord_dim;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
template <typename Adapter>
void Zoltan2_AlgMJ<Adapter>::boxAssign(
  int dim,
  typename Adapter::scalar_t *lower,
  typename Adapter::scalar_t *upper,
  size_t &nPartsFound,
  typename Adapter::part_t **partsFound) const
{
  // TODO:  Implement with cuts rather than boxes to reduce algorithmic
  // TODO:  complexity.  Or at least do a search through the boxes, using
  // TODO:  p x q x r x ... if possible.

  nPartsFound = 0;
  *partsFound = NULL;

  if (this->mj_keep_part_boxes) {

    // Get vector of part boxes
    RCP<mj_partBoxVector_t> partBoxes = this->getGlobalBoxBoundaries();

    size_t nBoxes = (*partBoxes).size();
    if (nBoxes == 0) {
      throw std::logic_error("no part boxes exist");
    }

    // Determine whether the box overlaps the globalBox at all
    RCP<mj_partBox_t> globalBox = this->mj_partitioner.get_global_box();

    if (globalBox->boxesOverlap(dim, lower, upper)) {

      std::vector<typename Adapter::part_t> partlist;

      // box overlaps the global box; find specific overlapping boxes
      for (size_t i = 0; i < nBoxes; i++) {
        try {
          if ((*partBoxes)[i].boxesOverlap(dim, lower, upper)) {
            nPartsFound++;
            partlist.push_back((*partBoxes)[i].getpId());
            /*
            std::cout << "Given box (";
            for (int j = 0; j < dim; j++)
              std::cout << lower[j] << " ";
            std::cout << ") x (";
            for (int j = 0; j < dim; j++)
              std::cout << upper[j] << " ";
            std::cout << ") overlaps PartBox "
                      << (*partBoxes)[i].getpId() << " (";
            for (int j = 0; j < dim; j++)
              std::cout << (*partBoxes)[i].getlmins()[j] << " ";
            std::cout << ") x (";
            for (int j = 0; j < dim; j++)
              std::cout << (*partBoxes)[i].getlmaxs()[j] << " ";
            std::cout << ")" << std::endl;
            */
          }
        }
        Z2_FORWARD_EXCEPTIONS;
      }
      if (nPartsFound) {
        *partsFound = new mj_part_t[nPartsFound];
        for (size_t i = 0; i < nPartsFound; i++)
          (*partsFound)[i] = partlist[i];
      }
    }
    else {
      // Box does not overlap the domain at all.  Find the closest part
      // Not sure how to perform this operation for MJ without having the
      // cuts.  With the RCB cuts, the concept of a part extending to
      // infinity was natural.  With the boxes, it is much more difficult.
      // TODO:  For now, return information indicating NO OVERLAP.
    }
  }
  else {
    throw std::logic_error("need to use keep_cuts parameter for boxAssign");
  }
}

/////////////////////////////////////////////////////////////////////////////
template <typename Adapter>
typename Adapter::part_t Zoltan2_AlgMJ<Adapter>::pointAssign(
  int dim,
  typename Adapter::scalar_t *point) const
{
  // TODO:  Implement with cuts rather than boxes to reduce algorithmic
  // TODO:  complexity.  Or at least do a search through the boxes, using
  // TODO:  p x q x r x ... if possible.

  if (this->mj_keep_part_boxes) {
    typename Adapter::part_t foundPart = -1;

    // Get vector of part boxes
    RCP<mj_partBoxVector_t> partBoxes = this->getGlobalBoxBoundaries();

    size_t nBoxes = (*partBoxes).size();
    if (nBoxes == 0) {
      throw std::logic_error("no part boxes exist");
    }

    // Determine whether the point is within the global domain
    RCP<mj_partBox_t> globalBox = this->mj_partitioner.get_global_box();

    if (globalBox->pointInBox(dim, point)) {

      // point is in the global domain; determine in which part it is.
      size_t i;
      for (i = 0; i < nBoxes; i++) {
        try {
          if ((*partBoxes)[i].pointInBox(dim, point)) {
            foundPart = (*partBoxes)[i].getpId();
            // std::cout << "Point (";
            // for (int j = 0; j < dim; j++) std::cout << point[j] << " ";
            //   std::cout << ") found in box " << i << " part " << foundPart
            //     << std::endl;
            // (*partBoxes)[i].print();
            break;
          }
        }
        Z2_FORWARD_EXCEPTIONS;
      }

      if (i == nBoxes) {
        // This error should never occur
        std::ostringstream oss;
        oss << "Point (";
        for (int j = 0; j < dim; j++) oss << point[j] << " ";
        oss << ") not found in domain";
        throw std::logic_error(oss.str());
      }
    }

    else {
      // Point is outside the global domain.
      // Determine to which part it is closest.
      // TODO:  with cuts, would not need this special case

      size_t closestBox = 0;
      mj_scalar_t minDistance = std::numeric_limits<mj_scalar_t>::max();
      mj_scalar_t *centroid = new mj_scalar_t[dim];
      for (size_t i = 0; i < nBoxes; i++) {
        (*partBoxes)[i].computeCentroid(centroid);
        mj_scalar_t sum = 0.;
        mj_scalar_t diff;
        for (int j = 0; j < dim; j++) {
          diff = centroid[j] - point[j];
          sum += diff * diff;
        }
        if (sum < minDistance) {
          minDistance = sum;
          closestBox = i;
        }
      }
      foundPart = (*partBoxes)[closestBox].getpId();
      delete [] centroid;
    }

    return foundPart;
  }
  else {
    throw std::logic_error("need to use keep_cuts parameter for pointAssign");
  }
}

template <typename Adapter>
void Zoltan2_AlgMJ<Adapter>::getCommunicationGraph(
  const PartitioningSolution<Adapter> *solution,
  ArrayRCP<typename Zoltan2_AlgMJ<Adapter>::mj_part_t> &comXAdj,
  ArrayRCP<typename Zoltan2_AlgMJ<Adapter>::mj_part_t> &comAdj)
{
  if(comXAdj_.getRawPtr() == NULL && comAdj_.getRawPtr() == NULL){
    RCP<mj_partBoxVector_t> pBoxes = this->getGlobalBoxBoundaries();
    mj_part_t ntasks =  (*pBoxes).size();
    int dim = (*pBoxes)[0].getDim();
    GridHash<mj_scalar_t, mj_part_t> grid(pBoxes, ntasks, dim);
    grid.getAdjArrays(comXAdj_, comAdj_);
  }
  comAdj = comAdj_;
  comXAdj = comXAdj_;
}

template <typename Adapter>
RCP<typename Zoltan2_AlgMJ<Adapter>::mj_partBoxVector_t>
Zoltan2_AlgMJ<Adapter>::getGlobalBoxBoundaries() const
{
  return this->mj_partitioner.get_kept_boxes();
}
} // namespace Zoltan2

#endif
