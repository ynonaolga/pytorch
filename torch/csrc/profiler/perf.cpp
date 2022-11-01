#include <unordered_set>

#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <torch/csrc/profiler/perf-inl.h>
#include <torch/csrc/profiler/perf.h>

namespace torch {
namespace profiler {
namespace impl {

// HACK: make sure we have new children pids to attach to
void ResetPThreadPool() {
#ifdef USE_PTHREADPOOL
  caffe2::PThreadPool* const tp = caffe2::pthreadpool();
  // check if the threadpool does exist yet and has additional threads
  if (tp && tp->get_thread_count() > 1) {
    tp->set_thread_count(tp->get_thread_count());
  }
#endif /* USE_PTHREADPOOL */
}

namespace linux_perf {

#if defined(__ANDROID__) || defined(__linux__)

/*
 * PerfEvent
 * ---------
 */

/*
 * Syscall wrapper for perf_event_open(2)
 */
inline long perf_event_open(
    struct perf_event_attr* hw_event,
    pid_t pid,
    int cpu,
    int group_fd,
    unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// TODO sync with Kineto level abstract events in profiler/events.h
static const std::unordered_map<
    std::string,
    std::pair<perf_type_id, /* perf event type */ uint32_t>>
    EventTable{
        {"cycles",
         std::make_pair(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES)},
        {"instructions",
         std::make_pair(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)},

        // Non Standard events for testing
        {"pagefaults",
         std::make_pair(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS)},
        {"backend-stall-cycles",
         std::make_pair(
             PERF_TYPE_HARDWARE,
             PERF_COUNT_HW_STALLED_CYCLES_BACKEND)},
        {"frontend-stall-cycles",
         std::make_pair(
             PERF_TYPE_HARDWARE,
             PERF_COUNT_HW_STALLED_CYCLES_FRONTEND)}};

PerfEvent::~PerfEvent() {
  if (fd_ > -1) {
    close(fd_);
  }
  fd_ = -1; // poison
}

void PerfEvent::Init() {
  TORCH_CHECK(!name_.empty(), "Invalid profiler event name");

  auto const it = EventTable.find(name_);
  if (it == EventTable.end()) {
    TORCH_CHECK(false, "Unsupported profiler event name: ", name_);
  }

  struct perf_event_attr attr {};
  memset(&attr, 0, sizeof(attr));

  attr.size = sizeof(perf_event_attr);
  attr.type = it->second.first;
  attr.config = it->second.second;
  attr.disabled = 1;
  attr.inherit = 1;
  attr.exclude_kernel = 1; // TBD
  attr.exclude_hv = 1;
  /*
   * These can be used to calculate estimated totals if the PMU is overcommitted
   * and multiplexing is happening
   */
  attr.read_format =
      PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

  pid_t pid = getpid(); // this pid
  int cpu = -1; // all cpus
  int group_fd = -1;
  unsigned long flags = 0;

  fd_ = static_cast<int>(perf_event_open(&attr, pid, cpu, group_fd, flags));
  if (fd_ == -1) {
    TORCH_CHECK(
        false, "perf_event_open() failed, error: ", std::strerror(errno));
  }
  Reset();
}

uint64_t PerfEvent::ReadCounter() const {
  PerfCounter counter{};
  long n = read(fd_, &counter, sizeof(PerfCounter));
  TORCH_CHECK(
      n == sizeof(counter),
      "Read failed for Perf event fd, event : ",
      name_,
      ", error: ",
      std::strerror(errno));
  TORCH_CHECK(
      counter.time_enabled == counter.time_running,
      "Hardware performance counter time multiplexing is not handled yet",
      ", name: ",
      name_,
      ", enabled: ",
      counter.time_enabled,
      ", running: ",
      counter.time_running);
  return counter.value;
}

#else /* __ANDROID__ || __linux__ */
/*
 * Shim class for unsupported platforms - this will always return 0 counter
 * value
 */

PerfEvent::~PerfEvent(){};

void PerfEvent::Init(){};

uint64_t PerfEvent::ReadCounter() const {
  return 0;
};

#endif /* __ANDROID__ || __linux__ */

/*
 * PerfProfiler
 * ------------
 */

void PerfProfiler::Configure(std::vector<std::string>& event_names) {
  TORCH_CHECK(
      event_names.size() <= MAX_EVENTS,
      "Too many events to configure, configured: ",
      event_names.size(),
      ", max allowed:",
      MAX_EVENTS);
  std::unordered_set<std::string> s(event_names.begin(), event_names.end());
  TORCH_CHECK(
      s.size() == event_names.size(), "Duplicate event names are not allowed!")
  for (auto name : event_names) {
    events_.emplace_back(name);
    events_.back().Init();
  }
  start_values_.resize(events_.size(), 0);
  ResetPThreadPool();
}

void PerfProfiler::Enable() {
  TORCH_CHECK(!is_enabled_, "Nested perf event counting is not supported yet");
  for (int i = 0; i < events_.size(); ++i) {
    start_values_[i] = events_[i].ReadCounter();
    events_[i].Enable();
  }
  is_enabled_ = true;
}

void PerfProfiler::Disable(perf_counters_t& vals) {
  TORCH_CHECK(
      vals.size() == events_.size(),
      "Can not fit all perf counters in the supplied container");
  TORCH_CHECK(is_enabled_, "Perf Profiler is not enabled");
  for (int i = 0; i < events_.size(); ++i) {
    events_[i].Disable();
    vals[i] = CalcDelta(start_values_[i], events_[i].ReadCounter());
  }
  is_enabled_ = false;
}
} // namespace linux_perf
} // namespace impl
} // namespace profiler
} // namespace torch
