#pragma once

#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <torch/torch.h>
#include <unistd.h>

#include "common.h"
#include "model.h"

#undef assert
#define assert(x)      \
  do {                 \
    if (!(x)) abort(); \
  } while (0);

struct GrantMsg {
  uint64_t clock_ns;
  uint64_t do_writeback;
};

class ExternalController {
 public:
  virtual GrantMsg BlockNextGrant() = 0;
  virtual void AckGrant(GrantMsg m) = 0;
  virtual bool IsDone() = 0;
};

class SameProcessController : public ExternalController {
 public:
  GrantMsg BlockNextGrant() {
    size_t val;
    while (last_grant_clock == train_clock && !terminated) {
      assert(read(notifier_efd_, &val, sizeof(val)) == sizeof(val));
    }
    last_grant_clock = train_clock;
    GrantMsg g = {train_clock, 0};
    return g;
  }
  void AckGrant(GrantMsg m) {}
  void Grant(uint64_t micros) {
    size_t val = 1;
    train_clock = get_now_ns() + 1000 * micros;
    assert(write(notifier_efd_, &val, sizeof(val)) == sizeof(val));
  }

  bool IsDone() { return terminated; }

  void Close() {
    terminated = true;
    size_t val = 1;
    train_clock = -1L;
    assert(write(notifier_efd_, &val, sizeof(val)) == sizeof(val));
  }

  SameProcessController() {
    notifier_efd_ = eventfd(0, 0);
    if (notifier_efd_ < 0) throw std::runtime_error("bad efd");
  }

  ~SameProcessController() { close(notifier_efd_); }

 private:
  int notifier_efd_;
  uint64_t last_grant_clock;
  volatile bool terminated;
  volatile uint64_t train_clock;
};

class UnixSocketController : public ExternalController {
 public:
  GrantMsg BlockNextGrant() {
    GrantMsg cur_grant;
    assert(read(sockfd_, &cur_grant, sizeof(cur_grant)) == sizeof(cur_grant));
    return cur_grant;
  }
  void AckGrant(GrantMsg m) {
    assert(write(sockfd_, &m.clock_ns, sizeof(m.clock_ns)) ==
           sizeof(m.clock_ns));
  }
  static UnixSocketController *Dial() {
    int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    assert(sockfd >= 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, "/tmp/training_server", sizeof(addr.sun_path) - 1);
    int ret =
        connect(sockfd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un));
    if (ret != 0) {
      std::cerr << "Could not dial" << std::endl;
      return nullptr;
    }
    return new UnixSocketController(sockfd);
  }
  bool IsDone() { return false; }

 private:
  UnixSocketController(int sockfd) : sockfd_(sockfd) {}
  int sockfd_;
};

std::vector<double> TimeLayers(std::shared_ptr<TrainableModel> model,
                               long batch_size,
                               std::string cached_timings_file);
void ConsumeOrBlock(ExternalController *c, uint64_t micros, uint64_t layern);
void Train(ExternalController *c, std::shared_ptr<TrainableModel> model,
           long bsize, uint64_t iters, int device_no);
