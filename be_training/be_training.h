#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <torch/torch.h>
#include <unistd.h>

#include "common.h"
#include "model.h"

struct GrantMsg {
  uint64_t granted_nanos;
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
    std::unique_lock<std::mutex> lck(m_);
    cv_.wait(lck, [&] { return !terminated_ && micros_to_train_ > 0; });
    if (terminated_) return {0, 0};
    auto micros = micros_to_train_;
    micros_to_train_ = 0;
    train_after_event_.block(c10::cuda::getCurrentCUDAStream());
    return {micros * 1000, 0};
  }
  void AckGrant(GrantMsg m) {}
  void Grant(uint64_t micros) {
    at::cuda::CUDAEvent event;
    event.record();
    std::unique_lock<std::mutex> lck(m_);
    micros_to_train_ = micros;
    train_after_event_ = std::move(event);
    cv_.notify_one();
  }
  bool IsDone() { return terminated_; }

  void Close() {
    std::unique_lock<std::mutex> lck(m_);
    terminated_ = true;
    cv_.notify_one();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  uint64_t micros_to_train_;
  at::cuda::CUDAEvent train_after_event_;
  volatile bool terminated_;
};

class UnixSocketController : public ExternalController {
 public:
  GrantMsg BlockNextGrant() {
    GrantMsg cur_grant;
    assert(read(sockfd_, &cur_grant, sizeof(cur_grant)) == sizeof(cur_grant));
    auto now = get_now_ns();
    if (cur_grant.granted_nanos > now)
      cur_grant.granted_nanos = cur_grant.granted_nanos - now;
    else
      cur_grant.granted_nanos = 0;
    return cur_grant;
  }
  void AckGrant(GrantMsg m) {
    assert(write(sockfd_, &m.granted_nanos, sizeof(m.granted_nanos)) ==
           sizeof(m.granted_nanos));
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
void Train(ExternalController *c, std::shared_ptr<TrainableModel> model,
           long bsize, uint64_t iters, std::vector<double> timings);
