//
// Created by baidu on 2017/6/5.
//

#ifndef NAIVENET_ERROR_H
#define NAIVENET_ERROR_H
#include <stdlib.h>
#include <memory>
#include <string>
namespace nnet {

class Error : public std::exception {
 public:
  /**
   * Construct a no-error value.
   */
  Error() {}

  /**
   * @brief Create an Error use printf syntax.
   */
  explicit Error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    constexpr size_t kBufferSize = 1024;
    char buffer[kBufferSize];
    vsnprintf(buffer, kBufferSize, fmt, ap);
    this->msg_.reset(new std::string(buffer));
    va_end(ap);
  }

  virtual ~Error(){};

  /**
   * @brief msg will return the error message. If no error, return nullptr.
   */
  const char* msg() const {
    if (msg_) {
      return msg_->c_str();
    } else {
      return "";
    }
  }

  const char* what() const _NOEXCEPT { return msg(); }

  /**
   * @brief operator bool, return True if there is something error.
   */
  operator bool() const { return !this->isOK(); }

  /**
   * @brief isOK return True if there is no error.
   * @return True if no error.
   */
  bool isOK() const { return msg_ == nullptr; }

 private:
  std::shared_ptr<std::string> msg_;
};
}

#endif  // NAIVENET_ERROR_H
