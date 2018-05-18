#ifndef NNC_EXCEPTION_H
#define NNC_EXCEPTION_H

#include <exception>
#include <string>
#include <boost/format.hpp>

namespace nnt {

class Exception : public std::exception {
 public:
  Exception(const boost::format& msg)
      : msg_(boost::str(msg)) {}

  Exception(const std::string& msg)
      : msg_(msg) {}

  virtual ~Exception() noexcept  = default;

  Exception(const Exception& rt_err)
      : msg_(rt_err.msg_) {}

  Exception& operator=(const Exception& rt_err) {
    msg_ = rt_err.msg_;

    return *this;
  }

  /**
   * @return the error description and the context as a text string.
   */
  virtual const char* what() const noexcept {
    return msg_.c_str();
  }

  const std::string& msg() const noexcept {
    return msg_;
  }

  std::string msg_;
};

#define FATAL(msg_arg)      \
  throw Exception(msg_arg);

}  // nnt

#endif  // NNC_EXCEPTION_H
