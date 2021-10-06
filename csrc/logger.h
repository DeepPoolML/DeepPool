// Copyright (c) 2021 MIT
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#ifndef LOGGER_H
#define LOGGER_H

#include <condition_variable>
#include <thread>
#include <memory>
#include "SpinLock.h"
#include "CodeLocation.h"

using CodeLocation = RAMCloud::CodeLocation;
using SpinLock = RAMCloud::SpinLock;

/**
 * The levels of verbosity for messages logged with #LOG.
 */
enum LogLevel {
  SILENT_LOG_LEVEL = 0,
  ERROR,
  WARNING,
  NOTICE,
  DEBUG,
  NUM_LOG_LEVELS // must be the last element in the enum
};

class Logger {
  explicit Logger(LogLevel level = WARNING);
 public:
  ~Logger();
  static Logger& get();

  void setLogFile(const char* path, bool truncate = false);
  void setLogFile(int fd);
  int getLogFile() { return fd; }
  void setLogLevel(LogLevel level);
  void sync();
  void logMessage(LogLevel level,
                  const CodeLocation& where,
                  const char* format, ...)
      __attribute__((format(printf, 4, 5)));
  
  /**
   * Return whether the current logging configuration includes messages of
   * the given level.
   */
  bool isLogging(LogLevel level) {
      return (level <= logLevel);
  }

 private:
  bool addToBuffer(const char* src, int length);
  static void printThreadMain(Logger* logger);

  /**
   * Monitor-style lock
   */
  RAMCloud::SpinLock mutex;
  typedef std::unique_lock<RAMCloud::SpinLock> Lock;


  /**
   * Log output gets written to this file descriptor (default is 2, for
   * stdout).
   */
  int fd;

  /**
   * True means that fd came from a file that we opened, so we must
   * eventually close fd when the Logger is destroyed. False means that
   * someone else provided this descriptor, so it's their responsibility
   * to close it.
   */
  bool mustCloseFd;

  LogLevel logLevel;

  /**
   * The print thread uses this to sleep when it runs out of log data
   * to print.
   */
  std::condition_variable_any logDataAvailable;

  /**
   * Used by waitIfCongested to wait for buffered log data to get
   * printed.
   */
  std::condition_variable_any bufferSpaceFreed;

  /**
   * Total number of bytes available in messageBuffer; modified only for
   * unit testing.
   */
  int bufferSize;

  /**
   * Buffer space (dynamically allocated, must be freed).
   */
  char* const messageBuffer;

  /**
   * Offset in messageBuffer of the location where the next log message
   * will be stored
   */
  int nextToInsert;

  /**
   * Offset in messageBuffer of the first byte of data that has not yet been
   * printed. Modified only by the printer thread (and this is the only
   * shared variable modified by the printer thread).
   */
  int nextToPrint;

  /**
   * Nonzero means that the most recently generated log entries had to be
   * discarded because we ran out of buffer space (the print thread got
   * behind). The value indicates how many entries were lost.
   */
  int discardedEntries;

  /**
   * This thread is responsible for invoking the (potentially blocking)
   * kernel calls to write out the log.
   */
  std::unique_ptr<std::thread> printThread;

  /**
   * Set to true to cause the print thread to exit during the Logger
   * destructor.
   */
  bool printThreadExit;

  /**
   * If non-zero, overrides default value for buffer size in logMessage
   * (used for testing).
   */
  int testingBufferSize;
};

/**
 * Log a message for the system administrator with (CLOG) or without (LOG)
 * collapsing frequent messages.
 * The #RAMCLOUD_CURRENT_LOG_MODULE macro should be set to the LogModule to
 * which the message pertains.
 * \param[in] level
 *      The level of importance of the message (LogLevel).
 * \param[in] format
 *      A printf-style format string for the message. It should not have a line
 *      break at the end, as RAMCLOUD_LOG will add one.
 * \param[in] ...
 *      The arguments to the format string.
 */
#define DP_LOG(level, format, ...) do { \
    Logger& _logger = Logger::get(); \
    if (level != DEBUG && _logger.isLogging(level)) { \
        _logger.logMessage(level, HERE, format "\n", ##__VA_ARGS__); \
    } \
} while (0)

/**
 * Log an ERROR message, dump a backtrace, and throw a #RAMCloud::FatalError.
 * The #RAMCLOUD_CURRENT_LOG_MODULE macro should be set to the LogModule to
 * which the message pertains.
 * \param[in] format_
 *      See #RAMCLOUD_LOG().
 * \param[in] ...
 *      See #RAMCLOUD_LOG().
 */
#define DIE(format_, ...) do { \
    DP_LOG(LogLevel::ERROR, format_, ##__VA_ARGS__); \
    Logger::get().sync(); \
    throw 1; \
} while (0)

#endif
