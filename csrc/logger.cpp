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

#include <cstdarg>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#include "logger.h"
#include "Cycles.h"
#include "ThreadId.h"

using Cycles = RAMCloud::Cycles;
using ThreadId = RAMCloud::ThreadId;

/**
 * Friendly names for each #LogLevel value.
 * Keep this in sync with the LogLevel enum.
 */
static const char* logLevelNames[] = {"(none)", "ERROR", "WARNING",
                                      "NOTICE", "DEBUG"};

/**
 * Constructs a logger.
 */
Logger::Logger(LogLevel level)
  : mutex("Logger::mutex")
  , fd(2)
  , mustCloseFd(false)
  , logLevel(level)
  , logDataAvailable()
  , bufferSpaceFreed()
  , bufferSize(1000000)
  , messageBuffer(new char[bufferSize])
  , nextToInsert(0)
  , nextToPrint(0)
  , discardedEntries(0)
  , printThread()
  , printThreadExit(false)
  , testingBufferSize(0)
{
  // Touch every page in the buffer
  for (int i = 0; i < bufferSize; i += 1000) {
    messageBuffer[i] = 'x';
  }

  printThread.reset(new std::thread(printThreadMain, this));
}

/**
 * Destructor for debug logs.
 */
Logger::~Logger()
{
  // Exit the print thread.
  {
    Lock lock(mutex);
    printThreadExit = true;
    logDataAvailable.notify_one();
  }
  printThread->join();

  // No lock needed: the print thread is finished
  if (mustCloseFd)
    close(fd);
  delete[] messageBuffer;
}

/**
 * Return the singleton shared instance that is normally used for logging.
 */
Logger&
Logger::get()
{
  // Use static local variable to achieve efficient thread-safe lazy
  // initialization. If multiple threads attempt to initialize sharedLogger
  // concurrently, the initialization is guaranteed to occur exactly once.
  static Logger sharedLogger;
  return sharedLogger;
}

/**
 * Arrange for future log messages to go to a particular file.
 *
 * This should only be invoked when one is sure that printThread is not
 * running (e.g. before any logMessage() invocations) due to a file
 * descriptor race that can occur (RAM-839).
 * \param path
 *      The pathname for the log file, which may or may not exist already.
 * \param truncate
 *      True means the log file should be truncated if it already exists;
 *      false means an existing log file is retained, and new messages are
 *      appended.
 */
void
Logger::setLogFile(const char* path, bool truncate)
{
  Lock lock(mutex);
  int newFd = open(path, O_CREAT | O_WRONLY | (truncate ? O_TRUNC : 0),
          0666);
  if (newFd < 0) {
    exit(1);
  }
  lseek(newFd, 0, SEEK_END);
  if (mustCloseFd) {
    close(fd);
  }
  fd = newFd;
  mustCloseFd = true;
}

/**
 * Arrange for future log messages to go to a particular file descriptor.
 * \param newFd
 *    The file descriptor for the log file, which must already have
 *    been opened by the caller. This file descriptor belongs to the caller,
 *    meaning that we will never close it.
 */
void
Logger::setLogFile(int newFd)
{
  Lock lock(mutex);
  if (mustCloseFd) {
    close(fd);
  }
  fd = newFd;
  mustCloseFd = false;
}

/**
 * Set the log level.
 * \param[in] level
 *    Messages at least as important as \a level will be logged.
 */
void
Logger::setLogLevel(LogLevel level)
{
  Lock lock(mutex);
  logLevel = level;
}

/**
 * Wait for all buffered log messages to be printed. This method is intended
 * only for tests and a few special situations such as application exit. It
 * should *not* be used in the normal course of logging messages, since it
 * can result in long delays that could potentially cause the server to be
 * considered crashed.
 */
void
Logger::sync()
{
  Lock lock(mutex);
  while (nextToInsert != nextToPrint) {
    mutex.unlock();
    usleep(100);
    mutex.lock();
  }
}

/**
 * Log a message for the system administrator.
 * \param[in] level
 *      See #LOG.
 * \param[in] where
 *      The result of #HERE.
 * \param[in] fmt
 *      See #LOG except the string should end with a newline character.
 * \param[in] ...
 *      See #LOG.
 */
void
Logger::logMessage(LogLevel level,
                   const CodeLocation& where,
                   const char* fmt, ...)
{
  uint64_t start = Cycles::rdtsc();
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);

  Lock lock(mutex);

#define MAX_MESSAGE_CHARS 2000
  // Extra space for a message about truncated characters, if needed.
#define TRUNC_MESSAGE_SPACE 50
  char buffer[MAX_MESSAGE_CHARS + TRUNC_MESSAGE_SPACE];
  int spaceLeft = MAX_MESSAGE_CHARS;
  int charsLost = 0;
  int charsWritten = 0;
  int actual;

  // Create the new log message in a local buffer. First write a standard
  // prefix containing timestamp, information about source file, etc.
  actual = snprintf(buffer+charsWritten, spaceLeft,
      "%010lu.%09lu %s:%d in %s %s[%d]: ",
      now.tv_sec, now.tv_nsec, where.baseFileName(), where.line,
      where.function, logLevelNames[level],
      ThreadId::get());
  if (actual >= spaceLeft) {
      // We ran out of space in the buffer (should never happen here).
      charsLost += 1 + actual - spaceLeft;
      actual = spaceLeft - 1;
  }
  charsWritten += actual;
  spaceLeft -= actual;

  // Last, add the caller's log message.
  va_list ap;
  va_start(ap, fmt);
  actual = vsnprintf(buffer + charsWritten, spaceLeft, fmt, ap);
  va_end(ap);
  if (actual >= spaceLeft) {
      // We ran out of space in the buffer.
      charsLost += 1 + actual - spaceLeft;
      actual = spaceLeft - 1;
  }
  charsWritten += actual;
  spaceLeft -= actual;

  if (charsLost > 0) {
      // Ran out of space: add a note about the lost info.
      charsWritten += snprintf(buffer + charsWritten, TRUNC_MESSAGE_SPACE,
              "... (%d chars truncated)\n", charsLost);
  }

  if (addToBuffer(buffer, charsWritten)) {
    discardedEntries = 0;
  } else {
    discardedEntries++;
  }

  // Make sure this method did not take very long to execute.  If the
  // logging gets backed up it is really bad news, because it can lock
  // up the server so that it appears dead and crash recovery happens
  // (e.g., the Dispatch thread might be blocked waiting to log a message).
  // Thus, generate a log message to help people trying to debug the "crash".
  double elapsedMs = Cycles::toSeconds(Cycles::rdtsc() - start)*1e3;
  if (elapsedMs > 10) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    CodeLocation here = HERE;
    snprintf(buffer, sizeof(buffer), "%010lu.%09lu %s:%d in %s "
        "ERROR[%d]: Logger got stuck for %.1f ms, which could "
        "hang server\n",
        now.tv_sec, now.tv_nsec, here.baseFileName(),
        here.line, here.function, ThreadId::get(),
        elapsedMs);
    addToBuffer(buffer, static_cast<int>(strlen(buffer)));
  }
}

/**
 * This method copies a block of data into the printBuffer and wakes up
 * the print thread to output it.
 * 
 * \param src
 *      First byte in block of data to add to the buffer.
 * \param length
 *      Total number of bytes of data to add.  Must be > 0.
 * 
 * \return
 *      The return value is true if the data was successfully added, and false
 *      if there wasn't enough space for all of it (in which case none is
 *      added).
 */
bool
Logger::addToBuffer(const char* src, int length)
{
  // No lock needed: already locked by caller

  // First, write data at the end of the buffer, if there's space there.
  if (nextToInsert >= nextToPrint) {
      int count = std::min(bufferSize - nextToInsert, length);
      memcpy(messageBuffer + nextToInsert, src, count);
      src += count;
      nextToInsert += count;
      if (nextToInsert == bufferSize) {
          nextToInsert = 0;
      }
      length -= count;
  }

  if (length > 0) {
    // We get here if the space at the end of the buffer was all
    // occupied, or if that wasn't enough to hold all of the data.
    // See if there's space at the beginning of the buffer. Note: we
    // can't use the last byte before nextToPrint; if we do, we'll
    // wrap around and lose all the buffered data.
    if (length > (nextToPrint - 1 - nextToInsert)) {
        return false;
    }
    memcpy(messageBuffer + nextToInsert, src, length);
    nextToInsert += length;
  }
  logDataAvailable.notify_one();
  return true;
}

/**
 * This method is the main program for a separate thread that runs in the
 * background to print log messages to secondary storage (this way, the
 * main RAMCloud threads aren't delayed for I/O).
 * 
 * \param logger
 *      The owning Logger. This thread accesses only information related
 *      to the buffer and the output file.
 */
void
Logger::printThreadMain(Logger* logger)
{
  Lock lock(logger->mutex);
  std::cout << "printThreadMain is starting.." << std::endl;
  int bytesToPrint;
  ssize_t bytesPrinted;
  while (true) {
    if (logger-> printThreadExit) {
      std::cout << "printThreadMain is exiting.." << std::endl;
      return;
    }

    // Handle buffer wraparound.
    if (logger->nextToPrint >= logger->bufferSize) {
      logger->nextToPrint = 0;
    }

    // See if there is new log data to print.
    bytesToPrint = logger->nextToInsert - logger->nextToPrint;
    if (bytesToPrint < 0) {
      // If we get here, it means that nextToInsert has wrapped back
      // to the start of the buffer.
      bytesToPrint = logger->bufferSize - logger->nextToPrint;
    }
    if (bytesToPrint == 0) {
      // The line below is not needed for correctness, but it
      // results in better cache performance: as long as the printer
      // keeps up, only the first part of the buffer will be used,
      // and the later parts will never be touched.
      logger->nextToPrint = logger->nextToInsert = 0;
      logger->logDataAvailable.wait(lock);
      continue;
    }

    // Print whatever data is available (be careful not to hold the
    // buffer lock during expensive kernel calls, since this will
    // block calls to logMessage).
    {
      int fd = logger->fd;
      int nextToPrint = logger->nextToPrint;
      logger->mutex.unlock();
      // Since we release the lock before write, there could be a race
      // condition when another thread acquires the lock and tries to
      // close fd. Therefore, we must ensure that setLogFile() will only
      // be called when the print thread is not running.
      bytesPrinted = write(fd, logger->messageBuffer + nextToPrint,
          bytesToPrint);
      logger->mutex.lock();
    }
    if (bytesPrinted < 0) {
        fprintf(stderr, "Error writing log: %s\n", strerror(errno));

        // Skip these bytes; otherwise we'll loop infinitely.
        logger->nextToPrint += static_cast<int>(bytesToPrint);
    } else {
        logger->nextToPrint += static_cast<int>(bytesPrinted);
    }
    logger->bufferSpaceFreed.notify_all();
  }
}
