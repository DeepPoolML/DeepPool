/* Copyright (c) 2011 Stanford University
 * Copyright (c) 2021 MIT
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "CodeLocation.h"
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wconversion"
// #pragma GCC diagnostic ignored "-Weffc++"
// #include <pcrecpp.h>
// #pragma GCC diagnostic pop
#include <assert.h>
#include <cstring>
#include "utils.h"

namespace RAMCloud {

namespace {

/**
 * Return the number of characters of __FILE__ that make up the path prefix.
 * That is, __FILE__ plus this value will be the relative path from the top
 * directory of the RAMCloud repo.
 */
int
length__FILE__Prefix()
{
    const char* start = __FILE__;
    const char* match = strstr(__FILE__, "src/CodeLocation.cc");
    assert(match != NULL);
    return static_cast<int>(match - start);
}

} // anonymous namespace


/**
 * Return the base name of the file (i.e., only the last component of the
 * file name, omitting any preceding directories).
 */
const char*
CodeLocation::baseFileName() const
{
    const char* lastSlash = strrchr(file, '/');
    if (lastSlash == NULL) {
        return file;
    }
    return lastSlash+1;
}

std::string
CodeLocation::relativeFile() const
{
    static int lengthFilePrefix = length__FILE__Prefix();
    // Remove the prefix only if it matches that of __FILE__. This check is
    // needed in case someone compiles different files using different paths.
    if (strncmp(file, __FILE__, lengthFilePrefix) == 0)
        return std::string(file + lengthFilePrefix);
    else
        return std::string(file);
}

/**
 * Return the name of the function, qualified by its surrounding classes and
 * namespaces. Note that this strips off the RAMCloud namespace to produce
 * shorter strings.
 *
 * Beware: this method is really really slow (10-20 microseconds); we no
 * longer use it in log messages because it wastes so much time.
 */
std::string
CodeLocation::qualifiedFunction() const
{
    // std::string ret;
    // const std::string pattern(
    //     format("\\s(?:RAMCloud::)?(\\S*\\b%s)\\(", function));
    // if (pcrecpp::RE(pattern).PartialMatch(prettyFunction, &ret))
    //     return ret;
    // else // shouldn't happen
        return function;
}

/**
 * Returns the string representation of CodeLocation.
 */
std::string
CodeLocation::str() const {
    return format("%s at %s:%d",
                    qualifiedFunction().c_str(),
                    relativeFile().c_str(),
                    line);
}

} // namespace RAMCloud
