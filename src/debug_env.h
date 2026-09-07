#pragma once

#include <cstdlib>

namespace r2ppnp::internal {

inline bool env_flag_enabled(const char* name)
{
#if defined(_MSC_VER)
    char* value = nullptr;
    std::size_t length = 0;
    const errno_t error = _dupenv_s(&value, &length, name);
    const bool enabled = (error == 0 && value != nullptr);
    std::free(value);
    return enabled;
#else
    return std::getenv(name) != nullptr;
#endif
}

} // namespace r2ppnp::internal
