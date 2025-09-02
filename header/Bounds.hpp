#pragma once
#include <cmath>

namespace geom {

struct AABBf {
    float  min_x, min_y, max_x, max_y;
    float width()  const noexcept { return max_x - min_x; }
    float height() const noexcept { return max_y - min_y; }
    bool valid() const noexcept {
        return std::isfinite(min_x) && std::isfinite(min_y) &&
               std::isfinite(max_x) && std::isfinite(max_y);
    }
};

struct AABBd {
    float min_x, min_y, max_x, max_y;
};

} // namespace geom

