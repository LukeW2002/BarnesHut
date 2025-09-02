#pragma once




class MortonCode {
public:
    static uint64_t encode_morton_2d(uint32_t x, uint32_t y) {
        return (expand_bits_2d(x) << 1) | expand_bits_2d(y);
    }
    

private:
    static uint64_t expand_bits_2d(uint32_t v) {
        uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FF;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2))  & 0x3333333333333333;
        x = (x | (x << 1))  & 0x5555555555555555;
        return x;
    }
    
    static uint32_t compact_bits_2d(uint64_t x) {
        x &= 0x5555555555555555;
        x = (x ^ (x >> 1))  & 0x3333333333333333;
        x = (x ^ (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
        x = (x ^ (x >> 4))  & 0x00FF00FF00FF00FF;
        x = (x ^ (x >> 8))  & 0x0000FFFF0000FFFF;
        x = (x ^ (x >> 16)) & 0x00000000FFFFFFFF;
        return static_cast<uint32_t>(x);
    }
};

class MortonEncoder {
public:
    static uint64_t encode_position(float x, float y, const geom::AABBf& b) noexcept {
        const double rx = std::max<double>(b.max_x - b.min_x, 1e-10);
        const double ry = std::max<double>(b.max_y - b.min_y, 1e-10);
        const float  nx = std::clamp<float>(float((x - b.min_x) / rx), 0.0f, 1.0f);
        const float  ny = std::clamp<float>(float((y - b.min_y) / ry), 0.0f, 1.0f);

        constexpr uint32_t maxc = (1u << 21) - 1;
        const uint32_t ix = uint32_t(nx * maxc);
        const uint32_t iy = uint32_t(ny * maxc);
        return MortonCode::encode_morton_2d(ix, iy);
    }

    static void encode_morton_keys(const float* xs, const float* ys,
                                   uint64_t* out_keys, std::size_t n,
                                   const geom::AABBf& b, bool parallel) noexcept
    {
    #ifdef _OPENMP
        if (parallel && n > 1000) {
            #pragma omp parallel for
            for (std::size_t i = 0; i < n; ++i)
                out_keys[i] = encode_position(xs[i], ys[i], b);
        } else
    #endif
        {
            for (std::size_t i = 0; i < n; ++i)
                out_keys[i] = encode_position(xs[i], ys[i], b);
        }
    }
};


