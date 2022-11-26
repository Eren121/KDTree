#pragma once

#include <cassert>
#include <cmath>

#ifndef CUDA_BOTH
    #ifdef __CUDACC__
        #define CUDA_BOTH __device__ __host__
    #else
        #define CUDA_BOTH
    #endif
#endif

/**
 * @{
 * Byte unit multiples.
 */
static inline constexpr int KB = 1'000;
static inline constexpr int MB = 1'000'000;
/**
 * @}
 */

struct Point
{
    float x, y, z;

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wreturn-type"

    CUDA_BOTH float& operator[](int i)
    {
        assert(i < 3 && i >= 0);

        switch(i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
        }

        assert(false && "Invalid index");
    }

    #pragma GCC diagnostic pop

    CUDA_BOTH const float& operator[](int i) const
    {
        return const_cast<Point&>(*this)[i];
    }

    [[nodiscard]] CUDA_BOTH float distance(const Point& other) const
    {
        return sqrtf(distanceSquared(other));
    }

    [[nodiscard]] CUDA_BOTH float distanceSquared(const Point& other) const
    {
        const float dx = (x - other.x);
        const float dy = (y - other.y);
        const float dz = (z - other.z);
        return dx*dx + dy*dy + dz*dz;
    }

    CUDA_BOTH Point operator-(const Point& other) const
    {
        return {x - other.x, y - other.y, z - other.z};
    }

    CUDA_BOTH Point operator*(float f) const
    {
        return {x * f, y * f, z * f};
    }

    CUDA_BOTH Point operator+(const Point& other) const
    {
        return {x + other.x, y + other.y, z + other.z};
    }

    friend CUDA_BOTH float dot(const Point& a, const Point& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    template<typename T> friend T& operator<<(T& lhs, const Point& rhs)
    {
        lhs << "(" << rhs.x << ", " << rhs.y << ", " << rhs.z << ")";
        return lhs;
    }
};

struct Triangle
{
    using ID = int;
    Point points[3];

    Triangle() = default;
    CUDA_BOTH Triangle(const Point& a, const Point& b, const Point& c) : points{a, b, c} {}

    Point& operator[](int i) { return points[i]; }
    const Point& operator[](int i) const { return points[i]; }
};

using Mesh = std::vector<Triangle>;
using MeshAsID = std::vector<Triangle::ID>;

struct AABB
{
    Point min, max;

    [[nodiscard]] CUDA_BOTH bool inside(const Point& p) const
    {
        return p.x >= min.x && p.y >= min.y && p.z >= min.z
               && p.x < max.x  && p.y < max.y  && p.z < max.z;
    }
};

enum Side
{
    NEAR, FAR
};

/**
 * Axis aligned straight line.
 */
struct Line
{
    /**
     * The position of the straight line in the init dimension.
     * As this is a straight line, it goes ad infinitum to the others 2 dimensions.
     */
    float p;

    /**
     * The dimension of the init.
     * 0 for X, 1 for Y, 2 for Z.
     */
    int dim;

    /**
     * Check on which side a point is from a straight line.
     * @param point The point to check.
     *
     * @return
     *      NEAR if the point is below the axe in the init dimension, otherwise FAR.
     *      For example, the straight line X=5.5 is represented by {p=5.5, dim=0}.
     *      Then for a point P=(3, 123, 456) query returns NEAR.
     *      For a point P'=(6, 987, 654) query returns FAR.
     */
    [[nodiscard]] CUDA_BOTH Side query(const Point& point) const
    {
        return point[dim] < p ? NEAR : FAR;
    }
};

struct NPQueryRet
{
    Point point;
    Triangle::ID id;
};