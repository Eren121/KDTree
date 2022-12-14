#pragma once

#include <chrono>
#include <vector>
#include <random>

class Timer
{
public:
    Timer(const char* title) : title_(title), beg_(clock_::now()) {}

    ~Timer()
    {
        std::cout << title_ << " elapsed: " << elapsed() << "s" << std::endl;
    }

    void reset() { beg_ = clock_::now(); }

    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

private:
    const char* title_;
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace
{
    std::vector<Point> randomPoints(int size, float bounds = 10.0f)
    {
        std::vector<Point> points;

        std::uniform_real_distribution<float> dist(-bounds, bounds);
        std::mt19937 engine;

        for(int i = 0; i < size; i++)
        {
            Point point;

            for(int dim = 0; dim < 3; dim++)
            {
                point[dim] = dist(engine);
            }

            points.push_back(point);
        }

        return points;
    }

    Mesh randomTriangles(int size, float bounds = 10.0f)
    {
        Mesh ret;

        auto vertices = randomPoints(size * 3, bounds);

        for(int i = 0; i < vertices.size() / 3; i++)
        {
            ret.emplace_back(
                vertices[i * 3 + 0],
                vertices[i * 3 + 0] + vertices[i * 3 + 1] * 0.05f,
                vertices[i * 3 + 0] + vertices[i * 3 + 2] * 0.05f
            );
        }

        return ret;
    }
}