#pragma once

#include <vector>
#include <array>
#include <string>
#include <memory>
#include "../Math.h"

namespace hou
{
    struct Mesh
    {
        explicit Mesh(const char* path);
        std::vector<Triangle> triangles;

        std::vector<NPQueryRet> query_np(const std::vector<Point>& query) const;

    private:
        std::shared_ptr<void> m_gdp;
    };
}