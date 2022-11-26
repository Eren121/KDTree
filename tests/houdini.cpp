#include "houdini.hpp"
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPolySoup.h>
#include <GU/GU_DetailInfo.h>
#include <UT/UT_Exit.h>
#include <GU/GU_RayIntersect.h>
#include <iostream>
#include "Utils.h"

namespace hou
{
    long getRayIntersectPrimitive(GU_RayIntersect& ray, UT_Vector3F p, float& u, float& v, float& w)
    {
        long polyId;

        GU_MinInfo mininfo;

        mininfo.init();
        ray.minimumPoint(p, mininfo);

        // We could not find the primitive intersection
        if(!mininfo.prim)
        {
            std::cout << "error" << std::endl;
            exit(1);
        }

        //Get the Primitive ID
        const GEO_Primitive *geoPrim = mininfo.prim;
        polyId = geoPrim->getMapOffset();

        //se uv values of the rayIntersect point
        u = mininfo.u1;
        v = mininfo.v1;
        w = mininfo.w1;

        // To determine the actual 3D coordinates of the closest point, the UV coords have to be evaluated by the primitive:
        //Solution from:
        //https://forums.odforce.net/topic/8335-hdk-spatial-geometry-storage-solved/
        UT_Vector4 result;
        if(mininfo.prim)
        {
            mininfo.prim->evaluateInteriorPoint(result, mininfo.u1, mininfo.v1);
            u = result.x();
            v = result.y();
            w = result.z();

            //mininfo.prim-&gt;evaluateInteriorPoint(result, mininfo.u, mininfo.v);
            //The 3D point is now stored in "result"!
            //Now you can add it to the GDP or do whatever!
        }
        else
        {
            std::cout << "error" << std::endl;
        }


        return polyId;
    }

    Mesh::Mesh(const char* path)
    {
        auto gdp = std::make_unique<GU_Detail>();

        {
            Timer t("Load file from disk with Houdini");

            const auto status = gdp->load(path);
            if(!status.success())
            {
                std::cout << "Error loading bgeo file" << std::endl;
                exit(1);
            }
        }

        // Iterate all triangles of the geometry with Houdini
        GEO_Primitive *prim;
        GA_FOR_ALL_PRIMITIVES(&*gdp, prim)
        {
            Triangle tri;
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    tri.points[i][j] = prim->getPos3(i)(j);
                }
            }

            triangles.push_back(tri);
        }

        m_gdp = std::move(gdp);
    }

    std::vector<NPQueryRet> Mesh::query_np(const std::vector<Point>& query) const
    {
        std::unique_ptr<GU_RayIntersect> ray;

        {
            Timer timer("Houdini Build AS");

            ray = std::make_unique<GU_RayIntersect>(static_cast<GU_Detail *>(m_gdp.get()));
        }

        std::vector<NPQueryRet> res(query.size());

        {
            Timer t2("Houdini queries");

            for(int i = 0; i < query.size(); i++)
            {
                const auto& q = query[i];
                res[i].id = getRayIntersectPrimitive(*ray, {q[0], q[1], q[2]},
                                                     res[i].point.x, res[i].point.y, res[i].point.z);
            }
        }

        return res;
    }
}