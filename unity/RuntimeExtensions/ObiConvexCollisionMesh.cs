using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
namespace Obi
{
    // MeshCollider.convex is a solid, even if its shared mesh has cavities.
    // Cache a closed outward hull for Obi rather than treating that source
    // mesh as independent two-sided walls.
    internal static class ObiConvexCollisionMesh
    {
        private sealed class Face
        {
            public int a,b,c;
            public readonly List<int> outside=new List<int>();
            private double nx,ny,nz;
            private Vector3 anchor;
            public double Distance(Vector3 point)=>nx*((double)point.x-anchor.x)+ny*((double)point.y-anchor.y)+nz*((double)point.z-anchor.z);
            public bool Valid;
            public Face(int a,int b,int c,Vector3[] points,double[] interior)
            {
                this.a=a;this.b=b;this.c=c;anchor=points[a];
                double ux=(double)points[b].x-anchor.x,uy=(double)points[b].y-anchor.y,uz=(double)points[b].z-anchor.z;
                double vx=(double)points[c].x-anchor.x,vy=(double)points[c].y-anchor.y,vz=(double)points[c].z-anchor.z;
                nx=uy*vz-uz*vy;ny=uz*vx-ux*vz;nz=ux*vy-uy*vx;
                double length=Math.Sqrt(nx*nx+ny*ny+nz*nz);Valid=length>0 && !double.IsInfinity(length);nx/=length;ny/=length;nz/=length;
                if(nx*(interior[0]-anchor.x)+ny*(interior[1]-anchor.y)+nz*(interior[2]-anchor.z)>0){this.b=c;this.c=b;nx=-nx;ny=-ny;nz=-nz;}
            }
        }
        private static readonly Dictionary<Mesh,Mesh> cache=new Dictionary<Mesh,Mesh>();
        internal static void Release(Mesh source,Mesh collision)
        {
            if(ReferenceEquals(source,null) || ReferenceEquals(source,collision))return;
            if(cache.TryGetValue(source,out var mesh) && mesh==collision)
            {cache.Remove(source);if(mesh!=null)UnityEngine.Object.Destroy(mesh);}
        }
        // Explicit opt-in for already decomposed closed convex inputs. Avoid
        // rebuilding their very thin facets with the general QuickHull path.
        public static void RegisterValidatedConvex(Mesh source)
        {
            if(source==null)throw new ArgumentNullException(nameof(source));
            var points=source.vertices;var triangles=source.triangles;
            if(points.Length<4 || points.Length>64 || triangles.Length<12)throw new ArgumentException("Invalid convex part");
            double[] interior={points.Average(p=>(double)p.x),points.Average(p=>(double)p.y),points.Average(p=>(double)p.z)};
            double tolerance=Math.Max(1e-9,source.bounds.size.magnitude*1e-4);
            var edges=new Dictionary<Tuple<int,int>,int>();int valid=0;
            for(int i=0;i<triangles.Length;i+=3)
            {
                var face=new Face(triangles[i],triangles[i+1],triangles[i+2],points,interior);
                if(face.Valid)
                {
                    valid++;
                    if(points.Any(p=>face.Distance(p)>tolerance))throw new ArgumentException("Input part is not convex within validated tolerance");
                }
                for(int j=0;j<3;j++)
                {
                    int a=triangles[i+j],b=triangles[i+(j+1)%3];var key=Tuple.Create(Math.Min(a,b),Math.Max(a,b));
                    edges[key]=edges.TryGetValue(key,out int count)?count+1:1;
                }
            }
            if(valid<4 || edges.Values.Any(count=>count!=2))throw new ArgumentException("Input convex part is not closed");
            cache[source]=source;
        }

        public static Mesh Get(Mesh source)
        {
            if(cache.TryGetValue(source,out var mesh) && mesh!=null)return mesh;
            // Weld numerical seams at one part per million of mesh extent,
            // independently of cloth particle radii.
            double weld=Math.Max(1e-9,source.bounds.size.magnitude*1e-6);
            var points=source.vertices.Select(p=>new Vector3((float)(Math.Round(p.x/weld)*weld),(float)(Math.Round(p.y/weld)*weld),(float)(Math.Round(p.z/weld)*weld))).Distinct().ToArray();
            if(points.Length<4)throw new ArgumentException("Convex collider needs four distinct vertices");
            float tolerance=Mathf.Max(1e-12f,source.bounds.size.magnitude*1e-10f);
            int a=0,b=0,c=0,d=0;float largest=-1;
            for(int i=0;i<points.Length;i++)if(points[i].x<points[a].x)a=i;
            for(int i=0;i<points.Length;i++)
            {float value=(points[i]-points[a]).sqrMagnitude;if(value>largest){largest=value;b=i;}}
            largest=-1;Vector3 line=points[b]-points[a];
            for(int i=0;i<points.Length;i++)
            {float value=Vector3.Cross(points[i]-points[a],line).sqrMagnitude;if(value>largest){largest=value;c=i;}}
            Vector3 plane=Vector3.Cross(points[b]-points[a],points[c]-points[a]);
            plane=plane/plane.magnitude;largest=-1;
            for(int i=0;i<points.Length;i++)
            {float value=Mathf.Abs(Vector3.Dot(plane,points[i]-points[a]));if(value>largest){largest=value;d=i;}}
            if(largest<=tolerance)throw new ArgumentException("Convex collider mesh is planar");
            double[] interior={((double)points[a].x+points[b].x+points[c].x+points[d].x)*.25,((double)points[a].y+points[b].y+points[c].y+points[d].y)*.25,((double)points[a].z+points[b].z+points[c].z+points[d].z)*.25};
            var faces=new List<Face>{new Face(a,b,c,points,interior),new Face(a,d,b,points,interior),new Face(a,c,d,points,interior),new Face(b,d,c,points,interior)};
            Face Assign(int point,IEnumerable<Face> candidates)
            {
                Face best=null;double greatest=tolerance;
                foreach(var face in candidates)
                {double distance=face.Distance(points[point]);if(distance>greatest){greatest=distance;best=face;}}
                if(best!=null)best.outside.Add(point);return best;
            }
            for(int p=0;p<points.Length;p++)Assign(p,faces);
            int iterations=0;
            while(true)
            {
                var selected=faces.FirstOrDefault(f=>f.outside.Count>0);
                if(selected==null)break;
                if(++iterations>points.Length)throw new InvalidOperationException("Convex hull failed to converge");
                int point=selected.outside.OrderByDescending(p=>selected.Distance(points[p])).First();
                var visible=faces.Where(f=>f.Distance(points[point])>tolerance).ToList();
                var orphan=new HashSet<int>(visible.SelectMany(f=>f.outside));orphan.Remove(point);
                var boundary=new HashSet<(int,int)>();
                foreach(var face in visible)
                    foreach(var edge in new[]{(face.a,face.b),(face.b,face.c),(face.c,face.a)})
                        if(!boundary.Remove((edge.Item2,edge.Item1)))boundary.Add(edge);
                foreach(var face in visible)faces.Remove(face);
                var added=new List<Face>();
                foreach(var edge in boundary)
                {
                    var face=new Face(edge.Item1,edge.Item2,point,points,interior);
                    if(face.Valid){faces.Add(face);added.Add(face);}
                }
                foreach(int p in orphan)if(Assign(p,added)==null)Assign(p,faces);
            }
            if(faces.Count<4 || faces.Any(f=>points.Any(p=>f.Distance(p)>tolerance*20)))
            {
                string log=Environment.GetEnvironmentVariable("SOCK_GRASP_MOTION_LOG");
                if(!string.IsNullOrEmpty(log))
                {
                    string path=System.IO.Path.Combine(System.IO.Path.GetDirectoryName(log),"convex-failed-"+Math.Abs(source.GetInstanceID())+".obj");
                    if(!System.IO.File.Exists(path))
                    {
                        var text=new System.Text.StringBuilder();
                        foreach(var point in source.vertices)text.AppendFormat(System.Globalization.CultureInfo.InvariantCulture,"v {0:R} {1:R} {2:R}\n",point.x,point.y,point.z);
                        var original=source.triangles;
                        for(int i=0;i<original.Length;i+=3)text.AppendFormat("f {0} {1} {2}\n",original[i]+1,original[i+1]+1,original[i+2]+1);
                        System.IO.File.WriteAllText(path,text.ToString());
                    }
                }
                throw new InvalidOperationException("Invalid convex collider hull: "+source.name+" points="+points.Length+" faces="+faces.Count+" outside="+faces.Max(f=>points.Max(p=>f.Distance(p))));
            }
            var indices=faces.SelectMany(f=>new[]{f.a,f.b,f.c}).ToArray();
            // Keep original local coordinates/scaling; hull only removes
            // concave/internal faces, as Unity's convex collision cooking does.
            mesh=new Mesh{name=source.name+" Obi convex hull",hideFlags=HideFlags.HideAndDontSave};
            mesh.vertices=points;mesh.triangles=indices;mesh.RecalculateBounds();
            cache[source]=mesh;return mesh;
        }
    }
}
