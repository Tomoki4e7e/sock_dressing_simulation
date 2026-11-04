#!/usr/bin/env python3
"""Decode convex vertices/planes from a Unity 2022.3 diagnostic asset bundle.

Run under Python 3.11 with UnityPy installed. The binary reader is deliberately
limited to the observed PhysX CVXM 13 / CLHL 9 little endian format; fail closed
on another layout. Layout: PhysX 4.1 GuConvexMesh.cpp convexHullLoad().
"""
import argparse
import json
import struct
from pathlib import Path


def decode_convex(data):
    if data[:8]!=b'NXS\x01CVXM' or data[16:24]!=b'ICE\x01CLHL':
        raise ValueError('Unsupported convex binary header or byte order')
    version=struct.unpack_from('<I',data,8)[0];hull_version=struct.unpack_from('<I',data,24)[0]
    if (version,hull_version)!=(13,9):raise ValueError('Unsupported convex binary version')
    nv,ne,nf,ni=struct.unpack_from('<4I',data,28)
    if not 4<=nv<=255 or not 4<=nf<=255:raise ValueError('Invalid convex counts')
    offset=44+nv*12
    if len(data)<offset+nf*20+ni:raise ValueError('Truncated convex binary')
    vertices=[struct.unpack_from('<3f',data,44+12*i) for i in range(nv)]
    planes=[struct.unpack_from('<4f',data,offset+20*i) for i in range(nf)]
    polygons=[dict(index_base=struct.unpack_from('<H',data,offset+20*i+16)[0],vertex_count=data[offset+20*i+18]) for i in range(nf)]
    indices=list(data[offset+nf*20:offset+nf*20+ni])
    if any(i>=nv for i in indices):raise ValueError('Polygon vertex index out of range')
    if any(p['index_base']+p['vertex_count']>ni for p in polygons):raise ValueError('Polygon index range out of bounds')
    return dict(vertices=vertices,planes=planes,polygons=polygons,indices=indices,convex_version=version,hull_version=hull_version,vertex_count=nv)


def main():
    import UnityPy
    ap=argparse.ArgumentParser();ap.add_argument('bundle',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args();rows=[]
    for obj in UnityPy.load(str(args.bundle)).objects:
        if obj.type.name!='Mesh':continue
        tree=obj.read_typetree();data=bytes(tree['m_BakedConvexCollisionMesh'])
        if not data:raise ValueError('Missing baked convex data: '+tree['m_Name'])
        rows.append(dict(name=tree['m_Name'],cooking_options=tree['m_CookingOptions'],**decode_convex(data)))
    output=args.output or args.bundle.parent/'decoded.json';output.write_text(json.dumps(dict(parts=rows,source=str(args.bundle)),indent=2)+'\n');print('decoded',len(rows),'convex meshes')


if __name__=='__main__':main()
