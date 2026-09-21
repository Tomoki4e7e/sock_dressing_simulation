# SockDressingPlayer Unity project

This is a greenfield, wire-compatible player. It does not contain or modify the
binary-only RCareWorld Unity project.

Required:

- Unity `2022.3.34f1` with Linux Build Support (Mono)
- A licensed Obi Cloth package
- `assets/generated/sock.obj`
- `assets/generated/dry_airec3_gripper_v2_rcareworld.urdf` and its meshes

Import Obi, add `SOCKDRESSING_OBI` to the Standalone scripting define symbols,
generate an Obi cloth blueprint from `sock.obj`, assign `ObiSockBackend` to
`SockClothRuntime.obiAdapter`, and populate the opening particle indices. The
Player build rejects missing Obi unless `SOCK_ALLOW_STUB_BUILD=1`; that override
is only for testing TCP communication and cannot pass physical acceptance.

Build from the package root:

```bash
./scripts/build_custom_player.sh development
./scripts/build_custom_player.sh release
```

Outputs are isolated under `Build/SockDressingPlayer/`. The runtime bootstrap
creates a diagnostic scene when no authored scene exists. That scene deliberately
reports `obi_available=false`; it is not a physical sock-dressing result.
