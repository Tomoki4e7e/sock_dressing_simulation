using UnityEngine;

namespace SockDressing
{
    public static class SockSceneBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void EnsureScene()
        {
            if (Object.FindObjectOfType<SockClothRuntime>() != null)
                return;

            GameObject sock = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            sock.name = "SockCloth";
            sock.transform.SetPositionAndRotation(
                new Vector3(-0.14f, 0.68f, 0.53f),
                Quaternion.Euler(90, 0, 0));
            sock.transform.localScale = new Vector3(0.08f, 0.15f, 0.08f);
            Object.Destroy(sock.GetComponent<Collider>());
            sock.GetComponent<Renderer>().material.color = Color.red;
            StableObjectId sockId = sock.AddComponent<StableObjectId>();
            sockId.objectId = 1200;
            sockId.region = "sock";

            SockClothRuntime runtime = sock.AddComponent<SockClothRuntime>();
            runtime.sockRenderer = sock.GetComponent<Renderer>();
            sock.AddComponent<SockSceneController>();
            sock.AddComponent<SockTcpBridge>();

            GameObject robot = AddRegion(
                null,
                "robot",
                1100,
                new Vector3(0, 1.35f, 0),
                new Vector3(.35f, .45f, .25f));
            robot.GetComponent<Renderer>().material.color = Color.gray;

            GameObject human = new("SeatedHuman");
            GameObject calf = AddRegion(human.transform, "calf", 2101, new Vector3(0, 0.72f, 0.62f), new Vector3(.12f, .38f, .12f));
            GameObject ankle = AddRegion(human.transform, "ankle", 2102, new Vector3(0, 0.55f, 0.73f), new Vector3(.11f, .14f, .12f));
            GameObject heel = AddRegion(human.transform, "heel", 2103, new Vector3(0, 0.49f, 0.78f), new Vector3(.13f, .10f, .15f));
            GameObject forefoot = AddRegion(human.transform, "forefoot", 2104, new Vector3(0, 0.47f, 0.93f), new Vector3(.15f, .09f, .22f));
            GameObject toes = AddRegion(human.transform, "toes", 2105, new Vector3(0, 0.47f, 1.08f), new Vector3(.15f, .08f, .10f));
            runtime.legRenderer = toes.GetComponent<Renderer>();
            runtime.legRenderers = new[]
            {
                calf.GetComponent<Renderer>(),
                ankle.GetComponent<Renderer>(),
                heel.GetComponent<Renderer>(),
                forefoot.GetComponent<Renderer>(),
                toes.GetComponent<Renderer>()
            };

            GameObject left = AddRegion(robot.transform, "left_gripper", 2201, new Vector3(-0.18f, .68f, .53f), Vector3.one * .04f);
            GameObject right = AddRegion(robot.transform, "right_gripper", 2202, new Vector3(-0.10f, .68f, .53f), Vector3.one * .04f);
            runtime.leftGripper = left.transform;
            runtime.rightGripper = right.transform;

            GameObject chair = AddRegion(null, "chair", 2301, new Vector3(0, .42f, -.18f), new Vector3(.62f, .10f, .58f));
            chair.GetComponent<Renderer>().material.color = new Color(.25f, .2f, .15f);
            AddRegion(null, "floor", 2302, new Vector3(0, -.03f, 0), new Vector3(4, .05f, 4));

            Camera camera = CreateCamera("SockRgbCamera", new Vector3(-1.8f, 1.05f, -1.5f), new Vector3(10, 42, 0));
            Camera mask = CreateCamera("SockMaskCamera", camera.transform.position, camera.transform.eulerAngles);
            mask.enabled = false;
            runtime.rgbCamera = camera;
            runtime.maskCamera = mask;
        }

        private static GameObject AddRegion(
            Transform parent,
            string region,
            int id,
            Vector3 position,
            Vector3 scale)
        {
            GameObject item = GameObject.CreatePrimitive(PrimitiveType.Cube);
            item.name = region;
            item.transform.SetPositionAndRotation(position, Quaternion.identity);
            item.transform.localScale = scale;
            if (parent != null)
                item.transform.SetParent(parent, true);
            StableObjectId stable = item.AddComponent<StableObjectId>();
            stable.objectId = id;
            stable.region = region;
            stable.obiColliderExpected = true;
            item.GetComponent<Renderer>().material.color = Color.green;
            return item;
        }

        private static Camera CreateCamera(string name, Vector3 position, Vector3 rotation)
        {
            GameObject item = new(name);
            Camera camera = item.AddComponent<Camera>();
            camera.transform.SetPositionAndRotation(position, Quaternion.Euler(rotation));
            camera.fieldOfView = 60;
            camera.nearClipPlane = .1f;
            camera.farClipPlane = 3;
            return camera;
        }
    }
}
