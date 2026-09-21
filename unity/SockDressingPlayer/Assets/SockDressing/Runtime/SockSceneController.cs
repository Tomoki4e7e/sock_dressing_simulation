using System;
using System.Collections.Generic;
using UnityEngine;

namespace SockDressing
{
    [RequireComponent(typeof(SockClothRuntime))]
    public sealed class SockSceneController : MonoBehaviour
    {
        private readonly Queue<List<object>> commands = new();
        private SockClothRuntime cloth;
        private bool collectRequested;
        private readonly float[] robotJoints = new float[29];
        private static readonly string[] RobotJointNames =
        {
            "base_aux_0", "base_aux_1", "base_aux_2", "base_aux_3", "base_aux_4",
            "base_aux_5", "base_aux_6", "torso/joint_1", "torso/joint_2",
            "torso/joint_3", "base_aux_10", "base_aux_11", "base_aux_12",
            "right_arm/joint_1", "right_arm/joint_2", "right_arm/joint_3",
            "right_arm/joint_4", "right_arm/joint_5", "right_arm/joint_6",
            "right_arm/joint_7", "right_gripper/finger_joint",
            "left_arm/joint_1", "left_arm/joint_2", "left_arm/joint_3",
            "left_arm/joint_4", "left_arm/joint_5", "left_arm/joint_6",
            "left_arm/joint_7", "left_gripper/finger_joint"
        };

        private void Awake()
        {
            cloth = GetComponent<SockClothRuntime>();
            Physics.autoSimulation = false;
        }

        public void Enqueue(List<object> message)
        {
            commands.Enqueue(message);
        }

        public void FlushStep(SockTcpBridge bridge)
        {
            collectRequested = false;
            while (commands.Count > 0)
                Dispatch(commands.Dequeue(), bridge);
            if (collectRequested)
                SendSceneState(bridge);
        }

        private void Dispatch(List<object> message, SockTcpBridge bridge)
        {
            if (message.Count == 0)
                return;
            string channel = Convert.ToString(message[0]);
            if (channel == "Env")
            {
                DispatchEnvironment(message, bridge);
                return;
            }
            if (channel == "Instance")
            {
                DispatchInstance(message);
                return;
            }
            // Debug/Message/Object channels are intentionally accepted as no-ops.
        }

        private void DispatchEnvironment(List<object> message, SockTcpBridge bridge)
        {
            if (message.Count < 2)
                return;
            string command = Convert.ToString(message[1]);
            switch (command)
            {
                case "Simulate":
                    float requested = message.Count > 2 ? Convert.ToSingle(message[2]) : -1;
                    int count = message.Count > 3 ? Convert.ToInt32(message[3]) : 1;
                    float timestep = requested > 0 ? requested : Time.fixedDeltaTime;
                    for (int i = 0; i < Math.Max(1, count); i++)
                        Physics.Simulate(timestep);
                    break;
                case "Collect":
                    collectRequested = true;
                    break;
                case "PreLoadAssetsAsync":
                case "LoadSceneAsync":
                    bridge.Send("Env", new Dictionary<string, object> { ["load_done"] = true });
                    break;
                case "Close":
                    Application.Quit();
                    break;
            }
        }

        private void DispatchInstance(List<object> message)
        {
            if (message.Count < 3)
                return;
            int objectId = Convert.ToInt32(message[1]);
            string command = Convert.ToString(message[2]);
            if (objectId == cloth.robotId && command == "SetJointPosition" &&
                message.Count > 3 && message[3] is List<object> values)
            {
                for (int i = 0; i < Math.Min(values.Count, robotJoints.Length); i++)
                    robotJoints[i] = Convert.ToSingle(values[i]);
                return;
            }
            if (objectId != cloth.sockId)
                return;
            switch (command)
            {
                case "GraspLeft":
                    cloth.Grasp("left", Convert.ToSingle(message[3]));
                    break;
                case "GraspRight":
                    cloth.Grasp("right", Convert.ToSingle(message[3]));
                    break;
                case "ReleaseLeft":
                    cloth.Release("left");
                    break;
                case "ReleaseRight":
                    cloth.Release("right");
                    break;
                case "ResetSock":
                    cloth.ResetSock();
                    Array.Clear(robotJoints, 0, robotJoints.Length);
                    break;
                case "SetTransform":
                    if (message.Count > 3 && message[3] is List<object> position &&
                        position.Count == 3)
                        cloth.transform.position = new Vector3(
                            Convert.ToSingle(position[0]),
                            Convert.ToSingle(position[1]),
                            Convert.ToSingle(position[2]));
                    if (message.Count > 4 && message[4] is List<object> rotation &&
                        rotation.Count == 3)
                        cloth.transform.eulerAngles = new Vector3(
                            Convert.ToSingle(rotation[0]),
                            Convert.ToSingle(rotation[1]),
                            Convert.ToSingle(rotation[2]));
                    break;
                // Get* commands are fulfilled by the next Collect snapshot.
            }
        }

        private void SendSceneState(SockTcpBridge bridge)
        {
            bridge.Send("Instance", cloth.sockId, "SockClothAttr", cloth.Snapshot());
            var types = new List<object>();
            var names = new List<object>();
            var joints = new List<object>();
            var forces = new List<object>();
            for (int i = 0; i < robotJoints.Length; i++)
            {
                types.Add(i == 20 || i == 28 ? "PrismaticJoint" : "RevoluteJoint");
                names.Add(RobotJointNames[i]);
                joints.Add(robotJoints[i]);
                forces.Add(0f);
            }
            bridge.Send(
                "Instance",
                cloth.robotId,
                "ControllerAttr",
                new Dictionary<string, object>
                {
                    ["name"] = "Dry-AIREC3",
                    ["names"] = names,
                    ["types"] = types,
                    ["joint_positions"] = joints,
                    ["joint_force"] = forces
                });
            bridge.Send(
                "Instance",
                cloth.humanId,
                "BaseAttr",
                new Dictionary<string, object> { ["name"] = "SeatedHuman" });
            bridge.Send(
                "Env",
                new Dictionary<string, object>
                {
                    ["protocol_version"] = SockClothRuntime.ProtocolVersion,
                    ["simulation_frame"] = Time.frameCount
                });
        }
    }
}
