using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace SockDressing
{
    [DefaultExecutionOrder(-1000)]
    public sealed class SockTcpBridge : MonoBehaviour
    {
        private readonly ConcurrentQueue<List<object>> incoming = new();
        private readonly ConcurrentQueue<object[]> outgoing = new();
        private TcpClient client;
        private NetworkStream stream;
        private Thread receiveThread;
        private volatile bool running;
        private SockSceneController controller;

        public bool Connected => client != null && client.Connected;

        private void Awake()
        {
            controller = GetComponent<SockSceneController>();
            if (controller == null)
                controller = gameObject.AddComponent<SockSceneController>();
            DontDestroyOnLoad(gameObject);
        }

        private void Start()
        {
            int port = CommandLinePort();
            try
            {
                client = new TcpClient { NoDelay = true };
                client.Connect("127.0.0.1", port);
                stream = client.GetStream();
                running = true;
                receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
                receiveThread.Start();
                Send("Env", new Dictionary<string, object>
                {
                    ["scene_init"] = true,
                    ["protocol_version"] = SockClothRuntime.ProtocolVersion
                });
                Send("StepEnd");
            }
            catch (Exception exception)
            {
                Debug.LogError($"Sock TCP bridge could not connect to port {port}: {exception}");
                enabled = false;
            }
        }

        private void Update()
        {
            while (incoming.TryDequeue(out List<object> message))
            {
                if (message.Count == 1 && Convert.ToString(message[0]) == "StepStart")
                {
                    controller.FlushStep(this);
                    Send("StepEnd");
                    continue;
                }
                controller.Enqueue(message);
            }
            while (outgoing.TryDequeue(out object[] message))
                Write(message);
        }

        public void Send(params object[] values)
        {
            outgoing.Enqueue(values);
        }

        private void ReceiveLoop()
        {
            try
            {
                var lengthBytes = new byte[4];
                while (running)
                {
                    ReadExactly(lengthBytes, 4);
                    int length = BitConverter.ToInt32(lengthBytes, 0);
                    if (length <= 0)
                        continue;
                    var payload = new byte[length];
                    ReadExactly(payload, length);
                    incoming.Enqueue(WireCodec.Decode(payload));
                }
            }
            catch (Exception exception)
            {
                if (running)
                    Debug.LogError($"Sock TCP receive loop stopped: {exception}");
            }
        }

        private void Write(object[] values)
        {
            if (stream == null)
                return;
            byte[] payload = WireCodec.Encode(values);
            byte[] length = BitConverter.GetBytes(payload.Length);
            try
            {
                stream.Write(length, 0, length.Length);
                stream.Write(payload, 0, payload.Length);
                stream.Flush();
            }
            catch (Exception exception)
            {
                Debug.LogError($"Sock TCP send failed: {exception}");
            }
        }

        private void ReadExactly(byte[] buffer, int length)
        {
            int offset = 0;
            while (offset < length)
            {
                int count = stream.Read(buffer, offset, length - offset);
                if (count == 0)
                    throw new EndOfStreamException("Python peer disconnected");
                offset += count;
            }
        }

        private static int CommandLinePort()
        {
            foreach (string argument in Environment.GetCommandLineArgs())
            {
                if (argument.StartsWith("-port:", StringComparison.Ordinal) &&
                    int.TryParse(argument.Substring(6), out int port))
                    return port;
            }
            return 5005;
        }

        private void OnDestroy()
        {
            running = false;
            stream?.Close();
            client?.Close();
            if (receiveThread != null && receiveThread.IsAlive)
                receiveThread.Join(200);
        }
    }
}
