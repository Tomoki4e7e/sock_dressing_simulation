using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace SockDressing
{
    public static class WireCodec
    {
        public static List<object> Decode(byte[] payload)
        {
            using var stream = new MemoryStream(payload, false);
            using var reader = new BinaryReader(stream, Encoding.UTF8, false);
            int count = reader.ReadInt32();
            var result = new List<object>(count);
            for (int i = 0; i < count; i++)
                result.Add(ReadObject(reader));
            return result;
        }

        public static byte[] Encode(params object[] values)
        {
            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(values.Length);
                foreach (object value in values)
                    WriteObject(writer, value);
            }
            return stream.ToArray();
        }

        private static object ReadObject(BinaryReader reader)
        {
            string type = ReadString(reader);
            switch (type)
            {
                case "none":
                case "null":
                    return null;
                case "int":
                    return reader.ReadInt32();
                case "float":
                    return reader.ReadSingle();
                case "bool":
                    return reader.ReadByte() != 0;
                case "string":
                    return ReadString(reader);
                case "bytes":
                    return reader.ReadBytes(reader.ReadInt32());
                case "list":
                case "tuple":
                {
                    int count = reader.ReadInt32();
                    var list = new List<object>(count);
                    for (int i = 0; i < count; i++)
                        list.Add(ReadObject(reader));
                    return list;
                }
                case "dict":
                {
                    int count = reader.ReadInt32();
                    var dictionary = new Dictionary<string, object>(count);
                    for (int i = 0; i < count; i++)
                        dictionary[Convert.ToString(ReadObject(reader))] = ReadObject(reader);
                    return dictionary;
                }
                case "array":
                {
                    int rank = reader.ReadInt32();
                    int length = 1;
                    var shape = new int[rank];
                    for (int i = 0; i < rank; i++)
                    {
                        shape[i] = reader.ReadInt32();
                        length *= shape[i];
                    }
                    var values = new float[length];
                    for (int i = 0; i < length; i++)
                        values[i] = reader.ReadSingle();
                    return values;
                }
                default:
                    throw new InvalidDataException($"Unsupported wire type: {type}");
            }
        }

        private static void WriteObject(BinaryWriter writer, object value)
        {
            if (value == null)
            {
                WriteString(writer, "none");
                return;
            }
            switch (value)
            {
                case bool boolean:
                    WriteString(writer, "bool");
                    writer.Write((byte)(boolean ? 1 : 0));
                    return;
                case byte byteNumber:
                    WriteString(writer, "int");
                    writer.Write((int)byteNumber);
                    return;
                case int intNumber:
                    WriteString(writer, "int");
                    writer.Write(intNumber);
                    return;
                case long longNumber:
                    WriteString(writer, "int");
                    writer.Write(checked((int)longNumber));
                    return;
                case float floatNumber:
                    WriteString(writer, "float");
                    writer.Write(floatNumber);
                    return;
                case double doubleNumber:
                    WriteString(writer, "float");
                    writer.Write((float)doubleNumber);
                    return;
                case string text:
                    WriteString(writer, "string");
                    WriteString(writer, text);
                    return;
                case byte[] bytes:
                    WriteString(writer, "bytes");
                    writer.Write(bytes.Length);
                    writer.Write(bytes);
                    return;
                case Vector3 vector:
                    WriteObject(writer, new List<object> { vector.x, vector.y, vector.z });
                    return;
                case Quaternion quaternion:
                    WriteObject(
                        writer,
                        new List<object>
                        {
                            quaternion.x, quaternion.y, quaternion.z, quaternion.w
                        });
                    return;
                case IDictionary dictionary:
                    WriteString(writer, "dict");
                    writer.Write(dictionary.Count);
                    foreach (DictionaryEntry item in dictionary)
                    {
                        WriteObject(writer, Convert.ToString(item.Key));
                        WriteObject(writer, item.Value);
                    }
                    return;
                case IEnumerable enumerable:
                {
                    var list = new List<object>();
                    foreach (object item in enumerable)
                        list.Add(item);
                    WriteString(writer, "list");
                    writer.Write(list.Count);
                    foreach (object item in list)
                        WriteObject(writer, item);
                    return;
                }
                default:
                    throw new InvalidDataException(
                        $"Unsupported wire value: {value.GetType().FullName}");
            }
        }

        private static string ReadString(BinaryReader reader)
        {
            int length = reader.ReadInt32();
            return Encoding.UTF8.GetString(reader.ReadBytes(length));
        }

        private static void WriteString(BinaryWriter writer, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }
    }
}
