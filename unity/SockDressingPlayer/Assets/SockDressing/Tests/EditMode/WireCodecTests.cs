using System.Collections.Generic;
using NUnit.Framework;

namespace SockDressing.Tests
{
    public sealed class WireCodecTests
    {
        [Test]
        public void RoundTripsPythonCompatibleValues()
        {
            byte[] payload = WireCodec.Encode(
                "Instance",
                1200,
                true,
                0.005f,
                new List<object> { "left", 3 },
                new Dictionary<string, object> { ["valid"] = false });
            List<object> decoded = WireCodec.Decode(payload);
            Assert.That(decoded[0], Is.EqualTo("Instance"));
            Assert.That(decoded[1], Is.EqualTo(1200));
            Assert.That(decoded[2], Is.EqualTo(true));
            Assert.That((float)decoded[3], Is.EqualTo(0.005f).Within(1e-7));
            Assert.That(decoded[4], Is.TypeOf<List<object>>());
            Assert.That(decoded[5], Is.TypeOf<Dictionary<string, object>>());
        }
    }
}
