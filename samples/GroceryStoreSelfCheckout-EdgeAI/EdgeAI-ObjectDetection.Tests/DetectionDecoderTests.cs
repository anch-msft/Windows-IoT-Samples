using EdgeAI_ObjectDetection.Controls;
using EdgeAI_ObjectDetection.Pipeline;

namespace EdgeAI_ObjectDetection.Tests;

public sealed class DetectionDecoderTests
{
    private static readonly string[] Labels = ["person", "bicycle"];

    [Fact]
    public void BoxesAreNormalizedWithinInferenceCrop()
    {
        var detection = YoloInferenceEngine.DecodeDetection(160, 80, 480, 400, .8f, 1, Labels);
        Assert.NotNull(detection);
        Assert.Equal(1, detection.ClassId);
        Assert.Equal("bicycle", detection.Label);
        Assert.Equal(new DetectionBox(.25f, .125f, .5f, .5f), detection.BoundingBox);
    }

    [Fact]
    public void BoxesAreClippedToCrop()
    {
        var detection = YoloInferenceEngine.DecodeDetection(-10, -20, 650, 700, .8f, 0, Labels);
        Assert.NotNull(detection);
        Assert.Equal(new DetectionBox(0, 0, 1, 1), detection.BoundingBox);
    }

    [Theory]
    [InlineData(.49f, 0)]
    [InlineData(float.NaN, 0)]
    [InlineData(1.1f, 0)]
    [InlineData(.8f, -1)]
    [InlineData(.8f, 2)]
    [InlineData(.8f, .5f)]
    [InlineData(.8f, float.PositiveInfinity)]
    public void InvalidOrLowConfidenceRowsAreIgnored(float confidence, float classId) =>
        Assert.Null(YoloInferenceEngine.DecodeDetection(0, 0, 100, 100, confidence, classId, Labels));

    [Theory]
    [InlineData(float.NaN, 0, 100, 100)]
    [InlineData(100, 0, 0, 100)]
    [InlineData(0, 0, 100, 0)]
    [InlineData(650, 0, 700, 100)]
    public void InvalidOrEmptyBoxesAreIgnored(float x1, float y1, float x2, float y2) =>
        Assert.Null(YoloInferenceEngine.DecodeDetection(x1, y1, x2, y2, .8f, 0, Labels));
}
