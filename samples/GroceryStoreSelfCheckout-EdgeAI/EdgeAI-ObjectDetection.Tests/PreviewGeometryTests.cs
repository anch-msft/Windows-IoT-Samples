using EdgeAI_ObjectDetection.Controls;

namespace EdgeAI_ObjectDetection.Tests;

public sealed class PreviewGeometryTests
{
    [Theory]
    [InlineData(800, 400)]
    [InlineData(400, 800)]
    [InlineData(640, 640)]
    [InlineData(100.1, 333.33)]
    [InlineData(0.5, 0.25)]
    public void StillImageAndNormalizedBoxesFitWithoutCropping(double width, double height)
    {
        var geometry = PreviewGeometry.FitSquare(640, width, height);
        double size = Math.Min(width, height);

        Assert.Equal(0, geometry.CropX);
        Assert.Equal(0, geometry.CropY);
        Assert.Equal(640, geometry.CropSize);
        Assert.Equal(size, geometry.OverlaySize, 8);
        Assert.Equal((width - size) / 2, geometry.OverlayX, 8);
        Assert.Equal((height - size) / 2, geometry.OverlayY, 8);
        Assert.Equal(width / 2, geometry.OverlayX + .5 * geometry.OverlaySize, 8);
        Assert.Equal(height / 2, geometry.OverlayY + .5 * geometry.OverlaySize, 8);
    }

    [Theory]
    [InlineData(0, 400)]
    [InlineData(400, 0)]
    [InlineData(double.NaN, 400)]
    [InlineData(400, double.PositiveInfinity)]
    public void InvalidStillImageLayoutIsRejected(double width, double height) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => PreviewGeometry.FitSquare(640, width, height));

    [Theory]
    [InlineData(640, 480, 640, 640, 1, 80, 0, 480)]
    [InlineData(640, 480, 640, 640, 1.3333333333333333, 80, 0, 480)]
    [InlineData(1920, 1080, 800, 400, 0.4166666666666667, 480, 60, 960)]
    [InlineData(1920, 1080, 400, 800, 0.7407407407407407, 690, 270, 540)]
    [InlineData(1920, 1080, 640, 480, 1, 720, 300, 480)]
    [InlineData(1920, 1080, 640, 480, 0.5, 480, 60, 960)]
    public void CropIsCenteredInsideVisibleCamera(int sw, int sh, double vw, double vh,
        double scale, int x, int y, int size)
    {
        var geometry = PreviewGeometry.Create(sw, sh, vw, vh, scale);
        Assert.Equal(x, geometry.CropX);
        Assert.Equal(y, geometry.CropY);
        Assert.Equal(size, geometry.CropSize);
        Assert.InRange(geometry.OverlayX, -0.000001, vw);
        Assert.InRange(geometry.OverlayY, -0.000001, vh);
        Assert.True(geometry.OverlayX + geometry.OverlaySize <= vw + 0.000001);
        Assert.True(geometry.OverlayY + geometry.OverlaySize <= vh + 0.000001);
    }

    [Fact]
    public void FractionalSizesNeverIncludePixelsOutsidePreview()
    {
        foreach (double width in new[] { 100.1, 333.33, 640.5, 900.9 })
        {
            foreach (double height in new[] { 125.7, 400.4, 800.8 })
            {
                foreach (double scale in new[] { 1.0, 0.5, Math.Min(width / 1920, height / 1080),
            Math.Max(width / 1920, height / 1080) })
                {
                    var crop = PreviewGeometry.Create(1920, 1080, width, height, scale);
                    Assert.InRange(crop.CropX, 0, 1920 - crop.CropSize);
                    Assert.InRange(crop.CropY, 0, 1080 - crop.CropSize);
                    Assert.InRange(crop.OverlayX, -0.000001, width);
                    Assert.InRange(crop.OverlayY, -0.000001, height);
                    Assert.True(crop.OverlayX + crop.OverlaySize <= width + 0.000001);
                    Assert.True(crop.OverlayY + crop.OverlaySize <= height + 0.000001);
                }
            }
        }
    }

    [Fact]
    public void ResizingChangesCropWithoutChangingModelDimensions()
    {
        var first = PreviewGeometry.Create(1920, 1080, 640, 480, 1);
        var resized = PreviewGeometry.Create(1920, 1080, 400, 800, 1);
        Assert.NotEqual(first, resized);
        Assert.Equal(480, first.CropSize);
        Assert.Equal(400, resized.CropSize);
    }

    [Theory]
    [InlineData(0, 480, 1)]
    [InlineData(640, 0, 1)]
    [InlineData(640, 480, 0)]
    [InlineData(double.NaN, 480, 1)]
    [InlineData(640, double.PositiveInfinity, 1)]
    public void InvalidGeometryIsRejected(double width, double height, double scale) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => PreviewGeometry.Create(640, 480, width, height, scale));
}
