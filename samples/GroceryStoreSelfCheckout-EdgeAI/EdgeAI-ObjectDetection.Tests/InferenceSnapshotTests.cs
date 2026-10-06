using EdgeAI_ObjectDetection.Controls;
using EdgeAI_ObjectDetection.Pipeline;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;

namespace EdgeAI_ObjectDetection.Tests;

public sealed class InferenceSnapshotTests
{
    [Theory]
    [InlineData(80, 0, 640)]
    [InlineData(240, 160, 320)]
    public void SnapshotPixelsMatchEveryModelInputChannel(int cropX, int cropY, int cropSize)
    {
        using var frame = CreateFrame();
        var geometry = new PreviewGeometry(cropX, cropY, cropSize, 1, 0, 0);
        var input = new YoloPreprocessor().Preprocess(frame, geometry, true, out var pixels);
        Assert.NotNull(pixels);

        using var snapshot = new InferenceSnapshot(Array.Empty<Detection>(), pixels);
        Assert.Equal(640, snapshot.Image.PixelWidth);
        Assert.Equal(640, snapshot.Image.PixelHeight);
        Assert.Equal(BitmapPixelFormat.Bgra8, snapshot.Image.BitmapPixelFormat);
        Assert.Equal(BitmapAlphaMode.Premultiplied, snapshot.Image.BitmapAlphaMode);
        byte[] actual = new byte[640 * 640 * 4];
        snapshot.Image.CopyToBuffer(actual.AsBuffer());

        for (int i = 0; i < 640 * 640; i++)
        {
            Assert.Equal(input[i], actual[i * 4 + 2] / 255f);
            Assert.Equal(input[640 * 640 + i], actual[i * 4 + 1] / 255f);
            Assert.Equal(input[2 * 640 * 640 + i], actual[i * 4] / 255f);
            Assert.Equal((byte)255, actual[i * 4 + 3]);
        }

        if (cropSize == 640)
        {
            Assert.Equal((byte)cropX, actual[2]);
            Assert.Equal((byte)cropY, actual[1]);
            Assert.Equal((byte)((cropX + 639) % 256), actual[^2]);
            Assert.Equal((byte)((cropY + 639) % 256), actual[^3]);
        }
    }

    [Fact]
    public void StreamingDoesNotAllocateSnapshotPixels()
    {
        using var frame = CreateFrame();
        var preprocessor = new YoloPreprocessor();
        var geometry = new PreviewGeometry(80, 0, 640, 1, 0, 0);
        var streaming = preprocessor.Preprocess(frame, geometry, false, out var streamingPixels);
        var oneShot = preprocessor.Preprocess(frame, geometry, true, out var oneShotPixels);

        Assert.Null(streamingPixels);
        Assert.NotNull(oneShotPixels);
        Assert.Equal(oneShot, streaming);
    }

    [Fact]
    public void SnapshotOwnsImageAndKeepsDetectionsAfterDisposal()
    {
        byte[] pixels;
        using (var frame = CreateFrame())
        {
            new YoloPreprocessor().Preprocess(frame, new PreviewGeometry(80, 0, 640, 1, 0, 0),
                true, out var capturedPixels);
            pixels = capturedPixels!;
        }

        var detection = DetectionDecoder.Decode(160, 80, 480, 400, .8f, 0, new[] { "person" });
        Assert.NotNull(detection);
        IReadOnlyList<Detection> detections = Array.AsReadOnly(new[] { detection });
        using var snapshot = new InferenceSnapshot(detections, pixels);
        Array.Clear(pixels);
        snapshot.Image.CopyToBuffer(pixels.AsBuffer());
        Assert.Equal((byte)80, pixels[2]);
        Assert.Equal((byte)255, pixels[3]);
        Assert.Same(detections, snapshot.Detections);
        Assert.Equal(new DetectionBox(.25f, .125f, .5f, .5f), snapshot.Detections[0].BoundingBox);

        snapshot.Dispose();
        snapshot.Dispose();
        Assert.Throws<ObjectDisposedException>(() => snapshot.Image);
        Assert.Same(detections, snapshot.Detections);
    }

    private static SoftwareBitmap CreateFrame()
    {
        byte[] pixels = new byte[800 * 640 * 4];
        for (int y = 0; y < 640; y++)
        {
            for (int x = 0; x < 800; x++)
            {
                int offset = (y * 800 + x) * 4;
                pixels[offset] = (byte)((x + y) % 256);
                pixels[offset + 1] = (byte)(y % 256);
                pixels[offset + 2] = (byte)(x % 256);
                pixels[offset + 3] = 255;
            }
        }
        return SoftwareBitmap.CreateCopyFromBuffer(pixels.AsBuffer(), BitmapPixelFormat.Bgra8,
            800, 640, BitmapAlphaMode.Ignore);
    }
}
