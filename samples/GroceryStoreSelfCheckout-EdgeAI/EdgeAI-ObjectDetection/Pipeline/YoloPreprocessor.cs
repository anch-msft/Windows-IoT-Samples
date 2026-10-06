using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;

namespace EdgeAI_ObjectDetection.Pipeline;

internal sealed class YoloPreprocessor
{
    public const int InputSize = YoloModelContract.InputSize;

    internal float[] Preprocess(SoftwareBitmap frame, PreviewGeometry crop,
        bool captureImage, out byte[]? imagePixels)
    {
        using SoftwareBitmap rgbaFrame = SoftwareBitmap.Convert(
            frame,
            BitmapPixelFormat.Rgba8,
            BitmapAlphaMode.Ignore);
        int sourceWidth = rgbaFrame.PixelWidth;
        int sourceHeight = rgbaFrame.PixelHeight;
        byte[] rgbaPixels = new byte[sourceWidth * sourceHeight * 4];
        rgbaFrame.CopyToBuffer(rgbaPixels.AsBuffer());

        using Image<Rgba32> image = Image.LoadPixelData<Rgba32>(rgbaPixels, sourceWidth, sourceHeight);
        image.Mutate(operation => operation
            .Crop(new Rectangle(crop.CropX, crop.CropY, crop.CropSize, crop.CropSize))
            .Resize(InputSize, InputSize));

        float[] tensor = new float[3 * InputSize * InputSize];
        byte[]? snapshotPixels = captureImage ? new byte[4 * InputSize * InputSize] : null;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < InputSize; y++)
            {
                Span<Rgba32> row = accessor.GetRowSpan(y);
                for (int x = 0; x < InputSize; x++)
                {
                    int pixelIndex = y * InputSize + x;
                    tensor[pixelIndex] = row[x].R / 255f;
                    tensor[InputSize * InputSize + pixelIndex] = row[x].G / 255f;
                    tensor[2 * InputSize * InputSize + pixelIndex] = row[x].B / 255f;
                    if (snapshotPixels is not null)
                    {
                        // Copy the same resized pixels used by the tensor, not a later camera frame.
                        int offset = pixelIndex * 4;
                        snapshotPixels[offset] = row[x].B;
                        snapshotPixels[offset + 1] = row[x].G;
                        snapshotPixels[offset + 2] = row[x].R;
                        snapshotPixels[offset + 3] = 255;
                    }
                }
            }
        });

        imagePixels = snapshotPixels;
        return tensor;
    }
}
