using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;

namespace EdgeAI_ObjectDetection.Pipeline;

public sealed class YoloPreprocessor
{
    public const int InputSize = 640;

    public YoloModelInput Preprocess(SoftwareBitmap frame)
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
        int cropSize = Math.Min(sourceWidth, sourceHeight);
        int cropX = (sourceWidth - cropSize) / 2;
        int cropY = (sourceHeight - cropSize) / 2;
        image.Mutate(operation => operation
            .Crop(new Rectangle(cropX, cropY, cropSize, cropSize))
            .Resize(InputSize, InputSize));

        float[] tensor = new float[3 * InputSize * InputSize];
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
                }
            }
        });

        return new YoloModelInput(tensor, sourceWidth, sourceHeight, cropX, cropY, cropSize);
    }
}
