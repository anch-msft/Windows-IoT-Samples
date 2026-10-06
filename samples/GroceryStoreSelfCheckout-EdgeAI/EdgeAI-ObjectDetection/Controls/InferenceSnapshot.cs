using EdgeAI_ObjectDetection.Pipeline;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;

namespace EdgeAI_ObjectDetection.Controls;

/// <summary>A one-shot result whose image is owned and disposed by the caller.</summary>
public sealed class InferenceSnapshot : IDisposable
{
    private SoftwareBitmap? _image;

    public IReadOnlyList<Detection> Detections { get; }

    /// <summary>The 640x640 model-input image, in BGRA8 premultiplied format for XAML display.</summary>
    public SoftwareBitmap Image => _image ?? throw new ObjectDisposedException(nameof(InferenceSnapshot));

    internal InferenceSnapshot(IReadOnlyList<Detection> detections, byte[] imagePixels)
    {
        Detections = detections;
        _image = SoftwareBitmap.CreateCopyFromBuffer(imagePixels.AsBuffer(),
            BitmapPixelFormat.Bgra8, YoloModelContract.InputSize, YoloModelContract.InputSize,
            BitmapAlphaMode.Premultiplied);
    }

    public void Dispose()
    {
        _image?.Dispose();
        _image = null;
    }
}
