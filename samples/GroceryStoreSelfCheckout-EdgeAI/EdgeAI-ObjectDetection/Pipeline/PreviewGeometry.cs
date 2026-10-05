using System;

namespace EdgeAI_ObjectDetection.Pipeline;

internal readonly record struct PreviewGeometry(
    int CropX, int CropY, int CropSize,
    double Scale, double ImageX, double ImageY)
{
    public double OverlayX => ImageX + CropX * Scale;
    public double OverlayY => ImageY + CropY * Scale;
    public double OverlaySize => CropSize * Scale;

    // Round inward so even fractional layout sizes cannot include off-preview pixels.
    public static PreviewGeometry Create(int sourceWidth, int sourceHeight,
        double viewportWidth, double viewportHeight, double scale)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0 ||
            !double.IsFinite(viewportWidth) || !double.IsFinite(viewportHeight) ||
            !double.IsFinite(scale) || viewportWidth <= 0 || viewportHeight <= 0 || scale <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(viewportWidth), "Preview geometry must be finite and positive.");
        }

        double imageX = (viewportWidth - sourceWidth * scale) / 2;
        double imageY = (viewportHeight - sourceHeight * scale) / 2;
        int left = (int)Math.Ceiling(Math.Max(0, -imageX / scale));
        int top = (int)Math.Ceiling(Math.Max(0, -imageY / scale));
        int right = (int)Math.Floor(Math.Min(sourceWidth, (viewportWidth - imageX) / scale));
        int bottom = (int)Math.Floor(Math.Min(sourceHeight, (viewportHeight - imageY) / scale));
        int size = Math.Min(right - left, bottom - top);
        if (size <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(viewportWidth), "Preview must contain at least one source pixel.");
        }

        return new(left + (right - left - size) / 2, top + (bottom - top - size) / 2,
            size, scale, imageX, imageY);
    }
}
