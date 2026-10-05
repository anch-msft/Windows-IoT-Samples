using System;
using System.Collections.Generic;

namespace EdgeAI_ObjectDetection.Controls;

/// <summary>Coordinates normalized to the analyzed square, not the full frame or view.</summary>
public readonly record struct DetectionBox(float X, float Y, float Width, float Height);

public sealed record Detection(int ClassId, string Label, float Confidence, DetectionBox BoundingBox);

public sealed class DetectionsUpdatedEventArgs(IReadOnlyList<Detection> detections) : EventArgs
{
    public IReadOnlyList<Detection> Detections
    {
        get;
    } = detections;
}

public sealed class InferenceFaultedEventArgs(Exception exception) : EventArgs
{
    public Exception Exception
    {
        get;
    } = exception;
}
